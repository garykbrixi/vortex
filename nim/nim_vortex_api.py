#!/usr/bin/env python3

# Copyright (c) 2024, Michael Poli.

# Copyright (c) Together
# This software is distributed under the terms of the Apache License, Version 2.0
# Author: Michael Poli

"""
This file is what NIM uses to call into Vortex.

- biology/arc/evo2/generate endpoint calls run_generation()
- biology/arc/evo2/forward endpoint calls run_forward()

"""

import torch
import yaml

from dataclasses import dataclass
from functools import lru_cache
from os import getenv

import logging
log = logging.getLogger(__name__)

def bool_env(env, default="", *, return_optional=False):
    if getenv(env) is None and return_optional:
        return None
    return getenv(env, str(default)).lower() in ["y", "yes", "1", "t", "true"]

@lru_cache
def is_fp8_supported():
    from transformer_engine.pytorch.fp8 import check_fp8_support
    log.info(f"{check_fp8_support()=}")
    return check_fp8_support()[0]

@lru_cache
def should_use_cached_generation():
    env = bool_env("NIM_EVO2_CACHED_GENERATION", return_optional=True)
    if env is not None:
        log.info(f"Set cached generation preference from env variable: {env=}")
        return env

    mem_gb = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024 // 1024
    # So far cached generation is only practical on A100/H100 and above.
    if mem_gb > 60:
        log.info(f"Will use cached generation, {mem_gb=}")
        return True
    gpus = torch.cuda.device_count()
    if gpus >= 2:
        log.info(f"Will use cached generation, {gpus=}")
        return True
    log.info(f"Will not use cached generation, {mem_gb=}")
    return False

@lru_cache(maxsize=1)
def get_model(*,
    config_path,
    dry_run,
    checkpoint_path,
):
    # Make sure we only have one model in memory at a time. (lru_cache creates
    # new netry before deleting old, which will double peak GPU mem usage.)
    get_model.cache_clear()
    import gc
    gc.collect()

    from vortex.model.model import StripedHyena
    from vortex.model.tokenizer import HFAutoTokenizer, CharLevelTokenizer
    from vortex.model.utils import dotdict

    torch.set_printoptions(precision=2, threshold=5)

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    config = dotdict(yaml.load(open(config_path), Loader=yaml.FullLoader))

    fp8_env = bool_env("NIM_EVO2_FP8", return_optional=True)
    if fp8_env is not None:
        log.info(f"Set fp8 preference from env variable: {fp8_env=}")
        config.use_fp8_input_projections = fp8_env
    elif config.use_fp8_input_projections and not is_fp8_supported():
        log.info("fp8 forced off as the support is not present")
        config.use_fp8_input_projections = False

    if config.tokenizer_type == "CharLevelTokenizer":
        tokenizer = CharLevelTokenizer(config.vocab_size)
    else:
        tokenizer = HFAutoTokenizer(config.vocab_file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.device(device):
        m = StripedHyena(config)

    if not dry_run:
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location=device)
            # inv_freq are instantiated as parameters
            m.custom_load_state_dict(state_dict, strict=False)

    m.to_bfloat16_except_pr_lc()

    print(f"Number of parameters: {sum(p.numel() for p in m.parameters())}")
    return m, tokenizer, device

def to_sampled_probs(sequence, logits) -> list[float]:
    probs = torch.softmax(logits, dim=-1)
    return [probs[pos][ord(c)].item() for pos, c in enumerate(sequence)]

@dataclass(kw_only=True)
class GenerationOutput:
    sequence: str
    logits: list[float]
    sampled_probs: list[float]

def run_generation(
    input_string,
    *,
    num_tokens=5,
    top_k=4,
    top_p=1,
    temperature=1,
    config_path="shc-evo2-7b-8k-2T-v2.yml",
    dry_run=True,
    checkpoint_path=None,
    timeout_s=int(getenv("NIM_EVO2_TIMEOUT_S", 600)),
) -> GenerationOutput:
    from vortex.model.generation import generate

    m, tokenizer, device = get_model(
        config_path=config_path,
        dry_run=dry_run,
        checkpoint_path=checkpoint_path,
    )

    print(f"Generation Prompt: {input_string}")

    from time import monotonic
    deadline = monotonic() + timeout_s
    def token_callback(i):
        if monotonic() > deadline:
            raise TimeoutError(f"Timed out on {i}th token out of {num_tokens} requested. Allowed to run for {timeout_s} seconds.")

    with torch.inference_mode():
        ret = generate(
            prompt_seqs=[input_string],
            n_tokens=num_tokens,
            model=m,
            tokenizer=tokenizer,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            cached_generation=should_use_cached_generation(),
            verbose=2,
            token_callback=token_callback,
            device=device,
        )
        return GenerationOutput(
            sequence=ret.sequences[0],
            logits=ret.logits[0][0].tolist(),
            sampled_probs=to_sampled_probs(ret.sequences[0], ret.logits[0][0]),
        )

def test_vortex_generation():
    out = str(run_generation("ATCG"))
    print("Test generation: ", out[:100], "...", out[-100:])

class LayerHook:
    def __init__(self, *, layer_name, store):
        self.layer_name = layer_name
        self.store = store
    def hook_fn(self, module, input, output):
        self.store[self.layer_name + ".output"] = output.cpu().tolist()

def run_forward(
    input_string,
    *,
    layers=["embedding_layer", "unembed", "blocks.0.mlp.l1"],
    config_path="shc-evo2-7b-8k-2T-v2.yml",
    dry_run=True,
    checkpoint_path=None,
):
    m, tokenizer, device = get_model(
        config_path=config_path,
        dry_run=dry_run,
        checkpoint_path=checkpoint_path,
    )

    print(f"Forward Prompt: {input_string}")

    store = {}
    hooks = []

    try:
        for l in layers:
            hooks.append(
                m.get_submodule(l).register_forward_hook(
                    LayerHook(layer_name=l, store=store).hook_fn
                )
            )

        with torch.no_grad():
            from torch import nn
            x = tokenizer.tokenize(input_string)
            x = torch.LongTensor(x).unsqueeze(0).to(device)
            m(x)
    finally:
        for h in hooks:
            h.remove()

    from io import BytesIO
    with BytesIO() as buffer:
        from numpy import savez_compressed
        savez_compressed(buffer, **store)
        from base64 import encodebytes
        return encodebytes(buffer.getvalue())

def test_vortex_forward():
    from base64 import decodebytes
    from io import BytesIO
    from numpy import load

    octs = run_forward("ATCG")
    deserialized = dict(load(BytesIO(decodebytes(octs))))
    print("Test forward: ", octs[:100], "...", octs[-100:], deserialized)

def test_all():
    test_vortex_generation()
    test_vortex_forward()

if __name__ == "__main__":
    from pathlib import Path
    from sys import argv, path as pythonpath
    from os import chdir

    script_dir = Path(argv[0]).resolve().parent
    project_dir = script_dir / ".."
    chdir(project_dir / "configs")
    pythonpath.append(str(project_dir))

    test_all()
