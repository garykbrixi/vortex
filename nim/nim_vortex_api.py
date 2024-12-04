#!/usr/bin/env python3

# Copyright (c) 2024, Michael Poli.

# Copyright (c) Together
# This software is distributed under the terms of the Apache License, Version 2.0
# Author: Michael Poli

"""
This file is what NIM uses to call into Vortex.

- biology/arc/evo2/generate endpoint calls run_generation()
- biology/arc/evo2/embeddings endpoint calls run_embeddings()

"""

import torch
import yaml

from dataclasses import dataclass
from functools import lru_cache

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
    cached_generation=False, # TODO: likely not tested
) -> GenerationOutput:
    from vortex.model.generation import Generator

    m, tokenizer, device = get_model(
        config_path=config_path,
        dry_run=dry_run,
        checkpoint_path=checkpoint_path,
    )

    print(f"Generation Prompt: {input_string}")

    with torch.inference_mode():
        g = Generator(m, tokenizer, top_k=top_k, top_p=top_p, temperature=temperature)
        tokens, logits = g.generate(
            num_tokens=num_tokens,
            cached_generation=cached_generation,
            input_string=input_string,
            device=device,
            verbose=True,
            print_generation=True,
            max_seqlen=8192,
        )
        sequence = tokenizer.detokenize_batch(tokens)[0]
        return GenerationOutput(
            sequence=sequence,
            logits=logits[0].tolist(),
            sampled_probs=to_sampled_probs(sequence, logits[0]),
        )

def test_vortex_generation():
    out = str(run_generation("ATCG"))
    print("Test generation: ", out[:100], "...", out[-100:])

def run_embeddings(
    input_string,
    *,
    layer_index=1,
    config_path="shc-evo2-7b-8k-2T-v2.yml",
    dry_run=True,
    checkpoint_path=None,
):
    m, tokenizer, device = get_model(
        config_path=config_path,
        dry_run=dry_run,
        checkpoint_path=checkpoint_path,
    )

    print(f"Embeddings Prompt: {input_string}")

    with torch.no_grad():
        from torch import nn
        m = nn.Sequential(*list(m.children())[:layer_index]) # TODO: catch and sanitize index error
        x = tokenizer.tokenize(input_string)
        x = torch.LongTensor(x).unsqueeze(0).to(device)
        t = m(x)

    from io import BytesIO
    with BytesIO() as buffer:
        from numpy import savez_compressed
        savez_compressed(
            buffer,
            input_tokens=x.cpu().numpy(),
            embeddings=t.cpu().float().numpy(),
        )
        from base64 import encodebytes
        return encodebytes(buffer.getvalue())

def test_vortex_embeddings():
    out = str(run_embeddings("ATCG"))
    print("Test embeddings: ", out[:100], "...", out[-100:])

def test_all():
    test_vortex_generation()
    test_vortex_embeddings()

if __name__ == "__main__":
    from pathlib import Path
    from sys import argv, path as pythonpath
    from os import chdir

    script_dir = Path(argv[0]).resolve().parent
    project_dir = script_dir / ".."
    chdir(project_dir / "configs")
    pythonpath.append(str(project_dir))

    test_all()
