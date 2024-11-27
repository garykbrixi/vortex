# Copyright (c) 2024, Michael Poli.

# Copyright (c) Together
# This software is distributed under the terms of the Apache License, Version 2.0
# Author: Michael Poli

# Barebones generation class for standalone inference.

import torch

from vortex.model.sample import sample
from vortex.model.tokenizer import CharLevelTokenizer
from vortex.model.utils import print_rank_0


class Generator:
    def __init__(self, model, tokenizer, top_k=50, top_p=0.7, temperature=1):
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.untils = ["\n\n"]

    def generate(
        self,
        device="cuda:0",  # Default to first device but mainly for tracking/metrics
        input_string=None,
        input_ids=None,
        num_tokens=32,
        cached_generation=False,
        print_generation=True,
        verbose=False,
        skip_special_tokens=False,
        stop_at_eos=True,
        max_seqlen=None,
    ):
        # EOS token handling - always keep on first device
        if isinstance(self.tokenizer.eos, int):
            eos_token_ids = torch.LongTensor([self.tokenizer.eos]).to("cuda:0")
        else:
            eos_token_ids = self.tokenizer.tokenize(self.tokenizer.eos).to("cuda:0")

        # Input processing - ensure starting on first device
        if input_ids is None:
            input = self.tokenizer.tokenize(input_string)
            if isinstance(input, list):
                input = torch.LongTensor(input).unsqueeze(0).to("cuda:0")
            else:
                input = input.unsqueeze(0).to("cuda:0")
        else:
            input = input_ids.to("cuda:0")
        x = input

        print(input_string, input)

        if max_seqlen is not None:
            x = x[:, -max_seqlen:]

        prompt_len = x.shape[-1]
        num_tokens = int(num_tokens)
        tot_length = prompt_len + num_tokens
        batch_size = x.shape[0]

        # Keep generation outputs on first device
        generation = torch.empty(
            x.shape[0],
            num_tokens,
            dtype=torch.long,
            device="cuda:0",
        )

        scores = torch.empty(
            x.shape[0],
            num_tokens,
            self.tokenizer.vocab_size,
            dtype=torch.float,
            device="cuda:0",
        )

        if cached_generation:
            inference_params_dict_out = self.model.initialize_inference_params()
            # Set batch size for each block type on their respective devices
            for block_type in ["mha", "hcl", "hcm", "hcs"]:
                for layer_idx, block in enumerate(self.model.blocks):
                    if block_type in str(block.__class__).lower():
                        device = self.model.block_idx_to_device[layer_idx]
                        with torch.device(device):
                            inference_params_dict_out[block_type].max_batch_size = batch_size
        else:
            inference_params_dict_out = None

        if verbose:
            # Track memory across all devices
            total_mem = sum(torch.cuda.memory_allocated(f"cuda:{i}") 
                        for i in range(torch.cuda.device_count())) / 1e9
            print_rank_0(f"Total memory after tokenization across devices: {total_mem:.2f} GB")
            print_rank_0("Starting generation...")
            if input_string is not None:
                print_rank_0("Prompt: " + input_string)
            else:
                print_rank_0(f"Prompt ids: {input_ids} {input_ids.shape}")

        for i in range(int(num_tokens)):
            post_prefill = cached_generation and i > 0
            
            if post_prefill:
                x = x[:, -1:]
                seqlen_offset = inference_params_dict_out["mha"].seqlen_offset

                if seqlen_offset == 0:
                    seqlen_offset = input.shape[-1]
                    # Update seqlen offset for each block type on their respective devices
                    for block_type in ["mha", "hcl", "hcm", "hcs"]:
                        for layer_idx, block in enumerate(self.model.blocks):
                            if block_type in str(block.__class__).lower():
                                device = self.model.block_idx_to_device[layer_idx]
                                with torch.device(device):
                                    inference_params_dict_out[block_type].seqlen_offset = seqlen_offset
                else:
                    # Increment seqlen offset for each block type on their respective devices
                    for block_type in ["mha", "hcl", "hcm", "hcs"]:
                        for layer_idx, block in enumerate(self.model.blocks):
                            if block_type in str(block.__class__).lower():
                                device = self.model.block_idx_to_device[layer_idx]
                                with torch.device(device):
                                    inference_params_dict_out[block_type].seqlen_offset += 1

            # Forward pass moves tensors through pipeline automatically
            with torch.no_grad():
                logits, inference_params_dict_out = self.model(
                    x,
                    inference_params_dict=inference_params_dict_out,
                )

            # Ensure logits are on first device for sampling
            logits = logits.to("cuda:0")
            last_logits = logits[:, -1]
            
            if print_generation and verbose and batch_size == 1:
                print(last_logits.shape, last_logits.min(), last_logits.max(), last_logits)

            new_idx = sample(
                last_logits,
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
            )

            if stop_at_eos and (generation[0, -2:] == eos_token_ids).all():
                print_rank_0("Stopping generation at EOS")

            if print_generation and verbose and batch_size == 1:
                print_rank_0(
                    f"{self.tokenizer.detokenize([new_idx.item()])}",
                    end=" ",
                )

            scores[:, i] = last_logits
            generation[:, i] = new_idx

            if post_prefill:
                x = new_idx[:, None].to("cuda:0")  # Ensure next input starts on first device
            else:
                x = torch.cat([x, new_idx[:, None]], dim=-1).to("cuda:0")

        if verbose:
            kwargs = {}
            if not isinstance(self.tokenizer, CharLevelTokenizer):
                kwargs["skip_special_tokens"] = skip_special_tokens
            y = self.tokenizer.detokenize_batch(generation[:, : i + 1], **kwargs)

            for until in self.untils:
                if until in y:
                    y = y.split(until)[0]
                    break

            print_rank_0(f"\nInput: {input_string}, Output: {y}")

            # Report memory usage across all devices
            device_mems = [torch.cuda.memory_allocated(f"cuda:{i}") / 1e9 
                        for i in range(torch.cuda.device_count())]
            total_mem = sum(device_mems)
            print_rank_0(f"Total memory after generation: {total_mem:.2f} GB")
            for i, mem in enumerate(device_mems):
                print_rank_0(f"Memory on cuda:{i}: {mem:.2f} GB")

        return generation[:, : i + 1], scores[:, : i + 1]