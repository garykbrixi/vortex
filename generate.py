# Copyright (c) 2024, Michael Poli.

# Copyright (c) Together
# This software is distributed under the terms of the Apache License, Version 2.0
# Author: Michael Poli

import argparse
import os

import torch
import yaml

from vortex.model.generation import Generator
from vortex.model.model import StripedHyena
from vortex.model.sample import sample
from vortex.model.tokenizer import HFAutoTokenizer, CharLevelTokenizer
from vortex.model.utils import dotdict, print_rank_0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run StripedHyena Model")
    parser.add_argument("--config_path", required=True, help="Path to configuration file")
    parser.add_argument("--checkpoint_path", default=None, help="Path to checkpoint file")
    parser.add_argument("--num_tokens", default=8000, help="Number of tokens to generate.")
    parser.add_argument("--input_file", default="./prompt.txt", help="Path to prompt file.")
    parser.add_argument("--temperature", default=1, type=float)
    parser.add_argument("--repetition_penalty", default=1, type=float)
    parser.add_argument("--penalty_alpha", default=0, type=float)
    parser.add_argument("--top_k", default=8, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument(
        "--cached_generation",
        action="store_true",
        help="Use caching to speed up generation.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Dry run the generation.")
    parser.add_argument("--debug", action="store_true", help="Debug mode.")

    torch.set_printoptions(precision=4, threshold=5)

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    args = parser.parse_args()

    config = dotdict(yaml.load(open(args.config_path), Loader=yaml.FullLoader))

    if config.tokenizer_type == "CharLevelTokenizer":
        tokenizer = CharLevelTokenizer(config.vocab_size)
    else:
        tokenizer = HFAutoTokenizer(config.vocab_file)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config.print_activations = True
    m = StripedHyena(config).to(torch.float32)

    if not args.dry_run:
        if args.checkpoint_path:
            state_dict = torch.load(args.checkpoint_path)
            # inv_freq are instantiated as parameters
            m.custom_load_state_dict(state_dict, strict=False)

    m.to_bfloat16_except_pr_lc()

    for name, module in m.named_modules():
        if next(module.parameters(), None) is not None:
            print(f"{name}: {next(module.parameters()).device}")

    print_rank_0(f"Number of parameters: {sum(p.numel() for p in m.parameters())}")

    with open(args.input_file, "r") as f:
        input_string = f.read()
    # input_string = 'TGGATAAGAAATACTCAATAGGCTTAGATATCGGCACAAATAGCGTCGGATGGGCGGTGATCACTGATGATTATAAGGTTCCGTCTAAAAAGTTCAAGGTTCTGGGAAATACAGACCGCCACAGTATCAAAAAAAATCTTATAGGGGCTCTTTTATTTGACAGTGGAGAGACAGCGGAAGCGACTCGTCTCAAACGGACAGCTCGTAGAAGGTATACACGTCGGAAGAATCGTATTTGTTATCTACAGGAGATTTTTTCAAATGAGATGGCGAAAGTAGATGATAGTTTCTTTCATCGACTTGAAGAGTCTTTTTTGGTGGAAGAAGACAAGAAGCATGAACGTCATCCTATTTTTGGAAATATAGTAGATGAAGTTGCTTATCATGAGAAATATCCAACTATCTATCATCTGCGAAAAAAATTGGCAGATTCTACTGATAAAGCGGATTTGCGCTTAATCTATTTGGCCTTAGCGCATATGATTAAGTTTCGTGGTCATTTTTTGATTGAGGGAGATTTAAATCCTGATAATAGTGATGTGGACAAACTATTTATCCAGTTGGTACAAACCTACAATCAATTATTTGAAGAAAACCCTATTAACGCAAGTAGAGTAGATGCTAAAGCGATTCTTTCTGCACGATTGAGTAAATCAAGACGATTAGAAAATCTCATTGCTCAGCTCCCCGGTGAGAAGAAAAATGGCTTATTTGGGAATCTCATTGCTTTGTTATTGGGATTGACCCCTAATTTTAAATCAAATTTTGATTTGGCAGAAGATGCTAAATTACAGCTTTCAAAAGATACTTACGATGATGATTTAGATAATTTATTGGCGCAAATTGGAGATCAATATGCTGATTTGTTTTTGGCAGCTAAGAATTTATCAGATGCTATTTTACTTTCAGATATCCTAAGAGTAAATAGTGAAATAACTAAGGCTCCCCTATCAGCTTCAATGATTAAACGCTACGATGAACATCATCAAGACTTGACTCTTTTAAAAGCTTTAGTTCGACAACAACTTCCAGAAAAGTATAAAGAAATCTTTTTTGATCAATCAAAAAACGGATATGCAGGTTATATTGATGGGGGAGCTAGCCAAGAAGAATTTTATAAATTTATCAAACCAATTTTAGAAAAAATGGATGGTACTGAGGAATTATTGGCGAAACTAAATCGTGAAGATTTGCTGCGCAAGCAACGGACCTTTGACAACGGCTCTATTCCCCATCAAATTCACTTGGGTGAGCTGCATGCTATTTTGAGAAGACAAGAAGACTTTTATCCATTTTTAAAAGACAATCGTGAGAAGATTGAAAAAATCTTGACTTTTCGAATTCCTTATTATGTTGGTCCATTGGCGCGTGGCAATAGTCGTTTTGCATGGATGACTCGGAAGTCTGAAGAAACAATTACCCCATGGAATTTTGAAGAAGTTGTCGATAAAGGTGCTTCAGCTCAATCATTTATTGAACGCATGACAAACTTTGATAAAAATCTTCCAAATGAAAAAGTACTACCAAAACATAGTTTGCTTTATGAGTATTTTACGGTTTATAACGAATTGACAAAGGTCAAATATGTTACTGAGGGAATGCGAAAACC'
    print_rank_0(f"Prompt: {input_string}", end="\n\n")

    inputs = tokenizer.tokenize(input_string)
    inputs = torch.tensor(inputs, dtype=torch.long, device='cuda:0')[None]
    logits_vortex = m.forward(inputs)
    predicted_indices = logits_vortex[0].argmax(dim=-1)
    print(inputs)
    print(predicted_indices)
    breakpoint()

    with torch.inference_mode():
        g = Generator(m, tokenizer, top_k=args.top_k, top_p=args.top_p, temperature=args.temperature)
        g.generate(
            num_tokens=args.num_tokens,
            cached_generation=args.cached_generation,
            input_string=input_string,
            device=device,
            verbose=True,
            print_generation=args.debug,
            max_seqlen=8192,
        )
