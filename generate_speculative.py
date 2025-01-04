import argparse
import os

import torch
import yaml

from vortex.model.speculative import SpeculativeGenerator
from vortex.model.model import StripedHyena
from vortex.model.sample import sample, modify_logits
from vortex.model.tokenizer import HFAutoTokenizer, CharLevelTokenizer
from vortex.model.utils import dotdict, print_rank_0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speculative decoding forStripedHyena Models")
    parser.add_argument("--target_config_path", required=True, help="Path to configuration file")
    parser.add_argument("--target_checkpoint_path", default=None, help="Path to checkpoint file")
    parser.add_argument("--draft_config_path", required=True, help="Path to configuration file")
    parser.add_argument("--draft_checkpoint_path", default=None, help="Path to checkpoint file")
    parser.add_argument("--gamma", default=4, type=int, help="Number of speculative samples to take")
    parser.add_argument("--num_tokens", default=84, help="Number of tokens to generate.")
    parser.add_argument("--input_file", default="./prompt.txt", help="Path to prompt file.")
    parser.add_argument("--temperature", default=1, type=float)
    parser.add_argument("--top_k", default=8, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_seqlen", default=8192, type=int)
    parser.add_argument(
        "--cached_generation",
        action="store_true",
        help="Use caching to speed up generation.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Dry run the generation.")
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    parser.add_argument("--skip_special_tokens", action="store_true", help="Skip special tokens.")
    parser.add_argument("--no_stop_at_eos", action="store_true", help="Stop at EOS.")

    torch.set_printoptions(precision=4, threshold=5)

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    args = parser.parse_args()

    if args.cached_generation:
        raise NotImplementedError("Cached generation is not currentlysupported for speculative generation")

    target_config = dotdict(yaml.load(open(args.target_config_path), Loader=yaml.FullLoader))
    draft_config = dotdict(yaml.load(open(args.draft_config_path), Loader=yaml.FullLoader))

    if target_config.tokenizer_type == "CharLevelTokenizer":
        tokenizer = CharLevelTokenizer(target_config.vocab_size)
    else:
        tokenizer = HFAutoTokenizer(target_config.vocab_file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.device(device):
        target_model = StripedHyena(target_config).to(torch.float32)
        draft_model = StripedHyena(draft_config).to(torch.float32)

    if not args.dry_run:
        if args.target_checkpoint_path:
            # inv_freq are instantiated as parameters
            target_model.custom_load_state_dict(torch.load(args.target_checkpoint_path, map_location=device), strict=False)

        target_model.to_bfloat16_except_pr_lc()

        print_rank_0(f"Number of parameters: {sum(p.numel() for p in target_model.parameters())}")

        if args.draft_checkpoint_path:
            draft_model.custom_load_state_dict(torch.load(args.draft_checkpoint_path, map_location=device), strict=False)

        draft_model.to_bfloat16_except_pr_lc()

        print_rank_0(f"Number of parameters: {sum(p.numel() for p in draft_model.parameters())}")

    with open(args.input_file, "r") as f:
        input_string = f.read()
    print_rank_0(f"Prompt: {input_string}", end="\n\n")

    with torch.inference_mode():
        g = SpeculativeGenerator(
            draft_model=draft_model,
            target_model=target_model,
            tokenizer=tokenizer,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            gamma=args.gamma,
        )
        g.generate(
            input_string=input_string,
            device=device,
            num_tokens=args.num_tokens,
            cached_generation=args.cached_generation,
            print_generation=args.debug,
            verbose=True,
            skip_special_tokens=args.skip_special_tokens,
            stop_at_eos=not args.no_stop_at_eos,
            max_seqlen=args.max_seqlen,
        )

