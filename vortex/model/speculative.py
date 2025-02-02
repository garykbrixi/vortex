import torch

from vortex.model.sample import sample, modify_logits
from vortex.model.tokenizer import CharLevelTokenizer
from vortex.model.utils import print_rank_0


@torch.no_grad()
def find_candidate_pred_tokens(input_ids, max_ngram_size=3, num_pred_tokens=10, verbose=False):
    input_length = input_ids.size(1)

    # Ensure max_ngram_size and num_pred_tokens are valid
    if max_ngram_size <= 0 or num_pred_tokens <= 0 or max_ngram_size > input_length:
        raise ValueError("Invalid max_ngram_size or num_pred_tokens")

    for ngram_size in range(max_ngram_size, 0, -1):
        # Extract the last n tokens as our search ngram
        ngram = input_ids[0, -ngram_size:].tolist()

        if verbose:
            print(f"Ngram: {ngram}")

        # Create sliding windows of size ngram_size
        windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)

        # Convert ngram to a tensor for comparison
        ngram_tensor = torch.tensor(ngram, device=input_ids.device).unsqueeze(0)

        # Find where the windows match the ngram
        matches = (windows == ngram_tensor).all(dim=2)

        # Get the indices of matches
        match_indices = matches.nonzero(as_tuple=True)[1]

        if verbose:
            print(f"Match indices: {match_indices}")

        # Iterate through match indices to find a valid continuation
        for idx in match_indices:
            start_idx = idx + ngram_size
            end_idx = start_idx + num_pred_tokens
            # Ensure we don't go beyond the length of input_ids and avoid self-match
            if end_idx <= input_length and start_idx < input_length - ngram_size:
                return input_ids[0, start_idx:end_idx]

    # If no match is found, return an empty tensor
    return torch.tensor([], dtype=torch.long, device=input_ids.device)


class SpeculativeNGramGenerator:
    def __init__(self, model, tokenizer, top_k=50, top_p=0.7, temperature=1, max_ngram_size=3, num_pred_tokens=10):
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.untils = ["\n\n"]
        self.max_ngram_size = max_ngram_size
        self.num_pred_tokens = num_pred_tokens

    def generate(
        self,
        device,
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
        # Prepare input
        if isinstance(self.tokenizer.eos, int):
            eos_token_ids = torch.LongTensor([self.tokenizer.eos]).to(device)
        else:
            # is a tensor
            eos_token_ids = self.tokenizer.tokenize(self.tokenizer.eos).to(device)

        if input_ids is None:
            input = self.tokenizer.tokenize(input_string)
            if isinstance(input, list):
                input = torch.LongTensor(input).unsqueeze(0).to(device)
            # is a tensor
            else:
                input = input.unsqueeze(0).to(device)

        else:
            input = input_ids
            
        x = input

        if max_seqlen is not None:
            x = x[:, -max_seqlen:]

        prompt_len = x.shape[-1]

        num_tokens = int(num_tokens)
        tot_length = prompt_len + num_tokens
        batch_size = x.shape[0]

        assert batch_size == 1, "Batch size must be 1"

        # initialize final output
        generation = torch.empty(
            x.shape[0],
            num_tokens,
            dtype=torch.long,
            device=x.device,
        )

        scores = torch.empty(
            x.shape[0],
            num_tokens,
            self.tokenizer.vocab_size,
            dtype=torch.float,
            device=x.device,
        )

        if cached_generation:
            inference_params_dict_out = self.model.initialize_inference_params()
            inference_params_dict_out["mha"].max_batch_size = batch_size
            inference_params_dict_out["hcl"].max_batch_size = batch_size
            inference_params_dict_out["hcm"].max_batch_size = batch_size
            inference_params_dict_out["hcs"].max_batch_size = batch_size
        else:
            inference_params_dict_out = None

        if verbose:
            mem_after_tok = torch.cuda.memory_allocated(device=x.device) / 1e9
            print_rank_0(f"Memory after tokenization: {mem_after_tok} GB")
            print_rank_0("Starting generation...")
            if input_string is not None:
                print_rank_0("Prompt: " + input_string)
            else:
                print_rank_0(f"Prompt ids: {input_ids} {input_ids.shape}")

        # main loop
        
        i = 0

        while i < int(num_tokens):
            # post_prefill = cached_generation and i > 0

            # if post_prefill:
            #     # if using cached generation and is not the first token, we only need the last token
            #     x = x[:, -1:]
            #     seqlen_offset = inference_params_dict_out["mha"].seqlen_offset

            #     if seqlen_offset == 0:
            #         seqlen_offset = input.shape[-1]
            #         inference_params_dict_out["mha"].seqlen_offset = seqlen_offset
            #         inference_params_dict_out["hcl"].seqlen_offset = seqlen_offset
            #         inference_params_dict_out["hcm"].seqlen_offset = seqlen_offset
            #         inference_params_dict_out["hcs"].seqlen_offset = seqlen_offset
            #     else:
            #         inference_params_dict_out["mha"].seqlen_offset += 1
            #         inference_params_dict_out["hcl"].seqlen_offset += 1
            #         inference_params_dict_out["hcm"].seqlen_offset += 1
            #         inference_params_dict_out["hcs"].seqlen_offset += 1
            if cached_generation:
                raise NotImplementedError("Cached generation + speculative decoding not yet supported.")

            # get candidate tokens by ngram matching
            draft_tokens = find_candidate_pred_tokens(x, self.max_ngram_size, self.num_pred_tokens, verbose=verbose)

            # follows apoorvumang/prompt-lookup-decoding and places a pad token if no candidate tokens are found
            if len(draft_tokens) == 0:
                draft_tokens = torch.tensor([[self.tokenizer.pad_idx]], dtype=torch.long, device=x.device)
            
            # NOTE: this only works for batch size 1
            draft_tokens = draft_tokens.unsqueeze(0).to(device)  # (1, num_pred_tokens)
            draft_size = draft_tokens.size(1)
            draft_tokens = torch.cat([x, draft_tokens], dim=1)

            # examine target model logits for the proposed ngram tokens
            # if using cache, shape is (1, 1 + num_pred_tokens), otherwise (1, seqlen + num_pred_tokens)
            # TODO: make sure this works when there are no draft tokens
            target_logits, inference_params_dict_out = self.model(
                draft_tokens,
                inference_params_dict=inference_params_dict_out
            )

            # excludes the input prompt if present
            target_logits = target_logits[:, -draft_size - 1:, :]

            # sample from the modified logits
            target_tokens = target_logits.argmax(dim=-1)
            draft_tokens[:, -draft_size:]
            n_matches = ((~(draft_tokens[:, -draft_size:] == target_tokens[:, :-1])).cumsum(dim=-1) < 1).sum()
            n_matches = min(n_matches, num_tokens - i)
            
            # if stop_at_eos and (generation[0, -2:] == eos_token_ids).all():
            #     print_rank_0("Stopping generation at EOS")

            # if print_generation and verbose and batch_size == 1:
            #     print_rank_0(
            #         f"{self.tokenizer.detokenize([new_idx.item()])}",
            #         end=" ",
            # )
            
            
            # TODO: trim KV cache
            scores[:, i: (i + n_matches), :] = target_logits[:, :n_matches, :]
            generation[:, i: (i + n_matches)] = target_tokens[:, :n_matches]
            x = torch.cat([x, target_tokens[:, n_matches:]], dim=1)

            i += n_matches

            # TODO: confirm EOS criteria
        
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

            mem_end = torch.cuda.memory_allocated(device=x.device) / 1e9
            print_rank_0(f"Memory after generation: {mem_end} GB")

        return generation[:, : i + 1], scores[:, : i + 1]
