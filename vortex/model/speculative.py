import torch
from torch.nn import Module
from typing import List, Tuple

from vortex.model.sample import sample, modify_logits
from vortex.model.tokenizer import CharLevelTokenizer
from vortex.model.utils import print_rank_0


def _format_eos_token_ids(tokenizer):
    if isinstance(tokenizer.eos, int):
        eos_token_ids = torch.LongTensor([tokenizer.eos])
    else:
        eos_token_ids = tokenizer.tokenize(tokenizer.eos)
    return eos_token_ids

 
def _input_string_to_ids(input_string, tokenizer, device):
    input = tokenizer.tokenize(input_string)
    if isinstance(input, list):
        input = torch.LongTensor(input).unsqueeze(0).to(device)
    # is a tensor
    else:
        input = input.unsqueeze(0).to(device)
    return input


class SpeculativeGenerator:
    def __init__(
        self,
        draft_model: Module,  
        target_model: Module,
        tokenizer: CharLevelTokenizer,
        top_k: int = 50,
        top_p: float = 0.7,
        temperature: float = 1,
        untils: List[str] = ["\n\n"],
        gamma: int = 5,
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.untils = untils
        self.gamma = gamma   # number of speculative tokens
        self.drafts_accepted = 0

    def get_logits(self, model, x, cur_i, inference_params_dict, cached_generation, prompt_len):
        post_prefill = cached_generation and cur_i > 0
        
        # prefill then process only the last token
        if post_prefill:
            x = x[:, -1:]

            # both target and draft models should have the same seqlen_offset
            seqlen_offset = inference_params_dict["mha"].seqlen_offset

            if seqlen_offset == 0:
                seqlen_offset = prompt_len
                for key in ["mha", "hcl", "hcm", "hcs"]:
                    inference_params_dict[key].seqlen_offset = seqlen_offset
            else:
                seqlen_offset += 1
                for key in ["mha", "hcl", "hcm", "hcs"]:
                    inference_params_dict[key].seqlen_offset = seqlen_offset

        # do forward pass with no gradient on draft model:
        with torch.no_grad():
            logits, inference_params_dict_out = model(
                x,
                inference_params_dict=inference_params_dict,
            )

        return logits, inference_params_dict_out
    
    def generate(self, input_string, device, num_tokens=32, cached_generation=False, print_generation=True, verbose=False, skip_special_tokens=False, stop_at_eos=True, max_seqlen=None):
        # tokenize input string if not already a tensor
        x = _input_string_to_ids(input_string, self.tokenizer, device)

        # truncate input if max_seqlen is provided
        if max_seqlen is not None:
            x = x[:, -max_seqlen:]

        # initialize useful variables and arrays 
        prompt_len = x.shape[-1]

        num_tokens = int(num_tokens)
        tot_length = prompt_len + num_tokens
        batch_size = x.shape[0]

        generation = torch.empty(
            x.shape[0],
            num_tokens,
            dtype=torch.long,
            device=x.device,
        )

        target_logits = torch.empty(
            x.shape[0],
            num_tokens,
            self.tokenizer.vocab_size,
            dtype=torch.float,
            device=x.device,
        )

        target_scores = torch.empty(
            x.shape[0],
            num_tokens,
            self.tokenizer.vocab_size,
            dtype=torch.float,
            device=x.device,
        )

        # set up inference params if doing cached generation
        if cached_generation:
            target_inference_params_dict_out = self.target_model.initialize_inference_params()
            draft_inference_params_dict_out = self.draft_model.initialize_inference_params()
            for key in ["mha", "hcl", "hcm", "hcs"]:
                target_inference_params_dict_out[key].max_batch_size = batch_size
                draft_inference_params_dict_out[key].max_batch_size = batch_size
        else:
            target_inference_params_dict_out = None
            draft_inference_params_dict_out = None
    
        # print memory usage
        if verbose:
            mem_after_tok = torch.cuda.memory_allocated(device=x.device) / 1e9
            print_rank_0(f"Memory after tokenization: {mem_after_tok} GB")
            print_rank_0("Starting generation...")
            if input_string is not None:
                print_rank_0("Prompt: " + input_string)
            else:
                print_rank_0(f"Prompt ids: {input_ids} {input_ids.shape}")
        
        ##########################################################
        # generate tokens
        ##########################################################
        
        for i in range(int(num_tokens)):
            gamma = min(self.gamma, num_tokens - i - 1)
            all_draft_logits = []
            all_draft_sampled_idx = []

            for j in range(gamma):
                draft_logits, draft_inference_params_dict_out = self.get_logits(self.draft_model, x[:, :i+j], i + j, draft_inference_params_dict_out, cached_generation, prompt_len)
                draft_last_logits = draft_logits[:, -1]  # (1, L, vocab_size) -> (1, vocab_size)
                draft_last_logits = modify_logits(draft_last_logits, top_k=self.top_k, top_p=self.top_p, temperature=self.temperature)
                all_draft_logits.append(draft_last_logits)

                draft_sampled_idx = sample(draft_last_logits, top_k=self.top_k, top_p=self.top_p, temperature=self.temperature, ignore_logit_modification=True)
                all_draft_sampled_idx.append(draft_sampled_idx)
            
            q = torch.stack(all_draft_logits, dim=1)
            all_draft_sampled_idx = torch.stack(all_draft_sampled_idx, dim=1)
            
            # get target logits for gamma + 1 tokens in parallel
            target_logits, target_inference_params_dict_out = self.get_logits(self.target_model, x[:, :i + gamma + 1], i + gamma + 1, target_inference_params_dict_out, cached_generation, prompt_len)
            p = target_logits[..., -(gamma + 1):, :]  # (1, gamma + 1, vocab_size)
            p = modify_logits(p, top_k=self.top_k, top_p=self.top_p, temperature=self.temperature)

            # compute the last accepted draft position (rejection sampling)
            r = torch.rand(gamma, device=target.device)
            fractions = p / q
            n = gamma  # number of accepted guesses

            for j in range(gamma):
                if r[j] > fractions[0, j, i+j]:
                    n = i
                    break
            
            self.drafts_accepted += n

            target_scores[:, i:i + n] = target_logits[:, i:i+n]
            target_generation[:, i:i + n] = all_draft_sampled_idx[:, i:i+n]

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

        return target_generation[:, : i + 1], target_scores[:, : i + 1]
