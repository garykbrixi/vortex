import torch
from torch.nn import Module
from typing import List, Tuple

from vortex.model.sample import sample, modify_logits_for_top_k_filtering, modify_logits_for_top_p_filtering
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

        target_generation = torch.empty(
            x.shape[0],
            num_tokens,
            dtype=torch.long,
            device=x.device,
        )

        target_scores = torch.empty(
            x.shape[0],
            num_tokens,
            self.tokenizer.vocab_size,
            dtype=torch.float,
            device=x.device,
        )

        draft_generation = target_generation.clone()
        draft_scores = target_scores.clone()

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
            all_draft_logits = []
            all_draft_sampled_idxs = []

            for j in range(self.gamma):
                draft_logits, draft_inference_params_dict_out = self.get_logits(self.draft_model, x, i, draft_inference_params_dict_out, cached_generation, prompt_len)
                draft_last_logits = draft_logits[:, -1, :]  # TODO: check shape
                draft_sampled_idx, draft_sampled_logits = sample(
                    draft_last_logits,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    return_logits=True,
                )
                all_draft_logits.append(draft_last_logits)
                all_draft_sampled_idxs.append(draft_sampled_idx)
                # all_draft_sampled_idxs.append(rearrange(draft_sampled_logits, "b -> b 1 1"))
           
            # cat or stack? TODO: check if this is correct
            all_draft_sampled_idxs = torch.stack(all_draft_sampled_idxs, dim=1)
            all_draft_logits = torch.stack(all_draft_logits, dim=1)

            # get target logits for gamma + 1 tokens in parallel
            target_logits, target_inference_params_dict_out = self.get_logits(self.target_model, x, i, target_inference_params_dict_out, cached_generation, prompt_len)
            target_last_logits = target_logits[..., -(self.gamma + 1):, :]
            target_sampled_idx, target_sampled_logits = sample(
                target_last_logits,
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
                return_logits=True,
            )

            target_prob = torch.softmax(target_sampled_logits, dim=-1)
            draft_prob = torch.softmax(all_draft_logits, dim=-1)

            # Equation X of paper
            r = torch.uniform(0, 1, size=(batch_size, self.gamma + 1, 1))

            
            

