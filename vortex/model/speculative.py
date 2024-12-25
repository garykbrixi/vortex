import torch
from torch.nn import Module
from typing import List, Tuple

from vortex.model.sample import sample
from vortex.model.tokenizer import CharLevelTokenizer
from vortex.model.generation import BaseGenerator


class SpeculativeGenerator(BaseGenerator):
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
        super().__init__(draft_model, tokenizer, top_k, top_p, temperature, untils)
        self.gamma = gamma

    def _generate_tokens(self, *args, **kwargs):
        pass

