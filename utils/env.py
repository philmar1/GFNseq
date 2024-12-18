from configs.config import *
from utils.reward import * 

import numpy as np 
import torch 

from typing import Tuple, List 
from jaxtyping import Float

class Env():
    def __init__(self, reward) -> None:
        assert isinstance(reward, Reward)
        self.reward = reward

    def calculate_forward_mask_from_state(self, state):
        """Here, we mask forward actions to prevent the selection of invalid configurations.
        """
        mask = np.ones(VOCAB_SIZE)
        mask[CHAR_TO_IDX['#']] = 0
        return torch.Tensor(mask).bool() 
    
    def calculate_backward_mask_from_state(self, state):
        """Here, we mask backward actions to only select parent nodes."""
        return torch.Tensor(
            [i == state[-1] for i in range(VOCAB_SIZE)]
        ).bool()
    
    def get_reward(self, state):
        return torch.tensor(self.reward.get_reward(state)).float() + 1e-8 #torch.tensor works with num value. For future uses, with batch size, use torch.Tensor
    
    def get_parent_states(self, state) -> Tuple[List, List]: # Note: unused in TB
        if len(state) == 0:
            return None
        else:     
            parent_states = state[:-1]
            parent_actions = state[-1] 
            return [parent_states], [parent_actions]
        