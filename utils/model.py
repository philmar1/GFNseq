from abc import ABC, abstractmethod
from torch import nn, Tensor
import torch 
from torch.distributions.categorical import Categorical

from typing import Optional, Literal

from configs.config import *
from utils.utils import * 
from utils.losses import * 

import numpy as np 

from tqdm import tqdm, trange

    
# --- Models as backbone of the gflownets
class FlowModel(nn.Module):
    def __init__(self, num_hid, name='flowmodel'):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(SEQ_LEN, num_hid),  # Input state is a vector representing char
            nn.LeakyReLU(),
            nn.Linear(num_hid, VOCAB_SIZE)  # Output 6 numbers for the 6 actions (child states).
        )

        self._name = name 
    
    @property
    def name(self):
        return self._name 
    
    def forward(self, x):
        return self.mlp(x).exp()  # Flows must be positive, so we take the exponential.


class TBModel(nn.Module):
    def __init__(self, num_hid, name='TBmodel'):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(SEQ_LEN, num_hid),  # SEQ_LEN input features.
            nn.LeakyReLU(),
            nn.Linear(num_hid, SEQ_LEN*2),  # SEQ_LEN*2 outputs: SEQ_LEN for P_F and SEQ_LEN for P_B. /!\ There is flow of information from PF to PB and vice versa
        )
        self.logZ = nn.Parameter(torch.ones(1))  # log Z is just a single number.

        self._name = name 
    
    @property
    def name(self):
        return self._name 
    
    def forward(self, x):
        logits = self.mlp(x)
        # Slice the logits into forward and backward policies.
        P_F = logits[..., :SEQ_LEN]
        P_B = logits[..., SEQ_LEN:]

        return P_F, P_B

    def get_action(self, P_F_s):
        # Here P_F is logits, so we use Categorical to compute a softmax.
        categorical = Categorical(logits=P_F_s)
        action = categorical.sample()
        
        return action

# --- Gflonwets model

class GFN(ABC):
    """Base GFN class"""
    def __init__(self, model, env) -> None:
        self.model = model
        self.env = env 
        self.total_log_P_F = 0
        self.total_log_P_B = 0
    
    @abstractmethod
    def make_trajectory(self, traj_length):
        pass 
    
    @abstractmethod
    def sample(self, n_samples):
        pass 
    
    @abstractmethod
    def train(self):
        pass
    
class TB_GFN(GFN):
    """Gflownet model working with the trajectory balance"""
    def __init__(self, model, env) -> None:
        super().__init__(model, env)
        
        
    def make_trajectory(self, traj_length):
        state = []
        self.total_log_P_F = 0
        self.total_log_P_B = 0
        
        P_F_s, P_B_s = self.model(state_to_tensor(state))  
        for t in range(traj_length):  # All trajectories are length SEQ_LEN (not including s0).
            # Here we mask the relevant forward actions.
            mask = self.env.calculate_forward_mask_from_state(state)
            P_F_s = torch.where(mask, P_F_s, -100)  # Removes invalid forward actions.

            categorical = Categorical(logits=P_F_s)
            action = self.model.get_action(P_F_s)
            
            new_state = state + [action.item()] 
            self.total_log_P_F += categorical.log_prob(action)  

            # We recompute P_F and P_B for new_state.
            P_F_s, P_B_s = self.model(state_to_tensor(new_state))
            mask = self.env.calculate_backward_mask_from_state(new_state)
            P_B_s = torch.where(mask, P_B_s, -100)  # Removes invalid backward actions.

            # Accumulate P_B, going backwards from `new_state`
            self.total_log_P_B += Categorical(logits=P_B_s).log_prob(action)

            state = new_state  
            
        return state
    
    def sample(self, n_samples):
        states = []
        for _ in range(n_samples):
            states.append(self.make_trajectory(SEQ_LEN))
        return states
    
    def train(self, opt, n_episodes):
        self.losses, self.logZs = [], []
        minibatch_loss = 0
        tbar = trange(n_episodes, desc="Training iter")
            
        for episode in tbar:
            
            state = self.make_trajectory(SEQ_LEN)
            reward = self.env.get_reward(state)

            minibatch_loss += trajectory_balance_loss(
                self.model.logZ,
                self.total_log_P_F,
                self.total_log_P_B,
                reward,
            )
            minibatch_loss = torch.clip(minibatch_loss, -20)


            if episode % UPDATE_FREQ == 0:
                self.losses.append(minibatch_loss.item())
                self.logZs.append(self.model.logZ.item())
                minibatch_loss.backward()
                opt.step()
                opt.zero_grad()
                tbar.set_description("Training iter (loss={:.6f})".format(minibatch_loss.item()))
                minibatch_loss = 0