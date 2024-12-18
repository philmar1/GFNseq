# %% 
from abc import ABC, abstractmethod
from jaxtyping import Float
from configs.config import * 

class Reward(ABC):
    """Base Reward class."""
    def __init__(self, name='ABC') -> None:
        self._name = name
        
    @property
    def name(self):
        return self._name
    
    @abstractmethod
    def get_reward(self):
        pass
    
    
class SimpleReward(Reward):
    """Simple Reward class that counts number of A in sequence"""
    def __init__(self, name='simple_reward') -> None:
        super().__init__(name)
        
    def get_reward(self, state: list[str]) -> Float:
        # Count occurrences of 'A' in state
        return float(state.count(CHAR_TO_IDX['A']))
        


        
# %%
