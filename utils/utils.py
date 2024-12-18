import torch 
from configs.config import * 

def state_to_tensor(state):
    state = state + [CHAR_TO_IDX['#']]*(SEQ_LEN-len(state)) # append end of token
    return torch.tensor(state).float()
