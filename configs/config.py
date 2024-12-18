SEQ_LEN = 5

VOCAB = ['A', 'C', 'G', 'T', '#'] # the last action always corresponds to the exit or terminate action
CHAR_TO_IDX = {char: idx for idx, char in enumerate(VOCAB)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)

UPDATE_FREQ = 4
N_EPISODES = 20000


# Model params
HIDDEN_SIZES = 10