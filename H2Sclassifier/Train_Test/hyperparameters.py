# HYPERPARAMETER DEFINITION
HIDDEN_SIZE = 1024
NUM_LAYERS = 1
BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
SEQ_LEN = 50
BIAS = True
TEST_FRACTION = 0.2 # Fraction of the data used for testing
NUM_EPOCHS = 1000
NUM_JOINTS = 26

# DEVICE
import torch
torch.manual_seed(1)
device = 'cpu'
if torch.cuda.is_available():
  device = 'cuda'
  torch.cuda.manual_seed_all(1)
