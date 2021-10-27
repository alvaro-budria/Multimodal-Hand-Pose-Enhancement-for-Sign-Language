import torch.optim as optim

# OPTIMIZER
optimizers = {"Adam": optim.Adam,
              "AdamW": optim.AdamW,
              "NAdam": optim.NAdam}

# DEVICE
import torch
torch.manual_seed(1)
device = 'cpu'
if torch.cuda.is_available():
  device = 'cuda'
  torch.cuda.manual_seed_all(1)
