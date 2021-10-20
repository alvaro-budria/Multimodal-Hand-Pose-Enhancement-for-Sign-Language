# IMPORTS
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# IMPORTS FROM OTHER PROJECT FILES
# Add the model folder to the sys path
sys.path.insert(1, '../Model')

# Do the imports from other files in the project
from DepthLSTM import DepthLSTM
from hyperparameters import *
from train_epoch import train_epoch
from test_epoch import test_epoch

# LOAD THE DATA
xyz_data = np.load('../Data/normalized_data.npy')

# PARAMETER DEFINITION
DATA_SIZE = xyz_data.shape[0]
NUM_JOINTS = 26
INPUT_SIZE = 2*NUM_JOINTS

# SEPARATION OF DATA AND GROUND TRUTH
# Separate x and y dimensions and convert into tensors
x_data = torch.tensor(xyz_data[:, :, 0].squeeze())
y_data = torch.tensor(xyz_data[:, :, 1].squeeze())
# Obtain the body_data tensor of shape [DATA_SIZE, 2*NUM_JOINTS]
body_data = torch.empty((DATA_SIZE, 2*NUM_JOINTS))
for i in range(0, 2*NUM_JOINTS):
  if i%2 == 0:      
    body_data[:, i] = x_data[:, i//2]
  else:
    body_data[:, i] = y_data[:, i//2]
# Obtain the ground_data tensor of shape [DATA_SIZE, NUM_JOINTS]
ground_data = torch.tensor(xyz_data[:, :, 2].squeeze())

# TRAIN/TEST SPLIT
# Use the last TEST_FRACTION of the data for testing
TRAIN_SIZE = np.int(np.ceil(0.8*DATA_SIZE))
TEST_SIZE = DATA_SIZE - TRAIN_SIZE
train_data = body_data[0:TRAIN_SIZE,:]
train_ground = ground_data[0:TRAIN_SIZE,:]
test_data = body_data[TRAIN_SIZE:,:]
test_ground = ground_data[TRAIN_SIZE:,:]

# DIVIDE TRAINING DATA INTO BATCHES
# Compute the number of samples in each chunk or batch
CHUNK_SIZE = TRAIN_SIZE // BATCH_SIZE
# Obtain the list of BATCH_SIZE chunks where each element is a tensor
# of shape [CHUNK_SIZE, 2*NUM_JOINTS]
trainset = [ train_data[beg_i:end_i] \
            for beg_i, end_i in zip(range(0, TRAIN_SIZE - CHUNK_SIZE, CHUNK_SIZE),
                                    range(CHUNK_SIZE, TRAIN_SIZE, CHUNK_SIZE)) ]
groundset = [ train_ground[beg_i:end_i] \
            for beg_i, end_i in zip(range(0, TRAIN_SIZE - CHUNK_SIZE, CHUNK_SIZE),
                                    range(CHUNK_SIZE, TRAIN_SIZE, CHUNK_SIZE)) ]

# DIVIDE THE TESTING DATA INTO BATCHES
# Compute the number of samples in each chunk or batch
CHUNK_TEST_SIZE = TEST_SIZE // BATCH_SIZE
# Obtain the list of BATCH_SIZE batches where each element is a tensor
# of shape [CHUNK_SIZE, 2*NUM_JOINTS]
testset = [ test_data[beg_i:end_i] \
            for beg_i, end_i in zip(range(0, TEST_SIZE - CHUNK_TEST_SIZE, CHUNK_TEST_SIZE),
                                    range(CHUNK_TEST_SIZE, TEST_SIZE, CHUNK_TEST_SIZE)) ]
groundtestset = [ test_ground[beg_i:end_i] \
            for beg_i, end_i in zip(range(0, TEST_SIZE - CHUNK_TEST_SIZE, CHUNK_TEST_SIZE),
                                    range(CHUNK_TEST_SIZE, TEST_SIZE, CHUNK_TEST_SIZE)) ]

# TRAIN AND TEST THE MODEL
# Initialize the model
model = DepthLSTM(HIDDEN_SIZE, NUM_LAYERS, SEQ_LEN, BATCH_SIZE, NUM_JOINTS)
model.to(device)
# Define the loss function and the optimizer
# loss_function = nn.MSELoss()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
tr_loss = []
tst_loss = []
state = None
timer_beg = timer()
# Train the model for NUM_EPOCHS epochs
for epoch in range(NUM_EPOCHS):
  print('Starting epoch: ', epoch)
  train_epoch_loss = train_epoch(model, trainset, groundset, optimizer, loss_function, CHUNK_SIZE)
  test_epoch_loss, predY = test_epoch(model, testset, groundtestset, CHUNK_TEST_SIZE)
  timer_end = timer()  
  if (epoch + 1) % 10 == 0:
    # Print the training loss of this epoch
    # It is calculated as the average of losses of every window
    print('Training loss in epoch {} is: {}'.format(epoch, sum(train_epoch_loss)/len(train_epoch_loss) ))
    print('Test loss in epoch {} is: {}'.format(epoch, sum(test_epoch_loss)/len(test_epoch_loss) ))
  tr_loss.append(train_epoch_loss)
  tst_loss.append(test_epoch_loss)
  timer_beg = timer()
