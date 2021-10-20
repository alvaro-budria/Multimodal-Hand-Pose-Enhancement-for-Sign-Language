from hyperparameters import *
import torch
import numpy as np
from timeit import default_timer as timer

# Define a function that trains the model for one epoch
def train_epoch(model, trainset, groundset, optimizer, loss_function, CHUNK_SIZE):
    state = None
    timer_beg = timer()
    model.train()
    epoch_loss = []
    # this loop slides over one chunk in windows of length SEQ_LEN
    # *first loop -> 0 : SEQ_LEN -1
    # *second loop-> SEQ_LEN : 2*SEQ_LEN
    # *...
    # *last loop -> CHUNK_SIZE - SEQ_LEN : CHUNK_SIZE - 1
    # Observe that SEQ_LEN is the length of the sequences we feed into the LSTM
    for beg_t, end_t in zip(range(0, CHUNK_SIZE - SEQ_LEN - 1, SEQ_LEN),
                            range(SEQ_LEN, CHUNK_SIZE, SEQ_LEN)):

        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        optimizer.zero_grad()

        dataX = torch.empty([BATCH_SIZE, SEQ_LEN, 52], device=device)
        dataY = torch.empty([BATCH_SIZE, SEQ_LEN, 26], device=device)
        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of one-hot sequences.
        # for every batch 
        for batch in range(BATCH_SIZE):
            # we obtain the batch
            batch_data = trainset[batch] # shape = [CHUNK_SIZE, NUM_JOINTS*2]
            batch_ground = groundset[batch] # shape = [CHUNK_SIZE, NUM_JOINTS]

            # we obtain the sequence by applying the window [beg_t:end_t] to the batch
            seq_data = batch_data[beg_t:end_t, :] # shape = [SEQ_LEN, NUM_JOINTS*2]
            seq_ground = batch_ground[beg_t:end_t, :] # shape = [SEQ_LEN, NUM_JOINTS]

            # Add the sequences to dataX and dataY
            dataX[batch, :, :] = seq_data # create batch-dim and append
            dataY[batch, : , :] = seq_ground # create batch-dim and append

        # Step 3. Run our forward pass.
        # Forward through model and carry the previous state forward in time (statefulness)
        y_, state = model(dataX, state)

        # detach the previous state graph to not backprop gradients further than the BPTT span
        state = (state[0].detach(), # detach h[t] 
                 state[1].detach()) # detach c[t]

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(y_, dataY)
        loss.backward()
        optimizer.step()  
        epoch_loss.append(loss)
    return epoch_loss
