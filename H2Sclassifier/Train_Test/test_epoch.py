from hyperparameters import *
import torch
import numpy as np
from timeit import default_timer as timer

def test_epoch(model, testset, groundtestset, CHUNK_TEST_SIZE):
  test_loss = []
  state = None
  timer_beg = timer()
  predY = []
  # Evaluation flag
  model.eval()
  # We don't need to keep track of the gradients
  with torch.no_grad():

    # this loop slides over one chunk in windows of length SEQ_LEN
    # *first loop -> 0 : SEQ_LEN -1
    # *second loop-> SEQ_LEN : 2*SEQ_LEN
    # *...
    # *last loop -> CHUNK_SIZE - SEQ_LEN : CHUNK_SIZE - 1
    # Observe that SEQ_LEN is the length of the sequences we feed into the LSTM
    for beg_t, end_t in zip(range(0, CHUNK_TEST_SIZE - SEQ_LEN - 1, SEQ_LEN),
                            range(SEQ_LEN, CHUNK_TEST_SIZE, SEQ_LEN)):
      
      dataX = torch.empty([BATCH_SIZE, SEQ_LEN, 2*NUM_JOINTS], device=device)
      dataY = torch.empty([BATCH_SIZE, SEQ_LEN, NUM_JOINTS], device=device)
      # Step 2. Get our inputs ready for the network, that is, turn them into
      # Tensors of one-hot sequences.
      # for every batch 
      for batch in range(BATCH_SIZE):
        # we obtain the batch
        batch_data = testset[batch] # shape = [CHUNK_SIZE, NUM_JOINTS*2]
        batch_ground = groundtestset[batch] # shape = [CHUNK_SIZE, NUM_JOINTS]

        # we obtain the sequence by applying the window [beg_t:end_t] to the batch
        seq_data = batch_data[beg_t:end_t, :] # shape = [SEQ_LEN, NUM_JOINTS*2]
        seq_ground = batch_ground[beg_t:end_t, :] # shape = [SEQ_LEN, NUM_JOINTS]

        # Add the sequences to dataX and dataY
        dataX[batch, :, :] = seq_data # create batch-dim and append
        dataY[batch, : , :] = seq_ground # create batch-dim and append

      # Step 3. Run our forward pass.
      # Forward through model and carry the previous state forward in time (statefulness)
      y_, state = model(dataX, None)
      predY.append(y_)
      # detach the previous state graph to not backprop gradients further than the BPTT span
      state = (state[0].detach(), # detach h[t] 
              state[1].detach()) # detach c[t]

      # Step 4. Compute the loss, gradients, and update the parameters by
      #  calling optimizer.step()
      loss = loss_function(y_, dataY)
      test_loss.append(loss.item())
    timer_end = timer()  
    return test_loss, predY
