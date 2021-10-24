import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from timeit import default_timer as timer
from hyperparameters import device


# Define a function that trains the model for one epoch
def train_epoch(model, train_X, train_Y, optimizer, loss_function, BATCH_SIZE, rng, clip_grad=False):
    model.train()
    epoch_loss = []
    epoch_acc = 0
    batchinds = np.arange(train_X.shape[0] // BATCH_SIZE)
    rng.shuffle(batchinds)
    for bii, bi in enumerate(batchinds):
        ## setting batch data
        idxStart = bi * BATCH_SIZE
        inputData = train_X[idxStart:(idxStart + BATCH_SIZE), :, :]
        outputGT = train_Y[idxStart:(idxStart + BATCH_SIZE)]
        inputData = Variable(torch.from_numpy(inputData).float()).to(device)
        # convert labels to one-hot encoding. subtract 1 from Y to make labels start from 0
        outputGT = Variable( F.one_hot(torch.from_numpy(outputGT-1), num_classes=10).to(device) )

        # Forward pass
        y_, _ = model(inputData)
        epoch_acc += torch.sum(y_[:,-1,:] == outputGT)

        # Set gradients to 0, compute the loss, gradients, and update the parameters
        optimizer.zero_grad()
        loss = loss_function(y_[:,-1,:], outputGT)
        epoch_loss.append(loss.item())
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    return epoch_loss, epoch_acc.cpu().detach().numpy()/(len(batchinds)*BATCH_SIZE)
