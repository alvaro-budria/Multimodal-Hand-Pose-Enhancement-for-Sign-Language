import torch
from torch.autograd import Variable
import numpy as np
from hyperparameters import device


# Define a function that trains the model for one epoch
def train_epoch(model, train_X, train_Y, optimizer, loss_function, BATCH_SIZE, rng, clip_grad=False):
    model.train()
    epoch_loss, epoch_acc = [], 0
    batchinds = np.arange(train_X.shape[0] // BATCH_SIZE)
    # rng.shuffle(batchinds)
    for bii, bi in enumerate(batchinds):
        ## setting batch data
        idxStart = bi * BATCH_SIZE
        inputData = train_X[idxStart:(idxStart + BATCH_SIZE), :, :]
        outputGT = train_Y[idxStart:(idxStart + BATCH_SIZE)]
        inputData = Variable(torch.from_numpy(inputData).float()).to(device)
        outputGT = Variable(torch.from_numpy(outputGT-1)).to(device)

        # Forward pass
        y_, _ = model(inputData)
        epoch_acc += sum(np.argmax(y_[:,-1,:].cpu().detach().numpy(), axis=1) == outputGT.cpu().detach().numpy())

        # Set gradients to 0, compute the loss, gradients, and update the parameters
        optimizer.zero_grad()
        loss = loss_function(y_[:,-1,:], outputGT)
        epoch_loss.append(loss.item())
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    return epoch_loss, epoch_acc/(len(batchinds)*BATCH_SIZE)
