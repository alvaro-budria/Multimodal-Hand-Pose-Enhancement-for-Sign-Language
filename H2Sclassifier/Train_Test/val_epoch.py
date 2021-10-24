from hyperparameters import *
import torch
from torch.autograd import Variable
import numpy as np


def val_epoch(model, train_X, train_Y, loss_function, BATCH_SIZE, rng):
    val_loss = []
    predY = []
    model.eval()
    batchinds = np.arange(train_X.shape[0] // BATCH_SIZE)
    rng.shuffle(batchinds)
    with torch.no_grad():
        for bii, bi in enumerate(batchinds):
            ## setting batch data
            idxStart = bi * BATCH_SIZE
            inputData = train_X[idxStart:(idxStart + BATCH_SIZE), :, :]
            outputGT = train_Y[idxStart:(idxStart + BATCH_SIZE)]
            inputData = Variable(torch.from_numpy(inputData)).to(device)
            outputGT = Variable(torch.from_numpy(outputGT)).to(device)

            # Forward pass.
            y_ = model(inputData)
            predY.append(y_)

            # Compute loss
            loss = loss_function(y_, outputGT)
            val_loss.append(loss.item())
    return val_loss, predY
