import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from hyperparameters import device


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
            inputData = Variable(torch.from_numpy(inputData).float()).to(device)
            # convert labels to one-hot encoding. subtract 1 from Y to make labels start from 0
            outputGT = Variable( F.one_hot(torch.from_numpy(outputGT - 1), num_classes=10).to(device) )

            # Forward pass.
            y_, _ = model(inputData)
            predY.append(y_)

            # Compute loss
            loss = loss_function(y_, outputGT)
            val_loss.append(loss.item())
    return val_loss, predY
