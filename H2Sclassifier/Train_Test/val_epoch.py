import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from hyperparameters import device


def val_epoch(model, train_X, train_Y, loss_function, BATCH_SIZE, rng):
    val_loss = []
    epoch_acc = 0
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
            #outputGT = Variable(torch.from_numpy(outputGT-1)).to(device)

            # Forward pass
            y_, _ = model(inputData)
            predY.append(y_)
            #epoch_acc += sum(np.argmax(y_[:,-1,:].cpu().detach().numpy(), axis=1) == outputGT.cpu().detach().numpy())
            epoch_acc += sum(np.argmax(y_[:,-1,:].cpu().detach().numpy(), axis=1) == (outputGT-1))

            # Compute loss
            loss = loss_function(y_[:,-1,:], outputGT)
            val_loss.append(loss.item())
    return val_loss,  epoch_acc/(len(batchinds)*BATCH_SIZE)
