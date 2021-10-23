import sys
import numpy as np
import torch.nn as nn
import torch.optim as optim

sys.path.insert(1, '../Model')
from ClassifLSTM import ClassifLSTM
from hyperparameters import *

from train_epoch import train_epoch
from test_epoch import test_epoch

sys.path.insert(1, "../../utils")
from postprocess_utils import *
from load_save_utils import load_binary


def main():
    # LOAD THE DATA
    data_dir = "../../video_data"
    def load_data(data_dir="video_data", filename="r6d_train.pkl"):
        data = load_binary(f"{data_dir}/r6d_train.pkl")
        data = make_equal_len(data, method="cutting+reflect")  # make sequences have equal length, as initially they have different lengths
        data, _, _ = rmv_clips_nan(data, r6d_train)  # remove those clips containing nan values
    r6d_train = load_data(data_dir=data_dir, filename="r6d_train.pkl")
    r6d_val = load_data(data_dir=data_dir, filename="r6d_val.pkl")
    Y_train, Y_val = load_binary(f"{data_dir}/categs_train.pkl"), load_binary(f"{data_dir}/categs_val.pkl")

    # PARAMETER DEFINITION
    NUM_ROTATIONS = r6d_train.shape[1]

    # TRAIN AND VAL THE MODEL
    # Initialize the model
    model = ClassifLSTM(HIDDEN_SIZE, NUM_LAYERS, SEQ_LEN, BATCH_SIZE, NUM_ROTATIONS)
    model.to(device)
    # Define the loss function and the optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    tr_loss, val_loss = [], []
    rng = np.random.RandomState(23456)  # for shuffling batches
    # Train the model for NUM_EPOCHS epochs
    for epoch in range(NUM_EPOCHS):
        print('Starting epoch: ', epoch)
        train_epoch_loss = train_epoch(model, r6d_train, Y_train, optimizer, loss_function, BATCH_SIZE, rng)
        val_epoch_loss = test_epoch(model, r6d_val, Y_val, loss_function, VAL_BATCH_SIZE, rng)
        if (epoch + 1) % 10 == 0:
            print('Training loss in epoch {} is: {}'.format(epoch, sum(train_epoch_loss)/len(train_epoch_loss) ))
            print('Val loss in epoch {} is: {}'.format(epoch, sum(val_epoch_loss)/len(val_epoch_loss) ))
        tr_loss.append(train_epoch_loss)  ##### should store mean loss?¿
        val_loss.append(val_epoch_loss)


if __name__ == "__main__":
    main()
