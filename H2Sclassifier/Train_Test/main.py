import sys
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim

sys.path.insert(1, '../Model')
from ClassifLSTM import ClassifLSTM
from hyperparameters import *

from train_epoch import train_epoch
from val_epoch import val_epoch

sys.path.insert(1, "../../utils")
from postprocess_utils import *
from load_save_utils import load_binary

# experiment logging
import wandb


def main(args):
    wandb.login()
    ## variables
    config = dict(
        num_epochs = args.num_epochs,
        batch_size = args.batch_size,
        learning_rate = args.learning_rate,
        data_dir=args.data_dir,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers)

    ## DONE variables
    with wandb.init(project="B2H-H2S", name=args.exp_name, id=args.exp_name, save_code=True, config=config):
        config = wandb.config

        X_train, Y_train = load_data(data_dir=config.data_dir, key="train")
        X_val, Y_val = load_data(data_dir=config.data_dir, key="val")
        print(f"X_train.shape, Y_train.shape {X_train.shape, Y_train.shape}", flush=True)
        print(f"X_val.shape, Y_val.shape {X_val.shape, Y_val.shape}", flush=True)
        
        # PARAMETER DEFINITION
        NUM_ROTATIONS = X_train.shape[2]
        SEQ_LEN = X_train.shape[1]  # number of frames per clip
        NUM_CLASSES = 10
        print(f"NUM_ROTATIONS: {NUM_ROTATIONS}, SEQ_LEN: {SEQ_LEN}, NUM_CLASSES: {NUM_CLASSES}", flush=True)

        # TRAIN AND VAL THE MODEL
        # Initialize the model
        model = ClassifLSTM(config.hidden_size, config.num_layers, SEQ_LEN, config.batch_size, NUM_ROTATIONS, NUM_CLASSES)
        model.to(device)
        # Define the loss function and the optimizer
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        tr_loss, val_loss = [], []
        rng = np.random.RandomState(23456)  # for shuffling batches
        # Train the model for NUM_EPOCHS epochs
        for epoch in range(config.num_epochs):
            print('Starting epoch: ', epoch)
            train_epoch_loss = train_epoch(model, X_train, Y_train, optimizer, loss_function, config.batch_size, rng)
            val_epoch_loss = val_epoch(model, X_val, Y_val, loss_function, config.batch_size, rng)
            wandb.log({"epoch": epoch, "loss_train": np.mean(train_epoch_loss)})
            wandb.log({"epoch": epoch, "loss_val": np.mean(val_epoch_loss)})
            if (epoch + 1) % 10 == 0:
                print('Training loss in epoch {} is: {}'.format(epoch, sum(train_epoch_loss)/len(train_epoch_loss) ))
                print('Val loss in epoch {} is: {}'.format(epoch, sum(val_epoch_loss)/len(val_epoch_loss) ))
            tr_loss.append(train_epoch_loss)  ##### should store mean loss?¿
            val_loss.append(val_epoch_loss)


# Data load helper
def load_data(data_dir="video_data", key="train"):
    X = load_binary(f"{data_dir}/r6d_{key}.pkl")
    Y = load_binary(f"{data_dir}/categs_{key}.pkl")
    X = make_equal_len(X, method="cutting+reflect")  # make sequences have equal length, as initially they have different lengths
    X, Y, _ = rmv_clips_nan(X, Y)  # remove those clips containing nan values
    return X, Y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="../../video_data" , help='directory where results should be stored to and loaded from')
    parser.add_argument('--exp_name', type=str, default='experiment', help='name for the experiment')
    parser.add_argument('--num_epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate for training G and D')
    parser.add_argument('--hidden_size', type=int , default=1024, help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int , default=10, help='Number of LSTM layers')

    args = parser.parse_args()
    print(args, flush=True)
    main(args)
