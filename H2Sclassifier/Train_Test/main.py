import sys
import argparse
import numpy as np
import torch.nn as nn

import gc

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
                data_dir=args.data_dir,
                categs_dir=args.categs_dir,
                num_epochs = args.num_epochs,
                batch_size = args.batch_size,
                learning_rate = args.learning_rate,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                bidir=args.bidir,
                dropout=args.dropout,
                optimizer=args.optimizer,
                weight_decay=args.weight_decay,
                log_step=args.log_step)

    args.exp_name = (f"{args.data_dir.split('/')[-1]}__{args.num_epochs}"
                     f"__{args.batch_size}__{args.learning_rate}"
                     f"__{args.hidden_size}__{args.num_layers}"
                     f"__bidir{str(args.bidir)}__{args.weight_decay}"
                     f"__{args.dropout}__{args.optimizer}")

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
        model = ClassifLSTM(config.hidden_size, config.num_layers, SEQ_LEN, config.batch_size,
                            NUM_ROTATIONS, NUM_CLASSES, bool(config.bidir), config.dropout)
        model.to(device)
        model.train()
        # Define the loss function and the optimizer
        loss_function = nn.CrossEntropyLoss()
        optimizer = optimizers[config.optimizer]
        optimizer = optimizer(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        # log model stats
        wandb.watch(model, loss_function, log="all", log_freq=10)

        tr_loss, val_loss = [], []
        rng = np.random.RandomState(23456)  # for shuffling batches
        # Train the model for NUM_EPOCHS epochs
        currBestLoss = 0
        for epoch in range(config.num_epochs):
            print("Starting epoch: ", epoch, flush=True)
            train_epoch_loss, train_acc = train_epoch(model, X_train, Y_train, optimizer, loss_function, config.batch_size, rng)
            val_epoch_loss, val_acc = val_epoch(model, X_val, Y_val, loss_function, config.batch_size, rng)
            wandb.log({"epoch": epoch,
                       "loss_train": np.mean(train_epoch_loss),
                       "loss_val": np.mean(val_epoch_loss),
                       "acc_train": train_acc,
                       "acc_val": val_acc})
            if epoch % config.log_step == 0:
                print(f"Epoch {epoch}:  Tr. loss={sum(train_epoch_loss)/len(train_epoch_loss)} Tr. acc.={train_acc}", flush=True)
                print(f"Epoch {epoch}: Val. loss={sum(val_epoch_loss)/len(val_epoch_loss)} Val. acc.={val_acc}", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
            tr_loss.append(train_epoch_loss)
            val_loss.append(val_epoch_loss)
            if np.mean(val_epoch_loss) < currBestLoss:
                checkpoint = {"epoch": epoch,
                              "state_dict": model.state_dict(),
                              "g_optimizer": optimizer.state_dict()}
                fileName = args.models_dir + "/{}_checkpoint.pth".format(args.exp_name)
                torch.save(checkpoint, fileName)
                currBestLoss = np.mean(val_epoch_loss)
            
            # Data shuffle
            I = np.arange(X_train.shape[0])
            rng.shuffle(I)
            X_train = X_train[I,:,:]
            Y_train = Y_train[I,:,:]


# Data load helper
def load_data(data_dir="../../video_data", categs_dir="../../video_data", key="train"):
    X = load_binary(f"{data_dir}/r6d_{key}.pkl")
    Y = load_binary(f"{categs_dir}/categs_{key}.pkl")
    X = make_equal_len(X, method="cutting+reflect")  # make sequences have equal length, as initially they have different lengths
    X, Y, _ = rmv_clips_nan(X, Y)  # remove those clips containing nan values
    return X, Y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="../../video_data" , help='Directory where results should be stored to and loaded from')
    parser.add_argument('--categs_dir', type=str, default="../../video_data" , help='Directory where categories for each sequence can be loaded from')
    parser.add_argument('--models_dir', type=str, default="models/" , help='Directory where checkpoints are stored.')
    parser.add_argument('--exp_name', type=str, default='experiment', help='Name for the experiment')
    parser.add_argument('--num_epochs', type=int, default=200, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--hidden_size', type=int , default=1024, help="LSTM hidden size")
    parser.add_argument('--num_layers', type=int , default=10, help="Number of LSTM layers")
    parser.add_argument('--bidir', type=str, default="False", help="If 'True' is passed, bidirectional LSTM cells are chosen.")
    parser.add_argument('--weight_decay', type=float, default=1e-3, help="Weight decay rate for regularization.")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout at the end of each LSTM for regularization.")
    parser.add_argument('--optimizer', type=str, default="Adam", help="Available optimizers are Adam, AdamW and NAdam.")
    parser.add_argument('--log_step', type=int , default=2, help="Print logs every log_step epochs")

    args = parser.parse_args()
    args.bidir = True if args.bidir in ["True", "T", "true"] else False
    print(args, flush=True)
    main(args)
