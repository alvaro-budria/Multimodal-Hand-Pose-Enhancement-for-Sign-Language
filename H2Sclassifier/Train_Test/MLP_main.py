import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch import functional as F
import gc
import wandb

sys.path.insert(1, "../../utils")
from load_save_utils import load_binary

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class SentenceClassifier(nn.Module):
    def __init__(self):
        super(SentenceClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.classifier(x)


def main(args):
    wandb.login()

    ## variables
    config = dict(
                data_dir=args.data_dir,
                num_epochs = args.num_epochs,
                batch_size = args.batch_size,
                learning_rate = args.learning_rate,
                optimizer=args.optimizer,
                weight_decay=args.weight_decay,
                log_step=args.log_step)

    args.exp_name = (f"{args.data_dir.split('/')[-1]}__{args.num_epochs}"
                     f"__{args.batch_size}__{args.learning_rate}"
                     f"__{args.weight_decay}"
                     f"__{args.optimizer}")


    with wandb.init(project="B2H-H2S", name=args.exp_name, id=args.exp_name, save_code=True, config=config):
        config = wandb.config

        X_train, Y_train = load_data(data_dir=config.data_dir, key="train")
        X_val, Y_val = load_data(data_dir=config.data_dir, key="val")
        print(f"X_train.shape, Y_train.shape {X_train.shape, Y_train.shape}", flush=True)
        print(f"X_val.shape, Y_val.shape {X_val.shape, Y_val.shape}", flush=True)

        model = SentenceClassifier()
        model.to(device)
        model.train()

        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        loss_function = F.CrossEntropyLoss()

        # log model stats
        wandb.watch(model, loss_function, log="all", log_freq=10)

        tr_loss, val_loss = [], []
        rng = np.random.RandomState(23456)  # for shuffling batches
        # Train the model for NUM_EPOCHS epochs
        currBestAcc = 0
        for epoch in range(config.num_epochs):
            print("Starting epoch: ", epoch, flush=True)
            train_epoch_loss, train_acc = train_epoch(model, X_train, Y_train, optimizer, loss_function, config.batch_size, rng)
            val_epoch_loss, val_acc, (GT, predY) = val_epoch(model, X_val, Y_val, loss_function, config.batch_size, rng)
            print(f"len(GT), len(predY) {len(GT), len(predY)}", flush=True)

            wandb.log({"epoch": epoch,
                       "loss_train": np.mean(train_epoch_loss),
                       "loss_val": np.mean(val_epoch_loss),
                       "acc_train": train_acc,
                       "acc_val": val_acc})
            if epoch % config.log_step == 0:
                print(f"Epoch {epoch}:  Tr. loss={sum(train_epoch_loss)/len(train_epoch_loss)} Tr. acc.={train_acc}", flush=True)
                print(f"Epoch {epoch}: Val. loss={val_epoch_loss} Val. acc.={val_acc}", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
            tr_loss.append(train_epoch_loss)
            val_loss.append(val_epoch_loss)

            print(f"val_epoch_loss, currBestAcc {val_epoch_loss, currBestAcc}", flush=True)
            if val_acc > currBestAcc:
                checkpoint = {"epoch": epoch,
                              "state_dict": model.state_dict(),
                              "g_optimizer": optimizer.state_dict()}
                fileName = args.models_dir + "/{}_checkpoint.pth".format(args.exp_name)
                torch.save(checkpoint, fileName)
                currBestAcc = val_acc

                # save predY here, in the format (GT, predY)
                import csv
                from itertools import zip_longest
                d = [GT, predY]
                export_data = zip_longest(*d, fillvalue='')
                with open('GT_predY.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
                    wr = csv.writer(myfile)
                    wr.writerow(("GT", "predY"))
                    wr.writerows(export_data)
                print("after saving .csv", flush=True)

            # Data shuffle
            I = np.arange(X_train.shape[0])
            rng.shuffle(I)
            X_train = X_train[I,:]
            Y_train = Y_train[I]


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
        y_ = model(inputData)
        epoch_acc += sum(np.argmax(y_[:,:].cpu().detach().numpy(), axis=1) == outputGT.cpu().detach().numpy())

        # Set gradients to 0, compute the loss, gradients, and update the parameters
        optimizer.zero_grad()
        loss = loss_function(y_[:,:], outputGT)
        epoch_loss.append(loss.item())
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    return epoch_loss, epoch_acc/(len(batchinds)*BATCH_SIZE)


def val_epoch(model, train_X, train_Y, loss_function, BATCH_SIZE, rng):
    val_loss = 0
    epoch_acc = 0
    predY, GT = [], []
    model.eval()
    batchinds = np.arange(train_X.shape[0] // BATCH_SIZE)
    rng.shuffle(batchinds)
    with torch.no_grad():
        for bii, bi in enumerate(batchinds):
            print(f"bii: {bii}", flush=True)
            ## setting batch data
            idxStart = bi * BATCH_SIZE
            inputData = train_X[idxStart:(idxStart + BATCH_SIZE), :, :]
            outputGT = train_Y[idxStart:(idxStart + BATCH_SIZE)]
            inputData = Variable(torch.from_numpy(inputData).float()).to(device)
            outputGT = Variable(torch.from_numpy(outputGT-1)).to(device)
            print(f"outputGT.shape {outputGT.shape}", flush=True)
            GT = GT + outputGT.cpu().numpy().tolist()

            # Forward pass
            y_ = model(inputData)
            print(f"y_.shape {y_.shape}", flush=True)
            print(f"np.argmax(y_[:,:].cpu().detach().numpy(), axis=1).shape {np.argmax(y_[:,:].cpu().detach().numpy(), axis=1).shape}", flush=True)
            predY = predY + np.argmax(y_[:,:].cpu().detach().numpy(), axis=1).tolist()
            epoch_acc += sum(np.argmax(y_[:,:].cpu().detach().numpy(), axis=1) == outputGT.cpu().detach().numpy())

            # Compute loss
            loss = loss_function(y_[:,:], outputGT)
            val_loss += loss.item()
    return val_loss,  epoch_acc/(len(batchinds)*BATCH_SIZE), (GT, predY)


# Data load helper
def load_data(data_dir="../../video_data_groupByClip", key="train"):
    X = load_binary(f"{data_dir}/True{key}_wordBert_sentEmbeddings.pkl")
    Y = load_binary(f"{data_dir}/Truecategs_{key}.pkl")
    print(f"type(X) {type(X)}", flush=True)
    print(f"type(Y) {type(Y)}", flush=True)
    return X, Y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="../../video_data" , help='Directory where results should be stored to and loaded from')
    parser.add_argument('--models_dir', type=str, default="models/" , help='Directory where checkpoints are stored.')
    parser.add_argument('--exp_name', type=str, default='experiment', help='Name for the experiment')
    parser.add_argument('--num_epochs', type=int, default=200, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-3, help="Weight decay rate for regularization.")
    parser.add_argument('--optimizer', type=str, default="Adam", help="Available optimizers are Adam, AdamW and NAdam.")
    parser.add_argument('--log_step', type=int , default=2, help="Print logs every log_step epochs")

    args = parser.parse_args()
    print(args, flush=True)
    main(args)
