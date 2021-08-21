import os
import shutil
import sys
import argparse

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

sys.path.append('./viz')
from track_grads import plot_grad_flow
import modelZoo
from utils import *


DATA_PATHS = {
        "train": "video_data/r6d_train.pkl",
        "val": "video_data/r6d_val.pkl",
        "test": "video_data/r6d_test.pkl"
        }

TEXT_PATHS = {
    "train": "video_data/train_sentence_embeddings.pkl",
    "val": "video_data/val_sentence_embeddings.pkl",
    "test": "video_data/test_sentence_embeddings.pkl"
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lastCheckpoint = ""

#######################################################
## main training function
#######################################################
def main(args):
    ## variables
    learning_rate = args.learning_rate
    pipeline = args.pipeline

    feature_in_dim, feature_out_dim = FEATURE_MAP[pipeline]
    feats = pipeline.split('2')
    in_feat, out_feat = feats[0], feats[1]
    currBestLoss = 1e3
    rng = np.random.RandomState(23456)
    torch.manual_seed(23456)
    torch.cuda.manual_seed(23456)
    ## DONE variables

    ## set up generator model
    args.model = 'regressor_fcn_bn_32'
    generator = getattr(modelZoo, args.model)()
    generator.build_net(feature_in_dim, feature_out_dim, require_text=args.require_text)
    generator.to(device)
    reg_criterion = nn.L1Loss()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, weight_decay=1e-5)
    g_scheduler = ReduceLROnPlateau(g_optimizer, 'min', patience=2*args.patience//(3*2), factor=0.5, min_lr=1e-8)
    generator.train()

    ## set up discriminator model
    args.model = 'regressor_fcn_bn_discriminator'
    discriminator = getattr(modelZoo, args.model)()
    discriminator.build_net(feature_out_dim)
    discriminator.to(device)
    gan_criterion = nn.MSELoss()
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=1e-5)
    d_scheduler = ReduceLROnPlateau(g_optimizer, 'min', patience=2*args.patience//(3*2), factor=0.5, min_lr=1e-8)
    discriminator.train()
    ## DONE model

    ## load data from saved files
    data_tuple = load_data(args, rng)
    if args.require_text:
        train_X, train_Y, val_X, val_Y, train_text, val_text = data_tuple
    else:
        train_X, train_Y, val_X, val_Y = data_tuple
        train_text, val_text = None, None
    ## DONE: load data from saved files

    ## setup results logger
    mkdir("logs/"); mkdir('logs/train/'); mkdir('logs/val/')
    train_log_dir = 'logs/train/' + args.tag
    val_log_dir   = 'logs/val/' + args.tag
    train_summary_writer = SummaryWriter(train_log_dir)
    val_summary_writer   = SummaryWriter(val_log_dir)
    mkdir(args.model_path) # create model checkpoints directory if it doesn't exist
    ## DONE setup logger 

    ## training job
    kld_weight = 0.05
    prev_save_epoch = 0
    patience = args.patience
    for epoch in range(args.num_epochs):
        args.epoch = epoch
        # train discriminator
        if epoch > 100 and (epoch - prev_save_epoch) > patience:
            print('early stopping at:', epoch-1, flush=True)
            break

        if epoch > 0 and epoch % 3 == 0:
            train_discriminator(args, rng, generator, discriminator, gan_criterion, d_optimizer, train_X, train_Y, train_text=train_text)
        else:
            train_generator(args, rng, generator, discriminator, reg_criterion, gan_criterion, g_optimizer, train_X, train_Y, epoch, train_summary_writer, train_text=train_text)
            currBestLoss, prev_save_epoch = val_generator(args, generator, discriminator, reg_criterion, g_optimizer, g_scheduler, d_scheduler, val_X, val_Y, currBestLoss, prev_save_epoch, epoch, val_summary_writer, val_text=val_text)
    
    shutil.copyfile(lastCheckpoint, args.model_path + "/lastCheckpoint.pth")  #  name last checkpoint as "lastCheckpoint.pth"

    train_summary_writer.flush()
    val_summary_writer.flush()


#######################################################
## local helper methods
#######################################################

## function to load data from external files
def load_data(args, rng):

    def fetch_data(set="train"):
        ## load from external files
        path = DATA_PATHS[set]
        text_path = TEXT_PATHS[set]
        data_path = os.path.join(args.base_path, path)
        curr_p0, curr_p1 = load_windows(data_path, args.pipeline, require_text=args.require_text, text_path=text_path)
        if args.require_text:
            text = curr_p0[1]
            curr_p0 = curr_p0[0]
            return curr_p0, curr_p1, text
        return curr_p0, curr_p1, None

    train_X, train_Y, train_text = fetch_data("train")
    val_X, val_Y, val_text = fetch_data("val")

    print(train_X.shape, train_Y.shape, flush=True)
    if args.require_text:
        print(train_text.shape)
    print(train_text)
    train_X, train_Y, train_text = rmv_clips_nan(train_X, train_Y, train_text)
    val_X, val_Y, val_text = rmv_clips_nan(val_X, val_Y, val_text)
    assert not np.any(np.isnan(train_X)) and not np.any(np.isnan(train_Y)) and not np.any(np.isnan(val_X)) and not np.any(np.isnan(val_Y))
    print(train_X.shape, train_Y.shape, flush=True)
    if args.require_text:
        print(train_text.shape)

    print("-"*20 + "train" + "-"*20, flush=True)
    print('===> in/out', train_X.shape, train_Y.shape, flush=True)
    print(flush=True)
    print("-"*20 + "val" + "-"*20, flush=True)
    print('===> in/out', val_X.shape, val_Y.shape, flush=True)
    if args.require_text:
        print("===> text", train_text.shape, flush=True)
    ## DONE load from external files

    train_X = np.swapaxes(train_X, 1, 2).astype(np.float32)
    train_Y = np.swapaxes(train_Y, 1, 2).astype(np.float32)
    val_X = np.swapaxes(val_X, 1, 2).astype(np.float32)
    val_Y = np.swapaxes(val_Y, 1, 2).astype(np.float32)
    body_mean_X, body_std_X, body_mean_Y, body_std_Y = calc_standard(train_X, train_Y, args.pipeline)
    
    mkdir(args.model_path)
    np.savez_compressed(os.path.join(args.model_path, '{}{}_preprocess_core.npz'.format(args.tag, args.pipeline)), 
                        body_mean_X=body_mean_X, body_std_X=body_std_X,
                        body_mean_Y=body_mean_Y, body_std_Y=body_std_Y)

    print(f"train_X: {train_X.shape}; val_X: {val_X.shape}", flush=True)
    print(f"body_mean_X: {body_mean_X.shape}; body_std_X: {body_std_X.shape}", flush=True)
    
    train_X = (train_X - body_mean_X) / body_std_X
    val_X = (val_X - body_mean_X) / body_std_X
    train_Y = (train_Y - body_mean_Y) / body_std_Y
    val_Y = (val_Y - body_mean_Y) / body_std_Y
    print("===> standardization done", flush=True)

    # Data shuffle
    I = np.arange(len(train_X))
    rng.shuffle(I)
    train_X = train_X[I]
    train_Y = train_Y[I]
    if args.require_text:
        train_text = train_text[I]
        return (train_X, train_Y, val_X, val_Y, train_text, val_text)
    ## DONE shuffle and set train/validation
    return (train_X, train_Y, val_X, val_Y)


## calc temporal deltas within sequences
def calc_motion(tensor):
    res = tensor[:,:,:1] - tensor[:,:,:-1]
    return res


## training discriminator function
def train_discriminator(args, rng, generator, discriminator, gan_criterion, d_optimizer, train_X, train_Y, train_text=None):
    generator.eval()
    discriminator.train()
    batchinds = np.arange(train_X.shape[0] // args.batch_size)   # integer division so drop last incomplete batch
    totalSteps = len(batchinds)
    rng.shuffle(batchinds)

    for bii, bi in enumerate(batchinds):
        ## setting batch data
        idxStart = bi * args.batch_size
        inputData_np = train_X[idxStart:(idxStart + args.batch_size), :, :]
        outputData_np = train_Y[idxStart:(idxStart + args.batch_size), :, :]
        inputData = Variable(torch.from_numpy(inputData_np)).to(device)
        outputGT = Variable(torch.from_numpy(outputData_np)).to(device)

        textData = None
        if args.require_text:
            textData_np = train_text[idxStart:(idxStart + args.batch_size), :]
            textData = Variable(torch.from_numpy(textData_np)).to(device)
        ## DONE setting batch data

        with torch.no_grad():
            fake_data = generator(inputData, text_=textData).detach()

        fake_motion = calc_motion(fake_data)
        real_motion = calc_motion(outputGT)
        fake_score = discriminator(fake_motion)
        real_score = discriminator(real_motion)

        d_loss = gan_criterion(fake_score, torch.zeros_like(fake_score)) + gan_criterion(real_score, torch.ones_like(real_score))
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()


## training generator function
def train_generator(args, rng, generator, discriminator, reg_criterion, gan_criterion, g_optimizer,
                    train_X, train_Y, epoch, train_summary_writer, clip_grad=False, train_text=None):
    discriminator.eval()
    generator.train()
    batchinds = np.arange(train_X.shape[0] // args.batch_size)
    totalSteps = len(batchinds)
    rng.shuffle(batchinds)
    avgLoss = 0.

    for bii, bi in enumerate(batchinds):
        ## setting batch data
        idxStart = bi * args.batch_size
        inputData_np = train_X[idxStart:(idxStart + args.batch_size), :, :]
        outputData_np = train_Y[idxStart:(idxStart + args.batch_size), :, :]
        inputData = Variable(torch.from_numpy(inputData_np)).to(device)
        outputGT = Variable(torch.from_numpy(outputData_np)).to(device)

        textData = None
        if args.require_text:
            textData_np = train_text[idxStart:(idxStart + args.batch_size), :]
            textData = Variable(torch.from_numpy(textData_np)).to(device)
        ## DONE setting batch data

        output = generator(inputData, text_=textData)
        fake_motion = calc_motion(output)
        with torch.no_grad():
            fake_score = discriminator(fake_motion)
        fake_score = fake_score.detach()

        g_loss = reg_criterion(output, outputGT) + gan_criterion(fake_score, torch.ones_like(fake_score))
        g_optimizer.zero_grad()
        g_loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1)
        if epoch % 1 == 0 and bii == 0:  # every epoch, generate gradient flow chart
            mkdir("viz_grads")
            plot_grad_flow(generator.named_parameters(), f'viz_grads/grad_graph_e{epoch}')
        g_optimizer.step()

        avgLoss += g_loss.item() * args.batch_size
        if bii % args.log_step == 0:
            print('Epoch [{}/{}], Step [{}/{}], Tr. Loss: {:.4f}, Tr. Perplexity: {:5.4f}'.format(args.epoch, args.num_epochs-1, bii+1, totalSteps,
                                                                                                  avgLoss / (totalSteps * args.batch_size), 
                                                                                                  np.exp(avgLoss / (totalSteps * args.batch_size))), flush=True)
    # Save data to tensorboard                             
    train_summary_writer.add_scalar('Tr. loss', avgLoss / (totalSteps * args.batch_size), epoch)


## validating generator function
def val_generator(args, generator, discriminator, reg_criterion, g_optimizer, g_scheduler, d_scheduler, val_X, val_Y, currBestLoss, prev_save_epoch, epoch, val_summary_writer, val_text=None):
    testLoss = 0
    generator.eval()
    discriminator.eval()
    batchinds = np.arange(val_X.shape[0] // args.batch_size)  # integer division so drop last incomplete batch
    totalSteps = len(batchinds)

    for bii, bi in enumerate(batchinds):
        ## setting batch data
        idxStart = bi * args.batch_size
        inputData_np = val_X[idxStart:(idxStart + args.batch_size), :, :]
        outputData_np = val_Y[idxStart:(idxStart + args.batch_size), :, :]
        inputData = Variable(torch.from_numpy(inputData_np)).to(device)
        outputGT = Variable(torch.from_numpy(outputData_np)).to(device)

        textData = None
        if args.require_text:
            textData_np = val_text[idxStart:(idxStart + args.batch_size), :]
            textData = Variable(torch.from_numpy(textData_np)).to(device)
        ## DONE setting batch data
        
        output = generator(inputData, text_=textData)
        g_loss = reg_criterion(output, outputGT)
        testLoss += g_loss.item() * args.batch_size

    testLoss /= totalSteps * args.batch_size
    print('Epoch [{}/{}], Step [{}/{}], Val. Loss: {:.4f}, Val. Perplexity: {:5.4f}, LR: {:e}'.format(args.epoch, args.num_epochs-1, bii+1, totalSteps, 
                                                                                                      testLoss, 
                                                                                                      np.exp(testLoss),
                                                                                                      g_optimizer.param_groups[0]["lr"]), flush=True)
    # Save data to tensorboard                             
    val_summary_writer.add_scalar('Val. loss', testLoss, epoch)
    print('----------------------------------', flush=True)
    g_scheduler.step(testLoss)
    d_scheduler.step(testLoss)
    if testLoss < currBestLoss:
        prev_save_epoch = args.epoch
        checkpoint = {'epoch': args.epoch,
                      'state_dict': generator.state_dict(),
                      'g_optimizer': g_optimizer.state_dict()}
        fileName = args.model_path + '/{}{}_checkpoint_e{}_loss{:.4f}.pth'.format(args.tag, args.pipeline, args.epoch, testLoss)
        torch.save(checkpoint, fileName)
        currBestLoss = testLoss
        global lastCheckpoint
        lastCheckpoint = fileName

        # for f in os.listdir(args.model_path):  # remove past checkpoints to avod blowing up disk memory
        #     if f != fileName:
        #         os.remove(os.path.join(args.model_path, f))

    return currBestLoss, prev_save_epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default="./", help='path to the directory where the data files are stored')
    parser.add_argument('--pipeline', type=str, default='arm2wh', help='pipeline specifying which input/output joints to use')
    parser.add_argument('--num_epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate for training G and D')
    parser.add_argument('--require_text', action="store_true", help="use additional text feature or not")
    parser.add_argument('--model_path', type=str, default="models/" , help='path for saving trained models')
    parser.add_argument('--log_step', type=int , default=25, help='step size for prining log info')
    parser.add_argument('--tag', type=str, default='', help='prefix for naming purposes')
    parser.add_argument('--patience', type=int, default=100, help='amount of epochs without loss improvement before termination') 

    args = parser.parse_args()
    print(args, flush=True)
    main(args)
