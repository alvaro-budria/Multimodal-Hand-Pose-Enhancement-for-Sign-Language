import os
import shutil
import sys
import argparse

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append('./viz')
import modelZoo

sys.path.append("./utils")
from constants import *
from load_save_utils import *
from standardization_utils import *

# experiment logging
import wandb


#######################################################
## main training function
#######################################################
def main(args):
    wandb.login()

    ## variables
    config = dict(
        epochs = args.num_epochs,
        batch_size = args.batch_size,
        learning_rate = args.learning_rate,
        model = args.model,
        pipeline = args.pipeline,
        epochs_train_disc = args.epochs_train_disc,
        disc_label_smooth = args.disc_label_smooth,
        data_dir=args.data_dir)

    ## DONE variables
    with wandb.init(project="B2H-H2S", name=args.exp_name, id=args.exp_name, save_code=True, config=config):
        config = wandb.config

        feature_in_dim, feature_out_dim = FEATURE_MAP[config.pipeline]
        currBestLoss = 1e9
        rng = np.random.RandomState(23456)
        torch.manual_seed(23456)
        torch.cuda.manual_seed(23456)
        ## load data from saved files
        data_tuple = load_data(args, rng, config.data_dir)
        if args.require_text or args.require_image:
            print("Using text/image embeds as input to the model.", flush=True)
            train_X, train_Y, val_X, val_Y, train_feats, val_feats = data_tuple
        else:
            train_X, train_Y, val_X, val_Y = data_tuple
            train_feats, val_feats = None, None
        ## DONE: load data from saved files

        ## set up generator model
        mod = MODELS[config.model]
        print(f"mod: {mod}", flush=True)
        generator = getattr(modelZoo, mod)()
        if mod == "regressor_fcn_bn_32_b2h":
            generator.build_net(feature_in_dim, feature_out_dim, require_image=args.require_image)
        else:
            generator.build_net(feature_in_dim, feature_out_dim, require_text=args.require_text)
        generator.to(device)
        g_optimizer = torch.optim.Adam(generator.parameters(), lr=config.learning_rate, weight_decay=0)
        if args.use_checkpoint:
            loaded_state = torch.load(os.path.join(args.model_path, f"lastCheckpoint_{args.exp_name}.pth"), map_location=lambda storage, loc: storage)
            generator.load_state_dict(loaded_state['state_dict'], strict=False)
            g_optimizer.load_state_dict(loaded_state['g_optimizer'])
        reg_criterion = LOSSES[args.loss]
        if args.loss=="RobustLoss":
            reg_criterion = reg_criterion(num_dims=train_Y.shape[1]*train_Y.shape[2],
                                          float_dtype=torch.float32,
                                          device="cuda:0")
        g_scheduler = ReduceLROnPlateau(g_optimizer, 'min', patience=1000000, factor=0.5, min_lr=1e-5)
        generator.train()
        wandb.watch(generator, reg_criterion, log="all", log_freq=10)

        ## set up discriminator model
        args.model = 'regressor_fcn_bn_discriminator'
        discriminator = getattr(modelZoo, args.model)()
        discriminator.build_net(feature_out_dim)
        discriminator.to(device)
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.learning_rate, weight_decay=0)#1e-5)
        if args.use_checkpoint:
            loaded_state = torch.load(os.path.join(args.model_path, f"discriminator_{args.exp_name}.pth"), map_location=lambda storage, loc: storage)
            discriminator.load_state_dict(loaded_state['state_dict'], strict=False)
            d_optimizer.load_state_dict(loaded_state['d_optimizer'])        
        gan_criterion = nn.MSELoss()
        d_scheduler = ReduceLROnPlateau(g_optimizer, 'min', patience=1000000, factor=0.5, min_lr=1e-5)
        discriminator.train()
        wandb.watch(discriminator, gan_criterion, log="all", log_freq=10)
        ## DONE model

        ## training job
        prev_save_epoch = 0
        patience = args.patience
        for epoch in range(args.num_epochs):
            args.epoch = epoch
            # train discriminator
            if epoch > 100 and (epoch - prev_save_epoch) > patience:
                print('early stopping at:', epoch-1, flush=True)
                break
            if epoch > 0 and (config.epochs_train_disc==0 or epoch % config.epochs_train_disc==0):
                train_discriminator(args, generator, discriminator, gan_criterion, d_optimizer, train_X, train_Y, epoch, train_feats=train_feats)
            else:
                train_generator(args, generator, discriminator, reg_criterion, gan_criterion, g_optimizer, train_X, train_Y, epoch, train_summary_writer, train_feats=train_feats)
                currBestLoss, prev_save_epoch = val_generator(args, generator, discriminator, reg_criterion, g_optimizer, d_optimizer, g_scheduler, d_scheduler, val_X, val_Y, currBestLoss, prev_save_epoch, epoch, val_summary_writer, val_feats=val_feats)
            # Data shuffle
            I = np.arange(len(train_X))
            rng.shuffle(I)
            train_X = train_X[I]
            train_Y = train_Y[I]
            if args.require_text or args.require_image:
                train_feats = train_feats[I]

    shutil.copyfile(lastCheckpoint, args.model_path + f"/lastCheckpoint_{args.exp_name}.pth")  #  name last checkpoint as "lastCheckpoint.pth"


#######################################################
## local helper methods
#######################################################

## function to load data from external files
def load_data(args, rng, data_dir):

    def fetch_data(set="train"):
        ## load from external files
        path = os.path.join(data_dir, DATA_PATHS_r6d[set])

        if args.embeds_type == "normal":
            text_path = f"{data_dir}/{set}_sentence_embeddings.pkl"
        elif args.embeds_type == "average":
            text_path = f"{data_dir}/average_{set}_sentence_embeddings.pkl"
        image_path = f"{data_dir}/{set}_vid_feats.pkl"
        print(f"image_path {image_path}", flush=True)

        data_path = os.path.join(args.base_path, path)
        curr_p0, curr_p1 = load_windows(data_path, args.pipeline, require_text=args.require_text, text_path=text_path,
                                        require_image=args.require_image, image_path=image_path)
        if args.require_text or args.require_image:
            feats = curr_p0[1]
            curr_p0 = curr_p0[0]
            #return curr_p0[:args.batch_size], curr_p1[:args.batch_size], text[:args.batch_size]
            return curr_p0, curr_p1, feats
        return curr_p0, curr_p1, None

    train_X, train_Y, train_feats = fetch_data("train")
    val_X, val_Y, val_feats = fetch_data("val")
    if args.pipeline == "wh2wh":
        train_X, val_X = train_X[:,:,6*6:], val_X[:,:,6*6:]  # keep hands for training

    print(f"train_X.shape, train_Y.shape {train_X.shape, train_Y.shape}", flush=True)
    if args.require_text or args.require_image:
        print(f"train_feats.shape: {train_feats.shape}", flush=True)
    train_X, train_Y, train_feats = rmv_clips_nan(train_X, train_Y, train_feats)
    val_X, val_Y, val_feats = rmv_clips_nan(val_X, val_Y, val_feats)
    assert not np.any(np.isnan(train_X)) and not np.any(np.isnan(train_Y)) and not np.any(np.isnan(val_X)) and not np.any(np.isnan(val_Y))
    print(f"train_X.shape, train_Y.shape {train_X.shape, train_Y.shape}", flush=True)
    if args.require_text or args.require_image:
        print(train_feats.shape, flush=True)

    print("-"*20 + "train" + "-"*20, flush=True)
    print('===> in/out', train_X.shape, train_Y.shape, flush=True)
    print(flush=True)
    print("-"*20 + "val" + "-"*20, flush=True)
    print('===> in/out', val_X.shape, val_Y.shape, flush=True)
    if args.require_text or args.require_image:
        print("===> feats", train_feats.shape, flush=True)
    ## DONE load from external files

    train_X = np.swapaxes(train_X, 1, 2).astype(np.float32)
    train_Y = np.swapaxes(train_Y, 1, 2).astype(np.float32)
    val_X = np.swapaxes(val_X, 1, 2).astype(np.float32)
    val_Y = np.swapaxes(val_Y, 1, 2).astype(np.float32)
    body_mean_X, body_std_X, body_mean_Y, body_std_Y = calc_standard(train_X, train_Y, args.pipeline)

    mkdir(args.model_path)
    np.savez_compressed(os.path.join(args.model_path, '{}{}_preprocess_core.npz'.format(args.exp_name, args.pipeline)), 
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
    if args.require_text or args.require_image:
        train_feats = train_feats[I]
        return (train_X, train_Y, val_X, val_Y, train_feats, val_feats)
    ## DONE shuffle and set train/validation
    return (train_X, train_Y, val_X, val_Y)


## calc temporal deltas within sequences
def calc_motion(tensor):
    res = tensor[:,:,:1] - tensor[:,:,:-1]
    return res


## training discriminator function
def train_discriminator(args, generator, discriminator, gan_criterion, d_optimizer, train_X, train_Y, epoch, train_feats=None):
    generator.eval()
    discriminator.train()
    batchinds = np.arange(train_X.shape[0] // args.batch_size)   # integer division so drop last incomplete minibatch
    totalSteps = len(batchinds)
    avgLoss = 0.
    for bii, bi in enumerate(batchinds):
        ## setting batch data
        idxStart = bi * args.batch_size
        inputData_np = train_X[idxStart:(idxStart + args.batch_size), :, :]
        outputData_np = train_Y[idxStart:(idxStart + args.batch_size), :, :]
        inputData = Variable(torch.from_numpy(inputData_np)).to(device)
        outputGT = Variable(torch.from_numpy(outputData_np)).to(device)

        featsData = None
        if args.require_text or args.require_image:
            featsData_np = train_feats[idxStart:(idxStart + args.batch_size), :]
            featsData = Variable(torch.from_numpy(featsData_np)).to(device)
        ## DONE setting batch data

        with torch.no_grad():
            fake_data = generator(inputData, feats_=featsData).detach()

        fake_motion = calc_motion(fake_data)
        real_motion = calc_motion(outputGT)
        fake_score = discriminator(fake_motion)
        real_score = discriminator(real_motion)
        target_fake = torch.zeros_like(fake_score)
        target_real = torch.ones_like(real_score)
        if args.disc_label_smooth:
            target_fake, target_real = target_fake.fill_(0.1), target_real.fill_(0.9)

        d_loss = gan_criterion(fake_score, target_fake) + gan_criterion(real_score, target_real)
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        avgLoss += d_loss.item() * args.batch_size

    print(f'Epoch [{epoch}/{args.num_epochs-1}], Tr. Disc. Loss: {avgLoss / (totalSteps * args.batch_size)}', flush=True)
    wandb.log({"epoch": epoch, "loss_train_disc": avgLoss / (totalSteps * args.batch_size)})


## training generator function
def train_generator(args, generator, discriminator, reg_criterion, gan_criterion, g_optimizer,
                    train_X, train_Y, epoch, train_summary_writer, clip_grad=False, train_feats=None):
    discriminator.eval()
    generator.train()
    batchinds = np.arange(train_X.shape[0] // args.batch_size)
    totalSteps = len(batchinds)
    avgLoss = 0.

    for bii, bi in enumerate(batchinds):
        ## setting batch data
        idxStart = bi * args.batch_size
        inputData_np = train_X[idxStart:(idxStart + args.batch_size), :, :]
        outputData_np = train_Y[idxStart:(idxStart + args.batch_size), :, :]
        inputData = Variable(torch.from_numpy(inputData_np)).to(device)
        outputGT = Variable(torch.from_numpy(outputData_np)).to(device)

        featsData = None
        if args.require_text or args.require_image:
            featsData_np = train_feats[idxStart:(idxStart + args.batch_size), :]
            featsData = Variable(torch.from_numpy(featsData_np)).to(device)
        ## DONE setting batch data

        output = generator(inputData, feats_=featsData)
        fake_motion = calc_motion(output)
        with torch.no_grad():
            fake_score = discriminator(fake_motion)
        fake_score = fake_score.detach()

        if args.loss=="RobustLoss":
            output2 = torch.reshape(output, (output.shape[0],-1))
            outputGT2 = torch.reshape(outputGT, (output.shape[0],-1))
            g_loss = torch.mean(reg_criterion.lossfun((output2 - outputGT2))) \
                     + gan_criterion(fake_score, torch.ones_like(fake_score))
        else:
            g_loss = reg_criterion(output, outputGT) + gan_criterion(fake_score, torch.ones_like(fake_score))
        g_optimizer.zero_grad()
        g_loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1)
        g_optimizer.step()

        avgLoss += g_loss.item() * args.batch_size
        if bii % args.log_step == 0 or bii ==len(batchinds):
            print('Epoch [{}/{}], Step [{}/{}], Tr. Loss: {:.4f}, Tr. Perplexity: {:5.4f}'.format(args.epoch, args.num_epochs-1, bii+1, totalSteps,
                                                                                                  avgLoss / (totalSteps * args.batch_size), 
                                                                                                  np.exp(avgLoss / (totalSteps * args.batch_size))), flush=True)
    
    print('Epoch [{}/{}], Step [{}/{}], Tr. Loss: {:.4f}, Tr. Perplexity: {:5.4f}'.format(args.epoch, args.num_epochs-1, bii+1, totalSteps,
                                                                                                  avgLoss / (totalSteps * args.batch_size), 
                                                                                                  np.exp(avgLoss / (totalSteps * args.batch_size))), flush=True)
    wandb.log({"epoch": epoch, "loss_train_gen": avgLoss / (totalSteps * args.batch_size)})


## validating generator function
def val_generator(args, generator, discriminator, reg_criterion, g_optimizer, d_optimizer, g_scheduler, d_scheduler,
                  val_X, val_Y, currBestLoss, prev_save_epoch, epoch, val_summary_writer, val_feats=None):
    testLoss = 0
    generator.eval()
    discriminator.eval()
    val_batch_size = args.batch_size // 2
    batchinds = np.arange(val_X.shape[0] // val_batch_size)  # integer division so last incomplete batch gets dropped
    totalSteps = len(batchinds)

    for bii, bi in enumerate(batchinds):
        ## setting batch data
        idxStart = bi * val_batch_size
        inputData_np = val_X[idxStart:(idxStart + val_batch_size), :, :]
        outputData_np = val_Y[idxStart:(idxStart + val_batch_size), :, :]
        inputData = Variable(torch.from_numpy(inputData_np)).to(device)
        outputGT = Variable(torch.from_numpy(outputData_np)).to(device)

        featsData = None
        if args.require_text or args.require_image:
            featsData_np = val_feats[idxStart:(idxStart + val_batch_size), :]
            featsData = Variable(torch.from_numpy(featsData_np)).to(device)
        ## DONE setting batch data
        
        output = generator(inputData, feats_=featsData)
        if args.loss=="RobustLoss":
            output2 = torch.reshape(output, (output.shape[0],-1))
            outputGT2 = torch.reshape(outputGT, (output.shape[0],-1))
            g_loss = torch.mean(reg_criterion.lossfun((output2 - outputGT2)))
        else:
            g_loss = reg_criterion(output, outputGT)
        testLoss += g_loss.item() * val_batch_size

    testLoss /= totalSteps * val_batch_size
    wandb.log({"loss_val_gen": testLoss})
    print('Epoch [{}/{}], Step [{}/{}], Val. Loss: {:.4f}, Val. Perplexity: {:5.4f}, LR: {:e}'.format(args.epoch, args.num_epochs-1, bii, totalSteps-1, 
                                                                                                      testLoss, 
                                                                                                      np.exp(testLoss),
                                                                                                      g_optimizer.param_groups[0]["lr"]), flush=True)
    print('----------------------------------', flush=True)
    g_scheduler.step(testLoss)
    d_scheduler.step(testLoss)
    if testLoss < currBestLoss:
        prev_save_epoch = args.epoch
        # store generator
        checkpoint = {'epoch': args.epoch,
                      'state_dict': generator.state_dict(),
                      'g_optimizer': g_optimizer.state_dict()}
        fileName = args.model_path + '/{}_checkpoint.pth'.format(args.exp_name)
        torch.save(checkpoint, fileName)
        currBestLoss = testLoss
        global lastCheckpoint
        lastCheckpoint = fileName

        # store discriminator
        checkpoint = {'epoch': args.epoch,
                      'state_dict': discriminator.state_dict(),
                      'd_optimizer': d_optimizer.state_dict()}
        fileName = args.model_path + f'/discriminator_{args.exp_name}.pth'
        torch.save(checkpoint, fileName)

    return currBestLoss, prev_save_epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default="./", help='path to the directory where the data files are stored')
    parser.add_argument('--pipeline', type=str, default='arm2wh', help='pipeline specifying which input/output joints to use')
    parser.add_argument('--num_epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate for training G and D')
    parser.add_argument('--require_text', action="store_true", help="use additional text embeddings or not")
    parser.add_argument('--require_image', action="store_true", help="use additional image features or not")
    parser.add_argument('--embeds_type', type=str, default="normal" , help='if "normal", use normal text embeds; if "avg", use avg text embeds')
    parser.add_argument('--model_path', type=str, default="models/" , help='path for saving trained models')
    parser.add_argument('--log_step', type=int , default=25, help='step size for prining log info')
    parser.add_argument('--tag', type=str, default='', help='prefix for naming purposes')
    parser.add_argument('--exp_name', type=str, default='experiment', help='name for the experiment')
    parser.add_argument('--patience', type=int, default=100, help='amount of epochs without loss improvement before termination')
    parser.add_argument('--use_checkpoint', action="store_true", help="use checkpoint from which to start training")
    parser.add_argument('--epochs_train_disc', type=int , default=3, help='train the discriminator every epochs_train_disc epochs')
    parser.add_argument('--model', type=str, default="v1" , help='model architecture to be used')
    parser.add_argument('--disc_label_smooth', action="store_true", help="if True, use label smoothing for the discriminator")
    parser.add_argument('--data_dir', type=str, default="video_data" , help='directory where results should be stored and loaded from')
    parser.add_argument('--loss', type=str, default="L1" , help='Loss to optimize the generator over')

    args = parser.parse_args()
    print(args, flush=True)

    main(args)
