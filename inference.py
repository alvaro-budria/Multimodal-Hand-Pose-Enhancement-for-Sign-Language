import argparse
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import wandb

sys.path.append('./3DposeEstimator')
import skeletalModel
from utils import *
import modelZoo
import viz.viz_3d as viz


def main(args):
    ## variable initializations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.RandomState(23456)
    torch.manual_seed(23456)
    torch.cuda.manual_seed(23456)
    pipeline = args.pipeline
    feature_in_dim, feature_out_dim = FEATURE_MAP[pipeline]
    print(f"feature_in_dim, feature_out_dim: {feature_in_dim}, {feature_out_dim}")
    ## DONE variable initializations

    ## set up model / load pretrained model
    if args.model == "v1":
        args.model = "regressor_fcn_bn_32"
    elif args.model == "v2":
        args.model = "regressor_fcn_bn_32_v2"
    model = getattr(modelZoo,args.model)()
    model.build_net(feature_in_dim, feature_out_dim, require_text=args.require_text)
    pretrained_model = args.checkpoint
    loaded_state = torch.load(pretrained_model, map_location=lambda storage, loc: storage)
    model.load_state_dict(loaded_state['state_dict'], strict=False)
    model = model.eval()
    model.to(device)
    criterion = nn.L1Loss()
    ## DONE set up model/ load pretrained model

    ## load/prepare data from external files
    args.data_dir = "video_data/r6d_train.pkl" #  to make inference on train set
    #test_X, test_Y = load_windows(args.data_dir, args.pipeline, require_text=args.require_text, text_path="video_data/test_sentence_embeddings.pkl")
    test_X, test_Y = load_windows(args.data_dir, args.pipeline, require_text=args.require_text, text_path="video_data/train_sentence_embeddings.pkl")
    text_text = None
    if args.require_text:
        test_text = test_X[1]
        test_X = test_X[0]
    print(test_X.shape, test_Y.shape, flush=True)
    if args.require_text:
        print(test_text.shape)
    test_X, test_Y, text_text = rmv_clips_nan(test_X, test_Y, text_text)
    assert not np.any(np.isnan(test_X)) and not np.any(np.isnan(test_Y))
    print(test_X.shape, test_Y.shape, flush=True)
    input_feats = test_X.copy()

    test_X = np.swapaxes(test_X, 1, 2).astype(np.float32)
    test_Y = np.swapaxes(test_Y, 1, 2).astype(np.float32)

    # standardize
    checkpoint_dir = os.path.split(pretrained_model)[0]
    model_tag = os.path.basename(args.checkpoint).split(args.pipeline)[0]
    preprocess = np.load(os.path.join(checkpoint_dir,'{}{}_preprocess_core.npz'.format(args.tag, args.pipeline)))
    body_mean_X = preprocess['body_mean_X']
    body_std_X = preprocess['body_std_X']
    body_mean_Y = preprocess['body_mean_Y']
    body_std_Y = preprocess['body_std_Y']
    test_X = (test_X - body_mean_X) / body_std_X
    test_Y = (test_Y - body_mean_Y) / body_std_Y
    ## DONE load/prepare data from external files

    ## pass loaded data into inference
    test_X, test_Y = torch.from_numpy(test_X), torch.from_numpy(test_Y)
    assert not torch.isnan(test_X).any() and not torch.isnan(test_Y).any()
    if args.require_text:
        test_text = torch.from_numpy(test_text)
    error = 0
    output = None
    model.eval()
    batchinds = np.arange(test_X.shape[0] // args.batch_size + 1)
    totalSteps = len(batchinds)
    for _, bi in enumerate(batchinds):
        ## setting batch data
        idxStart = bi * args.batch_size
        if idxStart >= test_X.shape[0]:
            break
        if bi > args.seqs_to_viz:
            break
        idxEnd = idxStart + args.batch_size if (idxStart + args.batch_size) <= test_X.shape[0] else test_X.shape[0]
        inputData_np = test_X[idxStart:idxEnd, :, :]
        outputData_np = test_Y[idxStart:idxEnd, :, :]
        inputData = Variable(inputData_np).to(device)
        outputGT = Variable(outputData_np).to(device)
        assert not torch.isnan(inputData).any() and not torch.isnan(outputGT).any()

        textData = None
        if args.require_text:
            textData_np = test_text[idxStart:(idxStart + args.batch_size), :]
            textData = Variable(textData_np).to(device)
        ## DONE setting batch data
        output_local = model(inputData, text_=textData)
        assert not torch.isnan(output_local).any()
        g_loss = criterion(output_local, outputGT)
        error += g_loss.item() * args.batch_size
        output = torch.cat((output, output_local), 0) if output is not None else output_local
        assert not torch.isnan(output).any()

    error /= totalSteps * args.batch_size
    ## DONE pass loaded data into inference

    print(">>> TOTAL ERROR: ", error, flush=True)
    print('----------------------------------', flush=True)

    ## preparing output for saving
    print("Saving results...", flush=True)
    output_np = output.data.cpu().numpy()
    assert not np.any(np.isnan(output_np))
    output_gt = outputGT.data.cpu().numpy()
    assert not np.any(np.isnan(output_gt))
    output_np = output_np * body_std_Y + body_mean_Y
    output_gt = output_gt * body_std_Y + body_mean_Y
    output_np = np.swapaxes(output_np, 1, 2).astype(np.float32)
    output_gt = np.swapaxes(output_gt, 1, 2).astype(np.float32)
    assert not np.any(np.isnan(input_feats))
    assert not np.any(np.isnan(output_np))
    save_results(input_feats, output_np, args.pipeline, args.base_path, tag=args.tag)
    print("Saved results.", flush=True)
    ## DONE preparing output for saving

    ## generating viz for qualitative assessment
    _inference_xyz = load_binary(os.path.join(args.base_path, f"results/{args.tag}_inference_xyz.pkl"))[0:args.seqs_to_viz]
    structure = skeletalModel.getSkeletalModelStructure()
    gifs_paths = viz.viz(_inference_xyz, structure, frame_rate=25, results_dir=f"viz_results_{args.exp_name}")
    with wandb.init(project="B2H-H2S", name=args.exp_name, id=args.exp_name, resume="must"):
        for path in gifs_paths:
            wandb.save(path)
    ## DONE generating viz


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="models/lastCheckpoint.pth", help='path to checkpoint file (pretrained model)')
    parser.add_argument('--base_path', type=str, default="./", help='absolute path to the base directory where all of the data is stored')
    parser.add_argument('--data_dir', type=str, default="video_data/r6d_test.pkl", help='path to test data directory')
    parser.add_argument('--pipeline', type=str, default='arm2wh', help='pipeline specifying which input/output joints to use')
    parser.add_argument('--require_text', action='store_true', help='whether text is used as input for the model')
    parser.add_argument('--tag', type=str, default='', help='prefix for naming purposes')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for inference')
    parser.add_argument('--seqs_to_viz', type=int, default=2, help='number of sequences to visualize')
    parser.add_argument('--exp_name', type=str, default='experiment', help='name for the experiment')
    parser.add_argument('--model', type=str, default="v1" , help='model architecture to be used')


    args = parser.parse_args()
    print(args, flush=True)
    main(args)
