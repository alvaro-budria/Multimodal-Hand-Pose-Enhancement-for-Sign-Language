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

sys.path.append("./utils")
from constants import *
from load_save_utils import *
from utils import save_results, load_binary
from postprocess_utils import *
import modelZoo
import viz.viz_3d as viz


def main(args):
    # variable initializations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(23456)
    torch.cuda.manual_seed(23456)
    pipeline = args.pipeline
    feature_in_dim, feature_out_dim = FEATURE_MAP[pipeline]
    print(f"feature_in_dim, feature_out_dim: {feature_in_dim}, {feature_out_dim}")
    ## DONE variable initializations

    ## set up model / load pretrained model
    args.model = MODELS[args.model]
    model = getattr(modelZoo,args.model)()
    if args.model == "regressor_fcn_bn_32_b2h":
        model.build_net(feature_in_dim, feature_out_dim, require_image=args.require_image)
    else:
        model.build_net(feature_in_dim, feature_out_dim, require_text=args.require_text)
    pretrained_model = args.checkpoint
    loaded_state = torch.load(pretrained_model, map_location=lambda storage, loc: storage)
    model.load_state_dict(loaded_state['state_dict'], strict=False)
    model = model.eval()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!", flush=True)
        model = nn.DataParallel(model)
    model.to(device)
    criterion = nn.L1Loss()
    ## DONE set up model/ load pretrained model

    ## load/prepare data from external files
    r6d_path = f"{args.data_dir}/r6d_{args.infer_set}.pkl"
    if args.embeds_type == "normal":
        text_path = f"{args.data_dir}/{args.infer_set}_sentence_embeddings.pkl"
    elif args.embeds_type == "average":
        text_path = f"{args.data_dir}/average_{args.infer_set}_sentence_embeddings.pkl"
    image_path = f"{args.data_dir}/{args.infer_set}_vid_feats.pkl"

    test_X, test_Y = load_windows(r6d_path, args.pipeline, require_text=args.require_text, text_path=text_path,
                                  require_image=args.require_image, image_path=image_path)

    test_feats = None
    if args.require_text or args.require_image:
        test_feats = test_X[1]
        test_X = test_X[0]
    print(f"test_X.shape, test_Y.shape {test_X.shape, test_Y.shape}", flush=True)
    if args.require_text or args.require_image:
        print(test_feats.shape, flush=True)
    test_X, test_Y, test_feats = rmv_clips_nan(test_X, test_Y, test_feats)
    assert not np.any(np.isnan(test_X)) and not np.any(np.isnan(test_Y))
    print(f"test_X.shape, test_Y.shape {test_X.shape, test_Y.shape}", flush=True)
    input_feats = test_X.copy()
    if pipeline=="wh2wh":
        test_X = test_X[:,:,6*6:]  # keep hands; input_feats still contains both hands and arms for saving purposes

    test_X = np.swapaxes(test_X, 1, 2).astype(np.float32)
    test_Y = np.swapaxes(test_Y, 1, 2).astype(np.float32)

    # standardize
    checkpoint_dir = os.path.split(pretrained_model)[0]
    preprocess = np.load(os.path.join(checkpoint_dir,'{}{}_preprocess_core.npz'.format(args.exp_name, args.pipeline)))
    body_mean_X = preprocess['body_mean_X']
    body_std_X = preprocess['body_std_X']
    body_mean_Y = preprocess['body_mean_Y']
    body_std_Y = preprocess['body_std_Y']
    test_X = (test_X - body_mean_X) / body_std_X
    test_Y = (test_Y - body_mean_Y) / body_std_Y
    ## DONE load/prepare data from external files

    ## pass loaded data into inference
    test_X, test_Y = torch.from_numpy(test_X), torch.from_numpy(test_Y)
    if args.require_text or args.require_image:
        test_feats = torch.from_numpy(test_feats)
    error = 0
    output = None
    batchinds = np.arange(test_X.shape[0] // args.batch_size + 1)
    totalSteps = 0
    for bii, bi in enumerate(batchinds):
        totalSteps += 1
        ## setting batch data
        idxStart = bi * args.batch_size
        print(f"idxStart {idxStart}", flush=True)
        if idxStart >= test_X.shape[0] or bi * args.batch_size >= args.num_samples:
            break
        idxEnd = idxStart + args.batch_size if (idxStart + args.batch_size) <= test_X.shape[0] else test_X.shape[0]
        inputData_np = test_X[idxStart:idxEnd, :, :]
        outputData_np = test_Y[idxStart:idxEnd, :, :]
        inputData = Variable(inputData_np).to(device)
        outputGT = Variable(outputData_np)

        featsData = None
        if args.require_text or args.require_image:
            featsData = Variable(test_feats[idxStart:(idxStart + args.batch_size), :]).to(device)
        ## DONE setting batch data
        output_local = model(inputData, feats_=featsData)
        g_loss = criterion(output_local.cpu(), outputGT)
        error += g_loss.item() * args.batch_size
        #output = np.concatenate((output, output_local.cpu().detach().numpy()), 0) if output is not None else output_local
        output = torch.cat((output, output_local.cpu()), 0) if output is not None else output_local.cpu()
        torch.cuda.empty_cache()

    error /= totalSteps * args.batch_size
    ## DONE pass loaded data into inference

    print(">>> TOTAL ERROR: ", error, flush=True)
    print('----------------------------------', flush=True)

    ## preparing output for saving
    print("Saving results...", flush=True)
    output_np = output.data.cpu().numpy() #####
    assert not np.any(np.isnan(output_np))
    output_np = output_np * body_std_Y + body_mean_Y
    output_np = np.swapaxes(output_np, 1, 2).astype(np.float32)
    assert not np.any(np.isnan(input_feats))
    assert not np.any(np.isnan(output_np))
    print(f"input_feats.shape: {input_feats.shape}; output_np.shape: {output_np.shape}", flush=True)
    print(f"input_feats[:output_np.shape[0],:,:].shape: {input_feats[:output_np.shape[0],:,:].shape}", flush=True)
    save_results(input_feats[:output_np.shape[0],:,:], output_np, args.pipeline, args.base_path,
                 data_dir=args.data_dir, tag=args.exp_name, infer_set=args.infer_set)
    print("Saved results.", flush=True)
    ## DONE preparing output for saving

    ## generating viz for qualitative assessment
    _inference_xyz = load_binary(os.path.join(args.base_path, f"results_{args.exp_name}/xyz_{args.infer_set}.pkl"))[0:args.seqs_to_viz]
    print(f"inference _inference_xyz[0].shape {_inference_xyz[0].shape}", flush=True)
    structure = skeletalModel.getSkeletalModelStructure()
    gifs_paths = viz.viz(_inference_xyz, structure, frame_rate=2, results_dir=f"viz_results_{args.exp_name}_{args.infer_set}")
    with wandb.init(project="B2H-H2S", name=args.exp_name, id=args.exp_name, resume="must"):
        for path in gifs_paths:
            wandb.save(path)
    ## DONE generating viz


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="models/lastCheckpoint.pth", help='path to checkpoint file (pretrained model)')
    parser.add_argument('--base_path', type=str, default="./", help='absolute path to the base directory where all of the data is stored')
    parser.add_argument('--data_dir', type=str, default="video_data", help='directory where results should be stored and loaded from')
    parser.add_argument('--pipeline', type=str, default='arm2wh', help='pipeline specifying which input/output joints to use')
    parser.add_argument('--require_text', action='store_true', help='whether text is used as input for the model')
    parser.add_argument('--require_image', action="store_true", help="use additional image features or not")
    parser.add_argument('--embeds_type', type=str, default="normal", help='if "normal", use normal text embeds; if "average", use average text embeds')
    parser.add_argument('--infer_set', type=str, default="test", help='if "test", infer using test set; if "train", infer using train set')
    parser.add_argument('--tag', type=str, default='', help='prefix for naming purposes')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for inference')
    parser.add_argument('--seqs_to_viz', type=int, default=2, help='number of sequences to visualize')
    parser.add_argument('--exp_name', type=str, default='experiment', help='name for the experiment')
    parser.add_argument('--model', type=str, default="v1" , help='model architecture to be used')
    parser.add_argument('--num_samples', type=int, default=3000, help='number of sequences to predict')


    args = parser.parse_args()
    print(args, flush=True)
    main(args)
