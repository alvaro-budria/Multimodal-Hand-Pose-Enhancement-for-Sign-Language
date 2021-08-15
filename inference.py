import argparse
import os
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from utils import *
import modelZoo


def main(args):
    ## variable initializations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.RandomState(23456)
    torch.manual_seed(23456)
    torch.cuda.manual_seed(23456)
    pipeline = args.pipeline
    feature_in_dim, feature_out_dim = FEATURE_MAP[pipeline]
    ## DONE variable initializations

    ## set up model/ load pretrained model
    args.model = 'regressor_fcn_bn_32'
    model = getattr(modelZoo,args.model)()
    model.build_net(feature_in_dim, feature_out_dim, require_text=args.require_text)
    pretrained_model = args.checkpoint
    loaded_state = torch.load(pretrained_model, map_location=lambda storage, loc: storage)
    model.load_state_dict(loaded_state['state_dict'], strict=False)
    model = model.eval()
    model.to(device)
    criterion = nn.MSELoss()
    ## DONE set up model/ load pretrained model

    ## load/prepare data from external files
    test_X, test_Y = load_windows(args.data_dir, args.pipeline, require_text=args.require_text)
    input_feats = test_X.copy()

    if args.require_text:
        test_text = test_X[1].astype(np.float32)
        test_X = test_X[0]

    test_X = np.swapaxes(test_X, 1, 2).astype(np.float32)
    test_Y = np.swapaxes(test_Y, 1, 2).astype(np.float32)

    # standardize
    checkpoint_dir = os.path.split(pretrained_model)[0]
    model_tag = os.path.basename(args.checkpoint).split(args.pipeline)[0]
    preprocess = np.load(os.path.join(checkpoint_dir,'{}{}_preprocess_core.npz'.format(model_tag, args.pipeline)))
    body_mean_X = preprocess['body_mean_X']
    body_std_X = preprocess['body_std_X']
    body_mean_Y = preprocess['body_mean_Y']
    body_std_Y = preprocess['body_std_Y']
    test_X = (test_X - body_mean_X) / body_std_X
    test_Y = (test_Y - body_mean_Y) / body_std_Y
    ## DONE load/prepare data from external files

    ## pass loaded data into training
    inputData = Variable(torch.from_numpy(test_X)).to(device)
    outputGT = Variable(torch.from_numpy(test_Y)).to(device)
    textData = None
    if args.require_text:
        textData = Variable(torch.from_numpy(test_text)).to(device)

    output = model(inputData, text_=textData)
    error = criterion(output, outputGT).data
    print(">>> TOTAL ERROR: ", error, flush=True)
    print('----------------------------------', flush=True)
    ## DONE pass loaded data into training

    ## preparing output for saving
    output_np = output.data.cpu().numpy()
    output_gt = outputGT.data.cpu().numpy()
    output_np = output_np * body_std_Y + body_mean_Y
    output_gt = output_gt * body_std_Y + body_mean_Y
    output_np = np.swapaxes(output_np, 1, 2).astype(np.float32)
    output_gt = np.swapaxes(output_gt, 1, 2).astype(np.float32)
    # inputData = np.swapaxes(inputData, 1, 2).numpy().astype(np.float32)
    save_results(input_feats, output_np, args.pipeline, args.base_path, tag=args.tag)
    ## DONE preparing output for saving


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='path to checkpoint file (pretrained model)')
    parser.add_argument('--base_path', type=str, default="./", help='absolute path to the base directory where all of the data is stored')
    parser.add_argument('--data_dir', type=str, default="video_data/r6d_test.pkl", help='path to test data directory')
    parser.add_argument('--pipeline', type=str, default='arm2wh', help='pipeline specifying which input/output joints to use')
    parser.add_argument('--require_text', action='store_true', help='whether text is used as input for the model')
    parser.add_argument('--tag', type=str, default='', help='prefix for naming purposes')

    args = parser.parse_args()
    print(args, flush=True)
    main(args)
