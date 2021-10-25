import os
import argparse
import os.path
import sys
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import numpy as np

sys.path.append(os.getcwd())
sys.path.append("./utils")
from load_save_utils import *
from postprocess_utils import *
from utils import *
from load_save_utils import *

sys.path.append("./3DposeEstimator")
import skeletalModel

import  wandb


# "frame" is a Numpy array containing the keypoints for a single frame
def plot_3d_lines(frame, structure, show=False):
    s = frame.shape

    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection="3d")
    # hide the axis
    ax._axis3don = False
    for iBone in range(len(structure)):
        id_p_J, id_p_E, _, _ = structure[iBone]
        ax.plot([frame[id_p_J*3], frame[id_p_E*3]],
                [frame[id_p_J*3+1], frame[id_p_E*3+1]],
             zs=[frame[id_p_J*3+2], frame[id_p_E*3+2]])
    ax.view_init(90, 90)
    if show:
        plt.show()
        Axes3D.plot()
    else:
        return fig, ax


def viz_clip(clip, clip_idx, structure, frame_rate=2, results_dir="viz_results"):
    mkdir(results_dir)
    mkdir(f"{results_dir}/{clip_idx}")
    files = []
    for frame_idx in range(clip.shape[0]):
        fig, ax = plot_3d_lines(clip[frame_idx,:], structure, show=False)
        
        filename = f"{results_dir}/"+str(clip_idx)+"/"+str(frame_idx)+".png"
        files.append(filename)
        plt.savefig(filename, dpi=75)
        plt.close(fig)
    # Create the frames
    frames = []
    for f in files:
        new_frame = Image.open(f)
        frames.append(new_frame)
    # Save into a GIF file that loops forever
    frames[0].save(f"{results_dir}/{clip_idx}.gif", format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=len(frames)/frame_rate, loop=0)

    # delete temporal dir
    shutil.rmtree(f"{results_dir}/{clip_idx}")
    return f"{results_dir}/{clip_idx}.gif"  # return animation filename


def viz(xyz, structure, frame_rate=2, results_dir="viz_results"):
    gifs_paths = []
    for clip_idx in range(len(xyz)):
        assert not np.any(np.isnan(xyz[clip_idx]))
        gifs_paths.append(viz_clip(xyz[clip_idx], clip_idx, structure, frame_rate=frame_rate, results_dir=results_dir))
    return gifs_paths


def viz_GT(args):
    r6d_path = f"{args.data_dir}/r6d_{args.infer_set}.pkl"
    X, Y = load_windows(r6d_path, args.pipeline)
    X, Y, _ = rmv_clips_nan(X[:args.seqs_to_viz+10,:,:], Y[:args.seqs_to_viz+10,:,:])

    save_results(X[:args.seqs_to_viz,:,:], Y[:args.seqs_to_viz,:,:], args.pipeline, args.base_path,
                 data_dir=args.data_dir, tag=args.exp_name+"_"+args.infer_set)
    print("Saved results.", flush=True)
    ## DONE preparing output for saving

    ## generating viz for qualitative assessment
    _inference_xyz = load_binary(os.path.join(args.base_path, f"results/{args.exp_name}_{args.infer_set}_inference_xyz.pkl"))[0:args.seqs_to_viz]
    print(f"inference _inference_xyz[0].shape {_inference_xyz[0].shape}", flush=True)
    structure = skeletalModel.getSkeletalModelStructure()
    gifs_paths = viz(_inference_xyz, structure, frame_rate=2, results_dir=f"viz_results_{args.exp_name}_{args.infer_set}")
    with wandb.init(project="B2H-H2S", name=args.exp_name, id=args.exp_name):
        for path in gifs_paths:
            wandb.save(path)
    ## DONE generating viz


if __name__ == '__main__':
    # Visualize inference results
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default="./", help='absolute path to the base directory where all of the data is stored')
    parser.add_argument('--file_path', type=str, default="results/_inference_xyz.pkl", help='path to the .pkl file containing results to visualize')
    parser.add_argument('--seqs_to_viz', type=int, default=20, help='number of sequences to visualize')
    parser.add_argument('--pipeline', type=str, default='arm2wh', help='pipeline specifying which input/output joints to use')
    parser.add_argument('--results_dir', type=str, default="viz_results", help="directory where visualizations should be stored")
    parser.add_argument('--data_dir', type=str, default="video_data" , help='directory where results should be stored and loaded from')
    parser.add_argument('--infer_set', type=str, default="test" , help='if "test", infer using test set; if "train", infer using train set')
    parser.add_argument('--require_text', action='store_true', help='whether text is used as input for the model')
    parser.add_argument('--require_image', action="store_true", help="use additional image features or not")
    parser.add_argument('--exp_name', type=str, default='experiment', help='name for the experiment')
    args = parser.parse_args()

    viz_GT(args)
