import os
import argparse
import os.path
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

sys.path.append(os.getcwd())
sys.path.append("../")
import utils
import shutil


# plots the
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


def viz_clip(clip, clip_idx, structure, frame_rate=27.5, results_dir="viz_results"):
    utils.mkdir(results_dir)
    utils.mkdir(f"{results_dir}/{clip_idx}")
    files = []
    for frame_idx in range(clip.shape[0]):
        if frame_idx == 0:
            print(f"clip[frame_idx,:].shape: {clip[frame_idx,:].shape}")
        fig, ax = plot_3d_lines(clip[frame_idx,:], structure, show=False)
        
        filename = f"{results_dir}/"+str(clip_idx)+"/"+str(frame_idx)+".png"
        files.append(filename)
        plt.savefig(filename, dpi=75)
        plt.close(fig)
    print(f"len(files): {len(files)}")
    # Create the frames
    frames = []
    for f in files:
        new_frame = Image.open(f)
        frames.append(new_frame)
    print(f"len(frames): {len(frames)}")
    # Save into a GIF file that loops forever
    frames[0].save(f"{results_dir}/{clip_idx}.gif", format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=len(frames)/frame_rate, loop=0)

    # delete temporal dir
    #shutil.rmtree(f"{results_dir}/{clip_idx}")
    return f"{results_dir}/{clip_idx}.gif"  # return animation filename

import numpy as np ########
def viz(xyz, structure, frame_rate=27.5, results_dir="viz_results"):
    print(f"len(xyz): {len(xyz)}")
    gifs_paths = []
    for clip_idx in range(len(xyz)):
        assert not np.any(np.isnan(xyz[clip_idx]))
        gifs_paths.append(viz_clip(xyz[clip_idx], clip_idx, structure, frame_rate=frame_rate, results_dir=results_dir))
    print(f"len(gifs_paths): {len(gifs_paths)}")
    return gifs_paths


if __name__ == '__main__':
    # Visualize inference results 
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default="results/_inference_xyz.pkl", help='path to the .pkl file containing results to visualize')
    args = parser.parse_args()
    _inference_xyz = utils.load_binary(args.file_path)
    import skeletalModel
    structure = skeletalModel.getSkeletalModelStructure()
    viz(_inference_xyz, structure, frame_rate=25, results_dir="viz_results")
