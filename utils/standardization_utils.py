import numpy

from load_save_utils import *
from constants import EPSILON


# loads a list of clips and computes the mean and standard deviation
def compute_mean_std(clips_list_path, data_dir):
    clip_list = load_binary(os.path.join(data_dir, clips_list_path))  # clip_list is expected to contain a list of TxCxHxWx2 arrays
    ####### COMPUTE MEAN / STD

    # placeholders
    psum    = np.array([0.0, 0.0, 0.0])
    psum_sq = np.array([0.0, 0.0, 0.0])
    pixel_count = 0

    # loop through images
    for clip in clip_list:
        psum        += np.sum(clip[:,:,:,:,0], axis=(0, 2, 3)) + np.sum(clip[:,:,:,:,1], axis=(0, 2, 3))
        psum_sq     += np.sum(clip[:,:,:,:,0].astype(np.float)**2, axis=(0, 2, 3)) + np.sum(clip[:,:,:,:,1].astype(np.float)**2, axis=(0, 2, 3))
        pixel_count += clip.shape[0]*clip.shape[2]*clip.shape[3]*clip.shape[4]  # T*H*W*2

    # mean and std
    total_mean = psum / pixel_count
    total_var  = (psum_sq / pixel_count) - (total_mean ** 2)
    total_std  = np.sqrt(total_var)

    # output
    print(f"mean: {total_mean}", flush=True)
    print(f"std:  {total_std}", flush=True)

    with open(f'{data_dir}/mean_std.npy', 'wb') as f:
        np.save(f, np.vstack((total_mean, total_std)))


## helper for calculating standardization stats
def calc_standard(train_X, train_Y, pipeline):
    rot_idx = -6
    feats = pipeline.split('2')
    in_feat, out_feat = feats[0], feats[1]
    body_mean_X, body_std_X = mean_std(in_feat, train_X, rot_idx)
    if in_feat == out_feat:
        body_mean_Y = body_mean_X
        body_std_Y = body_std_X
    else:
        body_mean_Y, body_std_Y = mean_std(out_feat, train_Y, rot_idx)
    return body_mean_X, body_std_X, body_mean_Y, body_std_Y


## helper for calculating mean and standard dev
def mean_std(feat, data, rot_idx):
    if feat == 'wh':
       mean = data.mean(axis=2).mean(axis=0)[np.newaxis,:, np.newaxis]
       std =  data.std(axis=2).std(axis=0)[np.newaxis,:, np.newaxis]
       std += EPSILON
    else:
        mean = data.mean(axis=2).mean(axis=0)[np.newaxis,:, np.newaxis]
        std = np.array([[[data.std()]]]).repeat(data.shape[1], axis=1)
    return mean, std
