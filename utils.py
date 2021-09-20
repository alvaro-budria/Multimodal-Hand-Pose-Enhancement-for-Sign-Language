import os
import sys
import json
import pickle
import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.append('./3DposeEstimator')
# 2D to 3D lifting
import skeletalModel
import pose2D
import pose2Dto3D
import pose3D

import viz.viz_3d as viz

import proc_text
import proc_vid

DATA_PATHS = {
        "train": "train/rgb_front/features/openpose_output/json",
        "val": "val/rgb_front/features/openpose_output/json",
        "test": "test/rgb_front/features/openpose_output/json"
    }

FEATURE_MAP = {
    'arm2wh': ((6*6), 42*6),
    "arm_wh2finger1": ( ((6+38)*6), (4*6) ),  # predict 5th finger of left hand given arms and rest of fingers
    "arm_wh2finger2": ( ((6+34)*6), (8*6) ),  # predict 4th and 5th fingers of left hand given arms and rest of fingers
    "arm_wh2finger3": ( ((6+30)*6), (12*6) ),
    "arm_wh2finger4": ( ((6+26)*6), (16*6) ),
    "arm_wh2finger5": ( ((6+22)*6), (20*6) ), # predict left hand
    "arm_wh2finger6": ( ((6+21)*6), (21*6) ), # predict left hand including wrist
    "arm_wh2finger7": ( ((6+17)*6), (25*6) ), # predict 5th finger of right hand given arms, left hand and rest of fingers
    "arm_wh2finger8": ( ((6+13)*6), (29*6) ),
    "arm_wh2finger9": ( ((6+9)*6), (33*6) ),
    "arm_wh2finger10": ( ((6+5)*6), (37*6) ),
    "arm_wh2finger11": ( ((6+1)*6), (41*6) ), 
    "arm_wh2finger12": ( ((6+0)*6), (42*6) ), # predict hands, including wrists, given arms
}

EPSILON = 1e-10

NECK = [0, 1]  # neck in Open Pose 25
WRIST = [[4, 7], [0, 21]]  # wrist in arms, wrist in hand
ARMS = [2, 3, 4, 5, 6, 7]  # arms in Open Pose 25
HANDS = list(range(21*2))  # hands in Open Pose


# removes those clips that contain at least one nan value
def rmv_clips_nan(X, Y, T=None):
    x = []
    y = []
    t = []
    for sample in range(X.shape[0]):
        if T is None:
            if not (np.isnan(X[sample,:,:]).any() | np.isnan(Y[sample,:,:]).any()):
                x.append(X[sample,:,:])
                y.append(Y[sample,:,:])
        else:
            if not (np.isnan(X[sample,:,:]).any() | np.isnan(Y[sample,:,:]).any() | np.isnan(T[sample,:]).any()):
                x.append(X[sample,:,:])
                y.append(Y[sample,:,:])
                t.append(T[sample,:])
    x = np.array(x)
    y = np.array(y)
    if T is not None:
        t = np.array(t)
    return x, y, t


def array_to_list(input):
    if type(input) != type(list()):  # convert 3D array to list of 2D arrays
        input = list(input)
    return input


def np_mat_to_rot6d(np_mat):
    """ Get 6D rotation representation for rotation matrix.
        Implementation base on
            https://arxiv.org/abs/1812.07035
        [Inputs]
            flattened rotation matrix (last dimension is 9)
        [Returns]
            6D rotation representation (last dimension is 6)
    """
    shape = np_mat.shape

    if not ((shape[-1] == 3 and shape[-2] == 3) or (shape[-1] == 9)):
        raise AttributeError("The inputs in tf_matrix_to_rotation6d should be [...,9] or [...,3,3], \
            but found tensor with shape {}".format(shape[-1]))

    np_mat = np.reshape(np_mat, [-1, 3, 3])
    np_r6d = np.concatenate([np_mat[...,0], np_mat[...,1]], axis=-1)

    if len(shape) == 1:
        np_r6d = np.reshape(np_r6d, [6])

    return np_r6d


## utility function to convert from r6d space to axis angle
def _rot6d_to_aa(r6ds):
    assert not np.any(np.isnan(r6ds))
    res = np.zeros((r6ds.shape[0], 3))
    assert not np.any(np.isnan(res))
    for i,row in enumerate(r6ds):
        np_r6d = np.expand_dims(row, axis=0)
        assert not np.any(np.isnan(np_r6d))
        np_mat = np.reshape(np_rot6d_to_mat(np_r6d)[0], (3,3))
        assert not np.any(np.isnan(np_rot6d_to_mat(np_r6d)))
        assert not np.any(np.isnan(np_mat))
        np_mat = R.from_matrix(np_mat)
        #assert not np.any(np.isnan(np_mat))
        aa = np_mat.as_rotvec()
        assert not np.any(np.isnan(aa))
        res[i,:] = aa
        assert not np.any(np.isnan(res[i,:]))
    return res


def clip_rot6d_to_aa(r6d_clip):
    assert not np.any(np.isnan(r6d_clip))
    aa_clip = np.empty((r6d_clip.shape[0], r6d_clip.shape[1]//2))
    for idx in range(0, r6d_clip.shape[1], 6):
        # print(f"r6d_clip.shape: {r6d_clip.shape}")
        # print(f"r6d_clip[:,idx:idx+6].shape: {r6d_clip[:,idx:idx+6].shape}")
        assert not np.any(np.isnan(r6d_clip[:,idx:idx+6]))
        assert not np.any(np.isnan(_rot6d_to_aa(r6d_clip[:,idx:idx+6])))
        aa_clip[:,idx//2:idx//2+3] = _rot6d_to_aa(r6d_clip[:,idx:idx+6])
    return aa_clip


def rot6d_to_aa(r6d):
    r6d = array_to_list(r6d)
    assert not np.any(np.isnan(r6d))
    aa = []
    with Pool(processes=24) as pool:
        aa = pool.starmap( clip_rot6d_to_aa, zip(r6d) )
    return aa


## utility function to convert from axis angle to r6d space
def _aa_to_rot6d(vecs):
    res = np.zeros((vecs.shape[0], 6))
    for i,row in enumerate(vecs):
        np_mat = R.from_rotvec(row)
        np_mat = np_mat.as_matrix()
        np_mat = np.expand_dims(np_mat, axis=0) #e.g. batch 1
        np_r6d = np_mat_to_rot6d(np_mat)[0]
        res[i,:] = np_r6d
    return res


# convert from axis angle to r6d space
def aa_to_rot6d(aa):
    aa = array_to_list(aa)
    r6d = []
    for clip in range(len(aa)):
        aa_clip = aa[clip]
        r6d_clip = np.empty((aa_clip.shape[0], aa_clip.shape[1]*2)) # from 3d to r6d
        for idx in range(0, aa_clip.shape[1], 3):
            r6d_clip[:,idx*2:idx*2+6] =  _aa_to_rot6d(aa_clip[:,idx:idx+3])
        r6d.append(r6d_clip)
    return r6d


# https://github.com/facebookresearch/body2hands/blob/0eba438b4343604548120bdb03c7e1cb2b08bcd6/utils/load_utils.py
## utility function to convert from r6d space to rotation matrix
def np_rot6d_to_mat(np_r6d):
    shape = np_r6d.shape
    np_r6d = np.reshape(np_r6d, [-1,6])
    assert not np.any(np.isnan(np_r6d))
    x_raw = np_r6d[:,0:3]
    assert not np.any(np.isnan(x_raw))
    y_raw = np_r6d[:,3:6]
    assert not np.any(np.isnan(y_raw))

    x = x_raw / (np.linalg.norm(x_raw, ord=2, axis=-1) + 1e-6)
    assert not np.any(np.isnan(x))
    z = np.cross(x, y_raw)
    assert not np.any(np.isnan(z))
    z = z / (np.linalg.norm(z, ord=2, axis=-1) + 1e-6)
    assert not np.any(np.isnan(z))
    y = np.cross(z, x)
    assert not np.any(np.isnan(y))

    x = np.reshape(x, [-1,3,1])
    y = np.reshape(y, [-1,3,1])
    z = np.reshape(z, [-1,3,1])
    np_matrix = np.concatenate([x,y,z], axis=-1)
    assert not np.any(np.isnan(np_matrix))
    if len(shape) == 1:
        np_matrix = np.reshape(np_matrix, [9])
        assert not np.any(np.isnan(np_matrix))
    else:
        output_shape = shape[:-1] + (9,)
        assert not np.any(np.isnan(output_shape))
        np_matrix = np.reshape(np_matrix, output_shape)
        assert not np.any(np.isnan(np_matrix))

    return np_matrix


# From a vector representing a rotation in axis-angle representation,
# retrieves the rotation angle and the rotation axis
def _retrieve_axis_angle(aa):
    th = np.linalg.norm(aa, axis=1)
    a = aa / th[:,np.newaxis]
    return a, th


def aa_to_xyz(aa, root, bone_len, structure):
    aa = array_to_list(aa)
    xyz = []
    for i in range(len(aa)):
        aa_clip = aa[i]
        print(f"aa_clip.shape: {aa_clip.shape}")
        xyz_clip = np.empty((aa_clip.shape[0], aa_clip.shape[1]+6), dtype="float32")  # add 6 values, corresponding to two keypoints defining the root bone
        xyz_clip[:,0:6] = root
        for iBone in range(1,len(structure)):
            id_p_J, id_p_E, _, id_p_B = structure[iBone]
            p_J, p_B = xyz_clip[:,id_p_J*3:id_p_J*3+3], xyz_clip[:,id_p_B*3:id_p_B*3+3]
            u = p_J - p_B
            u = u / np.linalg.norm(u, axis=1)[:, np.newaxis]
            a, th = _retrieve_axis_angle(aa_clip[:,(iBone-1)*3:(iBone-1)*3+3])
            # Rodrigues' rotation formula
            v = np.multiply(u, np.cos(th)[:, np.newaxis]) \
                + np.multiply(np.cross(a, u), np.sin(th)[:, np.newaxis]) \
                + np.multiply(np.multiply(a, np.einsum('ij,ij->i', a, u)[:, np.newaxis]), (1-np.cos(th))[:, np.newaxis])

            p_E = p_J + bone_len[iBone]*v
            xyz_clip[:,(iBone+1)*3:(iBone+1)*3+3] = p_E 
        xyz.append(xyz_clip)
    return xyz


def get_root_bone(xyz, structure):
    xyz = array_to_list(xyz)
    root = np.array([])
    for i in range(len(xyz)):
        xyz_clip = xyz[i]
        id_p_J, id_p_E, _, _ = structure[0]  # get initial and end joints indexes of root bone
        bone_points = np.hstack((xyz_clip[:,id_p_J*3:id_p_J*3+3], xyz_clip[:,id_p_E*3:id_p_E*3+3]))
        root = np.vstack( (root, bone_points) ) if root.shape!=(0,) else bone_points
    return np.average(root, axis=0)


def xyz_to_aa(xyz, structure, root_filename=None):
    xyz = array_to_list(xyz)
    aa = []
    for i in range(len(xyz)):
        xyz_clip = xyz[i]
        aa_clip = np.array([])
        for iBone in range(1,len(structure)):
            id_p_J, id_p_E, _, id_p_B = structure[iBone]
            u = xyz_clip[:,id_p_J*3:id_p_J*3+3] - xyz_clip[:,id_p_B*3:id_p_B*3+3]
            v = xyz_clip[:,id_p_E*3:id_p_E*3+3] - xyz_clip[:,id_p_J*3:id_p_J*3+3]
            th = np.arccos( np.einsum('ij,ij->i', u, v)/(np.linalg.norm(u, axis=1)*np.linalg.norm(v, axis=1) + 1e-6) )
            a = np.cross(u, v)
            a = a / np.linalg.norm(a, axis=1)[:,np.newaxis]  # rotation axis
            aa_clip = np.hstack(( aa_clip, np.multiply(a, th[:, np.newaxis]) )) if aa_clip.shape[0]!=0 else np.multiply(a, th[:, np.newaxis])
        aa.append(aa_clip)
    return aa


def _lift_2d_to_3d(inputSequence_2D):
    dtype = "float32"
    randNumGen = np.random.RandomState(1234)

    # Getting our structure of skeletal model.
    structure = skeletalModel.getSkeletalModelStructure()

    # Decomposition of the single matrix into three matrices: x, y, w (=likelihood)
    X = inputSequence_2D
    Xx = X[0:X.shape[0], 0:(X.shape[1]):3]
    Xy = X[0:X.shape[0], 1:(X.shape[1]):3]
    Xw = X[0:X.shape[0], 2:(X.shape[1]):3]

    # Normalization of the picture (so x and y axis have the same scale)
    Xx, Xy, mux, muy, sigma = pose2D.normalization(Xx, Xy)

    # Delete frames in which skeletal models have a lot of missing parts.

    ##### watchThis ?? per què (0, 1, 2, 3, 4, 5, 6, 7) ?
    Xx, Xy, Xw = pose2D.prune(Xx, Xy, Xw, (0, 1, 2, 3, 4, 5, 6, 7), 0.3, dtype)

    # Initial 3D pose estimation
    lines0, rootsx0, rootsy0, rootsz0, anglesx0, anglesy0, anglesz0, Yx0, Yy0, Yz0 = pose2Dto3D.initialization(
        Xx,
        Xy,
        Xw,
        structure,
        0.001, # weight for adding noise
        randNumGen,
        dtype
    )

    # Backpropagation-based filtering
    Yx, Yy, Yz = pose3D.backpropagationBasedFiltering_v2(
        lines0, 
        rootsx0,
        rootsy0, 
        rootsz0,
        anglesx0,
        anglesy0,
        anglesz0,   
        Xx,   
        Xy,
        Xw,
        structure,
        "float32",
        learningRate=20,
        nCycles=900
    )
    # # Backpropagation-based filtering
    # Yx, Yy, Yz = pose3D.backpropagationBasedFiltering(
    #     lines0, 
    #     rootsx0,
    #     rootsy0, 
    #     rootsz0,
    #     anglesx0,
    #     anglesy0,
    #     anglesz0,   
    #     Xx,   
    #     Xy,
    #     Xw,
    #     structure,
    #     "float32",
    #     learningRate=0.25,
    #     nCycles=900
    # )
    #_save("3D_keypoints.txt", [Yx, Yy, Yz])

    kp = np.empty((Yx.shape[0], Yx.shape[1] + Yy.shape[1] + Yz.shape[1]), dtype=dtype)
    kp[:,0::3], kp[:,1::3], kp[:,2::3] = Yx, Yy, Yz

    print("LIFTED SEQUENCE SUCCESSFULLY", flush=True)
    return kp


# input is a list of arrays, one array per clip
def lift_2d_to_3d(feats, filename="feats_3d", nPartitions=40):
    feats_3d = []
    if os.path.exists(filename):
        print(f" -> Found file with name {filename}. Appending results to this file.")
        feats_3d = load_binary(filename)
    idx = len(feats) // nPartitions + 1
    min_i = 0
    for i in range(min_i, nPartitions):
        feats_3d_sub = []
        with Pool(processes=24) as pool:
            feats_3d_sub = pool.starmap( _lift_2d_to_3d, zip(feats[idx*i:idx*(i+1)]) )
        feats_3d = feats_3d + feats_3d_sub
        save_binary(feats_3d, filename)
        print("*"*50, flush=True)
        print(f"PARTITION {i}", flush=True)
        print(flush=True)
        print(f"LIFTED {int(i / nPartitions *100)}%", flush=True)
        print("*"*50, flush=True)


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


# given a list of the form [X1, Y1, conf1, X2, Y2, conf2 ... Xn, Yn, conf_n]
# returns [X1, Y1, ... Xn, Yn] or [X1, Y1, W1 ... Xn, Yn, Wn] if keep_confidence=True

## @jit ?¿?
def retrieve_coords(keypoints, keep_confidence=False):
    coords = []
    for i in range(0, len(keypoints), 3):
        #coords.append([keypoints[i], keypoints[i+1]])
        coords.append(keypoints[i])
        coords.append(keypoints[i+1])
        if keep_confidence:
            coords.append(keypoints[i+2])
    return coords # [elem for singleList in coords for elem in singleList]


def load_clip(clip_path, pipeline, keep_confidence=True):
    feats = pipeline.split('2')
    in_feat, out_feat = feats[0], feats[1]
    in_kp, out_kp = np.array([]), np.array([])
    for frame in sorted(os.listdir(clip_path))[0:]:  # for each frame, there is an associated .json file
        if os.path.isfile(os.path.join(clip_path, frame)):
            f = open(os.path.join(clip_path, frame))
            data = json.load(f)
            f.close()
            if in_kp.shape != (0,):
                in_kp = np.append(in_kp, [retrieve_coords(data["people"][0]["pose_keypoints_2d"], keep_confidence=keep_confidence)], axis=0)
            else:
                in_kp = np.array([retrieve_coords(data["people"][0]["pose_keypoints_2d"], keep_confidence=keep_confidence)])
            
            temp = [retrieve_coords(data["people"][0]["hand_right_keypoints_2d"], keep_confidence=keep_confidence),
                    retrieve_coords(data["people"][0]["hand_left_keypoints_2d"], keep_confidence=keep_confidence)]
            temp = [elem for singleList in temp for elem in singleList]
            if out_kp.shape != (0,):
                out_kp = np.append(out_kp, [temp], axis=0)
            else:
                out_kp = np.array([temp])
    return in_kp, out_kp


def _join_ids(dir_list, clip_ids_text):
    return list(set(dir_list).intersection(clip_ids_text))

def _load(args):
    clip, dir, pipeline = args
    clip_path = os.path.join(dir, clip)
    in_kp, out_kp = load_clip(clip_path, pipeline)
    return clip, in_kp, out_kp

def _load_H2S_dataset(dir, pipeline, key, subset=0.1):  # subset allows to keep a certain % of the data only
    dir_list = os.listdir(dir)
    print(f"{key} len(dir_list): {len(dir_list)}", flush=True)

    clip_ids_text = proc_text.get_clip_ids(key=key)
    print(f"{key} len(clip_ids_text): {len(clip_ids_text)}", flush=True)

    clip_ids_vid = proc_vid.get_vid_ids(key=key)
    print(f"{key} len(clip_ids_vid): {len(clip_ids_vid)}", flush=True)

    ids = _join_ids(dir_list, clip_ids_text)
    ids = _join_ids(ids, clip_ids_vid)
    ids = sorted(ids)
    print(f"{key} len(ids): {len(ids)}", flush=True)

    idx_max = int(len(ids)*subset)
    print(f"{key} idx_max: {idx_max}", flush=True)
    print(f"{key} len(ids[:idx_max]): {len(ids[:idx_max])}", flush=True)
    #print(f"{key} len(ids[idx_max:]): {len(ids[idx_max:])}", flush=True)

    embeds = proc_text.obtain_embeddings(key, ids[0:idx_max])  # obtain text embeddings for each clip
    #embeds = proc_text.obtain_embeddings(key, ids[idx_max:])

    dir_ = [dir for _ in range(idx_max)]
    pipe_ = [pipeline for _ in range(idx_max)]
    with ProcessPoolExecutor() as executor:
        result = executor.map(_load, zip(ids[0:idx_max], dir_, pipe_))
        # result = executor.map(_load, zip(ids[idx_max:], dir_, pipe_))
    clips, in_features, out_features = map(list, zip(*result))
    print(f"Number of clips: {len(clips)}", flush=True)
    print(f"Number of input sequences (in_features): {len(in_features)}", flush=True)
    print(f"Number of output sequences (out_features): {len(out_features)}", flush=True)
    return in_features, out_features, embeds

def load_H2S_dataset(data_dir, pipeline="arm2wh", num_samples=None, require_text=False, require_audio=False, subset=0.1):
    train_path = os.path.join(data_dir, DATA_PATHS["train"])
    val_path = os.path.join(data_dir, DATA_PATHS["val"])
    test_path = os.path.join(data_dir, DATA_PATHS["test"])
    # load data
    in_train, out_train, in_val, out_val, in_test, out_test = None, None, None, None, None, None
    if os.path.exists(test_path):
        in_test, out_test, embeds_test = _load_H2S_dataset(test_path, pipeline=pipeline, key="test", subset=subset)
        print("LOADED RAW TEST DATA", flush=True)
    if os.path.exists(val_path):
        in_val, out_val, embeds_val = _load_H2S_dataset(val_path, pipeline=pipeline, key="val", subset=subset)
        print("LOADED RAW VAL DATA", flush=True)
    if os.path.exists(train_path):
        in_train, out_train, embeds_train = _load_H2S_dataset(train_path, pipeline=pipeline, key="train", subset=subset)
        print("LOADED RAW TRAIN DATA", flush=True)
    print("crashing", crash)
    return (in_train, out_train, embeds_train), (in_val, out_val, embeds_val), (in_test, out_test, embeds_test)


def obtain_vid_feats(kp_dir, key):
    kp_path = os.path.join(kp_dir, DATA_PATHS[key])
    kp_dir_list = os.listdir(kp_path)
    clip_ids_text = proc_text.get_clip_ids(key=key)
    ids = _join_ids(kp_dir_list, clip_ids_text)  # keep id that are present both in kp_dir_list (IDs for which keypoints are availabe)
                                                 # and in clip_ids_text (IDs for text sentences are availabe)
    clip_ids_vid = proc_vid.get_vid_ids(key=key)
    ids = _join_ids(ids, clip_ids_vid)
    ids = sorted(ids)
    hand_feats = proc_vid.obtain_feats(key, ids)
    return hand_feats


# returns the keypoints in the specified indexes
def get_joints(kp, idx):
    return kp[:,idx]


# selects the useful keypoints indicated by the indexes. Input is a list, each element containing the keypoints of a (video) clip
def select_keypoints(kp, idxs, keep_confidence=True):
    kp_cp = kp.copy()
    for i in range(len(kp)):
        new_kp_i = np.array([])
        for idx in idxs:
            if keep_confidence:
                new_kp_i = np.hstack((new_kp_i, kp[i][:,idx*3:idx*3+3])) if new_kp_i.shape[0]!=0 else kp[i][:,idx*3:idx*3+3]
            else:
                new_kp_i = np.hstack((new_kp_i, kp[i][:,idx*3:idx*3+2])) if new_kp_i.shape[0]!=0 else kp[i][:,idx*3:idx*3+2]
        kp_cp[i] = new_kp_i
    return kp_cp


def hconcat_feats(neck, arms, hands):
    assert [len(neck), len(arms)] == [len(hands), len(hands)]
    feats = []
    for i in range(len(neck)):  # for each frame, concatenate the features
        temp = np.hstack((neck[i], arms[i]))
        feats.append( np.hstack((temp, hands[i])) )
    return feats


def save_binary(obj, filename, append=False):
    if filename[-4:] != ".pkl":
        print("Adding .pkl extension as it was not found.", flush=True)
        filename = filename + ".pkl"

    if os.path.exists(filename) and append:
        print(f"Found file with name {filename}. Appending results to this file.")
        contents = load_binary(filename)
        if append=="embeds":
            obj = np.vstack((contents,obj))
        elif append:  # contents of filename are assumed to be contained in the form of a list. obj is assumed to be a list
            obj = contents + obj

    with open(filename, 'wb') as outfile:
        pickle.dump(obj, outfile, pickle.HIGHEST_PROTOCOL)


def load_binary(filename):
    result = None
    with open(filename, 'rb') as infile:
        result = pickle.load(infile)
    return result


def mkdir(dir):
    os.chdir(".")
    if not os.path.isdir(dir):
        os.mkdir(dir)


# given a list of arrays (each corresponding to a clip) with varying lengths,
# makes all of them have equal (pair) length. The result is a single array
def make_equal_len(data, pipeline="arm2wh", method="reflect", maxpad=192):
    sizes = [arr.shape[0] for arr in data]
    if method=="0pad":
        maxpad = np.amax(sizes) if maxpad=="maxlen" else maxpad
        maxpad = maxpad + 1 if maxpad % 2 == 1 else maxpad
        res = [np.vstack((arr, np.zeros((maxpad-arr.shape[0],arr.shape[1]),int))) for arr in data]
        res = np.stack(res)

    elif method=="cutting":
        # get shortest length, cut the rest
        min_T = np.amin([arr.shape[0] for arr in data])
        min_T = min_T - 1 if sizes % 2 == 1 else min_T
        res = np.array([arr[:min_T,:] for arr in data])

    elif method=="cutting+0pad":  # 0pad shorter sequences, cut longer sequences
        res = np.array([arr[:maxpad,:] if arr.shape[0] >= maxpad else np.vstack( (arr, np.zeros((maxpad-arr.shape[0],arr.shape[1]),int)) ) for arr in data])

    elif method=="cutting+reflect":
        res = np.array([arr[:maxpad,:] if arr.shape[0] >= maxpad else np.pad(arr, ((0, maxpad-arr.shape[0]), (0,0)), "reflect") for arr in data])

    else: # method=="wrap" or method=="reflect"
        max_T = np.amax(sizes) + 1 if np.amax(sizes) % 2 == 1 else np.amax(sizes)
        max_T = max(max_T, maxpad)
        res = [np.pad(arr, ((0, max_T-arr.shape[0]), (0,0)), method) for arr in data]
        res = np.stack(res)

    return res


def load_windows(data_path, pipeline, require_text=False, text_path=None, require_audio=False,
                 hand3d_image=False, use_lazy=False, test_smpl=False, temporal=False):
    feats = pipeline.split('2')
    p0_size, p1_size = FEATURE_MAP[pipeline]
    if os.path.exists(data_path):
        print('using super quick load', data_path, flush=True)
        data = load_binary(data_path)
        data = make_equal_len(data, method="cutting+reflect")
        if pipeline=="arm2wh" or pipeline[:13]=="arm_wh2finger":
            p0_windows = data[:,:,:p0_size]
            p1_windows = data[:,:,p0_size:p0_size+p1_size]
        if require_text:
            text_windows = load_binary(text_path)
            p0_windows = (p0_windows, text_windows)
        return p0_windows, p1_windows


## utility to save results
def save_results(input, output, pipeline, base_path, tag=''):
    feats = pipeline.split('2')
    out_feat = feats[1]
    mkdir(os.path.join(base_path, 'results/'))
    print(f"input.shape, output.shape: {input.shape}, {output.shape}")
    assert not np.any(np.isnan(input))
    assert not np.any(np.isnan(output))
    if pipeline in list(FEATURE_MAP.keys()) or out_feat == 'wh' or out_feat == 'fingerL':
        filename = os.path.join(base_path, f"results/{tag}_inference_r6d")
        save_binary(np.concatenate((input, output), axis=2), filename)  # save in r6d format
        filename = os.path.join(base_path, f"results/{tag}_inference_aa")
        input_aa, output_aa = np.array(rot6d_to_aa(input)), np.array(rot6d_to_aa(output))
        print(f"input_aa.shape, output_aa.shape: {input_aa.shape}, {output_aa.shape}")
        assert not np.any(np.isnan(input_aa))
        assert not np.any(np.isnan(output_aa))
        save_binary(np.concatenate(( input_aa, output_aa ), axis=2), filename)  # save in aa format

        structure = skeletalModel.getSkeletalModelStructure()
        xyz_train = load_binary("video_data/xyz_train.pkl")#[:input.shape[0]]
        xyz_train = make_equal_len(xyz_train, method="cutting+reflect")
        xyz_train, _, _ = rmv_clips_nan(xyz_train, xyz_train)  ####!##
        root = get_root_bone(xyz_train, structure)
        assert not np.any(np.isnan(root))
        # root = load_binary("video_data/xyz_train_root.pkl")  # use the bone lengths and root references from training
        bone_len = pose3D.get_bone_length(xyz_train, structure)
        assert not np.any(np.isnan(bone_len))
        # bone_len = load_binary("video_data/lengths_train.pkl")

        input_output_aa = load_binary(os.path.join(base_path, f"results/{tag}_inference_aa.pkl"))
        assert not np.any(np.isnan(input_output_aa))
        #input_output_aa = np.concatenate(( input_aa, output_aa ), axis=2)
        input_output_xyz = aa_to_xyz(input_output_aa, root, bone_len, structure)
        assert not np.any(np.isnan(input_output_xyz))
        filename = os.path.join(base_path, f"results/{tag}_inference_xyz")
        save_binary(input_output_xyz, filename)  # save in xyz format


def process_H2S_dataset(dir="./Green Screen RGB clips* (frontal view)"):
    mkdir("video_data")

    (in_train, out_train, embeds_train), (in_val, out_val, embeds_val), (in_test, out_test, embeds_test) = load_H2S_dataset(dir, subset=1)
    # print("Loaded raw data from disk", flush=True)
    # neck_train, neck_val, neck_test = select_keypoints(in_train, NECK), select_keypoints(in_val, NECK), select_keypoints(in_test, NECK)
    # print("Selected NECK keypoints", flush=True)
    # arms_train, arms_val, arms_test = select_keypoints(in_train, ARMS), select_keypoints(in_val, ARMS), select_keypoints(in_test, ARMS)
    # print("Selected ARMS keypoints", flush=True)
    # hands_train, hands_val, hands_test = select_keypoints(out_train, HANDS), select_keypoints(out_val, HANDS), select_keypoints(out_test, HANDS)
    # print("Selected HANDS keypoints", flush=True)

    # feats_train = hconcat_feats(neck_train, arms_train, hands_train)
    # feats_val = hconcat_feats(neck_val, arms_val, hands_val)
    # feats_test = hconcat_feats(neck_test, arms_test, hands_test)

    # save_binary(feats_train, "video_data/xy_train.pkl", append=False)
    # save_binary(feats_test, "video_data/xy_test.pkl", append=False)
    # save_binary(feats_val, "video_data/xy_val.pkl", append=False)

    # save_binary(embeds_train, "video_data/train_sentence_embeddings.pkl", append=False)
    # save_binary(embeds_test, "video_data/test_sentence_embeddings.pkl", append=False)
    # save_binary(embeds_val, "video_data/val_sentence_embeddings.pkl", append=False)
    # save_binary(proc_text.obtain_avg_embed(key="train", subset=1), "video_data/average_train_sentence_embeddings.pkl")
    # save_binary(proc_text.obtain_avg_embed(key="val", subset=1), "video_data/average_val_sentence_embeddings.pkl")
    # save_binary(proc_text.obtain_avg_embed(key="test", subset=1), "video_data/average_test_sentence_embeddings.pkl")

    # print()
    # print("saved xy original and text embeddings", flush=True)
    # print()

    # lift_2d_to_3d(load_binary("video_data/xy_train.pkl"), "video_data/xyz_train.pkl")
    # print("lifted train to 3d", flush=True)
    # lift_2d_to_3d(load_binary("video_data/xy_val.pkl"), "video_data/xyz_val.pkl")
    # print("lifted val to 3d", flush=True)
    # lift_2d_to_3d(load_binary("video_data/xy_test.pkl"), "video_data/xyz_test.pkl")
    # print("lifted test to 3d", flush=True)
 
    # print()
    # print("saved lifted xyz", flush=True)
    # print()

    # train_3d = load_binary("video_data/xyz_train.pkl")
    # val_3d = load_binary("video_data/xyz_val.pkl")
    # test_3d = load_binary("video_data/xyz_test.pkl")

    # structure = skeletalModel.getSkeletalModelStructure()
    # lengths = pose3D.get_bone_length(train_3d, structure)
    # save_binary(lengths, "video_data/lengths_train.pkl")
    # print("Obtained bone lengths.", flush=True)

    # train_aa = xyz_to_aa(train_3d, structure, root_filename="video_data/xyz_train_root.pkl")
    # save_binary(aa_to_rot6d(train_aa), "video_data/r6d_train.pkl")
    # print("Train xyz to r6d.", flush=True)
    # val_aa = xyz_to_aa(val_3d, structure, root_filename="video_data/xyz_val_root.pkl")
    # save_binary(aa_to_rot6d(val_aa), "video_data/r6d_val.pkl")
    # print("Val xyz to r6d.", flush=True)
    # test_aa = xyz_to_aa(test_3d, structure, root_filename="video_data/xyz_test_root.pkl")
    # save_binary(aa_to_rot6d(test_aa), "video_data/r6d_test.pkl")
    # print("Test xyz to r6d.", flush=True)

    # print()
    # print("saved r6d data", flush=True)
    # print()

    # save_binary(obtain_vid_feats(kp_dir=dir, key="train"), "video_data/train_vid_feats.pkl")
    # save_binary(obtain_vid_feats(kp_dir=dir, key="val"), "video_data/val_vid_feats.pkl")
    # save_binary(obtain_vid_feats(kp_dir=dir, key="test"), "video_data/test_vid_feats.pkl")
    
    # print()
    # print(f"obtained video features", flush=True)
    # print()

    # print(f"processed all H2S data in {dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="/mnt/gpid07/datasets/How2Sign/How2Sign/utterance_level/", help="path to the directory where the dataset is located")
    args = parser.parse_args()
    if args.dataset_path=="Green Screen RGB clips* (frontal view)":
        DATA_PATHS = {
        "train": "train_2D_keypoints/openpose_output/json",
        "val": "val_2D_keypoints/openpose_output/json",
        "test": "test_2D_keypoints/openpose_output/json"
    }

    ##
    # process_H2S_dataset(args.dataset_path)
    ##


    ## generating viz for qualitative assessment
    # import wandb
    # from glob import glob
    # # xyz_train = load_binary("video_data/xyz_train.pkl")[0:25]
    # # structure = skeletalModel.getSkeletalModelStructure()
    # # gifs_paths = viz.viz(xyz_train, structure, frame_rate=2, results_dir=f"viz_results_xyz_train")
    # gifs_paths = glob("viz_results_xyz_train/"+"*.gif")[0:25]
    # with wandb.init(project="B2H-H2S", name="viz_xyz_train"):
    #     for path in gifs_paths:
    #         wandb.save(path)

    # xyz_test = load_binary("video_data/xyz_test.pkl")[0:25]
    # structure = skeletalModel.getSkeletalModelStructure()
    # gifs_paths = viz.viz(xyz_test, structure, frame_rate=2, results_dir=f"viz_results_xyz_test")
    # gifs_paths = glob("viz_results_xyz_test/"+"*.gif")
    # with wandb.init(project="B2H-H2S", name="viz_xyz_test"):
    #     for path in gifs_paths:
    #         wandb.save(path)

    ## DONE generating viz
    ## save to wandb viz from existing folder

    # # obtain array where each row is the average sentence embedding
    # save_binary(proc_text.obtain_avg_embed(key="train", subset=1), "video_data/average_train_sentence_embeddings.pkl")

    # structure = skeletalModel.getSkeletalModelStructure()
    # Visualize inference results
    # _inference_xyz = load_binary("results/_inference_xyz.pkl")
    # viz.viz(_inference_xyz, structure, frame_rate=1, results_dir="viz_results")


    # testing that kp match video coordinates
    # path_json = "/home/alvaro/Documents/ML and DL/How2Sign/B2H-H2S/Green Screen RGB clips* (frontal view)/test_2D_keypoints/openpose_output/json/G42xKICVj9U_4-10-rgb_front"
    # in_kp, out_kp = load_clip(path_json, "arm2wh", keep_confidence=False)

    # print(type(in_kp), in_kp.shape)
    # arms = select_keypoints([in_kp], ARMS, keep_confidence=False)[0]
    # print(arms.shape)

    # video = proc_vid.load_clip("G42xKICVj9U_4-10-rgb_front.mp4")
    # video_overlap = proc_vid.overlap_vid_points(np.moveaxis(video, 1, -1), arms)
    # print(video_overlap.shape)
    # video_overlap = np.moveaxis(video_overlap, -1, 1)
    # print(video_overlap.shape)
    # proc_vid.save_as_mp4(video_overlap, fps=25, filename="testing_overlap.avi")