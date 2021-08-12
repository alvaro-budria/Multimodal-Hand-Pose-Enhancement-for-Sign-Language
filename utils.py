import os
import json
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys

sys.path.append('./3DposeEstimator')
# 2D to 3D lifting
import skeletalModel
import pose2D
import pose2Dto3D
import pose3D

import viz.viz_3d as viz


DATA_PATHS = {
    "train": "train_2D_keypoints/openpose_output/json",
    "val": "val_2D_keypoints/openpose_output/json",
    "test": "test_2D_keypoints/openpose_output/json"
}

FEATURE_MAP = {
    'arm2wh':((6*6), 42*6),
}

EPSILON = 1e-10

NECK = [0, 1]  # neck in Open Pose 25
WRIST = [[4, 7], [0, 21]]  # wrist in arms, wrist in hand
ARMS = [2, 3, 4, 5, 6, 7]  # arms in Open Pose 25
HANDS = list(range(21*2))  # hands in Open Pose


# # helper function for xyz_to_aa
# # arms to axis-angle
# def _arm_xyz_to_aa(in_kp, idxs, neck_up, neck_low):
#     p_B = neck_up  # parent joint
#     p_J = neck_low  # current join
#     p_E = None  # next joint
#     u = p_J - p_B
#     v = None
#     in_aa = np.array([])
#     for i in ARMS:
#         # start with right arm. Switch to left arm at the fourth (i==3) joint
#         if i == 5:
#             p_B = neck_up  # parent joint
#             p_J = neck_low  # current join
#             u = p_J - p_B
#         p_E = in_kp[:,i*3:i*3+3]
#         v = p_E - p_J
        
#         # rotation angle theta
#         th = np.arccos(np.einsum('ij,ij->i', u, v)/(np.linalg.norm(u, axis=1)*np.linalg.norm(v, axis=1)) + 1e-6)

#         a = np.cross(u, v)
#         a = a / np.linalg.norm(a, axis=1)  # rotation axis

#         in_aa = np.hstack(( in_aa, np.multiply(a, th[:, np.newaxis]) )) if in_aa.shape[0]!=0 else np.multiply(a, th[:, np.newaxis])

#         p_B = p_J
#         p_J = p_E
#         u = v
#     return in_aa


# # helper function for xyz_to_aa
# # hands to axis-angle
# def _wh_xyz_to_aa(out_kp, wrist_r, wrist_l):
#     #bone_L = {}
#     out_aa_r, out_aa_l = np.array([]), np.array([])
#     for i in range(5):  # a hand has 5 fingers
#         p_B_r, p_B_l = wrist_r[0], wrist_l[0]  # parent joint
#         p_J_r, p_J_l = wrist_r[1], wrist_l[1]  # current join
#         p_E_r, p_E_l = None, None  # next joint
#         u_r, u_l = p_J_r-p_B_r, p_J_l-p_B_l
#         v_r, v_l = None, None
#         for j in range(4):  # each finger has 4 joints
#             p_E_r = out_kp[:,(1+4*i+j)*3:(1+4*i+j)*3+3]
#             p_E_l = out_kp[:,(22+4*i+j)*3:(22+4*i+j)*3+3]
#             v_r, v_l = p_E_r-p_J_r, p_E_l-p_J_l

#             # rotation angle
#             th_r = np.arccos(np.einsum('ij,ij->i', u_r, v_r)/(np.linalg.norm(u_r, axis=1)*np.linalg.norm(v_r, axis=1))) + 1e-6
#             th_l = np.arccos(np.einsum('ij,ij->i', u_l, v_l)/(np.linalg.norm(u_l, axis=1)*np.linalg.norm(v_l, axis=1))) + 1e-6

#             a_r, a_l = np.cross(u_r, v_r), np.cross(u_l, v_l)
#             a_r, a_l = a_r / np.linalg.norm(a_r, axis=1), a_l / np.linalg.norm(a_l, axis=1)  # rotation axis

#             out_aa_r = np.hstack(( out_aa_r, np.multiply(a, th_r[:, np.newaxis]) )) if out_aa_r.shape[0]!=0 else np.multiply(a, th_r[:, np.newaxis])
#             out_aa_l = np.hstack(( out_aa_l, np.multiply(a, th_l[:, np.newaxis]) )) if out_aa_l.shape[0]!=0 else np.multiply(a, th_l[:, np.newaxis])

#             p_B_r, p_B_l = p_J_r, p_J_l
#             p_J_r, p_J_l = p_E_r, p_E_l
#             u_r, u_l = v_r, v_l
#     return np.hstack((out_aa_r, out_aa_l))


# # Converts the spatial representation of a skeleton into axis-angle representation
# # in_kp, out_kp: a list of arrays, one per clip, with arrays of dims NUM_FRAMES, KEYPOINTS; for each frame, a list of the form [X0, Y0, Z0, X1, Y1, Z1, ...] containing each joint's position
# def old_xyz_to_aa(in_kp, out_kp, pipeline="arm2wh"):
#     feats = pipeline.split('2')
#     in_feat, out_feat = feats[0], feats[1]
#     in_aa, out_aa = [], []
#     neck, wrist = np.array([]), np.array([])

#     if in_feat == "arm" and out_feat == "wh":
#         for i in range(len(in_kp)):
#             ## axis-angle - arms
#             # take the upper and lower points of the neck as initial reference
#             neck_up_i =  in_kp[i][:,NECK[0]*3:NECK[0]*3+3]
#             neck_low_i = in_kp[i][:,NECK[1]*3:NECK[1]*3+3]
#             neck_i = np.hstack((neck_up_i, neck_low_i))
#             neck = np.vstack((neck, neck_i)) if neck.shape!=(0,) else neck_i
#             in_aa_i = _arm_xyz_to_aa(in_kp[i], ARMS, neck_up_i, neck_low_i)
#             in_aa.append(in_aa_i)

#         for i in range(len(out_kp)):
#             # take the wrist of the arms and the hands as initial references
#             wrist_r_arm, wrist_r_hand = in_kp[i][:,4*3:4*3+3], out_kp[i][:,0*3:0*3+3]
#             wrist_l_arm, wrist_l_hand = in_kp[i][:,7*3:7*3+3], out_kp[i][:,21*3:21*3+3]
#             wrist_r, wrist_l = np.hstack((wrist_r_arm, wrist_r_hand)), np.hstack((wrist_l_arm, wrist_l_hand))
#             wrist_i = np.hstack((wrist_r, wrist_l))
#             wrist = np.vstack((wrist, wrist_i)) if wrist.shape!=(0,) else wrist_i
#             # axis-angle - hands
#             out_aa_i = _wh_xyz_to_aa(out_kp[i], (wrist_r_arm, wrist_r_hand), (wrist_l_arm, wrist_l_hand))
#             out_aa.append(out_aa_i)

#     return in_aa, out_aa, np.average(neck, axis=0), np.average(wrist, axis=0)
#     #return in_aa, out_aa, (neck_up, neck_low), (wrist_r_arm, wrist_r_hand), (wrist_l_arm, wrist_l_hand)


# # Converts keypoints in axis-angle representation to Cartesian coordinates
# def old_aa_to_xyz(in_aa=None, lengths_in=None, neck_up=None, neck_low=None, out_aa=None, wrist_r=None, wrist_l=None, lengths_out=None):
#     in_kp = np.full_like(in_aa, 1)
#     if None not in [in_aa, neck_up, neck_low, lengths]:
#         p_B = neck_up
#         p_J = neck_low
#         u = p_J - p_B
#         for i in range(len(ARMS)):
#             a, th = _retrieve_axis_angle(in_aa[:,i*3:i*3+3])
#             # Rodrigues' rotation formula
#             v = np.multiply(u, np.cos(th)[:, np.newaxis]) \
#                 + np.multiply(np.cross(a, u), np.sin(th)[:, np.newaxis]) \
#                 + np.multiply(np.multiply(a, np.einsum('ij,ij->i', a, u)[:, np.newaxis]), (1-np.cos(th))[:, np.newaxis])

#             p_E = p_J + lengths[i]*v
#             in_kp[:,i*3:i*3+3] = p_E
#             p_B = p_J
#             p_J = p_E
#             u = v

#     out_kp_r, out_kp_l = wrist_r[1], wrist_l[1]
#     #np.full_like(out_aa[:,:21], 1), np.full_like(out_aa[:,21:], 1)
#     if None not in [out_aa, wrist_r, wrist_l, lengths]:
#         for i in range(5):  # a hand has 5 fingers
#             p_B_r, p_B_l = wrist_r[0], wrist_l[0]  # parent joint
#             p_J_r, p_J_l = wrist_r[1], wrist_l[1]  # current join
#             p_E_r, p_E_l = None, None  # next joint
#             u_r, u_l = p_J_r-p_B_r, p_J_l-p_B_l
#             v_r, v_l = None, None
#             for j in range(4):  # each finger has 4 joints
#                 a_r, th_r = _retrieve_axis_angle(out_aa[:,(4*i+j)*3:(4*i+j)*3+3])
#                 a_l, th_l = _retrieve_axis_angle(out_aa[:,(20+4*i+j)*3:(20+4*i+j)*3+3])

#                 # Rodrigues' rotation formula
#                 v_r =   np.multiply(u_r, np.cos(th_r)[:, np.newaxis]) \
#                       + np.multiply(np.cross(a_r, u_r), np.sin(th_r)[:, np.newaxis]) \
#                       + np.multiply(np.multiply(a_r, np.einsum('ij,ij->i', a_r, u_r)[:, np.newaxis]), (1-np.cos(th_r))[:, np.newaxis])
#                 v_l =   np.multiply(u_l, np.cos(th_l)[:, np.newaxis]) \
#                       + np.multiply(np.cross(a_l, u_l), np.sin(th_l)[:, np.newaxis]) \
#                       + np.multiply(np.multiply(a_l, np.einsum('ij,ij->i', a_l, u_l)[:, np.newaxis]), (1-np.cos(th_l))[:, np.newaxis])

#                 p_E_r = p_J_r + lengths[i*4+j]*v_r
#                 p_E_l = p_J_l + lengths[20+i*4+j]*v_l

#                 p_E_r = out_kp[:,(1+4*i+j)*3:(1+4*i+j)*3+3]
#                 p_E_l = out_kp[:,(22+4*i+j)*3:(22+4*i+j)*3+3]
#                 v_r, v_l = p_E_r-p_J_r, p_E_l-p_J_l
    
#     out_kp = hstack((out_kp_r, out_kp_l))
#     pass



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
    res = np.zeros((r6ds.shape[0], 3))
    for i,row in enumerate(r6ds):
        np_r6d = np.expand_dims(row, axis=0)
        np_mat = np.reshape(np_rot6d_to_mat(np_r6d)[0], (3,3))
        np_mat = R.from_matrix(np_mat)
        aa = np_mat.as_rotvec()
        res[i,:] = aa
    return res


def rot6d_to_aa(r6d):
    aa = []
    for clip in range(len(r6d)):
        r6d_clip = r6d[clip]
        aa_clip = np.empty((r6d_clip.shape[0], r6d_clip.shape[1]//2))
        for idx in range(0, r6d_clip.shape[1], 6):
            aa_clip[:,idx//2:idx//2+3] = _rot6d_to_aa(r6d_clip[:,idx:idx+6])
        aa.append(aa_clip)
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
    x_raw = np_r6d[:,0:3]
    y_raw = np_r6d[:,3:6]

    x = x_raw / np.linalg.norm(x_raw, ord=2, axis=-1)
    z = np.cross(x, y_raw)
    z = z / np.linalg.norm(z, ord=2, axis=-1)
    y = np.cross(z, x)

    x = np.reshape(x, [-1,3,1])
    y = np.reshape(y, [-1,3,1])
    z = np.reshape(z, [-1,3,1])
    np_matrix = np.concatenate([x,y,z], axis=-1)

    if len(shape) == 1:
        np_matrix = np.reshape(np_matrix, [9])
    else:
        output_shape = shape[:-1] + (9,)
        np_matrix = np.reshape(np_matrix, output_shape)

    return np_matrix


# From a vector representing a rotation in axis-angle representation,
# retrieves the rotation angle and the rotation axis
def _retrieve_axis_angle(aa):
    th = np.linalg.norm(aa, axis=1)
    a = aa / th[:,np.newaxis]
    return a, th


def aa_to_xyz(aa, root, bone_len, structure):
    xyz = []
    for i in range(len(aa)):
        aa_clip = aa[i]
        xyz_clip = np.empty((aa_clip.shape[0], aa_clip.shape[1]+6), dtype="float32")
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
    root = np.array([])
    for i in range(len(xyz)):
        xyz_clip = xyz[i]
        id_p_J, id_p_E, _, _ = structure[0]  # get initial and end joints indexes of root bone
        bone_points = np.hstack((xyz_clip[:,id_p_J*3:id_p_J*3+3], xyz_clip[:,id_p_E*3:id_p_E*3+3]))
        root = np.vstack( (root, bone_points) ) if root.shape!=(0,) else bone_points
    return np.average(root, axis=0)


def xyz_to_aa(xyz, structure, root_filename=None):
    save_binary(get_root_bone(xyz, structure), root_filename)
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

    # Getting 2D data
    # The sequence is an N-tuple of
    #   (1st point - x, 1st point - y, 1st point - likelihood, 2nd point - x, ...)
    # a missing point should have x=0, y=0, likelihood=0
    #f = h5py.File("data/demo-sequence.h5", "r")
    #inputSequence_2D = numpy.array(f.get("20161025_pocasi"))
    #f.close()

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
        learningRate=25,
        nCycles=1000
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
    return kp


# input is a list of arrays, one array per clip
def lift_2d_to_3d(feats, filename="feats_3d"):
    feats_3d = []
    
    for arr in feats:
        kp_3d = _lift_2d_to_3d(arr)
        feats_3d.append(kp_3d)
    save_binary(feats_3d, filename)


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
# returns [X1, Y1, ... Xn, Yn] or [X1, Y1, W1 ... Xn, Yn, Wn] if keep_confidence=False

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


def load_clip(clip_path, pipeline):
    feats = pipeline.split('2')
    in_feat, out_feat = feats[0], feats[1]
    in_kp, out_kp = np.array([]), np.array([])
    for frame in sorted(os.listdir(clip_path))[0:500]:  # for each frame, there is an associated .json file
        if os.path.isfile(os.path.join(clip_path, frame)):
            f = open(os.path.join(clip_path, frame))
            data = json.load(f)
            f.close()
            if in_feat == "arm":  
                if in_kp.shape != (0,):
                    in_kp = np.append(in_kp, [retrieve_coords(data["people"][0]["pose_keypoints_2d"], keep_confidence=True)], axis=0)
                else:
                    in_kp = np.array([retrieve_coords(data["people"][0]["pose_keypoints_2d"], keep_confidence=True)])
            if out_feat == "wh": 
                temp = [retrieve_coords(data["people"][0]["hand_right_keypoints_2d"], keep_confidence=True),
                        retrieve_coords(data["people"][0]["hand_left_keypoints_2d"], keep_confidence=True)]
                temp = [elem for singleList in temp for elem in singleList]
                if out_kp.shape != (0,):
                    out_kp = np.append(out_kp, [temp], axis=0)
                else:
                    out_kp = np.array([temp])
    return in_kp, out_kp


def _load_H2S_dataset(dir, pipeline):
    in_features, out_features = [], []
    i = 1
    for clip in os.listdir(dir)[0:1]:  # each clip is stored in a separate folder
        print(i)
        i += 1
        clip_path = os.path.join(dir, clip)
        in_kp, out_kp = load_clip(clip_path, pipeline)
        in_features.append(in_kp)
        out_features.append(out_kp)
    #in_features = np.array([ elem for singleList in in_features for elem in singleList])
    return in_features, out_features


def load_H2S_dataset(data_dir, pipeline="arm2wh", num_samples=None, require_text=False, require_audio=False):
    train_path = os.path.join(data_dir, DATA_PATHS["train"])
    val_path = os.path.join(data_dir, DATA_PATHS["val"])
    test_path = os.path.join(data_dir, DATA_PATHS["test"])
    # load data
    in_train, out_train, in_val, out_val, in_test, out_test = None, None, None, None, None, None,
    if os.path.exists(train_path):
        in_train, out_train = _load_H2S_dataset(train_path, pipeline=pipeline)
    if os.path.exists(val_path):
        in_val, out_val = _load_H2S_dataset(val_path, pipeline=pipeline)
    if os.path.exists(test_path):
        in_test, out_test = _load_H2S_dataset(val_path, pipeline=pipeline)
    return (in_train, out_train), (in_val, out_val), (in_test, out_test)


# returns the keypoints in the specified indexes
def get_joints(kp, idx):
    return kp[:,idx]


# selects the useful keypoints indicated by the indexes. Input is a list, each element containing the keypoints of a (video) clip
def select_keypoints(kp, idxs):
    kp_cp = kp.copy()
    for i in range(len(kp)):
        new_kp_i = np.array([])
        for idx in idxs:
            new_kp_i = np.hstack((new_kp_i, kp[i][:,idx*3:idx*3+3])) if new_kp_i.shape[0]!=0 else kp[i][:,idx*3:idx*3+3]
        kp_cp[i] = new_kp_i
    return kp_cp


def hconcat_feats(neck, arms, hands):
    assert [len(neck), len(arms)] == [len(hands), len(hands)]
    feats = []
    for i in range(len(neck)):  # for each frame, concat the features
        temp = np.hstack((neck[i], arms[i]))
        feats.append( np.hstack((temp, hands[i])) )
    return feats


def save_binary(obj, filename):
    if filename[-4:] != ".pkl":
        print("Adding .pkl extension as it was not found.")
        filename = filename + ".pkl"
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


# given a list of arrays (corresponding to a clip) with varying lengths,
# makes all of them have equal length. The result is a single array
def make_equal_len(data, pipeline="arm2wh", method="reflect", maxpad=256):
    if method=="0pad":
        sizes = [arr.shape[0] for arr in data]
        maxpad = np.amax(sizes) if maxpad=="maxlen" else maxpad
        res = [np.vstack((arr, np.zeros((maxpad-arr.shape[0],arr.shape[1]),int))) for arr in data]
        res = np.stack(res)        

    elif method=="cutting":
        # get shortest length, cut the rest
        min_T = np.amin([arr.shape[0] for arr in data])
        res = np.array([arr[:min_T,:] for arr in data])

    else: # method=="wrap" or method=="reflect"
        sizes = [arr.shape[0] for arr in data]
        max_T = np.amax(sizes)
        res = [np.pad(arr, ((0,0), (0, max_T-arr.shape[0])), method) for arr in data]
    
    return res


def load_windows(data_path, pipeline, num_samples=None, use_euler=False, require_text=False, require_audio=False,
                 hand3d_image=False, use_lazy=False, test_smpl=False, temporal=False):
    feats = pipeline.split('2')
    in_feat, out_feat = feats[0], feats[1]
    p0_size, p1_size = FEATURE_MAP[pipeline]
    if os.path.exists(data_path):
        print('using super quick load', data_path)
        data = load_binary(data_path)
        data = make_equal_len(data, method="0pad")
        if pipeline=="arm2wh":
            p0_windows = data[:,:,:p0_size]
            p1_windows = data[:,:,p0_size:p0_size+p1_size]
            B,T = p0_windows.shape[0], p0_windows.shape[1]
        # if require_text:
        #   text_windows = ...
        #    p0_windows = (p0_windows, text_windows)
        return p0_windows, p1_windows


def process_H2S_dataset(dir="./Green Screen RGB clips* (frontal view)"):
    structure = skeletalModel.getSkeletalModelStructure()
    mkdir("video_data")

    (in_train, out_train), (in_val, out_val), (in_test, out_test) = load_H2S_dataset(dir)

    neck_train, neck_val, neck_test = select_keypoints(in_train, NECK), select_keypoints(in_val, NECK), select_keypoints(in_test, NECK)
    arms_train, arms_val, arms_test = select_keypoints(in_train, ARMS), select_keypoints(in_val, ARMS), select_keypoints(in_test, ARMS)
    hands_train, hands_val, hands_test = select_keypoints(out_train, HANDS), select_keypoints(out_val, HANDS), select_keypoints(out_test, HANDS)

    feats_train = hconcat_feats(neck_train, arms_train, hands_train)
    feats_val = hconcat_feats(neck_val, arms_val, hands_val)
    feats_test = hconcat_feats(neck_test, arms_test, hands_test)
    
    save_binary(feats_train, "video_data/xy_train.pkl")
    save_binary(feats_val, "video_data/xy_val.pkl")
    save_binary(feats_test, "video_data/xy_test.pkl")

    print()
    print("saved xy original")
    print()

    lift_2d_to_3d(load_binary("video_data/xy_train.pkl"), "video_data/xyz_train.pkl")
    lift_2d_to_3d(load_binary("video_data/xy_val.pkl"), "video_data/xyz_val.pkl")
    lift_2d_to_3d(load_binary("video_data/xy_test.pkl"), "video_data/xyz_test.pkl")

    print()
    print("saved lifted xyz")
    print()

    train_3d = load_binary("video_data/xyz_train.pkl")
    val_3d = load_binary("video_data/xyz_val.pkl")
    test_3d = load_binary("video_data/xyz_test.pkl")

    lengths = pose3D.get_bone_length(train_3d, structure)
    save_binary(lengths, "video_data/lengths_train.pkl")

            #  xyz_to_aa() also saves the root bone (first one in the skeletal structure)
    train_aa = xyz_to_aa(train_3d, structure, root_filename="video_data/xyz_train_root.pkl")
    save_binary(aa_to_rot6d(train_aa), "video_data/r6d_train.pkl")
    val_aa = xyz_to_aa(val_3d, structure, root_filename="video_data/xyz_val_root.pkl")
    save_binary(aa_to_rot6d(val_aa), "video_data/r6d_val.pkl")
    test_aa = xyz_to_aa(test_3d, structure, root_filename="video_data/xyz_test_root.pkl")
    save_binary(aa_to_rot6d(test_aa), "video_data/r6d_test.pkl")

    print(f"processed all H2S data in {dir}")


if __name__ == "__main__":
    process_H2S_dataset()
    
    #structure = skeletalModel.getSkeletalModelStructure()

    # (in_train, out_train), (in_val, out_val), (in_test, out_test) = load_H2S_dataset("./Green Screen RGB clips* (frontal view)")

    # neck_train, neck_val, neck_test = select_keypoints(in_train, NECK), select_keypoints(in_val, NECK), select_keypoints(in_test, NECK)
    # arms_train, arms_val, arms_test = select_keypoints(in_train, ARMS), select_keypoints(in_val, ARMS), select_keypoints(in_test, ARMS)
    # hands_train, hands_val, hands_test = select_keypoints(out_train, HANDS), select_keypoints(out_val, HANDS), select_keypoints(out_test, HANDS)

    # feats_train = hconcat_feats(neck_train, arms_train, hands_train)
    # feats_val = hconcat_feats(neck_val, arms_val, hands_val)
    # feats_test = hconcat_feats(neck_test, arms_test, hands_test)

    # save_binary(feats_train, "xy_train.pkl")
    # save_binary(feats_val, "xy_val.pkl")
    # save_binary(feats_test, "xy_test.pkl")

    # print()
    # print("saved xyz original")
    # print()

    # lift_2d_to_3d(load_binary("xy_train.pkl"), "xyz_train.pkl")
    # lift_2d_to_3d(load_binary("xy_val.pkl"), "xyz_val.pkl")
    # lift_2d_to_3d(load_binary("xy_test.pkl"), "xyz_test.pkl")


    #train_3d = load_binary("xyz_train.pkl")
    #viz_3d.viz(train_3d, structure)
    # val_3d = load_binary("xyz_val.pkl")
    # test_3d = load_binary("xyz_test.pkl")
    # print(len(train_3d), train_3d[0].shape)

    # lengths = pose3D.get_bone_length(train_3d, structure)
    # # print(lengths)
    # save_binary(lengths, "lengths_train.pkl")

             # xyz_to_aa() also saves the root bone (first one in the skeletal structure)
    # train_aa = xyz_to_aa(train_3d, structure, root_filename="xyz_train_root.pkl")
    # save_binary(train_aa, "aa_train.pkl")
    # print(len(train_aa), train_aa[0].shape)

    # root = get_root_bone(train_3d, structure)
    # bone_len = load_binary("lengths_train.pkl")
    # train_3d_from_aa = aa_to_xyz(train_aa, root, bone_len, structure)
    # print(len(train_3d_from_aa), train_3d_from_aa[0].shape)
    # viz_3d.viz(train_3d_from_aa, structure)
    #print(train_3d[1]-train_3d_aa[1])

    # train_rd6 = aa_to_rot6d(train_aa)
    # print(len(train_rd6), train_rd6[0].shape)

    # train_aa_from_r6d = rot6d_to_aa(train_rd6)
    # print(len(train_aa_from_r6d), train_aa_from_r6d[1].shape)


    #print(train_aa[0])
    # recover neck and wrist keypoints to reconstruct the model's output from aa to xyz
    #wrist_train = np.hstack(in_train[:,WRIST[0]], hands_train[:,WRIST[1]])