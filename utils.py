import os
import json
import pickle
import numpy as np

import sys
sys.path.append('./3DposeEstimator')

# 2D to 3D lifting
import skeletalModel
import pose2D
import pose2Dto3D
import pose3D


DATA_PATHS = {
    "train": "train_2D_keypoints/openpose_output/json",
    "val": "val_2D_keypoints/openpose_output/json",
    "test": "test_2D_keypoints/openpose_output/json"
}

NECK = [0, 1]  # neck in Open Pose 25
WRIST = [[4, 7], [0, 21]]  # wrist in arms, wrist in hand
ARMS = [2, 3, 4, 5, 6, 7]  # arms in Open Pose 25
HANDS = list(range(21*2))  # hands in Open Pose

# From a vector representing a rotation in axis-angle representation,
# retrieves the rotation angle and the rotation axis
def _retrieve_axis_angle(aa):
    th = np.linalg.norm(aa, axis=1)
    a = aa / th
    return a, th


# Converts keypoints in axis-angle representation to Cartesian coordinates
def aa_to_xyz(in_aa=None, lengths_in=None, neck_up=None, neck_low=None, out_aa=None, wrist_r=None, wrist_l=None, lengths_out=None):
    in_kp = np.full_like(in_aa, 1)
    if None not in [in_aa, neck_up, neck_low, lengths]:
        p_B = neck_up
        p_J = neck_low
        u = p_J - p_B
        for i in range(len(ARMS)):
            a, th = _retrieve_axis_angle(in_aa[:,i*3:i*3+3])
            # Rodrigues' rotation formula
            v = np.multiply(u, np.cos(th)[:, np.newaxis]) \
                + np.multiply(np.cross(a, u), np.sin(th)[:, np.newaxis]) \
                + np.multiply(np.multiply(a, np.einsum('ij,ij->i', a, u)[:, np.newaxis]), (1-np.cos(th))[:, np.newaxis])

            p_E = p_J + lengths[i]*v
            in_kp[:,i*3:i*3+3] = p_E
            p_B = p_J
            p_J = p_E
            u = v

    out_kp_r, out_kp_l = wrist_r[1], wrist_l[1]
    #np.full_like(out_aa[:,:21], 1), np.full_like(out_aa[:,21:], 1)
    if None not in [out_aa, wrist_r, wrist_l, lengths]:
        for i in range(5):  # a hand has 5 fingers
            p_B_r, p_B_l = wrist_r[0], wrist_l[0]  # parent joint
            p_J_r, p_J_l = wrist_r[1], wrist_l[1]  # current join
            p_E_r, p_E_l = None, None  # next joint
            u_r, u_l = p_J_r-p_B_r, p_J_l-p_B_l
            v_r, v_l = None, None
            for j in range(4):  # each finger has 4 joints
                a_r, th_r = _retrieve_axis_angle(out_aa[:,(4*i+j)*3:(4*i+j)*3+3])
                a_l, th_l = _retrieve_axis_angle(out_aa[:,(20+4*i+j)*3:(20+4*i+j)*3+3])

                # Rodrigues' rotation formula
                v_r =   np.multiply(u_r, np.cos(th_r)[:, np.newaxis]) \
                      + np.multiply(np.cross(a_r, u_r), np.sin(th_r)[:, np.newaxis]) \
                      + np.multiply(np.multiply(a_r, np.einsum('ij,ij->i', a_r, u_r)[:, np.newaxis]), (1-np.cos(th_r))[:, np.newaxis])
                v_l =   np.multiply(u_l, np.cos(th_l)[:, np.newaxis]) \
                      + np.multiply(np.cross(a_l, u_l), np.sin(th_l)[:, np.newaxis]) \
                      + np.multiply(np.multiply(a_l, np.einsum('ij,ij->i', a_l, u_l)[:, np.newaxis]), (1-np.cos(th_l))[:, np.newaxis])

                p_E_r = p_J_r + lengths[i*4+j]*v_r
                p_E_l = p_J_l + lengths[20+i*4+j]*v_l

                p_E_r = out_kp[:,(1+4*i+j)*3:(1+4*i+j)*3+3]
                p_E_l = out_kp[:,(22+4*i+j)*3:(22+4*i+j)*3+3]
                v_r, v_l = p_E_r-p_J_r, p_E_l-p_J_l
    
    out_kp = hstack((out_kp_r, out_kp_l))
    pass


# helper function for xyz_to_aa
# arms to axis-angle
def _arm_xyz_to_aa(in_kp, idxs, neck_up, neck_low):
    p_B = neck_up  # parent joint
    p_J = neck_low  # current join
    p_E = None  # next joint
    u = p_J - p_B
    v = None
    #bone_L = {}
    in_aa = np.array([])
    for i in ARMS:
        # start with right arm. Switch to left arm at the fourth (i==3) joint
        if i == 3:
            p_B = neck_up  # parent joint
            p_J = neck_low  # current join
            u = p_J - p_B
        p_E = in_kp[:,i*3:i*3+3]
        v = p_E - p_J
        #bone_L[] = 
        
        # rotation angle theta
        th = np.arccos(np.einsum('ij,ij->i', u, v)/(np.linalg.norm(u, axis=1)*np.linalg.norm(v, axis=1))) + 1e-6

        a = np.cross(u, v)
        a = a / np.linalg.norm(a, axis=1)  # rotation axis

        in_aa = np.hstack(( in_aa, np.multiply(a, th[:, np.newaxis]) )) if in_aa.shape[0]!=0 else np.multiply(a, th[:, np.newaxis])

        p_B = p_J
        p_J = p_E
        u = v
    return in_aa


# helper function for xyz_to_aa
# hands to axis-angle
def _wh_xyz_to_aa(out_kp, wrist_r, wrist_l):
    #bone_L = {}
    out_aa_r, out_aa_l = np.array([]), np.array([])
    for i in range(5):  # a hand has 5 fingers
        p_B_r, p_B_l = wrist_r[0], wrist_l[0]  # parent joint
        p_J_r, p_J_l = wrist_r[1], wrist_l[1]  # current join
        p_E_r, p_E_l = None, None  # next joint
        u_r, u_l = p_J_r-p_B_r, p_J_l-p_B_l
        v_r, v_l = None, None
        for j in range(4):  # each finger has 4 joints
            p_E_r = out_kp[:,(1+4*i+j)*3:(1+4*i+j)*3+3]
            p_E_l = out_kp[:,(22+4*i+j)*3:(22+4*i+j)*3+3]
            v_r, v_l = p_E_r-p_J_r, p_E_l-p_J_l

            # rotation angle
            th_r = np.arccos(np.einsum('ij,ij->i', u_r, v_r)/(np.linalg.norm(u_r, axis=1)*np.linalg.norm(v_r, axis=1))) + 1e-6
            th_l = np.arccos(np.einsum('ij,ij->i', u_l, v_l)/(np.linalg.norm(u_l, axis=1)*np.linalg.norm(v_l, axis=1))) + 1e-6

            a_r, a_l = np.cross(u_r, v_r), np.cross(u_l, v_l)
            a_r, a_l = a_r / np.linalg.norm(a_r, axis=1), a_l / np.linalg.norm(a_l, axis=1)  # rotation axis

            out_aa_r = np.hstack(( out_aa_r, np.multiply(a, th_r[:, np.newaxis]) )) if out_aa_r.shape[0]!=0 else np.multiply(a, th_r[:, np.newaxis])
            out_aa_l = np.hstack(( out_aa_l, np.multiply(a, th_l[:, np.newaxis]) )) if out_aa_l.shape[0]!=0 else np.multiply(a, th_l[:, np.newaxis])

            p_B_r, p_B_l = p_J_r, p_J_l
            p_J_r, p_J_l = p_E_r, p_E_l
            u_r, u_l = v_r, v_l
    return np.hstack((out_aa_r, out_aa_l))


# Converts the spatial representation of a skeleton into axis-angle representation
# in_kp, out_kp: dims NUM_FRAMES, KEYPOINTS; for each frame, a list of the form [X0, Y0, Z0, X1, Y1, Z1, ...] containing each joint's position
def xyz_to_aa(in_kp, out_kp, pipeline):
    feats = pipeline.split('2')
    in_feat, out_feat = feats[0], feats[1]

    if in_feat == "arm" and out_feat == "wh":
       
        ## axis-angle - arms
        # take the upper and lower points of the neck as initial reference
        neck_up =  in_kp[:,NECK[0]*3:NECK[0]*3+3]
        neck_low = in_kp[:,NECK[1]*3:NECK[1]*3+3]
        in_aa = _arm_xyz_to_aa(in_kp, ARMS, neck_up, neck_low)

        # take the wrist of the arms and the hands as initial references
        wrist_r_arm, wrist_r_hand = in_kp[:,4*3:4*3+3], out_kp[:,0*3:0*3+3]
        wrist_l_arm, wrist_l_hand = in_kp[:,7*3:7*3+3], out_kp[:,21*3:21*3+3]
        # axis-angle - hands
        out_aa = _wh_xyz_to_aa(out_kp, (wrist_r_arm, wrist_r_hand), (wrist_l_arm, wrist_l_hand))

        return in_aa, out_aa, (neck_up, neck_low), (wrist_r_arm, wrist_r_hand), (wrist_l_arm, wrist_l_hand)


# Normalize skeleton (given as collection of 3D keypoints)
def normalize():
    pass


# mirar què diuen l'Amanda Duarte i la literatura
def scale():
    pass


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
    for frame in sorted(os.listdir(clip_path))[0:105]:  # for each frame, there is an associated .json file
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


def _load_dataset(dir, pipeline):
    in_features, out_features = [], []
    i = 1
    for clip in os.listdir(dir)[0:2]:  # each clip is stored in a separate folder
        print(i)
        i += 1
        clip_path = os.path.join(dir, clip)
        in_kp, out_kp = load_clip(clip_path, pipeline)
        in_features.append(in_kp)
        out_features.append(out_kp)
    #in_features = np.array([ elem for singleList in in_features for elem in singleList])
    return in_features, out_features


def load_data(data_dir, pipeline="arm2wh", num_samples=None, require_image=False, require_audio=False):
    train_path = os.path.join(data_dir, DATA_PATHS["train"])
    val_path = os.path.join(data_dir, DATA_PATHS["val"])
    test_path = os.path.join(data_dir, DATA_PATHS["test"])
    # load data
    in_train, out_train, in_val, out_val, in_test, out_test = None, None, None, None, None, None,
    if os.path.exists(train_path):
        in_train, out_train = _load_dataset(train_path, pipeline=pipeline)
    if os.path.exists(val_path):
        in_val, out_val = _load_dataset(val_path, pipeline=pipeline)
    if os.path.exists(test_path):
        in_test, out_test = _load_dataset(val_path, pipeline=pipeline)
    return (in_train, out_train), (in_val, out_val), (in_test, out_test)


def _save(fname, lst):
    T, dim = lst[0].shape
    f = open(fname, "w")
    for t in range(T):
        for i in range(dim):
            for j in range(len(lst)):
                f.write("%e\t" % lst[j][t, i])
        f.write("\n")
    f.close()


# returns the keypoints in the specified indexes
def get_joints(kp, idx):
    return kp[:,idx]


# given a list of clips (possibly with variable length), lifts the keypoints to 3D
def clips_2d_to_3d():
    pass


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
    Yx, Yy, Yz = pose3D.backpropagationBasedFiltering(
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
        nCycles=10
    )
    #_save("3D_keypoints.txt", [Yx, Yy, Yz])
    

    kp = np.empty((Yx.shape[0], Yx.shape[1] + Yy.shape[1] + Yz.shape[1]), dtype=dtype)
    kp[:,0::3], kp[:,1::3], kp[:,2::3] = Yx, Yy, Yz
    return kp
    ######## cal implementar-ho
    #lengths = pose3D.get_bone_length(Yx, Yy, Yz)
    # compute mean length and store it
    # asdfdsfda


# input is a list of arrays, one array per clip
def lift_2d_to_3d(feats, filename="feats_3d"):
    feats_3d = []
    for arr in feats:
        kp_3d = _lift_2d_to_3d(arr)
        feats_3d.append(kp_3d)
    save_binary(feats_3d, filename)


if __name__ == "__main__":
    (in_train, out_train), (in_val, out_val), (in_test, out_test) = load_data("./Green Screen RGB clips* (frontal view)")

    neck_train, neck_val, neck_test = select_keypoints(in_train, NECK), select_keypoints(in_val, NECK), select_keypoints(in_test, NECK)
    arms_train, arms_val, arms_test = select_keypoints(in_train, ARMS), select_keypoints(in_val, ARMS), select_keypoints(in_test, ARMS)
    hands_train, hands_val, hands_test = select_keypoints(out_train, HANDS), select_keypoints(out_val, HANDS), select_keypoints(out_test, HANDS)

    feats_train = hconcat_feats(neck_train, arms_train, hands_train)
    feats_val = hconcat_feats(neck_val, arms_val, hands_val)
    feats_test = hconcat_feats(neck_test, arms_test, hands_test)

    save_binary(feats_train, "xy_train.pkl")
    save_binary(feats_val, "xy_val.pkl")
    save_binary(feats_test, "xy_test.pkl")

    lift_2d_to_3d(load_binary("xy_train.pkl"), "xyz_train.pkl")
    lift_2d_to_3d(load_binary("xy_val.pkl"), "xyz_val.pkl")
    lift_2d_to_3d(load_binary("xy_test.pkl"), "xyz_test.pkl")

    
    train_3d = load_binary("xyz_train.pkl")
    # val_3d = load_binary("xyz_val.pkl")
    # test_3d = load_binary("xyz_test.pkl")
    # print(len(train_3d), train_3d[0].shape)

    lengths = pose3D.get_bone_length(train_3d[0], skeletalModel.getSkeletalModelStructure())
    print(lengths, len(lengths))
    # recover neck and wrist keypoints to reconstruct the model's output, from aa to xyz
    #wrist_train = np.hstack(in_train[:,WRIST[0]], hands_train[:,WRIST[1]])