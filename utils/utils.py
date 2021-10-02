import os
import glob
import sys
import json
import pickle
import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool

import numpy as np

sys.path.append('./3DposeEstimator')
print(sys.path)
# 2D to 3D lifting
import skeletalModel
import pose2D
import pose2Dto3D
import pose3D

sys.path.append('./')
import viz.viz_3d as viz

import proc_text
import proc_vid

from conversion_utils import *
from load_save_utils import *
from standardization_utils import *
from postprocess_utils import *
from constants import *


def get_root_bone(xyz, structure):
    xyz = array_to_list(xyz)
    root = np.array([])
    for i in range(len(xyz)):
        id_p_J, id_p_E, _, _ = structure[0]  # get initial and end joints indexes of root bone
        bone_points = np.hstack((xyz_clip[:,id_p_J*3:id_p_J*3+3], xyz_clip[:,id_p_E*3:id_p_E*3+3]))
        root = np.vstack( (root, bone_points) ) if root.shape!=(0,) else bone_points
    return np.average(root, axis=0)


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

    ##### watchThis ?? per que (0, 1, 2, 3, 4, 5, 6, 7) ?
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
        print(f" -> Found file with name {filename}. Appending results to this file.", flush=True)
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


# given a list of the form [X1, Y1, conf1, X2, Y2, conf2 ... Xn, Yn, conf_n]
# returns [X1, Y1, ... Xn, Yn] or [X1, Y1, W1 ... Xn, Yn, Wn] if keep_confidence=True

## @jit
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
    return (in_train, out_train, embeds_train), (in_val, out_val, embeds_val), (in_test, out_test, embeds_test)


def obtain_vid_crops(kp_dir, key, data_dir):
    kp_path = os.path.join(kp_dir, DATA_PATHS[key])
    kp_dir_list = os.listdir(kp_path)
    clip_ids_text = proc_text.get_clip_ids(key=key)
    ids = _join_ids(kp_dir_list, clip_ids_text)  # keep id that are present both in kp_dir_list (IDs for which keypoints are availabe)
                                                 # and in clip_ids_text (IDs for which text sentences are availabe)
    clip_ids_vid = proc_vid.get_vid_ids(key=key)
    ids = _join_ids(ids, clip_ids_vid)
    ids = sorted(ids)
    print("Obtained ids! Entering proc_vid.obtain_crops", flush=True)
    size = 200
    start = 0
    for subset in range(start, len(ids), size):
        print(f"subset: {subset}", flush=True)
        hand_feats = proc_vid.obtain_crops(key, ids[subset:subset+size])
        save_binary(hand_feats, f"{data_dir}/{key}_vid_crops_{subset}-{subset+size}.pkl")

    # store all crops into a single file
    print("storing all crops into a single file...", flush=True)
    hand_feats = []
    vid_feats_files = glob.glob(f"{data_dir}/{key}_vid_crops_*.pkl")
    for file in vid_feats_files:
        hand_feats += load_binary(file)
        os.remove(file)  # remove batch files, leave only single whole file
    save_binary(hand_feats, f"{data_dir}/{key}_vid_crops.pkl")
    print("stored all crops into a single file", flush=True)


# obtains features from the image crops contained in data_dir
def obtain_vid_feats(key, data_dir):
    hand_crops_list = load_binary(f"{data_dir}/{key}_vid_crops.pkl")
    print(f"loaded {data_dir}/{key}_vid_crops.pkl", flush=True)
    feats_list = proc_vid.obtain_feats_crops_ResNet(hand_crops_list, data_dir)
    save_binary(feats_list, f"{data_dir}/{key}_vid_feats.pkl")


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


## utility to save results
def save_results(input, output, pipeline, base_path, data_dir, tag=''):
    feats = pipeline.split('2')
    out_feat = feats[1]
    mkdir(os.path.join(base_path, 'results/'))
    print(f"input.shape, output.shape: {input.shape}, {output.shape}", flush=True)
    assert not np.any(np.isnan(input))
    assert not np.any(np.isnan(output))
    if pipeline in list(FEATURE_MAP.keys()) or out_feat == 'wh' or out_feat == 'fingerL':
        filename = os.path.join(base_path, f"results/{tag}_inference_r6d")
        save_binary(np.concatenate((input, output), axis=2), filename)  # save in r6d format
        filename = os.path.join(base_path, f"results/{tag}_inference_aa")
        input_aa, output_aa = np.array(rot6d_to_aa(input)), np.array(rot6d_to_aa(output))
        print(f"input_aa.shape, output_aa.shape: {input_aa.shape}, {output_aa.shape}", flush=True)
        assert not np.any(np.isnan(input_aa))
        assert not np.any(np.isnan(output_aa))
        save_binary(np.concatenate(( input_aa, output_aa ), axis=2), filename)  # save in aa format

        structure = skeletalModel.getSkeletalModelStructure()
        xyz_train = load_binary(f"{data_dir}/xyz_train.pkl")#[:input.shape[0]]
        xyz_train = make_equal_len(xyz_train, method="cutting+reflect")
        xyz_train, _, _ = rmv_clips_nan(xyz_train, xyz_train)  ####!##
        root = get_root_bone(xyz_train, structure)
        assert not np.any(np.isnan(root))
        # root = load_binary(f"{data_dir}/xyz_train_root.pkl")  # use the bone lengths and root references from training
        bone_len = pose3D.get_bone_length(xyz_train, structure)
        assert not np.any(np.isnan(bone_len))
        # bone_len = load_binary(f"{data_dir}/lengths_train.pkl")

        input_output_aa = load_binary(os.path.join(base_path, f"results/{tag}_inference_aa.pkl"))
        assert not np.any(np.isnan(input_output_aa))
        #input_output_aa = np.concatenate(( input_aa, output_aa ), axis=2)
        input_output_xyz = aa_to_xyz(input_output_aa, root, bone_len, structure)
        assert not np.any(np.isnan(input_output_xyz))
        filename = os.path.join(base_path, f"results/{tag}_inference_xyz")
        save_binary(input_output_xyz, filename)  # save in xyz format


def process_H2S_dataset(dir, data_dir):
    mkdir(data_dir)

    (in_train, out_train, embeds_train), (in_val, out_val, embeds_val), (in_test, out_test, embeds_test) = load_H2S_dataset(dir, subset=1)
    print("Loaded raw data from disk", flush=True)
    neck_train, neck_val, neck_test = select_keypoints(in_train, NECK), select_keypoints(in_val, NECK), select_keypoints(in_test, NECK)
    print("Selected NECK keypoints", flush=True)
    arms_train, arms_val, arms_test = select_keypoints(in_train, ARMS), select_keypoints(in_val, ARMS), select_keypoints(in_test, ARMS)
    print("Selected ARMS keypoints", flush=True)
    hands_train, hands_val, hands_test = select_keypoints(out_train, HANDS), select_keypoints(out_val, HANDS), select_keypoints(out_test, HANDS)
    print("Selected HANDS keypoints", flush=True)

    #feats_train = hconcat_feats(neck_train, arms_train, hands_train)
    feats_val = hconcat_feats(neck_val, arms_val, hands_val)
    #feats_test = hconcat_feats(neck_test, arms_test, hands_test)

    #save_binary(feats_train, f"{data_dir}/xy_train.pkl", append=False)
    #save_binary(feats_test, f"{data_dir}/xy_test.pkl", append=False)
    save_binary(feats_val, f"{data_dir}/xy_val.pkl", append=False)

    #save_binary(embeds_train, f"{data_dir}/train_sentence_embeddings.pkl", append=False)
    #save_binary(embeds_test, f"{data_dir}/test_sentence_embeddings.pkl", append=False)
    #save_binary(embeds_val, f"{data_dir}/val_sentence_embeddings.pkl", append=False)
    #save_binary(proc_text.obtain_avg_embed(key="train", subset=1), f"{data_dir}/average_train_sentence_embeddings.pkl")
    #save_binary(proc_text.obtain_avg_embed(key="val", subset=1), f"{data_dir}/average_val_sentence_embeddings.pkl")
    #save_binary(proc_text.obtain_avg_embed(key="test", subset=1), f"{data_dir}/average_test_sentence_embeddings.pkl")

    print()
    print("saved xy original and text embeddings", flush=True)
    print()

    lift_2d_to_3d(load_binary(f"{data_dir}/xy_train.pkl"), f"{data_dir}/xyz_train.pkl")
    print("lifted train to 3d", flush=True)
    # lift_2d_to_3d(load_binary(f"{data_dir}/xy_val.pkl"), f"{data_dir}/xyz_val.pkl")
    # print("lifted val to 3d", flush=True)
    # lift_2d_to_3d(load_binary(f"{data_dir}/xy_test.pkl"), f"{data_dir}/xyz_test.pkl")
    # print("lifted test to 3d", flush=True)
 
    # print()
    # print("saved lifted xyz", flush=True)
    # print()

    # train_3d = load_binary(f"{data_dir}/xyz_train.pkl")
    # val_3d = load_binary(f"{data_dir}/xyz_val.pkl")
    # test_3d = load_binary(f"{data_dir}/xyz_test.pkl")

    # structure = skeletalModel.getSkeletalModelStructure()
    # lengths = pose3D.get_bone_length(train_3d, structure)
    # save_binary(lengths, f"{data_dir}/lengths_train.pkl")
    # print("Obtained bone lengths.", flush=True)

    # train_aa = xyz_to_aa(train_3d, structure, root_filename=f"{data_dir}/xyz_train_root.pkl")
    # save_binary(aa_to_rot6d(train_aa), f"{data_dir}/r6d_train.pkl")
    # print("Train xyz to r6d.", flush=True)
    # val_aa = xyz_to_aa(val_3d, structure, root_filename=f"{data_dir}/xyz_val_root.pkl")
    # save_binary(aa_to_rot6d(val_aa), f"{data_dir}/r6d_val.pkl")
    # print("Val xyz to r6d.", flush=True)
    # test_aa = xyz_to_aa(test_3d, structure, root_filename=f"{data_dir}/xyz_test_root.pkl")
    # save_binary(aa_to_rot6d(test_aa), f"{data_dir}/r6d_test.pkl")
    # print("Test xyz to r6d.", flush=True)

    # print()
    # print("saved r6d data", flush=True)
    # print()

    obtain_vid_crops(kp_dir=dir, key="val", data_dir=data_dir)
    print("vid crops val")
    # obtain_vid_crops(kp_dir=dir, key="test", data_dir=data_dir)
    # print("vid crops test")
    # obtain_vid_crops(kp_dir=dir, key="train", data_dir=data_dir)
    # print("vid feats train")

    # print()
    # print(f"obtained video crops", flush=True)
    # print()

    # compute_mean_std("train_vid_crops.pkl", data_dir)
    # print(f"saved mean and std for vids in train_vid_crops.pkl")

    obtain_vid_feats("val", data_dir)
    print("vid feats val", flush=True)
    # obtain_vid_feats("test", data_dir)
    # print("vid feats test", flush=True)
    # obtain_vid_feats("train", data_dir)
    # print("vid feats train", flush=True)

    # print()
    # print(f"obtained video features", flush=True)
    # print()

    # print(f"processed all H2S data in {dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="/mnt/gpid07/datasets/How2Sign/How2Sign/utterance_level/", help="path to the directory where the dataset is located")
    parser.add_argument('--data_dir', type=str, default="video_data", help="directory where results should be stored")
    args = parser.parse_args()
    if args.dataset_path=="Green Screen RGB clips* (frontal view)":
        DATA_PATHS = {
        "train": "train_2D_keypoints/openpose_output/json",
        "val": "val_2D_keypoints/openpose_output/json",
        "test": "test_2D_keypoints/openpose_output/json"
    }

    ##
    process_H2S_dataset(args.dataset_path, data_dir=args.data_dir)
    ##


    ## generating viz for qualitative assessment
    # import wandb
    # from glob import glob
    # # xyz_train = load_binary(f"{data_dir}/xyz_train.pkl")[0:25]
    # # structure = skeletalModel.getSkeletalModelStructure()
    # # gifs_paths = viz.viz(xyz_train, structure, frame_rate=2, results_dir=f"viz_results_xyz_train")
    # gifs_paths = glob("viz_results_xyz_train/"+"*.gif")[0:25]
    # with wandb.init(project="B2H-H2S", name="viz_xyz_train"):
    #     for path in gifs_paths:
    #         wandb.save(path)

    # xyz_test = load_binary(f"{data_dir}/xyz_test.pkl")[0:25]
    # structure = skeletalModel.getSkeletalModelStructure()
    # gifs_paths = viz.viz(xyz_test, structure, frame_rate=2, results_dir=f"viz_results_xyz_test")
    # gifs_paths = glob("viz_results_xyz_test/"+"*.gif")
    # with wandb.init(project="B2H-H2S", name="viz_xyz_test"):
    #     for path in gifs_paths:
    #         wandb.save(path)

    ## DONE generating viz
    ## save to wandb viz from existing folder

    # # obtain array where each row is the average sentence embedding
    # save_binary(proc_text.obtain_avg_embed(key="train", subset=1), f"{data_dir}/average_train_sentence_embeddings.pkl")

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