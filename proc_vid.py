import os
import argparse
import pickle
import glob
from multiprocessing import Pool
from re import S

import cv2
import clip  # to obtain CLIP embeddings

import numpy as np
import torch

VID_PATHS = {
    "train": "/mnt/gpid08/datasets/How2Sign/How2Sign/utterance_level/train/rgb_front/raw_videos/",
    "val": "/mnt/gpid08/datasets/How2Sign/How2Sign/utterance_level/val/rgb_front/raw_videos/",
    "test": "/mnt/gpid08/datasets/How2Sign/How2Sign/utterance_level/test/rgb_front/raw_videos/"
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# obtains frame-level embeddings from CxHxW numpy array
def obtain_embeds_img(img):
    model, preprocess = clip.load("ViT-B/32", device=device)
    #image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
    image = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features


# obtains frame-level embeddings from the given clip (list containing T CxHxW arrays)
def obtain_embeds(clip_list):
    clip_feats = []
    with Pool(processes=24) as pool:
        clip_feats = pool.starmap(obtain_embeds_img, zip(clip_list))
    return np.array(clip_feats)


def load_clip(path):
    frames = []
    cap = cv2.VideoCapture(path)
    ret = True
    while ret:
        ret, img = cap.read()  # read one frame from the 'capture' object; img is (H, W, C)
        if ret:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
    video = np.stack(frames, axis=0)  # dimensions (T, H, W, C)
    video = np.moveaxis(video, 3, 1)  # dimensions (T, C, H, W)
    return video

# loads the .mp4 files for the specified set/key and IDs
def load_clips(key, ids):
    path_ims = VID_PATHS[key]
    dict_vids = {}

    for id in ids:
        path = os.path.join(path_ims, id+".mp4")
        video = load_clip(path)
        dict_vids[id] = video
    video_list = [v for _, v in sorted(dict_vids.items())]
    print(f"len(video_list): {len(video_list)}", flush=True)
    return video_list


# returns the ID of those clips for which video is available
def get_vid_ids(key="train"):
    clips_ids = [x[:-4] for x in os.listdir(VID_PATHS[key]) if x.endswith(".mp4")]
    return clips_ids


def obtain_feats(key, ids):
    pass


def crop_frame(frame, middle, shape):
    '''
    Crops frame to given middle point and rectangle shape.
    :param frame: Input frame. Numpy array
    :param middle: Center point. [x, y]
    :param shape: [H', W']
    :return: cropped frame. Numpy array
    '''
    frame = np.array(frame)
    frame = np.pad(frame, ((shape[0], shape[0]), (shape[1], shape[1]), (0, 0)))

    # Adjust to padded image coords
    middle = [middle[0] + shape[0], middle[1] + shape[1]]

    # frame_ = cv2.circle(frame, (int(middle[0]), int(middle[1])),
    #                            radius=4, color=(0, 255, 0), thickness=-1)
    #
    x_0, y_0 = int(middle[0] - shape[0] / 2), int(middle[1] - shape[1] / 2)
    x_1, y_1 = int(middle[0] + shape[0] / 2), int(middle[1] + shape[1] / 2)
    crop = frame[y_0:y_1, x_0:x_1, :]

    # frame_ = cv2.circle(frame_, (x_0, y_0),
    #                     radius=4, color=(0, 255, 0), thickness=-1)
    # frame_ = cv2.circle(frame_, (x_1, y_1),
    #                     radius=4, color=(0, 255, 0), thickness=-1)

    return crop


# save a numpy array as a video
def save_as_mp4(vid, fps=25, filename="testing.avi"):
    T, _, H, W = vid.shape    
    writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'PIM1'), fps, (W, H), True)
    for i in range(T):
        x = np.moveaxis(vid[i,:,:,:], 0, -1)  # (C, H, W) -> (H, W, C)
        x_bgr = x[...,::-1].copy()  # from RGB to BGR
        writer.write(x_bgr)


# paints the specified points on the video.
# each point is represented T frames
def overlap_vid_points(vid, points):
    # (T, H, W, C)
    vid_overlap = vid.copy()
    print(vid.shape)
    for t in range(vid.shape[0]):
        p = points[t,:]
        for i in range(0, len(p), 2):
            vid_overlap[t,(int(p[i])-3):(int(p[i])+3), (int(p[i+1])-3):(int(p[i+1])+3),0] = 255
            vid_overlap[t,(int(p[i])-3):(int(p[i])+3), (int(p[i+1])-3):(int(p[i+1])+3),1:] = 0
    return vid_overlap


def get_hand_center(input_json):
    '''
    Returs the computed hand center given the hand keypoints. Implemented as
    average of MP joints points
    :param input_json:
    :return:
    '''
    # Get right hand keypoints
    right_hand_points = input_json["people"][0]["hand_right_keypoints_2d"]

    # format list shape from (N_point x 3,) to (N_points, 3)
    right_hand_points = [right_hand_points[3 * i:3 * i + 3] for i in
                         range(len(right_hand_points) // 3)]

    # Selecting only MP joints
    MP_JOINTS_INDEXES = [5, 9, 13, 17]
    mp_joints = [right_hand_points[i] for i in MP_JOINTS_INDEXES]

    mp_joints_coordinates = [[x[0], x[1]] for x in mp_joints]
    #plot_points(frame_large, mp_joints_coordinates)

    mp_joints_coordinates_numpy = np.array(mp_joints_coordinates)

    mp_joints_center = np.average(mp_joints_coordinates_numpy, axis=0)

    return mp_joints_center


def crop_video_main():
    import json
    input_video_path = "G3g0-BeFN3c_17-5-rgb_front.mp4"
    input_json_folder = "/home/alvaro/Documents/ML and DL/How2Sign/B2H-H2S/Green Screen RGB clips* (frontal view)/test_2D_keypoints/openpose_output/json/G3g0-BeFN3c_17-5-rgb_front"
    #output_file = "utterance_level/train/rgb_front/features/hand_openpose_video/CVmyQR31Dr4_5-3-rgb_front.mp4"
    output_file = "testing_G3g0-BeFN3c_17-5-rgb_front.mp4"
    utt_id = input_video_path.split("/")[-1].replace(".mp4", "")
    print(utt_id)
    cap = cv2.VideoCapture(input_video_path)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    n = 0

    writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'PIM1'), fps, (200, 200))
    while (cap.isOpened()):
        ret, frame_large = cap.read()

        if frame_large is None:
            break

        #frame_large = cv2.cvtColor(frame_large, cv2.COLOR_BGR2RGB)

        json_filename = utt_id + "_" + '{:012d}'.format(n) + "_keypoints.json"
        json_filename = input_json_folder + "/" + json_filename

        keypoints_json = json.load(open(json_filename))
        center_coords = get_hand_center(keypoints_json)
        crop = crop_frame(frame_large, center_coords, (200, 200))

        writer.write(crop)

        n += 1

    cap.release()
    cv2.destroyAllWindows()
    print("DONE crop_video_main")


if __name__=="__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--file_path', type=str, default="/mnt/gpid08/datasets/How2Sign/How2Sign/utterance_level/test/text/en/raw_text/test.text.id.en", help="path to the file where text dataset is located")
    # args = parser.parse_args()

    #load_clips("train", "some ID which must be filled in")
    
    # from PIL import Image
    # sacuanjoje = Image.open("flor_sacuanjoje.jpeg")
    # print(type(sacuanjoje), sacuanjoje.size)
    
    # sacuanjoje = np.array(sacuanjoje)
    # print(sacuanjoje.shape)
    # sacuanjoje = crop_frame(sacuanjoje, (200,200), (100, 100))
    # print(sacuanjoje.shape)
    # im = Image.fromarray(sacuanjoje)
    # im.save("sacuanjoje_original_cropped.jpeg")

    crop_video_main()
    # video = load_clip("G42xKICVj9U_4-10-rgb_front.mp4")
    # print(type(video), video.shape)
    # save_as_mp4(video, fps=25, filename="testing.mp4")
    #feats_scuanj = obtain_embeds_img()
    #print(type(feats_scuanj), feats_scuanj.shape)

    pass
