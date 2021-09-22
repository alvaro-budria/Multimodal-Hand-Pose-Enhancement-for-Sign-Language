import os
import json
import argparse
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

DATA_PATHS = {
        "train": "/mnt/gpid08/datasets/How2Sign/How2Sign/utterance_level/train/rgb_front/features/openpose_output/json/",
        "val": "/mnt/gpid08/datasets/How2Sign/How2Sign/utterance_level/val/rgb_front/features/openpose_output/json/",
        "test": "/mnt/gpid08/datasets/How2Sign/How2Sign/utterance_level/test/rgb_front/features/openpose_output/json/"
}

device = "cuda" if torch.cuda.is_available() else "cpu"


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
# loaded files are returned in a list of numpy arrays
def load_clips(key, ids):
    path_ims = VID_PATHS[key]
    dict_vids = {}
    print(f"len(ids): {len(ids)}", flush=True)
    i = 1
    for id in ids:
        path = os.path.join(path_ims, id+".mp4")
        #path = id+".mp4"
        video = load_clip(path)
        dict_vids[id] = video
        i += 1

    video_list = [v for _, v in sorted(dict_vids.items())]
    print(f"len(video_list): {len(video_list)}", flush=True)
    return video_list


# returns the ID of those clips for which video is available
def get_vid_ids(key="train"):
    clips_ids = [x[:-4] for x in os.listdir(VID_PATHS[key]) if x.endswith(".mp4")]
    return clips_ids


def crop_clip(clip, clip_id, input_json_folder):
    '''
    Returs the cropped frames where the hands are located.
    :param clip: size (T, C, H, W)
    :param clip_id: used to retrieve the clip's hand keypoints
    :param input_json_folder: the directory where the
    :return cropped clip: (T, C, 150, 150, 2), where first position for last index is right hand, second is left hand
    '''
    cropped_clip = np.empty((clip.shape[0], clip.shape[1], 150, 150, 2))
    hand = {0: "right", 1: "left"}
    for i in range(clip.shape[0]):
        json_filename = clip_id + "_" + '{:012d}'.format(i) + "_keypoints.json"
        #input_json_folder = "/home/alvaro/Documents/ML and DL/How2Sign/B2H-H2S/Green Screen RGB clips* (frontal view)/test_2D_keypoints/openpose_output/json"
        #json_filename = os.path.join(input_json_folder, clip_id) + "/" + json_filename
        json_filename = os.path.join(input_json_folder, json_filename)
        keypoints_json = json.load(open(json_filename))
        for j in range(2):  # for each hand
            center_coords_j = get_hand_center(keypoints_json, hand=hand[j])
            crop_j = crop_frame(np.moveaxis(clip[i,:,:,:], 0, -1), center_coords_j, (150, 150))
            crop_j = np.moveaxis(crop_j, -1, 0)
            cropped_clip[i,:,:,:,j] = crop_j
    return cropped_clip


from PIL import Image
def preprocess_clip(img, preprocess):
    '''
    :param img: numpy array of dims TxCxHxW
    :return preproc_list: preprocessed data as torch tensor of dims TxCxHxW
    '''
    img_8uint = img.astype(np.uint8)
    images = []
    for i in range(img_8uint.shape[0]):
        pil_img = Image.fromarray(img_8uint[i,:,:,:], 'RGB')
        pil_img = preprocess(pil_img)
        images.append(pil_img)
    return torch.tensor(np.stack(images))

# obtains frame-level embeddings from TxCxHxW numpy array
def obtain_embeds_img(img, model, preprocess):
    img_tensor = preprocess_clip(img, preprocess)
    with torch.no_grad():
        image_features = model.encode_image(img_tensor)
    return image_features.cpu().detach().numpy()


# obtains frame-level embeddings from the given clip (list containing T CxHxW arrays)
def obtain_embeds(clip_list):
    clip_feats = []
    with Pool(processes=24) as pool:
        clip_feats = pool.starmap(obtain_embeds_img, zip(clip_list))
    return np.array(clip_feats)


def obtain_feats_crops(crops_list):
    '''
    Obtains features for the input hand crops.
    :param crops_list: list containing arrays of dims TxCxHxWx2
    :return feats_list: list containing the hand features for each clip
    '''
    model, preprocess = clip.load("ViT-B/32", device=device, jit=True)
    feats_list = []
    for c in crops_list:  # parallelize this?¿? beware of memory overflow!
        embeds_r = obtain_embeds_img(c[:,:,:,:,0], model, preprocess)
        embeds_l = obtain_embeds_img(c[:,:,:,:,1], model, preprocess)
        feats_hands = np.hstack((embeds_r, embeds_l))
        feats_list.append(feats_hands)
    return feats_list


def obtain_feats(key, ids):
    s_ids = sorted(ids)
    print(f"sorted s_ids", flush=True)
    clip_list = load_clips(key, s_ids)
    print(f"Clips loaded for {key}!", flush=True)
    clip_list = obtain_cropped_clips(clip_list, key, s_ids)
    clip_list = obtain_feats_crops(clip_list)
    return clip_list


# # returns a list containing a Tx1024 hand features array for each clip
# def obtain_feats(key, ids):    
#     s_ids = sorted(ids)
#     print(f"sorted s_ids", flush=True)
#     clip_list = load_clips(key, s_ids)
#     print(f"Clips loaded for {key}!", flush=True)
#     feats_list = []
#     for i, clip in enumerate(clip_list):
#         input_json_folder = os.path.join(DATA_PATHS[key], s_ids[i])
#         crop = crop_clip(clip, s_ids[i], input_json_folder)
#         print(f"crop {i} done", flush=True)
#         embeds_r = np.squeeze( obtain_embeds(list(crop[:,:,:,:,0])) )
#         print(f"obtained embeds right")
#         print(embeds_r.shape, flush=True)
#         embeds_l = np.squeeze( obtain_embeds(list(crop[:,:,:,:,1])) )
#         print(f"obtained embeds left")
#         print(embeds_r.shape, flush=True)
#         feats_hands = np.hstack((embeds_r, embeds_l))
#         print(feats_hands.shape, flush=True)
#         feats_list.append(feats_hands)
#     return feats_list


# from keras.applications.resnet50 import ResNet50
# def obtain_embeds_img(clip):
#     '''
#     Crops frame to given middle point and rectangle shape.
#     :param frame: Input frame. Dims TxCxHxW. Numpy array
#     :return: Resnet features. Numpy array
#     '''
#     base_model = ResNet50(weights='imagenet', pooling=max, input_shape=(3,150,150), include_top=False)
#     input = Input(shape=clip.shape[1:], name='image_input')
#     x = base_model(input)


# returns a list containing cropped clips of dims TxCx150x150x2
def obtain_cropped_clips(clip_list, key, s_ids):
    crops_list = []
    for i, clip in enumerate(clip_list):
        input_json_folder = os.path.join(DATA_PATHS[key], s_ids[i])
        crop = crop_clip(clip, s_ids[i], input_json_folder)  # parallelize this?¿?
        crops_list.append(crop)
    return crops_list


# save a numpy array as a video
def save_as_mp4(vid, fps=25, filename="testing.avi"):
    T, _, H, W = vid.shape    
    writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'PIM1'), fps, (W, H), True)
    for i in range(T):
        x = np.moveaxis(vid[i,:,:,:], 0, -1)  # (C, H, W) -> (H, W, C)
        x_bgr = x[...,::-1].copy()  # from RGB to BGR
        writer.write(np.uint8(x_bgr))


# paints the specified points on the video.
# each point is represented T frames
def overlap_vid_points(vid, points):
    # (T, H, W, C)
    vid_overlap = vid.copy()
    print(vid.shape, flush=True)
    for t in range(vid.shape[0]):
        p = points[t,:]
        for i in range(0, len(p), 2):
            vid_overlap[t,(int(p[i])-3):(int(p[i])+3), (int(p[i+1])-3):(int(p[i+1])+3),0] = 255
            vid_overlap[t,(int(p[i])-3):(int(p[i])+3), (int(p[i+1])-3):(int(p[i+1])+3),1:] = 0
    return vid_overlap


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


def get_hand_center(input_json, hand="right"):
    '''
    Returs the computed hand center given the hand keypoints. Implemented as
    average of MP joints points
    :param input_json:
    :return:
    '''

    # Get right hand keypoints
    hand_points = input_json["people"][0][f"hand_{hand}_keypoints_2d"]

    # format list shape from (N_point x 3,) to (N_points, 3)
    hand_points = [hand_points[3 * i:3 * i + 3] for i in
                   range(len(hand_points) // 3)]

    # Selecting only MP joints
    MP_JOINTS_INDEXES = [5, 9, 13, 17]
    mp_joints = [hand_points[i] for i in MP_JOINTS_INDEXES]
    mp_joints_coordinates = [[x[0], x[1]] for x in mp_joints]
    mp_joints_coordinates_numpy = np.array(mp_joints_coordinates)
    mp_joints_center = np.average(mp_joints_coordinates_numpy, axis=0)
    return mp_joints_center


def crop_video_main():
    import json
    input_video_path = "G42xKICVj9U_4-10-rgb_front.mp4"
    input_json_folder = "/home/alvaro/Documents/ML and DL/How2Sign/B2H-H2S/Green Screen RGB clips* (frontal view)/test_2D_keypoints/openpose_output/json/G42xKICVj9U_4-10-rgb_front"
    #output_file = "utterance_level/train/rgb_front/features/hand_openpose_video/CVmyQR31Dr4_5-3-rgb_front.mp4"
    output_file = "testing_G42xKICVj9U_4-10-rgb_front.mp4"
    utt_id = input_video_path.split("/")[-1].replace(".mp4", "")
    print(utt_id, flush=True)
    cap = cv2.VideoCapture(input_video_path)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    n = 0

    writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'PIM1'), fps, (150, 150))
    while (cap.isOpened()):
        ret, frame_large = cap.read()

        if frame_large is None:
            break

        #frame_large = cv2.cvtColor(frame_large, cv2.COLOR_BGR2RGB)

        json_filename = utt_id + "_" + '{:012d}'.format(n) + "_keypoints.json"
        json_filename = input_json_folder + "/" + json_filename

        keypoints_json = json.load(open(json_filename))
        center_coords = get_hand_center(keypoints_json)
        crop = crop_frame(frame_large, center_coords, (150, 150))

        writer.write(crop)

        n += 1

    cap.release()
    cv2.destroyAllWindows()
    print("DONE crop_video_main", flush=True)


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
    # sacuanjoje = crop_frame(sacuanjoje, (150,150), (100, 100))
    # print(sacuanjoje.shape)
    # im = Image.fromarray(sacuanjoje)
    # im.save("sacuanjoje_original_cropped.jpeg")

    #crop_video_main()

    # video = load_clip("G42xKICVj9U_4-10-rgb_front.mp4")
    # video = crop_clip(video, "G42xKICVj9U_4-10-rgb_front", "/home/alvaro/Documents/ML and DL/How2Sign/B2H-H2S/Green Screen RGB clips* (frontal view)/test_2D_keypoints/openpose_output/json/G42xKICVj9U_4-10-rgb_front")
    # print(type(video), video.shape)
    # save_as_mp4(video[:,:,:,:,0], fps=25, filename="testing_crop0.mp4")
    # save_as_mp4(video[:,:,:,:,1], fps=25, filename="testing_crop1.mp4")
    #feats_scuanj = obtain_embeds_img()
    #print(type(feats_scuanj), feats_scuanj.shape)

    obtain_feats("test", ["G42xKICVj9U_4-10-rgb_front", "G3g0-BeFN3c_17-5-rgb_front"])

    pass
