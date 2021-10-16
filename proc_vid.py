import os
import json
import argparse
from multiprocessing import Pool
from re import I

from PIL import Image
import cv2
import clip  # to obtain CLIP embeddings

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

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
    :return cropped clip: (T, C, 120, 120, 2), where first position for last index is right hand, second is left hand
    '''
    cropped_clip = np.empty((clip.shape[0], clip.shape[1], 120, 120, 2))
    hand = {0: "right", 1: "left"}
    for i in range(clip.shape[0]):
        json_filename = clip_id + "_" + '{:012d}'.format(i) + "_keypoints.json"
        #input_json_folder = "/home/alvaro/Documents/ML and DL/How2Sign/B2H-H2S/Green Screen RGB clips* (frontal view)/test_2D_keypoints/openpose_output/json"
        #json_filename = os.path.join(input_json_folder, clip_id) + "/" + json_filename
        json_filename = os.path.join(input_json_folder, json_filename)
        try:
            keypoints_json = json.load(open(json_filename))
        except:  # could not load keypoints from json file
            keypoints_json = None
        for j in range(2):  # for each hand
            center_coords_j = get_hand_center(keypoints_json, hand=hand[j])
            crop_j = crop_frame(np.moveaxis(clip[i,:,:,:], 0, -1), center_coords_j, (120, 120))
            crop_j = np.moveaxis(crop_j, -1, 0)
            cropped_clip[i,:,:,:,j] = crop_j
        return cropped_clip.astype(np.uint8)


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
def obtain_embeds_img_CLIP(img, model, preprocess):
    img_tensor = preprocess_clip(img, preprocess)
    with torch.no_grad():
        image_features = model.encode_image(img_tensor)
    return image_features.cpu().detach().numpy()

def _obtain_feats_crops_CLIP(c):
    model, preprocess = clip.load("ViT-B/32", device=device, jit=True)
    embeds_r = obtain_embeds_img_CLIP(c[:,:,:,:,0], model, preprocess)
    embeds_l = obtain_embeds_img_CLIP(c[:,:,:,:,1], model, preprocess)
    feats_hands = np.hstack((embeds_r, embeds_l))
    return feats_hands

def obtain_feats_crops_CLIP(crops_list):
    '''
    Obtains features for the input hand crops.
    :param crops_list: list containing arrays of dims TxCxHxWx2
    :return feats_list: list containing the hand features for each clip
    '''
    feats_list = []
    # model_list = [model for _ in range(len(crops_list))]
    # preproc_list = [preprocess for _ in range(len(crops_list))]
    with Pool(processes=24) as pool:
        feats_list = pool.starmap(_obtain_feats_crops_CLIP, zip(crops_list))
    return feats_list    
    # Tsize_list = [crop.shape[0] for crop in crops_list]
    # print(f"Tsize_list: {Tsize_list}")
    # crops_list = _obtain_feats_crops( np.concatenate(crops_list, axis=0) )
    # print("after _obtain_feats_crops")
    # crops_list = np.split(crops_list, Tsize_list, axis=0)
    # print("after split")
    # return crops_list


def extract_feats_ResNet(tensor, model, batch_size=192):
    out_feats = torch.empty((tensor.shape[0], 1000)).cpu().numpy()
    for batch in range(0, tensor.shape[0], batch_size):
        model_out = model(tensor[batch:batch+batch_size,:,:,:])
        # print(f"model_out.shape {model_out.shape}", flush=True)
        out_feats[batch:batch+batch_size,:] = model(tensor[batch:batch+batch_size,:,:,:]).detach().cpu().numpy()
    return out_feats

# clip is a TxCxHxWx2
# output is a Tx2000 tensor (1000 for each hand)
def _obtain_feats_crops_ResNet(clip, model, transf):
    # print(f"clip.shape: {clip.shape}", flush=True)

    start = time.time()
    t_clip_r = transf(torch.from_numpy(clip[:,:,:,:,0]).to(dtype=torch.float))
    # print(f"transf r", flush=True)
    t_clip_l = transf(torch.from_numpy(clip[:,:,:,:,1]).to(dtype=torch.float))
    # print(f"transf l", flush=True)
    print(f"Time to transform inputs for ResNet: {time.time() - start}", flush=True)

    start = time.time()
    embed_r = extract_feats_ResNet(t_clip_r.to(device), model, batch_size=1)
    # embed_r = model(t_clip_r.to(device))
    # print(f"feats r", flush=True)
    embed_l = extract_feats_ResNet(t_clip_l.to(device), model, batch_size=1)
    # embed_l = model(t_clip_l.to(device))
    # print(f"feats l", flush=True)
    print(f"Time to extract feats from ResNet: {time.time() - start}", flush=True)

    feats_hands = np.hstack((embed_r, embed_l))
    return feats_hands

import time
def obtain_feats_crops_ResNet(crops_list, data_dir):
    start = time.time()
    model_ft = models.resnet50(pretrained=False)
    model_ft.load_state_dict(torch.load('./models/resnet50-0676ba61.pth'))
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_ft = nn.DataParallel(model_ft)
    model_ft.eval()
    model_ft.cuda()
    print("Sent model to CUDA", flush=True)
    #model_ft = torch.nn.Sequential(*list(model_ft.children())[:-1])  # keep feature extractor

    # mean_std = np.load(f"{data_dir}/mean_std.npy")
    # normalize = transforms.Normalize(mean=mean_std[0],
    #                                  std=mean_std[1])  # H2S mean and std
    normalize = transforms.Normalize(mean=[123.68, 116.779, 103.939 ],
                                     std= [58.393, 57.12,   57.375])  # ImageNet mean and std

    feats_list = []
    for crop in crops_list:
        print(f"crop.shape {crop.shape}", flush=True)
        feats = _obtain_feats_crops_ResNet(crop, model_ft, normalize)
        # print(f"feats.shape {feats.shape}", flush=True)
        feats_list.append(feats)
    print(f"Time to extract vid feats: {time.time() - start}", flush=True)
    return feats_list


def obtain_crops(key, ids):
    s_ids = sorted(ids)
    print(f"sorted s_ids", flush=True)

    start = time.time()
    clip_list = load_clips(key, s_ids)
    print(time.time() - start, flush=True)
    print(f"Clips loaded for {key}!", flush=True)

    start = time.time()
    clip_list = obtain_cropped_clips(clip_list, key, s_ids)
    print(time.time() - start, flush=True)
    print(f"Obtained cropped clips!", flush=True)
    return clip_list


def obtain_feats(key, ids):
    s_ids = sorted(ids)
    print(f"sorted s_ids", flush=True)

    start = time.time()
    clip_list = load_clips(key, s_ids)
    print(time.time() - start, flush=True)
    print(f"Clips loaded for {key}!", flush=True)

    start = time.time()
    clip_list = obtain_cropped_clips(clip_list, key, s_ids)
    print(time.time() - start, flush=True)
    print(f"Obtained cropped clips!", flush=True)

    start = time.time()
    clip_list = obtain_feats_crops(clip_list)
    print(time.time() - start, flush=True)
    print(f"Obtained features from crops!", flush=True)
    return clip_list


# returns a list containing cropped clips of dims TxCx100x100x2
def obtain_cropped_clips(clip_list, key, s_ids):
    crops_list = []
    # for i, clip in enumerate(clip_list):
    #     input_json_folder = os.path.join(DATA_PATHS[key], s_ids[i])
    #     crop = crop_clip(clip, s_ids[i], input_json_folder)  # parallelize this?¿?
    #     crops_list.append(crop)
    input_json_folder_list = [os.path.join(DATA_PATHS[key], s_ids[i]) for i in range(len(clip_list))]
    print(len(clip_list), len(s_ids), len(input_json_folder_list), flush=True)
    with Pool(processes=24) as pool:
        crops_list = pool.starmap( crop_clip, zip(clip_list, s_ids, input_json_folder_list) )
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

    # if the crop falls (partially) outside of the image frame, pad the crop to the desired size
    crop = np.pad( crop, ((0, max(0, shape[0]-crop.shape[0])), (0, max(0, shape[1]-crop.shape[1])), (0,0)) )
    return crop[:shape[0], :shape[1], :]


def get_hand_center(input_json, hand="right"):
    '''
    Returs the computed hand center given the hand keypoints. Implemented as
    average of MP joints points
    :param input_json:
    :return:
    '''

    if input_json is None:
        return np.array([700, 700])

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

    writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'PIM1'), fps, (100, 100))
    while (cap.isOpened()):
        ret, frame_large = cap.read()

        if frame_large is None:
            break

        #frame_large = cv2.cvtColor(frame_large, cv2.COLOR_BGR2RGB)

        json_filename = utt_id + "_" + '{:012d}'.format(n) + "_keypoints.json"
        json_filename = input_json_folder + "/" + json_filename

        keypoints_json = json.load(open(json_filename))
        center_coords = get_hand_center(keypoints_json)
        crop = crop_frame(frame_large, center_coords, (100, 100))

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
