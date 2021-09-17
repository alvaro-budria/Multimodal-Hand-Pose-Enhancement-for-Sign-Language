import os
import argparse
import pickle
import glob

import numpy as np
import torch

VID_PATHS = {
    "train": "/mnt/gpid08/datasets/How2Sign/How2Sign/utterance_level/train/rgb_front/raw_videos/",
    "val": "/mnt/gpid08/datasets/How2Sign/How2Sign/utterance_level/val/rgb_front/raw_videos/",
    "test": "/mnt/gpid08/datasets/How2Sign/How2Sign/utterance_level/test/rgb_front/raw_videos/"
}


device = "cuda" if torch.cuda.is_available() else "cpu"


# obtains frame-level embeddings from the given clip (TxE numpy array)
def obtain_embeds(clip):
    pass


# loads the .mp4 files for the specified set/key and IDs
def load_clips(key, ids):
    path_ims = VID_PATHS[key]
    dict_vids = {}

    import cv2
    for id in ids:
        frames = []
        path = os.path.join(path_ims, id+".mp4")
        cap = cv2.VideoCapture(path)
        ret = True
        while ret:
            ret, img = cap.read()  # read one frame from the 'capture' object; img is (H, W, C)
            if ret:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(img)
        video = np.stack(frames, axis=0)  # dimensions (T, H, W, C)
        video = np.moveaxis(video, 3, 1)  # dimensions (T, C, H, W)
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


# returns the cropped section of the image that most likely contains hands
def crop_image(im, H=32, W=32):
    pass

if __name__=="__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--file_path', type=str, default="/mnt/gpid08/datasets/How2Sign/How2Sign/utterance_level/test/text/en/raw_text/test.text.id.en", help="path to the file where text dataset is located")
    # args = parser.parse_args()
    pass
