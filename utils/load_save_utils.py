import pickle
import os
import numpy as np

from postprocess_utils import *
from constants import *


def save_binary(obj, filename, append=False):
    if filename[-4:] != ".pkl":
        print("Adding .pkl extension as it was not found.", flush=True)
        filename = filename + ".pkl"

    if os.path.exists(filename) and append:
        print(f"Found file with name {filename}. Appending results to this file.", flush=True)
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


def load_windows(data_path, pipeline, require_text=False, text_path=None, require_image=False, image_path=None,
                 require_audio=False, hand3d_image=False, use_lazy=False, test_smpl=False, temporal=False):
    feats = pipeline.split('2')
    p0_size, p1_size = FEATURE_MAP[pipeline]
    if os.path.exists(data_path):
        print('using super quick load', data_path, flush=True)
        data = load_binary(data_path)
        data = make_equal_len(data, method="cutting+reflect")
        if pipeline=="arm2wh" or pipeline[:13]=="arm_wh2finger":
            p0_windows = data[:,:,:p0_size]
            p1_windows = data[:,:,p0_size:p0_size+p1_size]
        if require_text and not require_image:
            text_windows = load_binary(text_path)
            p0_windows = (p0_windows, text_windows)
        elif require_image and not require_text:
            image_windows = load_binary(image_path)
            image_windows = make_equal_len(image_windows, method="cutting+reflect")
            p0_windows = (p0_windows, image_windows)
        return p0_windows, p1_windows