import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lastCheckpoint = ""

DATA_PATHS = {
        "train": "train/rgb_front/features/openpose_output/json",
        "val": "val/rgb_front/features/openpose_output/json",
        "test": "test/rgb_front/features/openpose_output/json"
}

FEATURE_MAP = {
    'arm2wh': ((6*6), 42*6),
    "arm_wh2wh": ( ((6+42)*6), (42*6) ), # predict hands, including wrists, given arms and hands
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
    "wh2wh":  ( (42*6), (42*6) )  # hand to hand
}

NECK = [0, 1]  # neck in Open Pose 25
WRIST = [[4, 7], [0, 21]]  # wrist in arms, wrist in hand
ARMS = [2, 3, 4, 5, 6, 7]  # arms in Open Pose 25
HANDS = list(range(21*2))  # hands in Open Pose

EPSILON = 1e-10

DATA_PATHS_r6d = {
        "train": "r6d_train.pkl",
        "val": "r6d_val.pkl",
        "test": "r6d_test.pkl"
        # "train": "video_data/r6d_train.pkl",
        # "val": "video_data/r6d_val.pkl",
        # "test": "video_data/r6d_test.pkl"
}

MODELS = {
        "v1": "regressor_fcn_bn_32",
        "b2h": "regressor_fcn_bn_32_b2h",
        "v2": "regressor_fcn_bn_32_v2",
        "v4": "regressor_fcn_bn_32_v4",
        "v4_deeper": "regressor_fcn_bn_32_v4_deeper"
}

import numpy as np
import torch.nn as nn
from robust_loss import adaptive
LOSSES = {"L1": nn.L1Loss(),
          "L2": nn.MSELoss(),
          "Huber1": nn.HuberLoss(delta=1.0),
          "RobustLoss": adaptive.AdaptiveLossFunction(num_dims=1, float_dtype=torch.float32, device="cuda:0")}
