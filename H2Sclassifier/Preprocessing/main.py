import numpy as np
import torch
import pickle
from skeleton_parts import *
from plot_3D_skeleton import plot_3D_skeleton
from rotate_skeleton import rotate_skeleton
from scale_axes import scale_axes



# Load the np arrays of dimension [NUM_VIDEOS, NUM_FRAMES, 2*NUM_JOINTS)
body_data = np.load('../Data/body_data.npy')
body_ground = np.load('../Data/body_ground.npy')

# Print the dataset basic parameters
print('Number of recorded videos: {}'.format(body_data.shape[0]))
print('Number of frames per video: {}'.format(body_data.shape[1]))
print('2*Number of keypoints: {}'.format(body_data.shape[2]))

# Define basic parameters
NUM_VIDEOS = body_data.shape[0]
NUM_FRAMES = body_data.shape[1]
NUM_JOINTS = 26
INPUT_SIZE = 2*NUM_JOINTS
HIDDEN_SIZE = 1024
NUM_LAYERS = 1
BATCH_SIZE = 256
SEQ_LEN = 50
BIAS = True

# Obtain a list xyz_data of len = NUM_VIDEOS of np arrays of shape
# [NUM_FRAMES, NUM_JOINTS, 3] (removed nan padding)
xyz_data = []
# For every video in the dataset
for vid_num in range(NUM_VIDEOS):
  print('Processing video number {}'.format(vid_num))
  # From the data extract the xy and z vectors
  xy_vec = body_data[vid_num,:,:]
  z_vec = body_ground[vid_num, :,:]

  # Obtain the x and y coordinate vectors from xy_vec
  x_vec = xy_vec[:,::2]
  y_vec = xy_vec[:,1::2]

  # Construct the vid_xyz vector of shape [num_joints, num_frames, 3]
  xyz_vec = np.transpose(np.asanyarray([x_vec, y_vec, z_vec]))
   
  # Rotate all the frames in the video so the skeleton is facing forwards
  rot_xyz_vec = np.empty(xyz_vec.shape)
  for i in range(0, xyz_vec.shape[1]):
    rot_xyz_vec[:,i,:] = rotate_skeleton(xyz_vec[:,i,:])

  # Scale axes so torso length is always 1.0
  norm_xyz_vec = scale_axes(rot_xyz_vec)

  # Remove nan padding
  padding = ~np.isnan(norm_xyz_vec[0,:,0])
  norm_xyz_vec = norm_xyz_vec[:,padding,:]

  # Add to the list
  xyz_data.append(torch.transpose(torch.tensor(norm_xyz_vec), 0, 1))
  print('Added video has shape: {}'.format(xyz_data[vid_num].shape))
  
# Cat the sequence of tensors along dim=0
# xyz_data has shape [TOTAL_NUM_FRAMES, NUM_JOINTS, 3]
xyz_data = torch.cat(xyz_data, dim=0)

#Save the data in a python pickle 
xyz_npdata = np.asanyarray(xyz_data)
np.save('../Data/normalized_data.npy', xyz_npdata)
