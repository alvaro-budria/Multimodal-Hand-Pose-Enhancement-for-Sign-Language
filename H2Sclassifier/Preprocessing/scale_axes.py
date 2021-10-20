import numpy as np
from skeleton_parts import *

# Function scale_axes takes as inputs a np.array xyz_vec of shape [NUM_FRAMES,NUM_JOINTS, 3] (equivalent to one whole video)
# and returns the normalized coordinates in a np.array of the same shape

def scale_axes(xyz_vec):

  xy_vec = xyz_vec[:,:,0:2]
  z_vec = xyz_vec[:,:,2]

  torso = xy_vec[bodypart_to_keypoint['Neck']] - xy_vec[bodypart_to_keypoint['MidHip']]
  torso_len = np.sqrt(torso[:,0]**2 + torso[:,1]**2)
  norm_xy_vec = np.empty(xy_vec.shape)
  norm_z_vec = np.empty(z_vec.shape)

  for i in range(0,26):
    x_vec = np.divide(xy_vec[i,:,0], torso_len)
    y_vec = np.divide(xy_vec[i,:,1], torso_len)
    norm_z_vec[i,:] = np.divide(z_vec[i,:], torso_len)
    norm_xy_vec[i,:,0] = x_vec
    norm_xy_vec[i,:,1] = y_vec

  norm_z_vec = np.expand_dims(norm_z_vec, axis=2)

  norm_xyz_vec = np.concatenate((norm_xy_vec, norm_z_vec), axis = 2)
  return norm_xyz_vec

