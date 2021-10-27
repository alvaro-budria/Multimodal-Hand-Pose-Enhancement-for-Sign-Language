from multiprocessing import Pool
import numpy as np
from scipy.spatial.transform import Rotation as R


def array_to_list(input):
    if type(input) != type(list()):  # convert 3D array to list of 2D arrays
        input = list(input)
    return input


def np_mat_to_rot6d(np_mat):
    """ Get 6D rotation representation for rotation matrix.
        Implementation base on
            https://arxiv.org/abs/1812.07035
        [Inputs]
            flattened rotation matrix (last dimension is 9)
        [Returns]
            6D rotation representation (last dimension is 6)
    """
    shape = np_mat.shape
    if not ((shape[-1] == 3 and shape[-2] == 3) or (shape[-1] == 9)):
        raise AttributeError("The inputs in tf_matrix_to_rotation6d should be [...,9] or [...,3,3], \
            but found tensor with shape {}".format(shape[-1]))
    np_mat = np.reshape(np_mat, [-1, 3, 3])
    np_r6d = np.concatenate([np_mat[...,0], np_mat[...,1]], axis=-1)
    if len(shape) == 1:
        np_r6d = np.reshape(np_r6d, [6])
    return np_r6d


## utility function to convert from r6d space to axis angle
def _rot6d_to_aa(r6ds):
    res = np.zeros((r6ds.shape[0], 3))
    for i,row in enumerate(r6ds):
        np_r6d = np.expand_dims(row, axis=0)
        np_mat = np.reshape(np_rot6d_to_mat(np_r6d)[0], (3,3))
        np_mat = R.from_matrix(np_mat)
        aa = np_mat.as_rotvec()
        res[i,:] = aa
    return res


def clip_rot6d_to_aa(r6d_clip):
    aa_clip = np.empty((r6d_clip.shape[0], r6d_clip.shape[1]//2))
    for idx in range(0, r6d_clip.shape[1], 6):
        aa_clip[:,idx//2:idx//2+3] = _rot6d_to_aa(r6d_clip[:,idx:idx+6])
    return aa_clip


def rot6d_to_aa(r6d):
    r6d = array_to_list(r6d)
    aa = []
    with Pool(processes=24) as pool:
        aa = pool.starmap( clip_rot6d_to_aa, zip(r6d) )
    return aa


## utility function to convert from axis angle to r6d space
def _aa_to_rot6d(vecs):
    res = np.zeros((vecs.shape[0], 6))
    for i,row in enumerate(vecs):
        np_mat = R.from_rotvec(row)
        np_mat = np_mat.as_matrix()
        np_mat = np.expand_dims(np_mat, axis=0) #e.g. batch 1
        np_r6d = np_mat_to_rot6d(np_mat)[0]
        res[i,:] = np_r6d
    return res


# convert from axis angle to r6d space
def aa_to_rot6d(aa):
    aa = array_to_list(aa)
    r6d = []
    for clip in range(len(aa)):
        aa_clip = aa[clip]
        r6d_clip = np.empty((aa_clip.shape[0], aa_clip.shape[1]*2)) # from 3d to r6d
        for idx in range(0, aa_clip.shape[1], 3):
            r6d_clip[:,idx*2:idx*2+6] =  _aa_to_rot6d(aa_clip[:,idx:idx+3])
        r6d.append(r6d_clip)
    return r6d


# https://github.com/facebookresearch/body2hands/blob/0eba438b4343604548120bdb03c7e1cb2b08bcd6/utils/load_utils.py
## utility function to convert from r6d space to rotation matrix
def np_rot6d_to_mat(np_r6d):
    shape = np_r6d.shape
    np_r6d = np.reshape(np_r6d, [-1,6])
    x_raw = np_r6d[:,0:3]
    y_raw = np_r6d[:,3:6]

    x = x_raw / (np.linalg.norm(x_raw, ord=2, axis=-1) + 1e-6)
    z = np.cross(x, y_raw)
    z = z / (np.linalg.norm(z, ord=2, axis=-1) + 1e-6)
    y = np.cross(z, x)

    x = np.reshape(x, [-1,3,1])
    y = np.reshape(y, [-1,3,1])
    z = np.reshape(z, [-1,3,1])
    np_matrix = np.concatenate([x,y,z], axis=-1)
    if len(shape) == 1:
        np_matrix = np.reshape(np_matrix, [9])
    else:
        output_shape = shape[:-1] + (9,)
        np_matrix = np.reshape(np_matrix, output_shape)

    return np_matrix


# From a vector representing a rotation in axis-angle representation,
# retrieves the rotation angle and the rotation axis
def _retrieve_axis_angle(aa):
    th = np.linalg.norm(aa, axis=1)
    a = aa / th[:,np.newaxis]
    return a, th

def aa_to_xyz(aa, root, bone_len, structure):
    aa = array_to_list(aa)
    xyz = []
    for i in range(len(aa)):
        aa_clip = aa[i]
        xyz_clip = np.empty((aa_clip.shape[0], aa_clip.shape[1]+6), dtype="float32")  # add 6 values, corresponding to two keypoints defining the root bone
        xyz_clip[:,0:6] = root
        for iBone in range(1,len(structure)):
            id_p_J, id_p_E, _, id_p_B = structure[iBone]
            p_J, p_B = xyz_clip[:,id_p_J*3:id_p_J*3+3], xyz_clip[:,id_p_B*3:id_p_B*3+3]
            u = p_J - p_B
            u = u / np.linalg.norm(u, axis=1)[:, np.newaxis]
            a, th = _retrieve_axis_angle(aa_clip[:,(iBone-1)*3:(iBone-1)*3+3])
            # Rodrigues' rotation formula
            v = np.multiply(u, np.cos(th)[:, np.newaxis]) \
                + np.multiply(np.cross(a, u), np.sin(th)[:, np.newaxis]) \
                + np.multiply(np.multiply(a, np.einsum('ij,ij->i', a, u)[:, np.newaxis]), (1-np.cos(th))[:, np.newaxis])
            p_E = p_J + bone_len[iBone]*v
            xyz_clip[:,(iBone+1)*3:(iBone+1)*3+3] = p_E 
        xyz.append(xyz_clip)
    return xyz


def xyz_to_aa(xyz, structure, root_filename=None):
    xyz = array_to_list(xyz)
    aa = []
    for i in range(len(xyz)):
        xyz_clip = xyz[i]
        aa_clip = np.array([])
        for iBone in range(1,len(structure)):
            id_p_J, id_p_E, _, id_p_B = structure[iBone]
            u = xyz_clip[:,id_p_J*3:id_p_J*3+3] - xyz_clip[:,id_p_B*3:id_p_B*3+3]
            v = xyz_clip[:,id_p_E*3:id_p_E*3+3] - xyz_clip[:,id_p_J*3:id_p_J*3+3]
            th = np.arccos( np.einsum('ij,ij->i', u, v)/(np.linalg.norm(u, axis=1)*np.linalg.norm(v, axis=1) + 1e-6) )
            a = np.cross(u, v)
            a = a / np.linalg.norm(a, axis=1)[:,np.newaxis]  # rotation axis
            aa_clip = np.hstack(( aa_clip, np.multiply(a, th[:, np.newaxis]) )) if aa_clip.shape[0]!=0 else np.multiply(a, th[:, np.newaxis])
        aa.append(aa_clip)
    return aa
