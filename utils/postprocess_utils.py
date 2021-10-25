import numpy as np


# removes those clips that contain at least one nan value
# def rmv_clips_nan(X, Y=None, T=None):
#     x, y, t = [], [], []
#     for sample in range(X.shape[0]):
#         if T is None and Y is None:
#             if not np.isnan(X[sample,:,:]).any():
#                 x.append(X[sample,:,:])
#         elif T is None:
#             if not (np.isnan(X[sample,:,:]).any() | np.isnan(Y[sample,:,:]).any()):
#                 x.append(X[sample,:,:])
#                 y.append(Y[sample,:,:])
#         else:
#             if not (np.isnan(X[sample,:,:]).any() | np.isnan(Y[sample,:,:]).any() | np.isnan(T[sample,:]).any()):
#                 x.append(X[sample,:,:])
#                 y.append(Y[sample,:,:])
#                 t.append(T[sample,:])
#     x = np.array(x)
#     if Y is not None:
#         y = np.array(y)
#     if T is not None:
#         t = np.array(t)
#     return x, y, t


# removes those clips that contain at least one nan value
def rmv_clips_nan(X, Y=None, T=None):
    idx_nan = np.argwhere(np.isnan(X).any(axis=(1,2))).squeeze().tolist()
    if type(idx_nan)==type(1):  # tolist() returned a single int
        idx_nan = [idx_nan]
    if Y is not None:
        if type(Y)==type([]):
            idx_nan_Y = np.argwhere(np.isnan(Y)).squeeze().tolist()
        else:
            idx_nan_Y = np.argwhere(np.isnan(Y).any(axis=(1,2))).squeeze().tolist()
        if type(idx_nan_Y)==type(1):  # tolist() returned a single int
            idx_nan_Y = [idx_nan_Y]
        idx_nan += idx_nan_Y
    if T is not None:
        idx_nan_T = np.argwhere(np.isnan(T).any(axis=(1))).squeeze().tolist()
        if type(idx_nan_T)==type(1):  # tolist() returned a single int
            idx_nan_T = [idx_nan_T]
        idx_nan += idx_nan_T
    idx_nan = sorted(set(idx_nan))  # remove duplicate indexes
    X = np.delete(X, idx_nan, axis=0)  # remove clips containing nan
    if Y is not None:
        Y = np.delete(Y, idx_nan, axis=0)
    if T is not None:
        T = np.delete(T, idx_nan, axis=0)
    return X, Y, T


# given a list of arrays (each corresponding to a clip) with varying lengths,
# makes all of them have equal (pair) length. The result is a single array
def make_equal_len(data, pipeline="arm2wh", method="cutting+reflect", maxpad=192):
    sizes = [arr.shape[0] for arr in data]
    if method=="0pad":
        maxpad = np.amax(sizes) if maxpad=="maxlen" else maxpad
        maxpad = maxpad + 1 if maxpad % 2 == 1 else maxpad
        res = [np.vstack((arr, np.zeros((maxpad-arr.shape[0],arr.shape[1]),int))) for arr in data]
        res = np.stack(res)

    elif method=="cutting":
        # get shortest length, cut the rest
        min_T = np.amin([arr.shape[0] for arr in data])
        min_T = min_T - 1 if sizes % 2 == 1 else min_T
        res = np.array([arr[:min_T,:] for arr in data])

    elif method=="cutting+0pad":  # 0pad shorter sequences, cut longer sequences
        res = np.array([arr[:maxpad,:] if arr.shape[0] >= maxpad else np.vstack( (arr, np.zeros((maxpad-arr.shape[0],arr.shape[1]),int)) ) for arr in data])

    elif method=="cutting+reflect":
        res = np.array([arr[:maxpad,:] if arr.shape[0] >= maxpad else np.pad(arr, ((0, maxpad-arr.shape[0]), (0,0)), "reflect") for arr in data])

    else: # method=="wrap" or method=="reflect"
        max_T = np.amax(sizes) + 1 if np.amax(sizes) % 2 == 1 else np.amax(sizes)
        max_T = max(max_T, maxpad)
        res = [np.pad(arr, ((0, max_T-arr.shape[0]), (0,0)), method) for arr in data]
        res = np.stack(res)
    return res
