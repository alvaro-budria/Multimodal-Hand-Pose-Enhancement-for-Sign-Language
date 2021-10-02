import numpy as np


# removes those clips that contain at least one nan value
def rmv_clips_nan(X, Y, T=None):
    x = []
    y = []
    t = []
    for sample in range(X.shape[0]):
        if T is None:
            if not (np.isnan(X[sample,:,:]).any() | np.isnan(Y[sample,:,:]).any()):
                x.append(X[sample,:,:])
                y.append(Y[sample,:,:])
        else:
            if not (np.isnan(X[sample,:,:]).any() | np.isnan(Y[sample,:,:]).any() | np.isnan(T[sample,:]).any()):
                x.append(X[sample,:,:])
                y.append(Y[sample,:,:])
                t.append(T[sample,:])
    x = np.array(x)
    y = np.array(y)
    if T is not None:
        t = np.array(t)
    return x, y, t


# given a list of arrays (each corresponding to a clip) with varying lengths,
# makes all of them have equal (pair) length. The result is a single array
def make_equal_len(data, pipeline="arm2wh", method="reflect", maxpad=192):
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