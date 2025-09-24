import copy

import numpy as np

def z_normalize(data):
    means = np.mean(data, axis=-1, keepdims=True)
    stds = np.std(data, axis=-1, keepdims=True)
    # Fix for extremely low standard deviations that can cause values to explode when denormalizing
    stds[stds < 1e-5] = 1

    return (data - means) / stds, means, stds

def z_denormalize(sequence, mean, std):
    return (sequence * std) + mean

def difference_subsequence(subsequence):
    return [v1 - v0 for v0, v1 in zip(subsequence[:-1], subsequence[1:])]

# First order differences an n dimensional list/ndarray
def difference_list(l):
    if (type(l[0]) == list or type(l[0]) == np.ndarray) and (type(l[0][0]) == list or type(l[0][0]) == np.ndarray):
        return np.array([difference_list(l_) for l_ in l])
    if type(l[0]) == list or type(l[0]) == np.ndarray:
        return np.array([difference_subsequence(i) for i in l])
    return np.array(difference_subsequence(l))

# Trims given sequences to multiple subsequences of a given length l
#  where each subsequence has a horizon reaching beyond the end of the original sequences
#  subsequences are extracted with a stride length of l by default
#  subsequences are returned ordered from latest to earliest

# Example: all_ts = [[1,2,3,4,5,6,7]] ; l=2,h=4
# returns: [[6,7],[4,5]]
def trim_to_length_chained(all_ts, l, h):
    res = []
    for ts in all_ts:
        sub = []
        sub.append(ts[-l:])
        i = 1
        while i*l + l <= len(ts) and h > i*l:
            sub.append(ts[-(i+1)*l:-i*l])
            i += 1
        res.append(sub)
    return np.array(res).astype(float)

# Trims given sequences to a given length l
# Note: The implementation is list-based, as all_ts can be ragged depending on the dataset
def trim_to_length(all_ts, l):
    return np.array([ts[-l:] for ts in all_ts]).astype(float)

# Merges training and validation data of a given dataset (for algorithms that don't require validation data)
def merge_train_val(dataset):
    dataset_copy = copy.deepcopy(dataset)
    x_train_merged = []
    y_train_merged = []
    
    for i in range(len(dataset["X_val"])):
        if len(dataset["X_val"][i]) > 0:
            concat_x = np.concatenate([dataset["X_train"][i], dataset["X_val"][i]])
            concat_y = np.concatenate([dataset["y_train"][i], dataset["y_val"][i]])
        else:
            concat_x = dataset["X_train"][i]
            concat_y = dataset["y_train"][i]
        x_train_merged.append(concat_x)
        y_train_merged.append(concat_y)

    dataset_copy["X_train"] = np.array(x_train_merged, dtype=object)
    dataset_copy["y_train"] = np.array(y_train_merged, dtype=object)
    return dataset_copy