import numpy as np

from utility import z_normalize, difference_list


# Weight horizons based on given distances
# Lower distances give higher weight, and vice versa
def weight_horizons(horizons, distances):
    if len(horizons) == 1:
        return horizons[0]
    # If one f-shapelet has a distance of 0, we return its horizon (perfect match)
    if np.min(distances) == 0:
        return horizons[np.argmin(distances)]

    # Weights for all horizons, with inverse distances
    weights = 1 / np.array(distances)
    sum_weights = np.sum(weights)
    # Element-wise product of weights and horizons
    weighted_horizons = np.reshape(weights, (len(weights), 1)) * np.array(horizons)
    # Sum of dot product
    summed_weighted_horizons = np.sum(weighted_horizons, axis=0)
    # Divide by sum of weights to normalize
    summed_weighted_horizons /= sum_weights

    return summed_weighted_horizons


# Generates first order differenced f-shapelets for a given subsequence length l with a given horizon h and a stride length
#  optionally z normalizes the f-shapelets before differencing
def generate_all_f_shapelets(
    series, l, h=1, stride_length=1, differencing=True, z_normalization=False
):
    if l < 2:
        print("[WARN] Cannot generate f-shapelets of a length smaller than 2")
        return None
    elif l > len(series):
        print(
            f"[WARN] F-shapelet length {l} greater than length of the series {len(series)}"
        )
        return None
    shapelets_horizons = np.lib.stride_tricks.sliding_window_view(series, l + h)[
        ::stride_length, :
    ]
    shapelets = []
    horizons = []
    if z_normalization:
        shapelets, means, stds = z_normalize(shapelets_horizons[:, :-h])
        horizons = (shapelets_horizons[:, -h - 1 :] - means) / stds
        if differencing:
            shapelets = difference_list(shapelets)
            horizons = difference_list(horizons)
    else:
        # Difference shapelet and horizon
        if differencing:
            shapelets_horizons = difference_list(shapelets_horizons)
        shapelets = shapelets_horizons[:, :-h]
        horizons = shapelets_horizons[:, -h:]

    return np.array(shapelets), np.array(horizons)


# Generates first order differenced f-shapelets for a given subsequence length l with a given horizon h and a stride length
#  optionally z normalizes the f-shapelets before differencing
def generate_f_shapelets(
    series,
    l,
    h=1,
    stride_length=1,
    z_normalization=False,
    differencing_order=1,
):
    if l < differencing_order + 1:
        print(
            f"[WARN] Cannot generate f-shapelets of order {differencing_order} a length smaller than {differencing_order + 1}"
        )
        return None
    elif l > len(series):
        print(
            f"[WARN] F-shapelet length {l} greater than length of the series {len(series)}"
        )
        return None
    shapelets_horizons = np.lib.stride_tricks.sliding_window_view(series, l + h)[
        ::stride_length, :
    ]
    shapelets = []
    horizons = []
    if z_normalization:
        shapelets, means, stds = z_normalize(shapelets_horizons[:, :-h])
        horizons = (shapelets_horizons[:, -h - 1 :] - means) / stds
        for _ in range(differencing_order):
            shapelets = difference_list(shapelets)
            horizons = difference_list(horizons)
    else:
        # Difference shapelet and horizon
        for _ in range(differencing_order):
            shapelets_horizons = difference_list(shapelets_horizons)
        shapelets = shapelets_horizons[:, :-h]
        horizons = shapelets_horizons[:, -h:]

    return np.array(shapelets), np.array(horizons)
