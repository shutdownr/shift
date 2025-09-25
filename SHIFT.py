import math

import numpy as np
from sklearn.neighbors import KDTree

from evaluation import smape
from f_shapelet_utils import weight_horizons, generate_all_f_shapelets
from utility import (
    z_normalize,
    z_denormalize,
    difference_list,
    trim_to_length_chained,
    trim_to_length,
)


class SHIFT:
    chain = True
    z_normalization = True
    # Search space for l for hyperparameter optimization
    L = np.array([3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20])
    # Search space for n for hyperparameter optimization
    N = np.arange(2, 9)
    DEFAULT_N = 4
    DEFAULT_L = 8

    # Outlier fallback setting:
    # k: number of stds from the max for an outlier (False -> No outlier fallback)
    k = False
    fallback = None

    distance_metric = "euclidean"

    _is_optimized = False
    # Boolean indicating whether the given horizon, backhorizon and shapelet length allow chaining
    _can_chain = None
    _can_chain_z = None
    # Optimized shapelet length l
    _optimal_l = None
    _optimal_l_z = None
    # Optimized number of similar shapelets n
    _optimal_n = None
    _optimal_n_z = None
    # Maximal distance measure (and its standard deviation) seen during optimization
    _optimal_max_distance = None
    _optimal_max_distance_z = None
    _optimal_std_distance = None
    _optimal_std_distance_z = None
    # Weights for normal / z-normalized / chained / chained+z-normalized forecasts
    _chain_z_weights = None

    _use_kd_tree = True
    leaf_size = 8

    def __init__(self):
        pass

    def fit_predict(self, timeseries, X_test, h, l=None, n=None):
        b = len(X_test[0])
        if self._is_optimized and h != self.h:
            print(
                f"[WARN] Model has been optimized for horizon {self.h}, while horizon {h} was passed"
            )
        if self._is_optimized:
            can_chain = self._can_chain
            can_chain_z = self._can_chain_z
            n = self._optimal_n
            n_z = self._optimal_n_z
            l = self._optimal_l
            l_z = self._optimal_l_z
            max_distance = self._optimal_max_distance
            max_distance_z = self._optimal_max_distance_z
            std_distance = self._optimal_std_distance
            std_distance_z = self._optimal_std_distance_z
            chain_z_weights = self._chain_z_weights
        else:
            n = self.DEFAULT_N if n == None else n
            max_n = len(timeseries) - b - h + 1
            if max_n < n:
                n = max_n
            n_z = n
            l = np.min([len(X_test[0]), self.DEFAULT_L]) if l == None else l
            l_z = l
            can_chain = np.min([b // l, math.ceil(h / l)]) > 1
            can_chain_z = can_chain
            max_distance = np.Inf
            max_distance_z = max_distance
            std_distance = np.Inf
            std_distance_z = std_distance
            chain_z_weights = np.ones(4)

        # normal shapelets
        shapelets, shapelet_horizons = generate_all_f_shapelets(
            timeseries, l, h, z_normalization=False
        )
        # Trim test instances (chained); Non-chained predictions take X_test_trimmed[:,0,:]
        X_test_trimmed = trim_to_length_chained(X_test, l, h)
        if self._use_kd_tree:
            tree = self._construct_shapelet_kdtree(shapelets)
            similar_horizons, distances = self._find_similar_shapelets(
                X_test_trimmed, tree, shapelet_horizons, n
            )
        else:
            similar_horizons, distances = self._find_similar_shapelets_brute_force(
                X_test_trimmed, shapelets, shapelet_horizons, n
            )
        self.distances = distances

        pred_s = self._train_predict_shapelets(
            X_test_trimmed[:, 0, :],
            h,
            similar_horizons[:, 0, :],
            distances[:, 0, :],
            l,
            self.k,
            max_distance,
            std_distance,
            chain=False,
            x_test_full=X_test,
        )
        predictions = [pred_s]
        weights = [chain_z_weights[0]]

        if self.z_normalization:
            X_test_trimmed_z = trim_to_length_chained(X_test, l_z, h)
            shapelets_z, shapelet_horizons_z = generate_all_f_shapelets(
                timeseries, l_z, h, z_normalization=True
            )
            X_test_normalized, means, stds = z_normalize(X_test_trimmed_z)
            if self._use_kd_tree:
                tree_z = self._construct_shapelet_kdtree(shapelets_z)
                similar_horizons_z, distances_z = self._find_similar_shapelets(
                    X_test_normalized, tree_z, shapelet_horizons_z, n_z
                )
            else:
                similar_horizons_z, distances_z = (
                    self._find_similar_shapelets_brute_force(
                        X_test_normalized, shapelets_z, shapelet_horizons_z, n_z
                    )
                )

            pred_z = self._train_predict_shapelets(
                X_test_normalized[:, 0, :],
                h,
                similar_horizons_z[:, 0, :],
                distances_z[:, 0, :],
                l_z,
                k=False,
                chain=False,
            )
            pred_z = np.array(
                [
                    z_denormalize(p, m, s)
                    for p, m, s in zip(pred_z, means[:, 0], stds[:, 0])
                ]
            )
            weights.append(chain_z_weights[1])
            predictions.append(pred_z)

        if self.chain and can_chain:
            pred_s_c = self._train_predict_shapelets(
                X_test_trimmed, h, similar_horizons, distances, l, k=False, chain=True
            )
            if not self.z_normalization:
                weights.append(chain_z_weights[1])
            else:
                weights.append(chain_z_weights[2])
            predictions.append(pred_s_c)

        if self.chain and can_chain_z and self.z_normalization:
            pred_z_c = self._train_predict_shapelets(
                X_test_normalized,
                h,
                similar_horizons_z,
                distances_z,
                l_z,
                k=False,
                chain=True,
            )
            pred_z_c = np.array(
                [
                    z_denormalize(p, m, s)
                    for p, m, s in zip(pred_z_c, means[:, 0], stds[:, 0])
                ]
            )
            weights.append(chain_z_weights[3])
            predictions.append(pred_z_c)

        return np.average(predictions, axis=0, weights=weights)

    def optimize_hyperparameters(self, timeseries, X_train, y_train):
        self.h = len(y_train[0])
        self.b = len(X_train[0])

        # Filter invalid l
        L = [l for l in self.L if l >= 2 and l <= self.b]
        # Filter invalid n (n has to be 1 <= n <= len(timeseries) - backhorizon - horizon + 1)
        max_n = len(timeseries) - self.b - self.h + 1
        N = [n for n in self.N if n >= 1 and n <= max_n]
        if N == [] or L == []:
            return
        if self.z_normalization:
            errors = np.full((len(L), len(N), 2), np.nan, dtype=float)
        else:
            errors = np.full((len(L), len(N), 1), np.nan, dtype=float)
        std_distances = np.full_like(errors, np.nan, dtype=float)
        max_distances = np.full_like(errors, np.nan, dtype=float)

        # Calculate error on training set for each combination of l and n
        max_n = np.max(self.N)
        for l_index, l in enumerate(L):
            # normal and z-normalized shapelets
            shapelets, shapelet_horizons = generate_all_f_shapelets(
                timeseries, l, self.h, z_normalization=False
            )
            if self._use_kd_tree:
                tree = self._construct_shapelet_kdtree(shapelets)
            if self.z_normalization:
                shapelets_z, shapelet_horizons_z = generate_all_f_shapelets(
                    timeseries, l, self.h, z_normalization=True
                )
                if self._use_kd_tree:
                    tree_z = self._construct_shapelet_kdtree(shapelets_z)

            # Trim test instances (chained); Non-chained predictions take X_test_trimmed[:,0,:]
            X_train_trimmed = trim_to_length(X_train, l)

            if self._use_kd_tree:
                similar_horizons, distances = self._find_similar_shapelets(
                    X_train_trimmed, tree, shapelet_horizons, max_n
                )
            else:
                similar_horizons, distances = self._find_similar_shapelets_brute_force(
                    X_train_trimmed, shapelets, shapelet_horizons, max_n
                )
            if self.z_normalization:
                X_train_normalized, means, stds = z_normalize(X_train_trimmed)
                # Similar horizons shape: (len(X_train), nr of chained shapelets, max_n, h)
                if self._use_kd_tree:
                    similar_horizons_z, distances_z = self._find_similar_shapelets(
                        X_train_normalized, tree_z, shapelet_horizons_z, max_n
                    )
                else:
                    similar_horizons_z, distances_z = (
                        self._find_similar_shapelets_brute_force(
                            X_train_normalized, shapelets_z, shapelet_horizons_z, max_n
                        )
                    )

            # Ignore one shapelet for hyperparameter optimization, unless N=[1]
            offset = 0 if len(N) == 1 and max_n == 1 else 1

            for n_index, n in enumerate(N):
                pred_s, dist_s = self._train_predict_shapelets(
                    X_train_trimmed,
                    self.h,
                    similar_horizons[:, offset : n + offset, :],
                    distances[:, offset : n + offset],
                    l,
                    k=False,
                    chain=False,
                    return_distance=True,
                )
                errors[l_index, n_index, 0] = smape(y_train, pred_s)
                std_distances[l_index, n_index, 0] = np.std(dist_s)
                max_distances[l_index, n_index, 0] = np.max(dist_s)

                if self.z_normalization:
                    pred_z, dist_z = self._train_predict_shapelets(
                        X_train_normalized,
                        self.h,
                        similar_horizons_z[:, offset : n + offset, :],
                        distances_z[:, offset : n + offset],
                        l,
                        k=False,
                        chain=False,
                        return_distance=True,
                    )
                    pred_z = np.array(
                        [z_denormalize(p, m, s) for p, m, s in zip(pred_z, means, stds)]
                    )
                    errors[l_index, n_index, 1] = smape(y_train, pred_z)
                    std_distances[l_index, n_index, 1] = np.std(dist_z)
                    max_distances[l_index, n_index, 1] = np.max(dist_z)

        # Best parameters for l and n have been estimated, check whether we can chain with these parameters
        optimal_index = np.nanargmin(errors[:, :, 0])
        l_index = optimal_index // errors.shape[1]
        n_index = optimal_index % errors.shape[1]
        self._optimal_l = self.L[l_index]
        self._optimal_n = self.N[n_index]
        self._optimal_max_distance = max_distances[l_index, n_index, 0]
        self._optimal_std_distance = std_distances[l_index, n_index, 0]
        self._can_chain = (
            np.min([self.b // self._optimal_l, math.ceil(self.h / self._optimal_l)]) > 1
        )

        if self.z_normalization:
            optimal_index_z = np.nanargmin(errors[:, :, 1])
            l_index_z = optimal_index_z // errors.shape[1]
            n_index_z = optimal_index_z % errors.shape[1]
            self._optimal_l_z = self.L[l_index_z]
            self._optimal_n_z = self.N[n_index_z]
            self._optimal_max_distance_z = max_distances[l_index_z, n_index_z, 0]
            self._optimal_std_distance_z = std_distances[l_index_z, n_index_z, 0]
            self._can_chain_z = (
                np.min([self.b // self._optimal_l, math.ceil(self.h / self._optimal_l)])
                > 1
            )

        min_errors = list(np.nanmin(np.nanmin(errors, axis=0), axis=0))
        if self._can_chain:
            X_train_trimmed = trim_to_length_chained(X_train, self._optimal_l, self.h)
            nr_chained = X_train_trimmed.shape[1]
            # normal shapelets
            (
                shapelets,
                shapelet_horizons,
            ) = generate_all_f_shapelets(
                timeseries, self._optimal_l, self.h, z_normalization=False
            )
            if self._use_kd_tree:
                tree = self._construct_shapelet_kdtree(shapelets)
                similar_horizons, distances = self._find_similar_shapelets(
                    X_train_trimmed,
                    tree,
                    shapelet_horizons,
                    self._optimal_n + nr_chained,
                )
            else:
                similar_horizons, distances = self._find_similar_shapelets_brute_force(
                    X_train_trimmed,
                    shapelets,
                    shapelet_horizons,
                    self._optimal_n + nr_chained,
                )

            pred_s_c = self._train_predict_shapelets(
                X_train_trimmed,
                self.h,
                similar_horizons[:, :, nr_chained : self._optimal_n + nr_chained, :],
                distances[:, :, nr_chained : self._optimal_n + nr_chained],
                self._optimal_l,
                k=False,
                chain=True,
            )
            min_errors.append(smape(y_train, pred_s_c))

        if self.z_normalization and self._can_chain_z:
            # Trim test instances (chained); Non-chained predictions take X_test_trimmed[:,0,:]
            X_train_trimmed = trim_to_length_chained(X_train, self._optimal_l_z, self.h)
            nr_chained = X_train_trimmed.shape[1]
            # z-normalized shapelets
            (
                shapelets_z,
                shapelet_horizons_z,
            ) = generate_all_f_shapelets(
                timeseries, self._optimal_l_z, self.h, z_normalization=True
            )
            X_train_normalized, means, stds = z_normalize(X_train_trimmed)

            if self._use_kd_tree:
                tree_z = self._construct_shapelet_kdtree(shapelets_z)
                similar_horizons_z, distances_z = self._find_similar_shapelets(
                    X_train_normalized,
                    tree_z,
                    shapelet_horizons_z,
                    self._optimal_n_z + nr_chained,
                )
            else:
                similar_horizons_z, distances_z = (
                    self._find_similar_shapelets_brute_force(
                        X_train_normalized,
                        shapelets_z,
                        shapelet_horizons_z,
                        self._optimal_n_z + nr_chained,
                    )
                )

            pred_z_c = self._train_predict_shapelets(
                X_train_normalized,
                self.h,
                similar_horizons_z[
                    :, :, nr_chained : self._optimal_n_z + nr_chained, :
                ],
                distances_z[:, :, nr_chained : self._optimal_n_z + nr_chained],
                self._optimal_l_z,
                k=False,
                chain=True,
            )
            pred_z_c = np.array(
                [
                    z_denormalize(p, m, s)
                    for p, m, s in zip(pred_z_c, means[:, 0], stds[:, 0])
                ]
            )
            min_errors.append(smape(y_train, pred_z_c))

        weights = np.zeros(len(min_errors))
        perfect_match = False
        error_sum = 0
        for i, val in enumerate(min_errors):
            if val == 0:
                weights = np.zeros(len(min_errors))
                weights[i] = 1
                perfect_match = True
                break
            else:
                weights[i] = val
                error_sum += val

        if not perfect_match:
            weights = error_sum / weights

        self._chain_z_weights = weights
        self._is_optimized = True

        return errors

    def _forecast_shapelets(
        self,
        X_test,
        similar_horizons,
        shapelet_distances,
        k,
        max_distance,
        std_distance,
        return_distance=False,
        x_test_full=None,
    ):
        if x_test_full is None:
            x_test_full = X_test
        predictions = []
        distances = []
        for x, best_horizons, shapelet_distances, x_full in zip(
            X_test, similar_horizons, shapelet_distances, x_test_full
        ):
            prediction = weight_horizons(best_horizons, shapelet_distances)
            if (
                type(k) != bool
                and np.min(shapelet_distances) > max_distance + k * std_distance
                and max_distance > 0
            ):
                # Outlier detected, use fallback!
                if self.fallback:
                    y_pred = self.fallback.predict(np.array([x_full]), verbose=0)
                    y_pred = y_pred.reshape(y_pred.shape[1])
                else:
                    # Naive fallback is a median of the test data + the weighted shapelet prediction, for optimal results use e.g. N-BEATS here
                    y_pred = np.median(x) + np.cumsum(prediction)
            else:
                y_pred = x[-1] + np.cumsum(prediction)
            predictions.append(y_pred)
            if return_distance:
                distances.append(np.mean(shapelet_distances))
        if return_distance:
            return np.array(predictions), np.array(distances)
        else:
            return np.array(predictions)

    # Weights chained predictions with nan
    def _weight_predictions_by_distance(self, predictions, distances):
        # Use distance based weights with a logarithmic decay by distance from the end of x_test
        if 0 in distances:
            weights = np.ones(predictions.shape[1])
        else:
            weights = 1 / distances
            weights = weights * np.log2(np.flip(np.arange(len(predictions)) + 2))

        means = []
        for i in range(predictions.shape[1]):
            non_nan_predictions = predictions[:, i]
            non_nan_predictions = non_nan_predictions[~np.isnan(non_nan_predictions)]
            w = weights[: len(non_nan_predictions)]
            means.append(np.average(non_nan_predictions, weights=w))
        return np.array(means)

    # Replaces the later values of chained predictions with nan
    def _replace_chained_predictions(self, predictions, h, l):
        chained_predictions = []
        for i, p in enumerate(predictions):
            pred_with_nan = np.full(h, np.nan)

            pred_with_nan[: h - i * l] = p[i * l :]
            chained_predictions.append(pred_with_nan)
        return np.array(chained_predictions)

    # Makes predictions for the test instances x_test based on similar f-shapelets' horizons and distances
    def _train_predict_shapelets(
        self,
        x_test,
        h,
        similar_horizons,
        distances,
        l=9,
        k=1,
        max_distance=np.Inf,
        std_distance=0,
        chain=True,
        return_distance=False,
        x_test_full=None,
    ):
        # Chain
        if chain:
            pred = []
            dists = []
            for x, horizons, dist_values in zip(x_test, similar_horizons, distances):
                prediction, dist = self._forecast_shapelets(
                    x,
                    horizons,
                    dist_values,
                    k,
                    max_distance,
                    std_distance,
                    return_distance=True,
                    x_test_full=x_test_full,
                )
                prediction = self._replace_chained_predictions(prediction, h, l)
                pred.append(self._weight_predictions_by_distance(prediction, dist))
                if return_distance:
                    dists.append(dist)
            if return_distance:
                return np.array(pred), np.array(dists)
            else:
                return np.array(pred)
        # Don't chain
        else:
            return self._forecast_shapelets(
                x_test,
                similar_horizons,
                distances,
                k,
                max_distance,
                std_distance,
                return_distance,
                x_test_full=x_test_full,
            )

    def _construct_shapelet_kdtree(self, shapelets):
        return KDTree(shapelets, leaf_size=self.leaf_size, metric=self.distance_metric)

    def _find_similar_shapelets(self, x, tree, horizons, n):
        x_differenced = difference_list(x)
        reshape = False
        if len(x_differenced.shape) > 2:
            reshape = True
            original_shape = x_differenced.shape
            x_differenced = x_differenced.reshape(
                original_shape[0] * original_shape[1], original_shape[2]
            )
        if len(tree.data) < n:
            n = len(tree.data)
        distances, similar_shapelets = tree.query(x_differenced, k=n)
        similar_horizons = horizons[similar_shapelets]

        if reshape:
            distances = distances.reshape(
                original_shape[0], original_shape[1], distances.shape[-1]
            )
            similar_horizons = similar_horizons.reshape(
                original_shape[0], original_shape[1], n, similar_horizons.shape[-1]
            )
        return similar_horizons, distances

    def _find_similar_shapelets_brute_force(self, x, shapelets, horizons, n):
        from wildboar.distance import argmin_distance

        x_differenced = difference_list(x)
        reshape = False
        if len(x_differenced.shape) > 2:
            reshape = True
            original_shape = x_differenced.shape
            x_differenced = x_differenced.reshape(
                original_shape[0] * original_shape[1], original_shape[2]
            )
        similar_shapelets, distances = argmin_distance(
            x_differenced,
            shapelets,
            k=n,
            metric=self.distance_metric,
            sorted=True,
            return_distance=True,
        )
        similar_horizons = horizons[similar_shapelets]

        if reshape:
            distances = distances.reshape(
                original_shape[0], original_shape[1], distances.shape[-1]
            )
            similar_horizons = similar_horizons.reshape(
                original_shape[0], original_shape[1], n, similar_horizons.shape[-1]
            )
        return similar_horizons, distances
