import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from dataset_config import Config
from data_wrangling import (
    read_dataset,
    flatten,
    scale,
    inverse_flatten_predictions,
    inverse_scale_predictions,
)
from basic_models import predict_knn, predict_shift
from evaluation import evaluate_errors

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Times New Roman"

# Benchmark algorithms
algorithms = {
    "SHIFT": predict_shift,
    "kNN": predict_knn,
}
datasets = Config.by_sample_dataset_names


def benchmark(horizon, backhorizon, algorithms, datasets, error_metrics):
    all_errors = {}
    all_inference_times = {}

    for dataset_name in datasets:
        dataset = read_dataset(
            dataset_name,
            Config.train_sizes[dataset_name],
            Config.val_sizes[dataset_name],
            Config.stride_lengths[dataset_name],
            backhorizon,
            horizon,
        )
        dataset_flat, _ = flatten(dataset)
        dataset_scaled, scalers = scale(dataset)
        dataset_scaled_flat, original_shapes = flatten(dataset_scaled)

        print(f"Starting dataset {dataset_name}\n")

        if len(dataset["y_train"]) == 0:
            print(
                f"All time series in dataset {dataset_name} are too short to generate train/test instances, skipping dataset"
            )
            continue

        algo_errors = {}
        algo_inference_times = {}
        for algo_name, algorithm in algorithms.items():
            print(f"Starting {algo_name}")
            if algo_name in ["SHIFT", "kNN_by_sample"]:
                y_pred, time = algorithm(dataset_scaled)
                y_pred = np.array(inverse_scale_predictions(y_pred, scalers))
            elif algo_name in ["kNN"]:
                y_pred, time = algorithm(dataset_scaled_flat)
                y_pred = np.array(
                    inverse_scale_predictions(
                        inverse_flatten_predictions(y_pred, original_shapes), scalers
                    )
                )

            errors = evaluate_errors(
                dataset["X_test"],
                dataset["y_test"],
                y_pred,
                error_metrics,
                dataset=dataset_flat,
            )
            algo_errors[algo_name] = errors
            algo_inference_times[algo_name] = time
            print(errors, time)
            print(f"DONE {algo_name}")
        all_errors[dataset_name] = algo_errors
        all_inference_times[dataset_name] = algo_inference_times
        print(f"DONE with dataset {dataset_name}\n")
    print(f"DONE with configuration - H: {horizon}; BH: {backhorizon}")
    return all_errors, all_inference_times


def update_results_file(results, filename):
    if os.path.exists(filename):
        with open(filename, "rb") as file:
            e = pickle.load(file)
            for dataset in results.keys():
                if dataset not in e:
                    e[dataset] = results[dataset]
                else:
                    for algo in results[dataset].keys():
                        e[dataset][algo] = results[dataset][algo]
        with open(filename, "wb") as file:
            pickle.dump(e, file)
    else:
        with open(filename, "wb") as file:
            pickle.dump(results, file)


results_path = f"./results/"
for h, bh in [(1, 10), (5, 10), (10, 10), (20, 10)]:
    print("Starting with configuration - H:", h, "; BH:", bh)

    all_errors, all_inference_times = benchmark(
        h, bh, algorithms, datasets, error_metrics=["owa"]
    )
    error_filename = f"{results_path}knn_errors_h_{h}_bh_{bh}.pkl"
    update_results_file(all_errors, error_filename)

    inference_time_filename = f"{results_path}knn_inference_times_h_{h}_bh_{bh}.pkl"
    update_results_file(all_inference_times, inference_time_filename)
