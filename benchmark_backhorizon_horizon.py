import os
import pickle 

import numpy as np

from basic_models import predict_naive, predict_theta, predict_arima, predict_shift, predict_nbeats, predict_gru, predict_dlinear, predict_timesnet, predict_fedformer, predict_nonstationary_transformer
from data_wrangling import read_dataset, flatten, scale, inverse_flatten_predictions, inverse_scale_predictions
from dataset_config import Config
from evaluation import evaluate_errors

results_path = "results/"

H = [1,5,10,20]
BH = [5,10,20,40]

# Error metrics used for model evaluation and comparison
error_metrics = ["smape", "mase", "owa"]

# Benchmark algorithms
algorithms = {
    "Naive": predict_naive,
    "Theta": predict_theta,
    "ARIMA": predict_arima,
    "SHIFT": predict_shift,
    "N-BEATS": predict_nbeats,
    "GRU": predict_gru,
    "DLinear": predict_dlinear,
    "NonStationaryT": predict_nonstationary_transformer,
    "FEDformer": predict_fedformer,
    "TimesNet": predict_timesnet,
}

# Datasets for the experiment
dataset_names = Config.dataset_names

# Random states for non-deterministic models
random_states = np.arange(3)

def benchmark(horizon, backhorizon, algorithms, datasets, error_metrics, random_states):
    all_errors = {}
    all_inference_times = {}

    for dataset_name in datasets:
        dataset = read_dataset(dataset_name, Config.train_sizes[dataset_name], Config.val_sizes[dataset_name], Config.stride_lengths[dataset_name], backhorizon, horizon)
        dataset_flat, _ = flatten(dataset)
        dataset_scaled, scalers = scale(dataset)
        dataset_scaled_flat, original_shapes = flatten(dataset_scaled)

        print(f"Starting dataset {dataset_name}\n")
        
        if len(dataset["y_train"]) == 0:
            print(f"All time series in dataset {dataset_name} are too short to generate train/test instances, skipping dataset")
            continue

        algo_errors = {}
        algo_inference_times = {}
        for algo_name, algorithm in algorithms.items():
            print(f"Starting {algo_name}")
            if algo_name == "SHIFT":
                y_pred, time = algorithm(dataset_scaled)
                y_pred = np.array(inverse_scale_predictions(y_pred, scalers))
            elif algo_name in ["NonStationaryT", "FEDformer", "TimesNet"]:
                y_pred, time = algorithm(dataset_scaled_flat, Config.timesnet_frequency_map[dataset_name], Config.timesnet_d_ff[dataset_name], Config.timesnet_d_model[dataset_name], random_states)
                y_pred = [inverse_flatten_predictions(pred, original_shapes) for pred in y_pred]
                y_pred = np.array([inverse_scale_predictions(pred, scalers) for pred in y_pred])
            elif algo_name in ["DLinear"]:
                y_pred, time = algorithm(dataset_scaled_flat, Config.timesnet_frequency_map[dataset_name], random_states)
                y_pred = [inverse_flatten_predictions(pred, original_shapes) for pred in y_pred]
                y_pred = np.array([inverse_scale_predictions(pred, scalers) for pred in y_pred])
            elif algo_name in ["N-BEATS", "GRU"]:
                y_pred, time = algorithm(dataset_scaled_flat, random_states)
                y_pred = [inverse_flatten_predictions(pred, original_shapes) for pred in y_pred]
                y_pred = np.array([inverse_scale_predictions(pred, scalers) for pred in y_pred])
            else:
                y_pred, time = algorithm(dataset_scaled_flat)
                y_pred = inverse_flatten_predictions(y_pred, original_shapes)
                y_pred = inverse_scale_predictions(y_pred, scalers)

            errors = evaluate_errors(dataset["X_test"], dataset["y_test"], y_pred, error_metrics, dataset=dataset_flat)
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

if __name__ == "__main__":
    # Base benchmarks
    for i,h in enumerate(H):
        for j,bh in enumerate(BH):
            errors, inference_times = benchmark(h, bh, algorithms, dataset_names, error_metrics, random_states)

            error_filename = f"{results_path}errors_h_{h}_bh_{bh}.pkl"
            update_results_file(errors, error_filename)

            inference_time_filename = f"{results_path}inference_times_h_{h}_bh_{bh}.pkl"
            update_results_file(inference_times, inference_time_filename)