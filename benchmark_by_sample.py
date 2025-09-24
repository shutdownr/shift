import os
import pickle 

import numpy as np

from basic_models import predict_shift, predict_nbeats, predict_dlinear
from benchmark_backhorizon_horizon import benchmark
from data_wrangling import read_dataset, flatten, scale, inverse_flatten_predictions, inverse_scale_predictions
from dataset_config import Config
from evaluation import evaluate_errors

results_path = "results/"

# Error metrics used for model evaluation and comparison
error_metrics = ["smape", "mase", "owa"]

# Benchmark algorithms
algorithms = {
    "SHIFT": predict_shift,
    "N-BEATS": predict_nbeats,
    "DLinear": predict_dlinear,
}
h = 5
bh = 10

# Random states for non-deterministic models
random_states = np.arange(3)

def benchmark_by_sample(algorithms, datasets, error_metrics, random_states):
    all_errors = {}

    for dataset_name in datasets:
        dataset = read_dataset(dataset_name, Config.train_sizes[dataset_name], Config.val_sizes[dataset_name], Config.stride_lengths[dataset_name], bh, h)
        dataset_flat, original_shapes = flatten(dataset)
        dataset_scaled, scalers = scale(dataset)
        print(f"Starting dataset {dataset_name}\n")
        algo_errors = {}
        for algo_name, algorithm in algorithms.items():
            if algo_name == "SHIFT":
                y_pred = []
                for i in range(len(dataset_scaled["ts_raw"])):
                    sample = {}
                    for k,v in dataset_scaled.items():
                        sample[k] = np.array([v[i]])
                    pred, _ = algorithm(sample)
                    y_pred.append(pred[0])
                y_pred = np.array(inverse_scale_predictions(np.array(y_pred), scalers))
            elif algo_name == "DLinear":
                y_pred = [[] for _ in range(len(random_states))]
                for i in range(len(dataset_scaled["ts_raw"])):
                    sample = {}
                    for k,v in dataset_scaled.items():
                        sample[k] = np.array(v[i])
                    pred, _ = algorithm(sample, Config.timesnet_frequency_map[dataset_name], random_states)
                    for index in range(len(pred)):
                        y_pred[index].extend(pred[index])
                y_pred = [inverse_flatten_predictions(pred, original_shapes) for pred in y_pred]
                y_pred = np.array([inverse_scale_predictions(np.array(pred), scalers) for pred in y_pred])
            elif algo_name == "N-BEATS":
                y_pred = [[] for _ in range(len(random_states))]
                for i in range(len(dataset_scaled["ts_raw"])):
                    sample = {}
                    for k,v in dataset_scaled.items():
                        sample[k] = np.array(v[i])
                    pred, _ = algorithm(sample, random_states)
                    for index in range(len(pred)):
                        y_pred[index].extend(pred[index])
                y_pred = [inverse_flatten_predictions(pred, original_shapes) for pred in y_pred]
                y_pred = np.array([inverse_scale_predictions(np.array(pred), scalers) for pred in y_pred])
            else:
                y_pred = []
                for i in range(len(dataset_scaled["ts_raw"])):
                    sample = {}
                    for k,v in dataset_scaled.items():
                        sample[k] = np.array([v[i]])
                    pred, _ = algorithm(sample)
                    y_pred.append(pred)
                y_pred = np.array(inverse_scale_predictions(np.array(y_pred), scalers))
            errors = evaluate_errors(dataset["X_test"], dataset["y_test"], y_pred, error_metrics, dataset=dataset_flat)
            algo_errors[algo_name] = errors
            print(errors)
            print(f"DONE {algo_name}")
        all_errors[dataset_name] = algo_errors
        print(f"DONE with dataset {dataset_name}\n")
    return all_errors

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
    errors_overall, _ = benchmark(h, bh, algorithms, Config.by_sample_dataset_names, error_metrics, random_states)
    update_results_file(errors_overall, f"{results_path}benchmark_by_sample_global.pkl")
    
    errors = benchmark_by_sample(algorithms, Config.by_sample_dataset_names, error_metrics, random_states)
    update_results_file(errors, f"{results_path}benchmark_by_sample.pkl")