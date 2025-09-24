import os
import pickle 

import numpy as np

from basic_models import predict_shift_ablation, train_nbeats
from data_wrangling import read_dataset, flatten, scale, inverse_flatten_predictions, inverse_scale_predictions
from dataset_config import Config
from evaluation import evaluate_errors

results_path = "results/"

# Error metrics used for model evaluation and comparison
error_metrics = ["smape","mase", "owa"]

# SHIFT algorithm for benchmark and ablation study
shift_algorithm = {
    "SHIFT": predict_shift_ablation
}

bh = 40
h = 20

# Random states for non-deterministic models
random_states = np.arange(3)

# Benchmark function running a benchmark for all passed algorithms and evaluation them by given error metrics
def benchmark(algorithms, error_metrics, random_states, k=1, chain=True, z_normalization=True, fallback=None):
    all_errors = {}
    
    for dataset_name in Config.dataset_names:
        dataset = read_dataset(dataset_name, Config.train_sizes[dataset_name], Config.val_sizes[dataset_name], Config.stride_lengths[dataset_name], bh, h)
        dataset_flat, _ = flatten(dataset)
        dataset_scaled, scalers = scale(dataset)
        dataset_scaled_flat, original_shapes = flatten(dataset_scaled)

        print(f"Starting dataset {dataset_name}\n")
        if len(dataset["y_train"]) == 0:
            print(f"All time series in dataset {dataset_name} are too short to generate train/test instances, skipping dataset")
            continue

        algo_errors = {}
        for algo_name, algorithm in algorithms.items():
            if algo_name == "SHIFT":
                if fallback:
                    fallback_model = train_nbeats(dataset_scaled_flat["X_train"], dataset_scaled_flat["y_train"], dataset_scaled_flat["X_val"], dataset_scaled_flat["y_val"])
                else:
                    fallback_model = None
                # SHIFT works with a non-flattened dataset since inference is done on a by-sample basis
                y_pred, _ = algorithm(dataset_scaled, k=k, chain=chain, z_normalization=z_normalization, fallback=fallback_model)
                y_pred = np.array(inverse_scale_predictions(y_pred, scalers))
            elif algo_name == "TimesNet":
                y_pred, _ = algorithm(dataset_scaled_flat, Config.timesnet_frequency_map[dataset_name], Config.timesnet_d_ff[dataset_name], Config.timesnet_d_model[dataset_name], random_states)
                y_pred = [inverse_flatten_predictions(pred, original_shapes) for pred in y_pred]
                y_pred = np.array([inverse_scale_predictions(pred, scalers) for pred in y_pred])
            elif algo_name == "DLinear":
                y_pred, _ = algorithm(dataset_scaled_flat, Config.timesnet_frequency_map[dataset_name], random_states)
                y_pred = [inverse_flatten_predictions(pred, original_shapes) for pred in y_pred]
                y_pred = np.array([inverse_scale_predictions(pred, scalers) for pred in y_pred])
            elif algo_name in ["N-BEATS", "GRU"]: 
                y_pred, _ = algorithm(dataset_scaled_flat, random_states)
                y_pred = [inverse_flatten_predictions(pred, original_shapes) for pred in y_pred]
                y_pred = np.array([inverse_scale_predictions(pred, scalers) for pred in y_pred])
            else:
                y_pred, _ = algorithm(dataset_scaled_flat)
                y_pred = inverse_flatten_predictions(y_pred, original_shapes)
                y_pred = inverse_scale_predictions(y_pred, scalers)

            errors = evaluate_errors(dataset["X_test"], dataset["y_test"], y_pred, error_metrics, dataset=dataset_flat)
            algo_errors[algo_name] = errors
            print(errors)
        all_errors[dataset_name] = algo_errors
        print(f"Done with dataset {dataset_name}\n")
    
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

#=======================================================================
#======================Benchmark model performance======================
#=======================================================================

print("Starting benchmark")

benchmark_errors = benchmark(shift_algorithm, error_metrics, random_states, k=False, z_normalization=True, chain=True)
update_results_file(benchmark_errors, f"{results_path}benchmark_base.pkl")

print("Benchmark done")

#=======================================================================
#===================Ablation study: Outlier fallback====================
#=======================================================================

print("Starting ablation study: Outlier fallback")

print("Using fallback: Naive")

# Outlier fallback with k=0, k=1, and k=2, and without outlier fallback
shift_errors = {}
for k in [0,1,2, False]:
    print(f"k={k}")
    errors = benchmark(shift_algorithm, error_metrics, random_states, k=k, z_normalization=False, chain=False, fallback=None)
    # Flatten the error dictionary since we only evaluate one algorithm
    errors_flat = {}
    for key, val in errors.items():
        errors_flat[key] = val["SHIFT"]
    shift_errors[f"SHIFT_k={k}"] = errors_flat
    
update_results_file(shift_errors, f"{results_path}ablation_k_naive.pkl")

print("Using fallback: N-BEATS")

# Outlier fallback with k=0, k=1, and k=2, and without outlier fallback
shift_errors = {}
for k in [0,1,2, False]:
    print(f"k={k}")
    errors = benchmark(shift_algorithm, error_metrics, random_states, k=k, z_normalization=False, chain=False, fallback=True)
    # Flatten the error dictionary since we only evaluate one algorithm
    errors_flat = {}
    for key, val in errors.items():
        errors_flat[key] = val["SHIFT"]
    shift_errors[f"SHIFT_k={k}"] = errors_flat
    
update_results_file(shift_errors, f"{results_path}ablation_k_nbeats.pkl")

print("Done ablation study: Outlier fallback")

#=======================================================================
#===================Ablation study: f-shapelet chaining===================
#=======================================================================

print("Starting ablation study: f-shapelet chaining")

shift_errors = {}
for chaining in [True, False]:
    print(f"chaining={chaining}")
    errors = benchmark(shift_algorithm, error_metrics, random_states, k=False, z_normalization=False, chain=chaining, fallback=None)
    # Flatten the error dictionary since we only evaluate one algorithm
    errors_flat = {}
    for key, val in errors.items():
        errors_flat[key] = val["SHIFT"]
    shift_errors[f"SHIFT_chaining={chaining}"] = errors_flat
    
update_results_file(shift_errors, f"{results_path}ablation_chaining.pkl")

print("Done ablation study: f-shapelet chaining")

#=======================================================================
#====================Ablation study: Z-normalization====================
#=======================================================================

print("Starting ablation study: Z-normalization")

shift_errors = {}
for z_normalization in [True, False]:
    print(f"z_normalization={z_normalization}")
    errors = benchmark(shift_algorithm, error_metrics, random_states, k=False, z_normalization=z_normalization, chain=False, fallback=None)
    # Flatten the error dictionary since we only evaluate one algorithm
    errors_flat = {}
    for key, val in errors.items():
        errors_flat[key] = val["SHIFT"]
    shift_errors[f"SHIFT_z-norm={z_normalization}"] = errors_flat
    
update_results_file(shift_errors, f"{results_path}ablation_z-norm.pkl")

print("Done ablation study: Z-normalization")
