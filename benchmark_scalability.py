import os
import pickle 
import time

import numpy as np

from SHIFT import SHIFT

results_path = "results_multivariate/"

bh = 40
h = 20

random_states = np.arange(3)

# Benchmark function running a benchmark for scalability
def benchmark():
    h = 5
    b = 10
    stride_length = 1

    all_inference_times = {}

    for length in [1e3,5e3,1e4,5e4,1e5,5e5]:
        print(f"Starting length {length}\n")
        # Generate random data
        sine_data = np.random.rand(int(length)).astype(float)

        # Split data into train and test
        train_length = 0.8
        split_point = int(train_length * len(sine_data))
        timeseries_train = sine_data[:split_point]
        timeseries_test = sine_data[split_point:]

        # Split train/test part of the time series into instances
        x_test = np.lib.stride_tricks.sliding_window_view(timeseries_test[:-h], b)[::stride_length, :]
    
        inference_times = {}

        model_naive = SHIFT()
        model_naive._use_kd_tree = False
        model_kdtree = SHIFT()
    
        # No need for hyperparameter optimization for this benchmark, as only inference time matters
        # model.optimize_hyperparameters(timeseries_train, x_train, y_train)
        # Make a forecast for each test instance
        times_naive = []
        times_kdtree = []
        for _ in random_states:
            t0 = time.time()
            _ = model_naive.fit_predict(timeseries_train, x_test, h)
            times_naive.append(time.time() - t0)

            t0 = time.time()
            _ = model_kdtree.fit_predict(timeseries_train, x_test, h)
            times_kdtree.append(time.time() - t0)
        inference_times["SHIFT_naive_mean"] = np.mean(times_naive)
        inference_times["SHIFT_naive_std"] = np.std(times_naive)
        inference_times["SHIFT_kdtree_mean"] = np.mean(times_kdtree)
        inference_times["SHIFT_kdtree_std"] = np.std(times_kdtree)
        print(inference_times)
        all_inference_times[f"{length}"] = inference_times
    
    return all_inference_times

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
#=========================Benchmark scalability=========================
#=======================================================================

print("Starting scalability benchmark")

inference_times = benchmark()
update_results_file(inference_times, f"{results_path}benchmark_scalability.pkl")

print("Scalability benchmark done")