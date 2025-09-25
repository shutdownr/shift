import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from dataset_config import Config
from data_wrangling import (
    read_dataset,
    flatten,
    scale,
    inverse_scale_predictions,
)
from basic_models import predict_shift_distances
from evaluation import evaluate_errors

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Times New Roman"

horizon = 10
backhorizon = 20
dataset_name = "m4_h"
error_metrics = ["mae", "mse", "smape", "mase", "owa"]
metric = "owa"
num_bins = 10

all_errors = {}
all_inference_times = {}

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

algo_errors = {}
algo_inference_times = {}


y_pred, distances, time = predict_shift_distances(dataset_scaled)
y_pred = np.array(inverse_scale_predictions(y_pred, scalers))

y_pred_flat = np.array([y for series in y_pred for y in series])
x_test_flat = np.array([x for series in dataset["X_test"] for x in series])
y_test_flat = np.array([y for series in dataset["y_test"] for y in series])
distances_flat = np.array([d for series in distances for d in series])

# calculate error per instances
errors_per_instance = []
for i, x_test_instance, y_pred_instance, y_true_instance in zip(
    range(len(x_test_flat)), x_test_flat, y_pred_flat, y_test_flat
):
    dataset_trimmed = dataset_flat.copy()
    dataset_trimmed["X_test"] = np.array([dataset_trimmed["X_test"][i]])
    error_instance = evaluate_errors(
        np.array([x_test_instance]),
        np.array([y_true_instance]),
        np.array([y_pred_instance]),
        error_metrics,
        dataset=dataset_trimmed,
    )
    errors_per_instance.append(error_instance[error_metrics.index(metric)])
errors_per_instance = np.array(errors_per_instance)

bins = pd.qcut(distances_flat, q=num_bins, labels=False, duplicates="drop")
# Recover bin edges for labeling
try:
    _, bin_edges = pd.qcut(distances_flat, q=num_bins, retbins=True, duplicates="drop")
except Exception:
    # Fallback: unique quantiles
    qs = np.unique(np.linspace(0, 1, num_bins + 1))
    bin_edges = np.quantile(distances_flat, qs)

df = pd.DataFrame(
    {"distances": distances_flat, "error": errors_per_instance, "bin": bins}
)
grouped = df.groupby("bin", dropna=True)

# Mean error and representative distances for each bin
mean_error = grouped["error"].mean()
mean_similarity = grouped["distances"].mean()

rho, pval = spearmanr(distances_flat, errors_per_instance, nan_policy="omit")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(mean_similarity.values, mean_error.values, marker="o", lw=2, markersize=7)
ax.set_xlabel("Mean distance $\delta^\star$", fontsize="large")
ax.set_ylabel("Mean forecast OWA error", fontsize="large")

fig.text(0.81, 0.2, f"$\\rho=${rho:.3f}", fontsize="large")
fig.text(0.81, 0.15, f"$p=${pval:.3f}", fontsize="large")

fig.savefig("plots/similarity_error_reliability.pdf", bbox_inches="tight")

print("spearman_rho_error_vs_similarity", float(rho))
print("spearman_p_value", float(pval))
