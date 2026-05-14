"""Empirical validation of SHIFT's noise robustness on a synthetic periodic signal.

Tests whether forecast MAE and mean(delta*) scale predictably with additive
Gaussian noise sigma. Produces noise_results.csv, noise_scaling.pdf,
delta_vs_sigma.pdf, and noise_summary.md.
"""

import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from SHIFT import SHIFT


SIGMAS = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
N_SEEDS = 30
TS_LENGTH = 2000
B = 20            # input window
H = 10            # horizon
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20   # test = remaining 0.20

OUT_DIR = "artifacts/noise_robustness"
CSV_PATH = os.path.join(OUT_DIR, "noise_results.csv")
SCALING_PDF = os.path.join(OUT_DIR, "noise_scaling.pdf")
DELTA_PDF = os.path.join(OUT_DIR, "delta_vs_sigma.pdf")
SUMMARY_MD = os.path.join(OUT_DIR, "noise_summary.md")


def generate_series(sigma, seed, length=TS_LENGTH):
    rng = np.random.default_rng(seed)
    i = np.arange(length, dtype=float)
    clean = 0.001 * i + np.sin(2 * np.pi * i / 24)
    noise = rng.normal(loc=0.0, scale=sigma, size=length) if sigma > 0 else np.zeros(length)
    return clean + noise, clean


def build_windows(series, b, h):
    """Sliding windows with stride 1: each (X_i, y_i) is (b inputs, h targets)."""
    n_windows = len(series) - b - h + 1
    X = np.empty((n_windows, b), dtype=float)
    y = np.empty((n_windows, h), dtype=float)
    for i in range(n_windows):
        X[i] = series[i : i + b]
        y[i] = series[i + b : i + b + h]
    return X, y


def split_series(noisy, clean, b, h):
    """60/20/20 split on the time axis; window each region; carry clean test targets."""
    n = len(noisy)
    n_train = int(round(TRAIN_FRAC * n))
    n_val = int(round(VAL_FRAC * n))

    train_seg = noisy[:n_train]
    val_seg = noisy[n_train - b : n_train + n_val]
    test_seg = noisy[n_train + n_val - b :]
    test_seg_clean = clean[n_train + n_val - b :]

    X_train, y_train = build_windows(train_seg, b, h)
    X_val, y_val = build_windows(val_seg, b, h)
    X_test, _ = build_windows(test_seg, b, h)
    _, y_test_clean = build_windows(test_seg_clean, b, h)

    # Training timeseries for SHIFT's f-shapelet bank: train+val region.
    ts_for_model = noisy[: n_train + n_val]
    return {
        "ts_raw": ts_for_model,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test_clean": y_test_clean,
    }


def run_one(sigma, seed):
    noisy, clean = generate_series(sigma, seed)
    data = split_series(noisy, clean, B, H)

    # Merge train+val for SHIFT hyperparameter optimization, mirroring predict_shift.
    X_opt = np.concatenate([data["X_train"], data["X_val"]], axis=0)
    y_opt = np.concatenate([data["y_train"], data["y_val"]], axis=0)

    model = SHIFT()
    # Defaults: chain=True, z_normalization=True, euclidean, k-d tree on.
    # Override search space per task spec: l in 3..20, n in 2..8.
    model.L = np.arange(3, 21)
    model.N = np.arange(2, 9)
    model.optimize_hyperparameters(data["ts_raw"], X_opt, y_opt)

    y_pred = model.fit_predict(data["ts_raw"], data["X_test"], H)

    # MAE between forecast and ground truth (clean targets — true signal).
    mean_mae = float(np.mean(np.abs(y_pred - data["y_test_clean"])))

    # delta* = mean of the n smallest f-shapelet distances per test instance.
    # model.distances is set inside fit_predict; shape varies but the non-chained
    # head [:, 0, :n] holds the n nearest distances per test instance.
    distances = model.distances
    nearest = distances[:, 0, :]  # (n_test, n)
    mean_delta_star = float(np.mean(nearest))

    return mean_mae, mean_delta_star


def fit_loglog_slope(xs, ys):
    """Linear regression in log-log space; skip non-positive entries."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = (xs > 0) & (ys > 0) & np.isfinite(xs) & np.isfinite(ys)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    lx = np.log10(xs[mask])
    ly = np.log10(ys[mask])
    slope, intercept = np.polyfit(lx, ly, 1)
    return float(slope), float(intercept)


def classify(slope):
    if not np.isfinite(slope):
        return "undetermined"
    if 0.8 <= slope <= 1.2:
        return "approximately linear"
    if slope < 0.8:
        return "sublinear"
    return "superlinear"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    seeds = list(range(N_SEEDS))

    t0 = time.time()
    total = len(SIGMAS) * len(seeds)
    done = 0
    for sigma in SIGMAS:
        for seed in seeds:
            tic = time.time()
            mean_mae, mean_delta = run_one(sigma, seed)
            done += 1
            rows.append({
                "sigma": sigma,
                "seed": seed,
                "mean_mae": mean_mae,
                "mean_delta_star": mean_delta,
            })
            elapsed = time.time() - t0
            rate = elapsed / done
            eta = rate * (total - done)
            bar_width = 30
            filled = int(bar_width * done / total)
            bar = "#" * filled + "-" * (bar_width - filled)
            print(
                f"[{bar}] {done:3d}/{total} ({100*done/total:5.1f}%) "
                f"sigma={sigma:<7g} seed={seed:<3d} "
                f"mae={mean_mae:.4e} delta*={mean_delta:.4e} "
                f"step={time.time() - tic:.1f}s elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sigma", "seed", "mean_mae", "mean_delta_star"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {CSV_PATH}")

    # Aggregate per sigma.
    rows_arr = np.array(
        [(r["sigma"], r["seed"], r["mean_mae"], r["mean_delta_star"]) for r in rows],
        dtype=float,
    )
    sigmas = np.array(SIGMAS, dtype=float)
    mae_means = np.array([rows_arr[rows_arr[:, 0] == s, 2].mean() for s in sigmas])
    mae_stds = np.array([rows_arr[rows_arr[:, 0] == s, 2].std(ddof=1) for s in sigmas])
    delta_means = np.array([rows_arr[rows_arr[:, 0] == s, 3].mean() for s in sigmas])
    delta_stds = np.array([rows_arr[rows_arr[:, 0] == s, 3].std(ddof=1) for s in sigmas])

    # Slopes use sigma > 0 only.
    mae_slope, mae_intercept = fit_loglog_slope(sigmas, mae_means)
    delta_slope, delta_intercept = fit_loglog_slope(sigmas, delta_means)

    # --- Plot: MAE vs sigma ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    pos = sigmas > 0
    ax.errorbar(
        sigmas[pos], mae_means[pos], yerr=mae_stds[pos],
        marker="o", linestyle="none", capsize=3, label="mean MAE (±1 std)",
    )
    if pos.any():
        x_line = np.array([sigmas[pos].min(), sigmas[pos].max()])
        y_line = 10 ** (mae_intercept + mae_slope * np.log10(x_line))
        ax.plot(
            x_line, y_line, linestyle="--",
            label=f"log-log fit, slope = {mae_slope:.3f}",
        )
    # Show sigma=0 point at the lowest positive sigma's location, annotated.
    if (sigmas == 0).any():
        zero_mae = mae_means[sigmas == 0][0]
        ax.axhline(zero_mae, color="grey", linewidth=0.8, linestyle=":",
                   label=f"σ=0 floor MAE = {zero_mae:.2e}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Noise std $\sigma$")
    ax.set_ylabel("Mean MAE on test (vs clean signal)")
    ax.set_title("SHIFT forecast error vs additive noise")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.6)
    fig.tight_layout()
    fig.savefig(SCALING_PDF)
    plt.close(fig)
    print(f"Wrote {SCALING_PDF}")

    # --- Plot: delta* vs sigma ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(
        sigmas[pos], delta_means[pos], yerr=delta_stds[pos],
        marker="s", linestyle="none", capsize=3, label="mean δ* (±1 std)",
    )
    if pos.any():
        x_line = np.array([sigmas[pos].min(), sigmas[pos].max()])
        y_line = 10 ** (delta_intercept + delta_slope * np.log10(x_line))
        ax.plot(
            x_line, y_line, linestyle="--",
            label=f"log-log fit, slope = {delta_slope:.3f}",
        )
    if (sigmas == 0).any():
        zero_delta = delta_means[sigmas == 0][0]
        ax.axhline(zero_delta, color="grey", linewidth=0.8, linestyle=":",
                   label=f"σ=0 floor δ* = {zero_delta:.2e}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Noise std $\sigma$")
    ax.set_ylabel(r"Mean $\delta^*$ (n nearest f-shapelet distances)")
    ax.set_title("F-shapelet nearest-neighbour distance vs additive noise")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.6)
    fig.tight_layout()
    fig.savefig(DELTA_PDF)
    plt.close(fig)
    print(f"Wrote {DELTA_PDF}")

    # --- Summary ---
    lines = []
    lines.append("# SHIFT noise robustness — empirical summary\n")
    lines.append(f"- Seeds per sigma: {N_SEEDS}\n")
    lines.append(f"- Series length: {TS_LENGTH}, split 60/20/20, b={B}, h={H}\n")
    lines.append(f"- Sigmas: {SIGMAS}\n")
    lines.append("\n## Per-sigma aggregates\n\n")
    lines.append("| sigma | mean MAE | std MAE | mean δ* | std δ* |\n")
    lines.append("|------:|---------:|--------:|--------:|-------:|\n")
    for s, mm, ms, dm, ds in zip(sigmas, mae_means, mae_stds, delta_means, delta_stds):
        lines.append(f"| {s:g} | {mm:.6f} | {ms:.6f} | {dm:.6f} | {ds:.6f} |\n")

    lines.append("\n## Log-log slopes (fit over sigma > 0)\n")
    lines.append(f"- **MAE vs sigma slope:** {mae_slope:.3f} ({classify(mae_slope)})\n")
    lines.append(f"- **δ\\* vs sigma slope:** {delta_slope:.3f} ({classify(delta_slope)})\n")

    lines.append("\n## Interpretation\n")
    if 0.8 <= mae_slope <= 1.2:
        lines.append(
            "- MAE scaling is approximately linear in sigma (slope in [0.8, 1.2]), "
            "which empirically supports a linear bound of the form "
            "`forecast_error <= C * sigma` (Theorem 2-style).\n"
        )
    elif mae_slope < 0.8:
        lines.append(
            "- MAE grows **sublinearly** with sigma (slope < 0.8). SHIFT is more "
            "robust than a linear bound would predict over this sigma range.\n"
        )
    else:
        lines.append(
            "- MAE grows **superlinearly** with sigma (slope > 1.2). A linear bound "
            "is weaker than observed scaling over this sigma range.\n"
        )

    if 0.8 <= delta_slope <= 1.2:
        lines.append(
            "- δ* tracks sigma approximately linearly, consistent with the "
            "expectation that identical signal templates differ only by their noise.\n"
        )
    else:
        lines.append(
            f"- δ* slope of {delta_slope:.3f} deviates from the expected ~1 — "
            "templates do not appear to differ purely by noise at these scales "
            "(e.g., shapelet search snaps to different periodic alignments, or "
            "differences are dominated by trend / numerical floor at low sigma).\n"
        )

    if (sigmas == 0).any():
        lines.append(
            f"- At sigma=0 the noise floor is MAE={mae_means[sigmas == 0][0]:.2e}, "
            f"δ*={delta_means[sigmas == 0][0]:.2e}; this is excluded from the log-log "
            "fits but shown as a horizontal reference in each plot.\n"
        )

    with open(SUMMARY_MD, "w") as f:
        f.writelines(lines)
    print(f"Wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
