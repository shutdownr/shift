import matplotlib.pyplot as plt
import numpy as np

from data_wrangling import train_test_split
from SHIFT import SHIFT

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Times New Roman"

segment_1_length = 1000
segment_2_length = 700
segment_3_length = 1300

start_frequency = 64
end_frequency = 128

w1 = np.pi / float(start_frequency)
w2 = np.pi / float(end_frequency)

# Segment 1: constant
k1 = np.arange(segment_1_length, dtype=float)
seg1 = np.sin(k1 * w1)
phase_after_seg1 = segment_1_length * w1

# Segment 2: linear chirp
steps2 = np.linspace(w1, w2, segment_2_length, endpoint=True, dtype=float)
phase2 = phase_after_seg1 + np.cumsum(steps2)  # absolute phase for each sample
seg2 = np.sin(phase2)
phase_after_seg2 = phase2[-1]

# Segment 3: constant
k3 = np.arange(segment_3_length, dtype=float)
seg3 = np.sin(phase_after_seg2 + (k3 + 1) * w2)  # +1 because seg2 consumed one step

ts_full = np.concatenate([seg1, seg2, seg3], axis=0)

h = 10
b = 20
data = train_test_split(
    ts_full, horizon=h, input_size=b, train_size=1 / 6, val_size=0, stride_length=1
)
timeseries = data["ts_raw"]

model = SHIFT()
model.optimize_hyperparameters(timeseries, data["X_train"], data["y_train"])
pred = model.fit_predict(timeseries, data["X_test"], h)
y_pred = np.array(pred, dtype=object)

data_retrain = train_test_split(
    ts_full, horizon=h, input_size=b, train_size=2 / 3, val_size=0, stride_length=1
)
timeseries_retrain = data_retrain["ts_raw"]
model.fit_predict(timeseries_retrain, data_retrain["X_test"], h)
pred_retrain = model.fit_predict(timeseries_retrain, data_retrain["X_test"], h)
y_pred_retrain = np.array(pred_retrain, dtype=object)

mae = np.mean(np.abs(y_pred[:, :] - data["y_test"][:, :]), axis=1)[: 1500 - b]
rolling_mae = np.convolve(mae, np.ones(50) / 50, mode="same")

mae_retrain = np.mean(
    np.abs(y_pred_retrain[:, :] - data_retrain["y_test"][:, :]), axis=1
)

mae = np.concatenate([mae, mae_retrain])

v1, v2 = 1000, 2000
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"hspace": 0.25}
)

ax1.plot(ts_full, linewidth=1.5)
ax1.axvline(v1, linestyle="--", color="black", linewidth=1.25)
ax1.axvline(v2, linestyle="--", color="black", linewidth=1.25)

y1 = ts_full[v1] if v1 < ts_full.size else 0.0
y2 = ts_full[v2] if v2 < ts_full.size else 0.0

ax1.set_ylabel("Value", fontsize="large")
ax1.margins(x=0)

ax2.plot(np.arange(len(mae)) + 500 + b, mae, linewidth=1.8)
ax2.axvline(v1, linestyle="--", color="black", linewidth=1.25)
ax2.axvline(v2, linestyle="--", color="black", linewidth=1.25)


pos1 = ax1.get_position()  # Bbox in figure fraction
pos2 = ax2.get_position()
y_center_between_axes = 0.5 * (
    pos1.y0 + pos2.y1
)  # halfway between ax1 bottom and ax2 top

to_fig = fig.transFigure.inverted()
x_v1_fig = to_fig.transform(ax1.transData.transform((v1, 0.0)))[0]
x_v2_fig = to_fig.transform(ax1.transData.transform((v2, 0.0)))[0]

txt1 = fig.text(
    x_v1_fig,
    y_center_between_axes,
    "Distribution shift begins",
    ha="center",
    va="center",
)
txt2 = fig.text(
    x_v2_fig + 0.01, y_center_between_axes, "Retraining", ha="center", va="center"
)

y_annotation = 0.038
ax1.annotate(
    "",
    xy=(v1, ax1.get_ylim()[0]),
    xycoords="data",
    xytext=(v1, -1.31),
    textcoords="data",
    arrowprops=dict(arrowstyle="->", lw=1),
    annotation_clip=False,
)
ax2.annotate(
    "",
    xy=(v1, ax2.get_ylim()[1]),
    xycoords="data",
    xytext=(v1, y_annotation),
    textcoords="data",
    arrowprops=dict(arrowstyle="->", lw=1),
    annotation_clip=False,
)

ax1.annotate(
    "",
    xy=(v2, ax1.get_ylim()[0]),
    xycoords="data",
    xytext=(v2, -1.31),
    textcoords="data",
    arrowprops=dict(arrowstyle="->", lw=1),
    annotation_clip=False,
)
ax2.annotate(
    "",
    xy=(v2, ax2.get_ylim()[1]),
    xycoords="data",
    xytext=(v2, y_annotation),
    textcoords="data",
    arrowprops=dict(arrowstyle="->", lw=1),
    annotation_clip=False,
)

ax2.axvspan(0, 500, facecolor="0.9", alpha=0.6, zorder=0)

ymin, ymax = ax2.get_ylim()
ax2.text(
    250,
    ymin + 0.5 * (ymax - ymin),
    "Training",
    ha="center",
    va="center",
    fontsize="large",
)

ax2.set_xlabel("Timestep", fontsize="large")
ax2.set_ylabel("MAE", fontsize="large")
ax2.margins(x=0)

y_annotation = 1.27

for tx in [0, 1000]:
    ax1.annotate(
        "",
        xy=(tx, y_annotation),
        xycoords="data",
        xytext=(500, y_annotation),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", lw=1, linestyle="--"),
        annotation_clip=False,
    )
ax1.text(
    500,
    y_annotation,
    "Constant frequency",
    ha="center",
    va="center",
    fontsize="medium",
    backgroundcolor="white",
)

for tx in [1000, 1700]:
    ax1.annotate(
        "",
        xy=(tx, y_annotation),
        xycoords="data",
        xytext=(1350, y_annotation),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", lw=1, linestyle="--"),
        annotation_clip=False,
    )
ax1.text(
    1350,
    y_annotation,
    "Increasing frequency\n (distribution shift)",
    ha="center",
    va="center",
    fontsize="medium",
    backgroundcolor="white",
)


for tx in [1700, 3000]:
    ax1.annotate(
        "",
        xy=(tx, y_annotation),
        xycoords="data",
        xytext=(2350, y_annotation),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", lw=1, linestyle="--"),
        annotation_clip=False,
    )
ax1.text(
    2350,
    y_annotation,
    "Constant frequency",
    ha="center",
    va="center",
    fontsize="medium",
    backgroundcolor="white",
)

plt.tight_layout()
plt.show()
fig.savefig("plots/distribution_shift_experiment.pdf", bbox_inches="tight")
