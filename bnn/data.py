import os
import csv
import json
import datetime
import subprocess
import numpy as np
import h5py

here = os.path.dirname(os.path.abspath(__file__))
h5_file = os.path.join(here, "..", "hrrpuv_data_500_runs_float32.h5")
split_file = os.path.join(here, "split.json")
log_file = os.path.join(here, "experiments.csv")
results_dir = os.path.join(here, "results")

n_train = 128         # training runs: Run_1 ... Run_n_train (at most 400)
seed = 0
n_times = 101         # t = 0, 2, ..., 200 s

# sampling bounds from input_space_sampling.py: E, mma_com, mma_radf, cp, em
lower = np.array([0.9 * 1.64e5, 0.9 * 25610, 0.9 * 0.33, 0.9 * 1.0, 0.9 * 0.96])
upper = np.array([1.1 * 1.64e5, 1.1 * 25610, 1.1 * 0.33, 1.1 * 1.0, 1.0 * 0.96])

os.makedirs(results_dir, exist_ok=True)


# run numbers (Run_1 -> 1) from split.json; validation and test runs never change
def split_runs():
    with open(split_file) as f:
        s = json.load(f)
    assert n_train <= len(s["train_pool"])
    train = np.array(s["train_pool"][:n_train])
    return train, np.array(s["validation"]), np.array(s["test"])


# scaled parameters and fields (float32) for a list of run numbers
def load_runs(runs):
    with h5py.File(h5_file, "r") as f:
        doe = f["Parameter/DOE"][:]
        g = f["slices/HRRPUV_PBY=0.04"]
        Y = np.stack([g["Run_" + str(r)][:] for r in runs])
    assert Y.shape == (len(runs), n_times, 28, 70)
    return scale_params(doe[runs - 1]), Y.astype(np.float32)


def scale_params(doe):
    return ((doe - lower) / (upper - lower)).astype(np.float32)


# log1p, then divide by the max of the training runs -> roughly [0, 1]
def fit_scale(Y_train):
    return float(np.log1p(np.maximum(Y_train, 0)).max())


def to_scaled(Y, s):
    return np.log1p(np.maximum(Y, 0)) / s


def to_physical(Z, s):
    return np.expm1(np.maximum(Z, 0) * s)


# plot one (28, 70) field: x (28 cells) horizontal, z (70 cells) vertical, 4 cm cells
cell = 0.04


def show_field(ax, field, **kw):
    im = ax.imshow(field.T, origin="lower", extent=[0, 28 * cell, 0, 70 * cell], **kw)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    return im


# experiment log: one row per evaluated model, with the git commit it ran on
log_cols = ["date", "commit", "model", "n_train", "seed", "epochs", "lr", "batch_size", "best_epoch"] + \
           [s + "_" + k for s in ("val", "test") for k in ("rel_l2_mean", "rel_l2_max", "rmse", "mae", "r2")]


def git_version():
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=here, capture_output=True, text=True).stdout.strip()
        return (h or "unknown") + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def add_metrics(row, name, res):
    for k, v in res.items():
        row[name + "_" + k] = round(v, 5)


def log_experiment(row):
    row = dict(row, date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), commit=git_version())
    new = not os.path.exists(log_file)
    with open(log_file, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log_cols, restval="")
        if new:
            w.writeheader()
        w.writerow(row)
    print("logged to", os.path.basename(log_file), "commit", row["commit"])
