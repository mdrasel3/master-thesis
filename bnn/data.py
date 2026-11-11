import os
import numpy as np
import h5py

here = os.path.dirname(os.path.abspath(__file__))
h5_file = os.path.join(here, "..", "hrrpuv_data_500_runs_float32.h5")
results_dir = os.path.join(here, "results")

n_runs = 128          # Run_1 ... Run_128
n_train, n_val, n_test = 102, 13, 13
seed = 0
n_times = 101         # t = 0, 2, ..., 200 s

# sampling bounds from input_space_sampling.py: E, mma_com, mma_radf, cp, em
lower = np.array([0.9 * 1.64e5, 0.9 * 25610, 0.9 * 0.33, 0.9 * 1.0, 0.9 * 0.96])
upper = np.array([1.1 * 1.64e5, 1.1 * 25610, 1.1 * 0.33, 1.1 * 1.0, 1.0 * 0.96])

assert n_train + n_val + n_test == n_runs
os.makedirs(results_dir, exist_ok=True)


def load_data():
    with h5py.File(h5_file, "r") as f:
        doe = f["Parameter/DOE"][:n_runs]
        g = f["slices/HRRPUV_PBY=0.04"]
        Y = np.stack([g["Run_" + str(i)][:] for i in range(1, n_runs + 1)])
    assert doe.shape == (n_runs, 5) and Y.shape == (n_runs, n_times, 28, 70)
    return doe, Y.astype(np.float32)


def scale_params(doe):
    return ((doe - lower) / (upper - lower)).astype(np.float32)


def split_runs():
    idx = np.random.default_rng(seed).permutation(n_runs)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


# log1p, then divide by the max of the training runs -> roughly [0, 1]
def fit_scale(Y_train):
    return float(np.log1p(np.maximum(Y_train, 0)).max())


def to_scaled(Y, s):
    return np.log1p(np.maximum(Y, 0)) / s


def to_physical(Z, s):
    return np.expm1(np.maximum(Z, 0) * s)
