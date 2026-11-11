import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from data import *

# summary of the training runs (Run_1 ... Run_n_train)
train, validation, test = split_runs()
P, Y = load_runs(train)
n = len(train)
t = np.arange(n_times) * 2

print("training runs", n, "| validation", len(validation), "| test", len(test))
print("Y", Y.shape, "max", Y.max(), "min", Y.min(), "zero fraction", (Y == 0).mean())
print("scaled input params: \n min", P.min(axis=0), "\n max", P.max(axis=0))

run_max = Y.max(axis=(1, 2, 3))
print("max values across all runs: \n min", run_max.min(), "\n median", np.median(run_max), "\n max", run_max.max())

first = np.array([np.argmax(Y[i].max(axis=(1, 2)) > 0) if Y[i].max() > 0 else -1 for i in range(n)])
print("\n\nfirst nonzero time index: min", first.min(), "max", first.max())

print("log1p scale (max, only training data):", fit_scale(Y))

std_t = Y.std(axis=0).mean(axis=(1, 2))
mean_t = Y.mean(axis=0).mean(axis=(1, 2))
print("[(Time, Average Spread, Average Mean)]:\n",[(int(k*2), float(std_t[k]), float(mean_t[k])) for k in (25,50,75,100)])
print("Spread across runs (std)/ mean field at t=50,100,150,200 s:\n", [round(float(std_t[k] / max(mean_t[k], 1e-12)), 3) for k in (25, 50, 75, 100)])

fig, ax = plt.subplots(2, 4, figsize=(16, 14))
for j, k in enumerate((25, 50, 75, 100)):
    im = show_field(ax[0, j], Y[0, k])
    ax[0, j].set_title(f"Run_{train[0]}, t={t[k]} s")
    plt.colorbar(im, ax=ax[0, j], fraction=0.046)
im = show_field(ax[1, 0], Y[:, 100].mean(axis=0)); ax[1, 0].set_title("mean over runs, t=200 s")
plt.colorbar(im, ax=ax[1, 0], fraction=0.046)
im = show_field(ax[1, 1], Y[:, 100].std(axis=0)); ax[1, 1].set_title("std over runs, t=200 s")
plt.colorbar(im, ax=ax[1, 1], fraction=0.046)
for i in range(n):
    ax[1, 2].plot(t, Y[i].sum(axis=(1, 2)), lw=0.5)
ax[1, 2].set_title("slice sum of HRRPUV vs time"); ax[1, 2].set_xlabel("t [s]")
ax[1, 3].hist(np.log1p(Y[:, 100].ravel()), bins=50); ax[1, 3].set_title("log1p(HRRPUV), t=200 s")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "data_summary.png"), dpi=120)

print("saved data_summary.png")
