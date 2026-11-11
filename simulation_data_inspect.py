# Understanding the data in the .npy file.
# This script loads a specific .npy file containing
# HRRPUV data for a given timestep and run, analyzes
# its properties, and visualizes it using a heatmap.
# Date: 2026-10-12
# Author: Md Rasel
# %% Imports
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# %% Loading the data
data_root = Path("data_raw/data_npy_HRRPUV_slice_PBY=0.04/")
desired_timestep = 200
desired_file = (data_root/f"Run_1/Run_1_HRRPUV_slice_PBY=0.04_{desired_timestep}.npy")

arr = np.load(desired_file, allow_pickle=True)

# %% Analyzing the data
print("Shape:", arr.shape)
print("Dtype:", arr.dtype)
print("Min:", np.nanmin(arr))
print("Max:", np.nanmax(arr))
print("Mean:", np.nanmean(arr))
print("NaNs:", np.isnan(arr).sum())
# %% Plotting
plt.figure(figsize=(6, 4))
plt.imshow(arr, origin="lower", cmap="inferno")
plt.colorbar(label="HRRPUV")
plt.title(f"Run 1, HRRPUV slice PBY=0.04, t={desired_timestep} s")
plt.xlabel("Grid index x")
plt.ylabel("Grid index z")
plt.tight_layout()
plt.show()

# %% 
runs = [1, 10, 50, 100]
times = [0, 50, 100, 150, 200]
fig, axes = plt.subplots(len(runs), len(times), figsize=(14, 8))

for r_idx, run_id in enumerate(runs):
    for t_idx, t in enumerate(times):
        file = data_root / f"Run_{run_id}" / f"Run_{run_id}_HRRPUV_slice_PBY=0.04_{t}.npy"
        arr = np.load(file)

        ax = axes[r_idx, t_idx]
        im = ax.imshow(arr, origin="lower", cmap="inferno")
        ax.set_title(f"Run {run_id}, t={t}")
        ax.axis("off")

plt.tight_layout()
plt.show()
# %% EOF