#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 11:54:47 2025

@author: hamza
"""

import numpy as np
import chaospy
#import chaospy as cp

num_samples=500



num_samples_file= 500

E_actual = 1.64E5 # same for all PMMA
mma_com_actual = 25610
mma_radf_actual = 0.33


A1_actual = 4.95E16
A2_actual = 1.35E11 
E_actual = 1.64E5 # same for all PMMA

rho_actual = 1210 # same for all PMMA
h1_actual =  5
h2_actual = 817

em_actual = 0.96 # same for all PMMA
absp_actual = 2870 # same for all PMMA
k_actual = 1 # ramp

cp_actual = 1 # ramp
mma_com_actual = 25610
mma_radf_actual = 0.33

mma_cp_actual = 1 # ramp

E = chaospy.Uniform(0.9*E_actual, 1.1*E_actual)
mma_com = chaospy.Uniform(0.9*mma_com_actual, 1.1*mma_com_actual)
mma_radf = chaospy.Uniform(0.9*mma_radf_actual, 1.1*mma_radf_actual)
cp = chaospy.Uniform(0.9*cp_actual, 1.1*cp_actual)
em= chaospy.Uniform(0.9*em_actual, 1*em_actual)

def transform(samples):
    return distribution_q.inv(distribution_r.fwd(samples))
joint = chaospy.J(E,mma_com,mma_radf,cp,em)

distribution_q = joint
distribution_r = chaospy.J(chaospy.Uniform(0, 1), chaospy.Uniform(0, 1), chaospy.Uniform(0, 1),chaospy.Uniform(0, 1),chaospy.Uniform(0, 1))


samples_r = distribution_r.sample(num_samples, rule="sobol")

mesh_data = np.load("data_raw/data_npy_HRRPUV_slice_PBY=0.04/Run_1/Run_1_HRRPUV_slice_PBY=0.04_0.npy", allow_pickle=True)


timestep = 200
mesh_shape = np.shape(mesh_data)
dataRoot = "data_raw/data_npy_HRRPUV_slice_PBY=0.04/"
HRRPUV_per_time = np.zeros((num_samples,mesh_shape[0], mesh_shape[1]))

for sam in range(num_samples):
    file = (dataRoot+ f"Run_{sam+1}/Run_{sam+1}_HRRPUV_slice_PBY=0.04_{timestep}.npy"    )
    HRRPUV_per_time[sam, :, :] = np.load(file)

from ezyrb import POD, RBF, Database
from ezyrb import ReducedOrderModel as ROM
from ezyrb import  GPR, KNeighborsRegressor, RadiusNeighborsRegressor, Linear, ANN
#### ROM to create new results####
num_new_samples = 10000
new_samples = distribution_r.sample(num_new_samples, rule="sobol")

pred_sol = np.zeros((num_new_samples,mesh_shape[0], mesh_shape[1]))
####



snapshots = HRRPUV_per_time

#### Transform snapshots ##############

snap_min = np.min(snapshots)
snap_max = np.max(snapshots)

snap_scaled = (snapshots - snap_min) / (snap_max - snap_min)
snapshots = snap_scaled
############################

import time

param = samples_r.T
print(snapshots.shape, param.shape)

start = time.perf_counter()


db = Database(param, snapshots)
pod = POD(method='correlation_matrix')
#pod = POD('svd')

kernel = "cubic"
#kernel = "multiquadric"
#kernel = "quintic"
#kernel= "gaussian"
#kernel ='linear'
#kernel="thin_plate_spline"
#kernel = 'inverse_multiquadric'
#kernel = "inverse_quadratic"
rbf =  RBF(kernel=kernel)
#approx= KNeighborsRegressor(n_neighbors=5)
#kernel = "KN"
rom = ROM(db, pod, rbf)
rom.fit();

end = time.perf_counter()
elapsed = end - start
print("Time elapsed for POD fit: ",elapsed)

new_mu = new_samples.T
prediction = rom.predict(new_mu)
pred_sol = np.asarray(getattr(prediction, "snapshots_matrix", prediction))


######### inverse transform ##############
pred_sol = pred_sol* (snap_max - snap_min) + snap_min
##########################################


pred_sol_reshaped = pred_sol.reshape(num_new_samples,mesh_shape[0], mesh_shape[1])


#pred_sol = pred_sol.reshape






from scipy.stats import skew, kurtosis
QMC_mean_mesh = np.mean(HRRPUV_per_time, axis=0)
QMC_std_mesh = np.std(HRRPUV_per_time, axis=0)
QMC_skew_mesh = skew(HRRPUV_per_time, axis=0)
QMC_kurtosis_mesh = kurtosis(HRRPUV_per_time, axis=0)
QMC_skew_mesh= np.nan_to_num(QMC_skew_mesh, nan=0.0)
QMC_kurtosis_mesh = np.nan_to_num(QMC_kurtosis_mesh, nan=0.0)

threshold = 0.01 * np.max(QMC_mean_mesh) 
mask = QMC_mean_mesh > threshold

QMC_skew_mesh[~mask] = 0.0
QMC_kurtosis_mesh[~mask] = 0.0

np.savetxt(f'mean_QMC_{num_samples}_samples.txt', QMC_mean_mesh)
np.savetxt(f'sdt_QMC_{num_samples}_samples.txt', QMC_std_mesh)
np.savetxt(f'skew_QMC_{num_samples}_samples.txt', QMC_skew_mesh)
np.savetxt(f'kurt_QMC_{num_samples}_samples.txt', QMC_kurtosis_mesh)


mean_mesh = np.mean(pred_sol, axis=0)  # Shape: (28, 23)
std_mesh = np.std(pred_sol, axis=0)    # Shape: (28, 23)

import matplotlib.pyplot as plt


# Compute mean and std for predicted meshes
pred_mean = np.mean(pred_sol_reshaped, axis=0)  # (28, 23)
pred_std = np.std(pred_sol_reshaped, axis=0)    # (28, 23)
pred_skew = skew(pred_sol_reshaped, axis=0)        # (28, 23)
pred_kurt = kurtosis(pred_sol_reshaped, axis=0)    # (28, 23)
pred_skew = np.nan_to_num(pred_skew, nan=0.0)
pred_kurt = np.nan_to_num(pred_kurt, nan=0.0)


threshold = 0.01 * np.max(pred_mean) 
mask = pred_mean > threshold
pred_skew[~mask] = 0.0
pred_kurt[~mask] = 0.0


np.savetxt(f'mean_PODI_{kernel}_{num_samples}_{num_new_samples}_samples.txt', pred_mean)
np.savetxt(f'sdt_PODI_{kernel}_{num_samples}_{num_new_samples}_samples.txt', pred_std)
np.savetxt(f'skew_PODI_{kernel}_{num_samples}_{num_new_samples}_samples.txt', pred_skew)
np.savetxt(f'kurt_PODI_{kernel}_{num_samples}_{num_new_samples}_samples.txt', pred_kurt)


# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot predicted mean and std
im1 = axes[0, 0].imshow(pred_mean.T, cmap='viridis', aspect='auto',origin='lower')
axes[0, 0].set_title("PODI Mean HRRPUV")
axes[0, 0].set_xlabel('X Coordinate')
axes[0, 0].set_ylabel('Z Coordinate')
cbar1 = fig.colorbar(im1, ax=axes[0, 0])
cbar1.set_label('HRRPUV (°C)')

im2 = axes[0, 1].imshow(pred_std.T, cmap='plasma', aspect='auto',origin='lower')
axes[0, 1].set_title("PODI HRRPUV Standard Deviation")
axes[0, 1].set_xlabel('X Coordinate')
axes[0, 1].set_ylabel('Z Coordinate')
cbar2 = fig.colorbar(im2, ax=axes[0, 1])
cbar2.set_label('Standard Deviation (°C)')

# Plot QMC mean and std
im3 = axes[1, 0].imshow(QMC_mean_mesh.T, cmap='viridis', aspect='auto',origin='lower')
axes[1, 0].set_title("QMC Mean HRRPUV")
axes[1, 0].set_xlabel('X Coordinate')
axes[1, 0].set_ylabel('Z Coordinate')
cbar3 = fig.colorbar(im3, ax=axes[1, 0])
cbar3.set_label('HRRPUV (°C)')

im4 = axes[1, 1].imshow(QMC_std_mesh.T, cmap='plasma', aspect='auto',origin='lower')
axes[1, 1].set_title("QMC HRRPUV Standard Deviation")
axes[1, 1].set_xlabel('X Coordinate')
axes[1, 1].set_ylabel('Z Coordinate')
cbar4 = fig.colorbar(im4, ax=axes[1, 1])
cbar4.set_label('Standard Deviation (°C)')
plt.savefig(f'PODI_stats_{kernel}_{num_samples}_{num_new_samples}_samples.pdf')
plt.tight_layout()
plt.show()

# Calculate MSE for mean and std
mse_mean = np.mean((pred_mean - QMC_mean_mesh) ** 2)
mse_std = np.mean((pred_std - QMC_std_mesh) ** 2)

print(f"MSE (Mean): {mse_mean:.6f}")
print(f"MSE (Std): {mse_std:.6f}")

#MSE (Mean): 3051.528109
#MSE (Std): 4906.852388
