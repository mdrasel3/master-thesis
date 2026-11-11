# Define the input space for the PMMA material properties
# Date: 2026-10-12
# Author: Md Rasel
# %% Imports
import numpy as np
import chaospy
# import csv
import h5py

# %% Define the actual values for PMMA properties
E_actual = 1.64E5 # same for all PMMA
mma_com_actual = 25610
mma_radf_actual = 0.33
em_actual = 0.96 # same for all PMMA
cp_actual = 1 # ramp

# %% Define the input distributions for the PMMA properties
# Type: 'chaospy.Uniform'
E = chaospy.Uniform(0.9 * E_actual, 1.1 * E_actual)
mma_com = chaospy.Uniform(0.9 * mma_com_actual, 1.1 * mma_com_actual)
mma_radf = chaospy.Uniform(0.9 * mma_radf_actual, 1.1 * mma_radf_actual)
cp = chaospy.Uniform(0.9 * cp_actual, 1.1 * cp_actual)
em = chaospy.Uniform(0.9 * em_actual, 1.0 * em_actual)

# %% Define the joint distribution, Type: 'chaospy.J'
joint_distribution = chaospy.J(E, mma_com, mma_radf, cp, em)
distribution_r = chaospy.J(chaospy.Uniform(0, 1), chaospy.Uniform(0, 1), chaospy.Uniform(0, 1),chaospy.Uniform(0, 1),chaospy.Uniform(0, 1))
# %% Sobol sampling in normalized space
# Initialize Uniform distribution for Sobol sampling, Type: 'chaospy.J'
# Sample from the uniform distribution using Sobol sequence, Type: 'numpy.ndarray'
num_samples = 500
samples_r = distribution_r.sample(num_samples, rule="sobol")
params_normalized = samples_r.T # shape: (num_samples, 5)

# Transform the normalized (0,1) samples to the physical parameter space
samples_q = joint_distribution.inv(samples_r) # shape: (5, num_samples)

params_physical = samples_q.T # shape: (num_samples, 5)

# %%
# Save the samples to a output file
names = ["E", "mma_com", "mma_radf", "cp", "em"]
"""
output_file = f"input_space_samples_{num_samples}.csv"

with open(output_file, "w", newline="") as f:

    writer = csv.writer(f)
    writer.writerow(["id"] + names)
    for i in range(num_samples):
        writer.writerow([i] + list(samples_q[:, i]))

print(f"Saved {num_samples} samples to {output_file}")
"""
output_file = f"input_space_samples_{num_samples}.h5"

f = h5py.File(output_file, "w")
d = f.create_dataset("samples", data=params_physical)
d.attrs["names"] = names
d.attrs["num_samples"] = num_samples
d.attrs["rule"] = "sobol"
f.close()

print("saved", params_physical.shape, "samples to", output_file)
# %% EOF