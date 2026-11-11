# %%
import numpy as np
import h5py
import os
# %% Convert .npy files to HDF5 format
data_root = "../Data/5P_data_npy_HRRPUV_slice_PBY=0.04/data_npy_HRRPUV_slice_PBY=0.04"
run_start = 1
run_end = 100
total_runs = run_end - run_start + 1
# timestep = 200
out_file = "hrrpuv_fds_sim_data_" + str(total_runs) + "_runs.h5"

f = h5py.File(out_file, "w")

for i in range(run_start, run_end + 1):
    run_name = "Run_" + str(i)
    fname = os.path.join(
        data_root, run_name,
        run_name + "_HRRPUV_slice_PBY=0.04_" + str(timestep) + ".npy"
    )

    if not os.path.exists(fname):
        raise FileNotFoundError(fname)

    field = np.load(fname)
    grp = f.create_group("run_" + str(i).zfill(3))
    grp.create_dataset("field", data=field)
    grp.attrs["run_number"] = i
    grp.attrs["timestep"] = timestep

    print("run", i, "done, shape", field.shape)

f.close()
print("saved", out_file)
# %%
