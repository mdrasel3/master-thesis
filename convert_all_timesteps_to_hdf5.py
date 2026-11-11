# %% Build one h5 file with all runs and all timesteps
import numpy as np
import h5py
import os

# settings
data_root = "data_raw/data_npy_HRRPUV_slice_PBY=0.04"
param_file = "input_space_samples_500.h5"
out_file = "hrrpuv_data_500_runs_float32.h5"
run_start = 1
run_end = 500
times = list(range(0, 201, 2))   # 0, 2, ..., 200
dtype = "float32"

# %% parameters
with h5py.File(param_file, "r") as p:
    samples = p["samples"][:]
    names = [n.decode() if isinstance(n, bytes) else n for n in p["samples"].attrs["names"]]

# %% write
with h5py.File(out_file, "w") as f:
    f.create_dataset("Parameter/DOE", data=samples)
    f["Parameter/DOE"].attrs["names"] = names
    # f.create_dataset("Time", data=np.array(times))

    grp = f.create_group("slices/HRRPUV_PBY=0.04")
    for i in range(run_start, run_end + 1):
        run_name = "Run_" + str(i)
        fields = []
        for t in times:
            fname = os.path.join(data_root, run_name,
                                 run_name + "_HRRPUV_slice_PBY=0.04_" + str(t) + ".npy")
            if not os.path.exists(fname):
                raise FileNotFoundError(fname)
            fields.append(np.load(fname))
        data = np.stack(fields).astype(dtype)      # (101, 28, 70)
        d = grp.create_dataset(run_name, data=data)
        d.attrs["param_index"] = i - 1             # row in Parameter/samples
        if i % 10 == 0:
            print("run", i, "done, shape", data.shape)

print("saved", out_file)
# %%
