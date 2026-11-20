import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import *
from metrics import *

# settings
windows = [3, 5, 9]        # snapshots in the moving average (3 = 6 s, 5 = 10 s, 9 = 18 s)
check_net = True           # also compare the trained network with the smoothed FDS
ckpt_file = os.path.join(results_dir, "best_model.pt")

train, validation, test = split_runs()
P_te, Y_te = load_runs(test)


# moving average over time for every cell; near t=0 and t=200 s the window gets shorter
def smooth(Y, w):
    h = w // 2
    S = np.empty_like(Y)
    for k in range(Y.shape[1]):
        S[:, k] = Y[:, max(0, k - h):k + h + 1].mean(axis=1)
    return S


# smoothed FDS = an ideal smooth model; its error against raw FDS = rough error floor
out = {}
for w in windows:
    out[f"floor_w{w}"] = report(f"smoothed FDS (w={w}) vs raw FDS, test", smooth(Y_te, w), Y_te)

if check_net and os.path.exists(ckpt_file):
    import torch
    from model import Decoder, predict_runs
    ck = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    if "test" in ck and np.array_equal(ck["test"], test):
        model = Decoder(**ck.get("sizes", {}))
        model.load_state_dict(ck["model"])
        pred = to_physical(predict_runs(model, P_te), ck["s"])
        out["net_vs_raw"] = report("net vs raw FDS, test", pred, Y_te)
        for w in windows:
            out[f"net_vs_smooth_w{w}"] = report(f"net vs smoothed FDS (w={w}), test", pred, smooth(Y_te, w))
    else:
        print("best_model.pt was trained on a different split: train again to include the network")

with open(os.path.join(results_dir, "noise_check.json"), "w") as f:
    json.dump(out, f, indent=1)

# one test run: slice sum raw vs smoothed
sec = np.arange(n_times) * 2
plt.figure(figsize=(9, 4))
plt.plot(sec, Y_te[0].sum(axis=(1, 2)), "k-", lw=1, label="FDS raw")
for w in windows:
    plt.plot(sec, smooth(Y_te[:1], w)[0].sum(axis=(1, 2)), label=f"smoothed, w={w}")
plt.title(f"Run_{test[0]}: slice sum of HRRPUV"); plt.xlabel("t [s]"); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "noise_check.png"), dpi=120)
print("saved noise_check.json and noise_check.png")
