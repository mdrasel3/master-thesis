import json
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import *
from metrics import *
from model import Decoder, predict_runs

ckpt_file = os.path.join(results_dir, "best_model.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ck = torch.load(ckpt_file, map_location=device, weights_only=False)
s, train, validation, test = ck["s"], ck["train"], ck["validation"], ck["test"]
_, val_now, test_now = split_runs()
if not (np.array_equal(validation, val_now) and np.array_equal(test, test_now)):
    print("warning: this model was trained with a different split.json")
model = Decoder().to(device)
model.load_state_dict(ck["model"])

_, Y_train = load_runs(train)
mean_field = Y_train.mean(axis=0)
del Y_train

res = {}
row = {k: ck[k] for k in ("n_train", "seed", "epochs", "lr", "batch_size", "best_epoch")}
row["model"] = "deterministic"
for name, runs in (("val", validation), ("test", test)):
    P, Y = load_runs(runs)
    pred = to_physical(predict_runs(model, P), s)
    res[name] = report("net " + name, pred, Y)
    res[name + "_baseline"] = report("baseline " + name, np.broadcast_to(mean_field, Y.shape), Y)
    add_metrics(row, name, res[name])
pred_te, true_te = pred, Y

with open(os.path.join(results_dir, "eval_metrics.json"), "w") as f:
    json.dump(res, f, indent=1)
log_experiment(row)

sec = np.arange(n_times) * 2
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
h = np.load(os.path.join(results_dir, "loss_history.npy"))
ax[0].semilogy(h[:, 0], label="train"); ax[0].semilogy(h[:, 1], label="val")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("MSE (scaled)"); ax[0].legend()
for k in range(min(4, len(test))):
    ax[1].plot(sec, total_vs_time(true_te)[k], "-", color=f"C{k}")
    ax[1].plot(sec, total_vs_time(pred_te)[k], "--", color=f"C{k}")
ax[1].set_title("slice sum of HRRPUV (solid FDS, dashed net)"); ax[1].set_xlabel("t [s]")
ax[2].plot(sec, rmse_vs_time(pred_te, true_te), label="net")
ax[2].plot(sec, rmse_vs_time(np.broadcast_to(mean_field, true_te.shape), true_te), label="baseline")
ax[2].set_title("RMSE vs time (test) [kW/m3]"); ax[2].set_xlabel("t [s]"); ax[2].legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "eval_curves.png"), dpi=120)

worst = int(np.argmax(rel_l2(pred_te, true_te)))
fig, ax = plt.subplots(3, 3, figsize=(15, 8))
for j, k in enumerate((25, 50, 100)):
    vmax = true_te[worst, k].max()
    ax[0, j].imshow(true_te[worst, k], origin="lower", vmin=0, vmax=vmax); ax[0, j].set_title(f"FDS t={sec[k]} s")
    ax[1, j].imshow(pred_te[worst, k], origin="lower", vmin=0, vmax=vmax); ax[1, j].set_title("net")
    ax[2, j].imshow(np.abs(pred_te[worst, k] - true_te[worst, k]), origin="lower"); ax[2, j].set_title("abs error")
plt.suptitle(f"worst test run (Run_{test[worst]})")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "eval_fields.png"), dpi=120)
print("saved plots")
