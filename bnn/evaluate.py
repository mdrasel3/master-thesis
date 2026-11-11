import json
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import *
from metrics import *
from model import Decoder

ckpt_file = os.path.join(results_dir, "best_model.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

doe, Y = load_data()
P = scale_params(doe)
ck = torch.load(ckpt_file, map_location=device, weights_only=False)
s, tr, va, te = ck["s"], ck["tr"], ck["va"], ck["te"]
model = Decoder().to(device)
model.load_state_dict(ck["model"])
model.eval()
t = np.arange(n_times) / (n_times - 1)


def predict(idx):
    out = []
    with torch.no_grad():
        for i in idx:
            X = np.concatenate([np.repeat(P[i][None], n_times, axis=0), t[:, None]], axis=1)
            Zp = model(torch.tensor(X, dtype=torch.float32, device=device))
            out.append(Zp.squeeze(1).cpu().numpy())
    return to_physical(np.array(out), s)


mean_field = Y[tr].mean(axis=0)
res = {}
for name, idx in (("val", va), ("test", te)):
    pred = predict(idx)
    res[name] = report("net " + name, pred, Y[idx])
    res[name + "_baseline"] = report("baseline " + name, np.broadcast_to(mean_field, Y[idx].shape), Y[idx])
with open(os.path.join(results_dir, "eval_metrics.json"), "w") as f:
    json.dump(res, f, indent=1)

pred_te = predict(te)
true_te = Y[te]
sec = np.arange(n_times) * 2

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
h = np.load(os.path.join(results_dir, "loss_history.npy"))
ax[0].semilogy(h[:, 0], label="train"); ax[0].semilogy(h[:, 1], label="val")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("MSE (scaled)"); ax[0].legend()
for k in range(min(4, len(te))):
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
plt.suptitle(f"worst test run (Run_{te[worst] + 1})")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "eval_fields.png"), dpi=120)
print("saved plots")
