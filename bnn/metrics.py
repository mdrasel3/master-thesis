import numpy as np


# pred, true: (n_runs, 101, 28, 70) in kW/m3
def rel_l2(pred, true):
    num = np.sqrt(((pred - true) ** 2).sum(axis=(1, 2, 3)))
    den = np.sqrt((true ** 2).sum(axis=(1, 2, 3)))
    return num / den


def rmse(pred, true):
    return float(np.sqrt(((pred - true) ** 2).mean()))


def mae(pred, true):
    return float(np.abs(pred - true).mean())


def r2(pred, true):
    return float(1 - ((pred - true) ** 2).sum() / ((true - true.mean()) ** 2).sum())


def rmse_vs_time(pred, true):
    return np.sqrt(((pred - true) ** 2).mean(axis=(0, 2, 3)))


def total_vs_time(field):
    return field.sum(axis=(2, 3))


def report(name, pred, true):
    r = rel_l2(pred, true)
    out = {"rel_l2_mean": float(r.mean()), "rel_l2_max": float(r.max()),
           "rmse": rmse(pred, true), "mae": mae(pred, true), "r2": r2(pred, true)}
    print(name, " ".join(f"{k}={v:.4g}" for k, v in out.items()))
    return out
