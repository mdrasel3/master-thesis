import json
import os
import numpy as np
from data import *
from metrics import *

# baseline: at each time step, predict the mean field of the training runs
train, validation, test = split_runs()
_, Y_train = load_runs(train)
mean_field = Y_train.mean(axis=0)

out = {}
row = {"model": "baseline", "n_train": n_train}
for name, runs in (("val", validation), ("test", test)):
    _, Y = load_runs(runs)
    pred = np.broadcast_to(mean_field, Y.shape)
    out[name] = report("baseline " + name, pred, Y)
    add_metrics(row, name, out[name])

with open(os.path.join(results_dir, "baseline_metrics.json"), "w") as f:
    json.dump(out, f, indent=1)
log_experiment(row)
