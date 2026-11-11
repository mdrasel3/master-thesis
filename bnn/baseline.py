import json
import os
import numpy as np
from data import *
from metrics import *

# baseline: at each time step, predict the mean field of the training runs
doe, Y = load_data()
train, validation, test = split_runs()

mean_field = Y[train].mean(axis=0)
out = {}
for name, idx in (("val", validation), ("test", test)):
    pred = np.broadcast_to(mean_field, Y[idx].shape)
    out[name] = report("baseline " + name, pred, Y[idx])

with open(os.path.join(results_dir, "baseline_metrics.json"), "w") as f:
    json.dump(out, f, indent=1)
