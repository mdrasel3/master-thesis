import os
import json
import numpy as np

# fixed validation and test runs from Run_401-500, training pool Run_1-400
# run once; the result split.json is kept in git so every machine uses the same runs

current_dir = os.path.dirname(os.path.abspath(__file__))
seed = 0

held_out = np.random.default_rng(seed).permutation(np.arange(401, 501))
split = {"train_pool": list(range(1, 401)),
         "validation": sorted(held_out[:50].tolist()),
         "test": sorted(held_out[50:].tolist())}

with open(os.path.join(current_dir, "split.json"), "w") as f:
    json.dump(split, f, indent=1)
print("validation", split["validation"])
print("test", split["test"])
