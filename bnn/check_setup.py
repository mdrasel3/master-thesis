import os
import sys
import numpy as np
import h5py
import matplotlib
import torch
from data import *
from model import Decoder

print("python", sys.version.split()[0])
print("numpy", np.__version__, "| h5py", h5py.__version__, "| matplotlib", matplotlib.__version__)
print("torch", torch.__version__, "| cuda build", torch.version.cuda, "| cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))

print("h5 file found:", os.path.exists(h5_file), "| split.json found:", os.path.exists(split_file))
with h5py.File(h5_file, "r") as f:
    print("DOE", f["Parameter/DOE"].shape, "| runs", len(f["slices/HRRPUV_PBY=0.04"].keys()))

train, validation, test = split_runs()
print("train", len(train), "(Run_1 to Run_%d)" % train[-1], "| validation", len(validation), "| test", len(test))
print("first test runs:", test[:5].tolist())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Decoder().to(device)
out = model(torch.rand(256, 6, device=device))
out.mean().backward()
print("output shape", tuple(out.shape), "| one training step ok on", device)
