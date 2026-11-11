import os
import numpy as np
import torch
from data import *
from model import Decoder

# settings
epochs = 300
batch_size = 256
lr = 1e-3
ckpt_file = os.path.join(results_dir, "best_model.pt")

torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)

doe, Y = load_data()
P = scale_params(doe)
train, validation, test = split_runs()
s = fit_scale(Y[train])
Z = to_scaled(Y, s)
t = np.arange(n_times) / (n_times - 1)


# one row per (run, time step): [5 scaled params, t] -> scaled field
def make_pairs(idx):
    n = len(idx)
    X = np.concatenate([np.repeat(P[idx], n_times, axis=0), np.tile(t, n)[:, None]], axis=1)
    T = Z[idx].reshape(n * n_times, 1, 28, 70)
    return torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(T, dtype=torch.float32, device=device)


X_tr, T_tr = make_pairs(train)
X_va, T_va = make_pairs(validation)
print("train pairs", X_tr.shape[0], "val pairs", X_va.shape[0])

model = Decoder().to(device)
print("parameters", sum(p.numel() for p in model.parameters()))
opt = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.MSELoss()

best = float("inf")
history = []
for epoch in range(1, epochs + 1):
    model.train()
    perm = torch.randperm(X_tr.shape[0], device=device)
    tot = 0.0
    for k in range(0, len(perm), batch_size):
        b = perm[k:k + batch_size]
        opt.zero_grad()
        loss = loss_fn(model(X_tr[b]), T_tr[b])
        loss.backward()
        opt.step()
        tot += loss.item() * len(b)
    train_loss = tot / len(perm)

    model.eval()
    with torch.no_grad():
        val_loss = np.mean([loss_fn(model(X_va[k:k + 1024]), T_va[k:k + 1024]).item()
                            for k in range(0, X_va.shape[0], 1024)])
    history.append([train_loss, val_loss])

    if val_loss < best:
        best = val_loss
        torch.save({"model": model.state_dict(), "s": s, "train": train, "validation": validation, "test": test}, ckpt_file)
    if epoch % 10 == 0 or epoch == 1:
        print(f"epoch {epoch} train {train_loss:.5f} val {val_loss:.5f} best {best:.5f}")

np.save(os.path.join(results_dir, "loss_history.npy"), np.array(history))
