import numpy as np
import torch
import torch.nn as nn


# (5 params + t) -> 28x70 field: dense layers, then 2 transposed convs
class Decoder(nn.Module):
    def __init__(self, n_input=6, n_hidden=256, c0=32, c1=16, c2=8):
        super().__init__()
        self.c0 = c0
        self.fc = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, c0 * 7 * 35), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(c0, c1, kernel_size=4, stride=2, padding=1)   # 7x35 -> 14x70
        self.up2 = nn.ConvTranspose2d(c1, c2, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1))  # 14x70 -> 28x70
        self.out = nn.Conv2d(c2, 1, kernel_size=3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.fc(x).view(-1, self.c0, 7, 35)
        h = self.act(self.up1(h))
        h = self.act(self.up2(h))
        return self.out(h)


# all 101 time steps for each run; P = scaled parameters (n_runs, 5) -> scaled fields (n_runs, 101, 28, 70)
def predict_runs(model, P, n_times=101):
    device = next(model.parameters()).device
    t = np.arange(n_times) / (n_times - 1)
    out = []
    model.eval()
    with torch.no_grad():
        for p in P:
            X = np.concatenate([np.repeat(p[None], n_times, axis=0), t[:, None]], axis=1)
            out.append(model(torch.tensor(X, dtype=torch.float32, device=device)).squeeze(1).cpu().numpy())
    return np.array(out)
