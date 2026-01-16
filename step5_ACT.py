
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Config
# -------------------------
OBS_DELAY = 3          # 2 or 3
DRIFT = (0.05, 0.05)   # keep small for stability demo
AMAX = 0.25            # acceleration clip

# dynamics
DT = 1.0
FRICTION = 0.02        # small damping on velocity (stabilizes)
ROLLOUT_STEPS = 240
TEST_START = (5.0, 5.0)

# dataset
N_EPISODES = 600
EP_HORIZON = 120
START_RANGE = (-5.0, 5.0)

# ACT
H = 20
K = 5
OBS_NOISE_STD = 0.02

# training
EPOCHS = 35
LR = 3e-4
BATCH = 256

# -------------------------
# Expert (has access to hidden v)
# -------------------------
KP = 0.12   # position gain
KD = 0.35   # velocity damping

def expert_accel(x, y, vx, vy, amax=AMAX):
    ax = -KP * x - KD * vx
    ay = -KP * y - KD * vy
    ax = float(np.clip(ax, -amax, amax))
    ay = float(np.clip(ay, -amax, amax))
    return ax, ay

# -------------------------
# Simulate expert trajectories (NO drift) for training
# state true: (x,y,vx,vy) ; observation: (x,y)
# -------------------------
def generate_expert_trajectories(n_episodes, horizon, start_range):
    lo, hi = start_range
    obs_xy = []    # (E, T+1, 2)
    acts_a = []    # (E, T,   2) acceleration
    for _ in range(n_episodes):
        x = np.random.uniform(lo, hi)
        y = np.random.uniform(lo, hi)
        vx = 0.0
        vy = 0.0

        ep_obs = [(x, y)]
        ep_act = []

        for _t in range(horizon):
            ax, ay = expert_accel(x, y, vx, vy)
            ep_act.append((ax, ay))

            # dynamics (no drift in dataset)
            vx = (1.0 - FRICTION) * vx + ax * DT
            vy = (1.0 - FRICTION) * vy + ay * DT
            x = x + vx * DT
            y = y + vy * DT

            ep_obs.append((x, y))

        obs_xy.append(ep_obs)
        acts_a.append(ep_act)

    return np.array(obs_xy, np.float32), np.array(acts_a, np.float32)

# -------------------------
# Build BC dataset with delayed observation
# X: (x_{t-d}, y_{t-d}) -> Y: a_t
# -------------------------
def build_bc_dataset_delayed(obs_xy, acts_a, obs_delay, obs_noise_std=0.0):
    E, T1, _ = obs_xy.shape
    T = acts_a.shape[1]
    X, Y = [], []
    for e in range(E):
        for t in range(T):
            t_obs = max(0, t - obs_delay)
            o = obs_xy[e, t_obs].copy()
            if obs_noise_std > 0:
                o = o + np.random.randn(2).astype(np.float32) * obs_noise_std
            X.append(o)
            Y.append(acts_a[e, t])
    return np.array(X, np.float32), np.array(Y, np.float32)

# -------------------------
# Build ACT dataset with delayed observation HISTORY
# X: last H delayed obs -> Y: next K accelerations
# -------------------------
def build_act_dataset_delayed(obs_xy, acts_a, H, K, obs_delay, obs_noise_std=0.0):
    E, T1, _ = obs_xy.shape
    T = acts_a.shape[1]
    X, Y = [], []
    for e in range(E):
        for t in range(0, T - K + 1):
            end = max(0, t - obs_delay)
            start = max(0, end - (H - 1))
            hist = obs_xy[e, start:end+1]
            if len(hist) < H:
                pad = np.tile(hist[0:1], (H - len(hist), 1))
                hist = np.concatenate([pad, hist], axis=0)
            if obs_noise_std > 0:
                hist = hist + np.random.randn(*hist.shape).astype(np.float32) * obs_noise_std
            tgt = acts_a[e, t:t+K]
            X.append(hist.astype(np.float32))
            Y.append(tgt.astype(np.float32))
    return np.array(X, np.float32), np.array(Y, np.float32)

# -------------------------
# Models
# -------------------------
class BCPolicy(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )
    def forward(self, x):
        return self.net(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L, :]

class ACTPolicy(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.1, K=K):
        super().__init__()
        self.K = K
        self.in_proj = nn.Linear(2, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, K * 2)
        )
    def forward(self, x):
        z = self.in_proj(x)
        z = self.pos(z)
        z = self.encoder(z)
        last = z[:, -1, :]
        out = self.head(last)
        return out.view(out.size(0), self.K, 2)

# -------------------------
# Train helpers
# -------------------------
def train_regression(model, X, Y, epochs=EPOCHS, lr=LR, batch=BATCH, tag=""):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    X_t = torch.from_numpy(X).to(device)
    Y_t = torch.from_numpy(Y).to(device)
    n = X_t.shape[0]

    for ep in range(1, epochs+1):
        idx = torch.randperm(n, device=device)
        total = 0.0
        model.train()
        for i in range(0, n, batch):
            b = idx[i:i+batch]
            pred = model(X_t[b])
            loss = F.mse_loss(pred, Y_t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * b.numel()
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"[{tag} epoch {ep:02d}] mse={total/n:.6f}")
    return model

# -------------------------
# Rollout in DRIFT env with delayed obs
# true dynamics includes drift on position
# -------------------------
@torch.no_grad()
def rollout_bc(policy, start_xy, steps=ROLLOUT_STEPS, obs_delay=OBS_DELAY):
    x, y = float(start_xy[0]), float(start_xy[1])
    vx, vy = 0.0, 0.0

    traj = [(x, y)]
    obs_buf = [np.array([x, y], np.float32) for _ in range(obs_delay + 1)]

    for _ in range(steps):
        obs = obs_buf[0]
        s = torch.tensor([obs], dtype=torch.float32).to(device)
        ax, ay = policy(s).cpu().numpy()[0]
        ax = float(np.clip(ax, -AMAX, AMAX))
        ay = float(np.clip(ay, -AMAX, AMAX))

        vx = (1.0 - FRICTION) * vx + ax * DT
        vy = (1.0 - FRICTION) * vy + ay * DT
        x = x + vx * DT + DRIFT[0]
        y = y + vy * DT + DRIFT[1]

        traj.append((x, y))
        obs_buf.pop(0)
        obs_buf.append(np.array([x, y], np.float32))

    return np.array(traj, np.float32)

@torch.no_grad()
def rollout_act(policy, start_xy, steps=ROLLOUT_STEPS, obs_delay=OBS_DELAY):
    x, y = float(start_xy[0]), float(start_xy[1])
    vx, vy = 0.0, 0.0

    traj = [(x, y)]
    obs_buf = [np.array([x, y], np.float32) for _ in range(obs_delay + 1)]
    delayed = obs_buf[0]
    hist = np.tile(delayed[None, :], (H, 1)).astype(np.float32)

    for _ in range(steps):
        inp = torch.from_numpy(hist[None, :, :]).to(device)
        chunk = policy(inp).cpu().numpy()[0]
        ax, ay = chunk[0]
        ax = float(np.clip(ax, -AMAX, AMAX))
        ay = float(np.clip(ay, -AMAX, AMAX))

        vx = (1.0 - FRICTION) * vx + ax * DT
        vy = (1.0 - FRICTION) * vy + ay * DT
        x = x + vx * DT + DRIFT[0]
        y = y + vy * DT + DRIFT[1]
        traj.append((x, y))

        obs_buf.pop(0)
        obs_buf.append(np.array([x, y], np.float32))
        delayed = obs_buf[0]
        hist = np.roll(hist, -1, axis=0)
        hist[-1] = delayed

    return np.array(traj, np.float32)

def final_dist(traj):
    return float(np.linalg.norm(traj[-1] - np.array([0.0, 0.0], np.float32)))

def plot_overlay(traj_bc, traj_act):
    d_bc = final_dist(traj_bc)
    d_act = final_dist(traj_act)
    plt.figure(figsize=(6,6))
    plt.plot(traj_bc[:,0], traj_bc[:,1], "-o", ms=3, label=f"BC (final={d_bc:.2f})")
    plt.plot(traj_act[:,0], traj_act[:,1], "-o", ms=3, label=f"ACT (final={d_act:.2f})")
    plt.scatter(TEST_START[0], TEST_START[1], s=90, label=f"start {TEST_START}")
    plt.scatter(0, 0, c="red", s=140, marker="x", label="goal (0,0)")
    plt.axis("equal")
    plt.grid(alpha=0.3)
    plt.title(f"Inertial system | drift={DRIFT}, aclip={AMAX}, obs_delay={OBS_DELAY}")
    plt.legend()
    plt.show()

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    obs_xy, acts_a = generate_expert_trajectories(N_EPISODES, EP_HORIZON, START_RANGE)
    print("Expert data:", obs_xy.shape, acts_a.shape)

    Xbc, Ybc = build_bc_dataset_delayed(obs_xy, acts_a, OBS_DELAY, obs_noise_std=0.0)
    Xact, Yact = build_act_dataset_delayed(obs_xy, acts_a, H, K, OBS_DELAY, obs_noise_std=OBS_NOISE_STD)
    print("BC dataset:", Xbc.shape, Ybc.shape)
    print("ACT dataset:", Xact.shape, Yact.shape)

    bc = train_regression(BCPolicy(), Xbc, Ybc, tag="BC")
    act = train_regression(ACTPolicy(), Xact, Yact, tag="ACT")

    traj_bc = rollout_bc(bc, TEST_START)
    traj_act = rollout_act(act, TEST_START)

    plot_overlay(traj_bc, traj_act)
