
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Config
# =========================
# Expert
ALPHA_EXPERT = 0.25
NOISE_STD_EXPERT = 0.05
AMAX = 0.5

# Dataset collection (NO drift)
N_EPISODES = 300
EP_HORIZON = 80
START_RANGE = (-5.0, 5.0)

# Rollout (DRIFT env)
DRIFT = (0.30, 0.30)
ROLLOUT_STEPS = 180
TEST_START = (40.0, 40.0)

# BC train
BC_EPOCHS = 25
BC_LR = 3e-4
BC_BATCH = 256

# ACT settings
H = 10
K = 5
OBS_NOISE_STD = 0.03  # observation noise for ACT history
ACT_EPOCHS = 25
ACT_LR = 3e-4
ACT_BATCH = 256


# =========================
# 1) Expert oracle
# =========================
def expert_action(x, y, alpha=ALPHA_EXPERT, noise_std=NOISE_STD_EXPERT, amax=AMAX):
    dx = -alpha * x + np.random.randn() * noise_std
    dy = -alpha * y + np.random.randn() * noise_std
    dx = float(np.clip(dx, -amax, amax))
    dy = float(np.clip(dy, -amax, amax))
    return dx, dy


# =========================
# 2) Generate expert trajectories (NO drift)
# =========================
def generate_expert_trajectories(n_episodes, horizon, start_range):
    lo, hi = start_range
    states = []
    actions = []

    for _ in range(n_episodes):
        x = np.random.uniform(lo, hi)
        y = np.random.uniform(lo, hi)

        ep_states = [(x, y)]
        ep_actions = []
        
        for _t in range(horizon):
            dx, dy = expert_action(x, y)
            ep_actions.append((dx, dy))
            x += dx
            y += dy
            ep_states.append((x, y))

        states.append(ep_states)   # (horizon+1,2)
        actions.append(ep_actions) # (horizon,2)

    return (np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32))


# =========================
# 3) Build BC dataset: (x,y) -> (dx,dy)
# =========================
def build_bc_dataset(states, actions):
    # use state[t] -> action[t]
    X = states[:, :-1, :].reshape(-1, 2).astype(np.float32)
    Y = actions.reshape(-1, 2).astype(np.float32)
    return X, Y


# =========================
# 4) Build ACT dataset: last H states -> next K actions
# =========================
def build_act_dataset(states, actions, H, K, obs_noise_std=0.0):
    X_list, Y_list = [], []
    E, T1, _ = states.shape      # T1 = horizon+1
    T = actions.shape[1]         # horizon
    assert T1 == T + 1

    for e in range(E):
        for t in range(H-1, T-K+1):
            hist = states[e, t-(H-1):t+1]  # (H,2)
            tgt  = actions[e, t:t+K]       # (K,2)

            if obs_noise_std > 0:
                hist = hist + np.random.randn(*hist.shape).astype(np.float32) * obs_noise_std

            X_list.append(hist)
            Y_list.append(tgt)

    return np.array(X_list, np.float32), np.array(Y_list, np.float32)


# =========================
# 5) BC model + training
# =========================
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

def train_bc(X, Y, epochs=BC_EPOCHS, lr=BC_LR, batch=BC_BATCH):
    model = BCPolicy(hidden=64).to(device)
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
            print(f"[BC epoch {ep:02d}] mse={total/n:.6f}")

    return model


# =========================
# 6) ACT model + training (minimal transformer encoder)
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

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
        # x: (B,H,2)
        z = self.in_proj(x)      # (B,H,D)
        z = self.pos(z)
        z = self.encoder(z)
        last = z[:, -1, :]       # (B,D)
        out = self.head(last)    # (B,K*2)
        return out.view(out.size(0), self.K, 2)

def train_act(X, Y, epochs=ACT_EPOCHS, lr=ACT_LR, batch=ACT_BATCH):
    model = ACTPolicy(d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.1, K=K).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    X_t = torch.from_numpy(X).to(device)  # (N,H,2)
    Y_t = torch.from_numpy(Y).to(device)  # (N,K,2)
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
            print(f"[ACT epoch {ep:02d}] mse={total/n:.6f}")

    return model


# =========================
# 7) Rollouts in DRIFT env
# =========================
@torch.no_grad()
def rollout_bc(policy, start, steps=ROLLOUT_STEPS, drift=DRIFT, amax=AMAX):
    x, y = float(start[0]), float(start[1])
    traj = [(x, y)]
    for _ in range(steps):
        s = torch.tensor([[x, y]], dtype=torch.float32).to(device)
        dx, dy = policy(s).cpu().numpy()[0]
        dx = float(np.clip(dx, -amax, amax))
        dy = float(np.clip(dy, -amax, amax))
        x += dx + drift[0]
        y += dy + drift[1]
        traj.append((x, y))
    return np.array(traj, np.float32)

@torch.no_grad()
def rollout_act(policy, start, steps=ROLLOUT_STEPS, drift=DRIFT, amax=AMAX, H=H, K=K):
    x, y = float(start[0]), float(start[1])
    hist = np.tile(np.array([[x, y]], dtype=np.float32), (H, 1))
    traj = [(x, y)]

    for _ in range(steps):
        inp = torch.from_numpy(hist[None, :, :]).to(device)  # (1,H,2)
        chunk = policy(inp).cpu().numpy()[0]                 # (K,2)
        dx, dy = chunk[0]
        dx = float(np.clip(dx, -amax, amax))
        dy = float(np.clip(dy, -amax, amax))
        x += dx + drift[0]
        y += dy + drift[1]
        traj.append((x, y))

        hist = np.roll(hist, shift=-1, axis=0)
        hist[-1] = np.array([x, y], dtype=np.float32)

    return np.array(traj, np.float32)

def final_dist(traj):
    return float(np.linalg.norm(traj[-1] - np.array([0.0, 0.0], dtype=np.float32)))


# =========================
# 8) Plot overlay
# =========================
def plot_overlay(traj_bc, traj_act, start, drift):
    d_bc = final_dist(traj_bc)
    d_act = final_dist(traj_act)

    plt.figure(figsize=(6,6))
    plt.plot(traj_bc[:,0], traj_bc[:,1], "-o", ms=3, label=f"BC (final={d_bc:.2f})")
    plt.plot(traj_act[:,0], traj_act[:,1], "-o", ms=3, label=f"ACT (final={d_act:.2f})")

    plt.scatter(start[0], start[1], s=90, label=f"start {start}")
    plt.scatter(0, 0, c="red", s=140, marker="x", label="goal (0,0)")

    plt.axis("equal")
    plt.grid(alpha=0.3)
    plt.title(f"BC vs ACT @ start={start}, drift={drift}, clip={AMAX}")
    plt.legend()
    plt.show()


# =========================
# Main
# =========================
if __name__ == "__main__":
    # 1) Collect expert demonstrations (NO drift)
    states, actions = generate_expert_trajectories(
        n_episodes=N_EPISODES,
        horizon=EP_HORIZON,
        start_range=START_RANGE
    )
    print("Expert traj data:", states.shape, actions.shape)

    # 2) Build datasets
    X_bc, Y_bc = build_bc_dataset(states, actions)
    X_act, Y_act = build_act_dataset(states, actions, H=H, K=K, obs_noise_std=OBS_NOISE_STD)
    print("BC dataset:", X_bc.shape, Y_bc.shape)
    print("ACT dataset:", X_act.shape, Y_act.shape)

    # 3) Train BC and ACT
    bc_model = train_bc(X_bc, Y_bc)
    act_model = train_act(X_act, Y_act)

    # 4) Rollout BOTH from the same start in the SAME drift env
    traj_bc = rollout_bc(bc_model, TEST_START)
    traj_act = rollout_act(act_model, TEST_START)

    # 5) Overlay plot
    plot_overlay(traj_bc, traj_act, TEST_START, DRIFT)
