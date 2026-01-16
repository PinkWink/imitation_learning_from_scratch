
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Config (tune for stronger effect)
# =========================
ALPHA_EXPERT = 0.25
NOISE_STD_EXPERT = 0.05     # demo imperfection (set 0.0 for perfect expert)
AMAX = 0.5                  # action clip magnitude (per-axis)
DRIFT = (0.25, 0.25)        # env bias per step (distribution shift amplifier)

HORIZON_EXPERT = 25         # expert demo length for initial dataset
HORIZON_ROLLOUT = 160       # rollout length in drift env (longer -> more shift)
DAGGER_ITERS = 8            # number of DAgger iterations
ROLLOUT_EPISODES_PER_ITER = 8  # how many rollouts to collect per DAgger iter

TRAIN_EPOCHS_PRE = 800
TRAIN_EPOCHS_DAGGER = 400
LR = 1e-3

# =========================
# 1) Expert (oracle) policy
# =========================
def expert_action(x, y, alpha=ALPHA_EXPERT, noise_std=NOISE_STD_EXPERT, amax=AMAX):
    dx = -alpha * x + np.random.randn() * noise_std
    dy = -alpha * y + np.random.randn() * noise_std
    dx = float(np.clip(dx, -amax, amax))
    dy = float(np.clip(dy, -amax, amax))

    return dx, dy

# =========================
# 2) Initial expert dataset generator (no drift)
# =========================
def generate_expert_dataset(start_points, horizon=HORIZON_EXPERT):
    states, actions = [], []
    for (x0, y0) in start_points:
        x, y = float(x0), float(y0)
        for _ in range(horizon):
            dx, dy = expert_action(x, y)
            states.append([x, y])
            actions.append([dx, dy])
            # no drift in expert dataset collection
            x += dx
            y += dy
    return np.array(states, np.float32), np.array(actions, np.float32)

# =========================
# 3) Policy model (BC)
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

# =========================
# 4) Train helper
# =========================
def train_policy(model, X, Y, epochs, lr=LR, batch_size=256, verbose_every=100):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.from_numpy(X).to(device)
    Y_t = torch.from_numpy(Y).to(device)
    n = X_t.shape[0]

    for ep in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(n, device=device)
        total = 0.0

        for i in range(0, n, batch_size):
            b = idx[i:i+batch_size]
            pred = model(X_t[b])
            loss = F.mse_loss(pred, Y_t[b])

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * b.numel()

        if (ep % verbose_every == 0) or (ep == 1) or (ep == epochs):
            print(f"  [epoch {ep:04d}] mse={total/n:.6f}")

    return model

# =========================
# 5) Rollout in DRIFT env
# =========================
@torch.no_grad()
def rollout_policy(policy, start, horizon=HORIZON_ROLLOUT, amax=AMAX, drift=DRIFT):
    x, y = float(start[0]), float(start[1])
    traj = [(x, y)]

    for _ in range(horizon):
        s = torch.tensor([[x, y]], dtype=torch.float32).to(device)
        dx, dy = policy(s).cpu().numpy()[0]

        # action clip at rollout
        dx = float(np.clip(dx, -amax, amax))
        dy = float(np.clip(dy, -amax, amax))

        # environment transition includes drift
        x += dx + drift[0]
        y += dy + drift[1]

        traj.append((x, y))

    return np.array(traj, dtype=np.float32)

def final_dist(traj):
    return float(np.linalg.norm(traj[-1] - np.array([0.0, 0.0], dtype=np.float32)))

# =========================
# 6) Visualization
# =========================
def plot_trajectories(trajs, labels, title):
    plt.figure(figsize=(5.5,5.5))
    for traj, lab in zip(trajs, labels):
        plt.plot(traj[:,0], traj[:,1], "-o", ms=3, label=lab, alpha=0.9)
        plt.scatter(traj[0,0], traj[0,1], s=70, label=f"start {lab}", alpha=0.9)
    plt.scatter(0, 0, c="red", s=140, marker="x", label="goal (0,0)")
    plt.axis("equal")
    plt.grid(alpha=0.3)
    plt.title(title)
    plt.legend()
    plt.show()

def expert_action_batch(states_xy):
    acts = []
    for x, y in states_xy:
        acts.append(expert_action(float(x), float(y)))

    return np.array(acts, dtype=np.float32)

# =========================
# 7) DAgger loop
# =========================
def dagger_train( X_init, Y_init, start_sampler_fn,
                dagger_iters=DAGGER_ITERS, rollouts_per_iter=ROLLOUT_EPISODES_PER_ITER ):
    # ---- Baseline BC (expert-only) ----
    print("\n=== Train BC baseline (expert-only) ===")
    bc_model = BCPolicy(hidden=64)
    bc_model = train_policy(bc_model, X_init, Y_init, epochs=TRAIN_EPOCHS_PRE, 
                                                                verbose_every=200)

    # ---- DAgger model starts from BC weights (common practice) ----
    dagger_model = BCPolicy(hidden=64)
    dagger_model.load_state_dict(bc_model.state_dict())

    # Aggregated dataset begins with expert dataset
    X_agg = X_init.copy()
    Y_agg = Y_init.copy()

    history = []

    for it in range(1, dagger_iters + 1):
        print(f"\n=== DAgger iter {it}/{dagger_iters} ===")

        # 1) Rollout current policy in drift env, collect visited states
        collected_states = []
        for _ in range(rollouts_per_iter):
            st = start_sampler_fn()
            traj = rollout_policy(dagger_model, st)
            # Use all visited states except the last (either is fine)
            collected_states.append(traj[:-1])

        S = np.concatenate(collected_states, axis=0)  # (M,2)

        # 2) Query expert to label those states
        A = expert_action_batch(S)  # (M,2)

        # 3) Aggregate
        X_agg = np.concatenate([X_agg, S.astype(np.float32)], axis=0)
        Y_agg = np.concatenate([Y_agg, A.astype(np.float32)], axis=0)

        print(f"  Aggregated dataset size: {len(X_agg)}")

        # 4) Retrain / finetune on aggregated dataset
        dagger_model = train_policy(
            dagger_model, X_agg, Y_agg,
            epochs=TRAIN_EPOCHS_DAGGER,
            verbose_every=200
        )

        # 5) Evaluate quickly on a fixed hard start
        test_start = (40.0, 40.0)
        traj_test = rollout_policy(dagger_model, test_start)
        dist = final_dist(traj_test)

        history.append({
            "iter": it,
            "agg_size": int(len(X_agg)),
            "test_start": test_start,
            "final_dist": dist
        })
        print(f"  [Eval] start={test_start}, final_dist={dist:.2f}")

    return bc_model, dagger_model, history, (X_agg, Y_agg)

# =========================
# 8) Run everything
# =========================
if __name__ == "__main__":
    # ---- Initial expert dataset: small & narrow (intentionally) ----
    start_points = [(1.0,1.0), (2.0,2.0), (3.0,3.0)]
    X0, Y0 = generate_expert_dataset(start_points, horizon=HORIZON_EXPERT)
    print("Initial expert dataset:", X0.shape, Y0.shape)

    # Start sampler for DAgger rollouts:
    # Important: sample "hard" starts so policy visits far-away states (e.g., around 40)
    def start_sampler():
        # Mix: mostly far starts + some medium starts
        if np.random.rand() < 0.7:
            x = np.random.uniform(25.0, 45.0)
            y = np.random.uniform(25.0, 45.0)
        else:
            x = np.random.uniform(-10.0, 10.0)
            y = np.random.uniform(-10.0, 10.0)
        return (float(x), float(y))

    bc_model, dagger_model, hist, (Xagg, Yagg) = dagger_train(
        X0, Y0, start_sampler_fn=start_sampler
    )

    # ---- Compare BC vs DAgger on the same test starts ----
    tests = [(40.0, 40.0)]
    for st in tests:
        traj_bc = rollout_policy(bc_model, st)
        traj_dg = rollout_policy(dagger_model, st)

        d_bc = final_dist(traj_bc)
        d_dg = final_dist(traj_dg)

        plot_trajectories(
            [traj_bc, traj_dg],
            [f"BC (final={d_bc:.2f})", f"DAgger (final={d_dg:.2f})"],
            title=f"Comparison @ start={st}, drift={DRIFT}, clip={AMAX}"
        )

    # ---- Print DAgger progress ----
    print("\nDAgger progress (final_dist from (40,40)):")
    for row in hist:
        print(f"  iter={row['iter']:02d}  agg_size={row['agg_size']:6d}  final_dist={row['final_dist']:.2f}")
