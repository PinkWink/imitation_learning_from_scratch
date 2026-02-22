"""Minimal scratch module for imitation learning class.

This module provides:
1) Simple behavior cloning (BC) dataset and rollout environment.
2) Drifted rollout setting to visualize BC limitation.
3) DAgger training loop for distribution-shift mitigation.
4) Inertial + delayed-observation setting where ACT can outperform BC/DAgger.

Designed for notebook import and classroom demos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Global helpers
# -----------------------------

def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Section A: BC / DAgger world (1st-order point mass)
# -----------------------------

@dataclass
class BCWorldConfig:
    alpha: float = 0.25
    noise_std: float = 0.03
    action_clip: float = 0.5
    horizon: int = 25


@dataclass
class DriftConfig:
    dx: float = 0.25
    dy: float = 0.25


DEFAULT_BC_WORLD = BCWorldConfig()
DEFAULT_DRIFT = DriftConfig()


def expert_action_bc(
    x: float,
    y: float,
    alpha: float = DEFAULT_BC_WORLD.alpha,
    noise_std: float = DEFAULT_BC_WORLD.noise_std,
    action_clip: float = DEFAULT_BC_WORLD.action_clip,
) -> Tuple[float, float]:
    dx = -alpha * x + np.random.randn() * noise_std
    dy = -alpha * y + np.random.randn() * noise_std
    dx = float(np.clip(dx, -action_clip, action_clip))
    dy = float(np.clip(dy, -action_clip, action_clip))
    return dx, dy


def generate_bc_training_data(
    start_points: Sequence[Tuple[float, float]],
    horizon: int = DEFAULT_BC_WORLD.horizon,
    alpha: float = DEFAULT_BC_WORLD.alpha,
    noise_std: float = DEFAULT_BC_WORLD.noise_std,
    action_clip: float = DEFAULT_BC_WORLD.action_clip,
) -> Tuple[np.ndarray, np.ndarray]:
    states, actions = [], []
    for x0, y0 in start_points:
        x, y = float(x0), float(y0)
        for _ in range(horizon):
            dx, dy = expert_action_bc(x, y, alpha, noise_std, action_clip)
            states.append([x, y])
            actions.append([dx, dy])
            x += dx
            y += dy
    return np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def get_bc_test_starts() -> List[Tuple[float, float]]:
    return [(4.0, 4.0), (10.0, 10.0), (-10.0, 8.0)]


def get_drift_test_starts() -> List[Tuple[float, float]]:
    return [(40.0, 40.0), (-40.0, 40.0), (-35.0, -35.0)]


class BCPolicy(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_bc_model(
    X: np.ndarray,
    Y: np.ndarray,
    epochs: int = 600,
    lr: float = 1e-3,
    hidden: int = 64,
    verbose_every: int = 100,
) -> BCPolicy:
    device = get_device()
    model = BCPolicy(hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    x_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(Y).to(device)

    for ep in range(1, epochs + 1):
        pred = model(x_t)
        loss = F.mse_loss(pred, y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep == 1 or ep % verbose_every == 0 or ep == epochs:
            print(f"[BC epoch {ep:04d}] mse={loss.item():.6f}")

    return model


@torch.no_grad()
def rollout_bc_policy(
    policy: nn.Module,
    start: Tuple[float, float],
    horizon: int = 120,
    action_clip: float = DEFAULT_BC_WORLD.action_clip,
    drift: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    device = get_device()
    x, y = float(start[0]), float(start[1])
    traj = [(x, y)]

    for _ in range(horizon):
        s = torch.tensor([[x, y]], dtype=torch.float32, device=device)
        dx, dy = policy(s).detach().cpu().numpy()[0]
        dx = float(np.clip(dx, -action_clip, action_clip))
        dy = float(np.clip(dy, -action_clip, action_clip))
        x += dx + drift[0]
        y += dy + drift[1]
        traj.append((x, y))

    return np.asarray(traj, dtype=np.float32)


def final_distance_to_goal(traj: np.ndarray, goal: Tuple[float, float] = (0.0, 0.0)) -> float:
    return float(np.linalg.norm(traj[-1] - np.asarray(goal, dtype=np.float32)))


def expert_action_batch_bc(states: np.ndarray) -> np.ndarray:
    acts = [expert_action_bc(float(x), float(y)) for x, y in states]
    return np.asarray(acts, dtype=np.float32)


def train_dagger_model(
    X_init: np.ndarray,
    Y_init: np.ndarray,
    start_sampler: Callable[[], Tuple[float, float]],
    dagger_iters: int = 6,
    rollouts_per_iter: int = 8,
    rollout_horizon: int = 120,
    drift: Tuple[float, float] = (DEFAULT_DRIFT.dx, DEFAULT_DRIFT.dy),
    pretrain_epochs: int = 500,
    finetune_epochs: int = 250,
    lr: float = 1e-3,
) -> Tuple[BCPolicy, BCPolicy, List[Dict[str, float]], Tuple[np.ndarray, np.ndarray]]:
    print("\n=== Train BC baseline ===")
    bc_model = train_bc_model(X_init, Y_init, epochs=pretrain_epochs, lr=lr, verbose_every=200)

    dagger_model = BCPolicy()
    dagger_model.load_state_dict(bc_model.state_dict())

    x_agg = X_init.copy()
    y_agg = Y_init.copy()
    history: List[Dict[str, float]] = []

    for it in range(1, dagger_iters + 1):
        collected = []
        for _ in range(rollouts_per_iter):
            st = start_sampler()
            traj = rollout_bc_policy(
                dagger_model,
                start=st,
                horizon=rollout_horizon,
                drift=drift,
            )
            collected.append(traj[:-1])

        s = np.concatenate(collected, axis=0).astype(np.float32)
        a = expert_action_batch_bc(s)

        x_agg = np.concatenate([x_agg, s], axis=0)
        y_agg = np.concatenate([y_agg, a], axis=0)

        print(f"\n=== DAgger iter {it}/{dagger_iters} | agg size={len(x_agg)} ===")
        dagger_model = train_bc_model(x_agg, y_agg, epochs=finetune_epochs, lr=lr, verbose_every=125)

        probe_start = (40.0, 40.0)
        probe_traj = rollout_bc_policy(dagger_model, probe_start, rollout_horizon, drift=drift)
        probe_dist = final_distance_to_goal(probe_traj)
        history.append({"iter": float(it), "agg_size": float(len(x_agg)), "probe_dist": probe_dist})
        print(f"[DAgger eval] start={probe_start} final_dist={probe_dist:.2f}")

    return bc_model, dagger_model, history, (x_agg, y_agg)


# -----------------------------
# Section B: ACT world (2nd-order + delayed observation)
# -----------------------------

@dataclass
class ACTWorldConfig:
    dt: float = 1.0
    friction: float = 0.02
    accel_clip: float = 0.25
    drift_x: float = 0.05
    drift_y: float = 0.05
    obs_delay: int = 3
    obs_noise_std: float = 0.02


DEFAULT_ACT_WORLD = ACTWorldConfig()

# PD expert with hidden velocity state
KP = 0.12
KD = 0.35


def expert_action_act(
    x: float,
    y: float,
    vx: float,
    vy: float,
    accel_clip: float = DEFAULT_ACT_WORLD.accel_clip,
) -> Tuple[float, float]:
    ax = -KP * x - KD * vx
    ay = -KP * y - KD * vy
    ax = float(np.clip(ax, -accel_clip, accel_clip))
    ay = float(np.clip(ay, -accel_clip, accel_clip))
    return ax, ay


def generate_act_expert_trajectories(
    n_episodes: int = 500,
    horizon: int = 120,
    start_range: Tuple[float, float] = (-5.0, 5.0),
    world: ACTWorldConfig = DEFAULT_ACT_WORLD,
) -> Tuple[np.ndarray, np.ndarray]:
    lo, hi = start_range
    obs_xy = []
    acts = []

    for _ in range(n_episodes):
        x = float(np.random.uniform(lo, hi))
        y = float(np.random.uniform(lo, hi))
        vx, vy = 0.0, 0.0

        ep_obs = [(x, y)]
        ep_act = []

        for _ in range(horizon):
            ax, ay = expert_action_act(x, y, vx, vy, world.accel_clip)
            ep_act.append((ax, ay))

            vx = (1.0 - world.friction) * vx + ax * world.dt
            vy = (1.0 - world.friction) * vy + ay * world.dt
            x = x + vx * world.dt
            y = y + vy * world.dt

            ep_obs.append((x, y))

        obs_xy.append(ep_obs)
        acts.append(ep_act)

    return np.asarray(obs_xy, dtype=np.float32), np.asarray(acts, dtype=np.float32)


def build_bc_dataset_delayed(
    obs_xy: np.ndarray,
    acts: np.ndarray,
    obs_delay: int = DEFAULT_ACT_WORLD.obs_delay,
    obs_noise_std: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    e_count, _, _ = obs_xy.shape
    t_count = acts.shape[1]
    X, Y = [], []

    for e in range(e_count):
        for t in range(t_count):
            t_obs = max(0, t - obs_delay)
            o = obs_xy[e, t_obs].copy()
            if obs_noise_std > 0:
                o += np.random.randn(2).astype(np.float32) * obs_noise_std
            X.append(o)
            Y.append(acts[e, t])

    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def build_act_dataset_delayed(
    obs_xy: np.ndarray,
    acts: np.ndarray,
    history_len: int = 20,
    chunk_len: int = 5,
    obs_delay: int = DEFAULT_ACT_WORLD.obs_delay,
    obs_noise_std: float = DEFAULT_ACT_WORLD.obs_noise_std,
) -> Tuple[np.ndarray, np.ndarray]:
    e_count, _, _ = obs_xy.shape
    t_count = acts.shape[1]
    X, Y = [], []

    for e in range(e_count):
        for t in range(0, t_count - chunk_len + 1):
            end = max(0, t - obs_delay)
            start = max(0, end - (history_len - 1))

            hist = obs_xy[e, start : end + 1]
            if len(hist) < history_len:
                pad = np.tile(hist[0:1], (history_len - len(hist), 1))
                hist = np.concatenate([pad, hist], axis=0)

            if obs_noise_std > 0:
                hist = hist + np.random.randn(*hist.shape).astype(np.float32) * obs_noise_std

            tgt = acts[e, t : t + chunk_len]
            X.append(hist.astype(np.float32))
            Y.append(tgt.astype(np.float32))

    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class ACTPolicy(nn.Module):
    def __init__(
        self,
        history_len: int = 20,
        chunk_len: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.history_len = history_len
        self.chunk_len = chunk_len
        self.in_proj = nn.Linear(2, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max(512, history_len + 4))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, chunk_len * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(x)
        z = self.pos(z)
        z = self.encoder(z)
        out = self.head(z[:, -1, :])
        return out.view(out.shape[0], self.chunk_len, 2)


def train_regression_model(
    model: nn.Module,
    X: np.ndarray,
    Y: np.ndarray,
    epochs: int = 35,
    lr: float = 3e-4,
    batch_size: int = 256,
    tag: str = "model",
    verbose_every: int = 5,
) -> nn.Module:
    device = get_device()
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    x_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(Y).to(device)
    n = x_t.shape[0]

    for ep in range(1, epochs + 1):
        idx = torch.randperm(n, device=device)
        total = 0.0
        model.train()

        for i in range(0, n, batch_size):
            b = idx[i : i + batch_size]
            pred = model(x_t[b])
            loss = F.mse_loss(pred, y_t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * b.numel()

        if ep == 1 or ep % verbose_every == 0 or ep == epochs:
            print(f"[{tag} epoch {ep:03d}] mse={total / n:.6f}")

    return model


@torch.no_grad()
def rollout_bc_on_act_world(
    policy: nn.Module,
    start_xy: Tuple[float, float] = (5.0, 5.0),
    steps: int = 240,
    world: ACTWorldConfig = DEFAULT_ACT_WORLD,
) -> np.ndarray:
    device = get_device()
    x, y = float(start_xy[0]), float(start_xy[1])
    vx, vy = 0.0, 0.0
    traj = [(x, y)]

    obs_buf = [np.asarray([x, y], dtype=np.float32) for _ in range(world.obs_delay + 1)]

    for _ in range(steps):
        obs = obs_buf[0]
        s = torch.from_numpy(obs[None, :]).to(device)
        ax, ay = policy(s).detach().cpu().numpy()[0]
        ax = float(np.clip(ax, -world.accel_clip, world.accel_clip))
        ay = float(np.clip(ay, -world.accel_clip, world.accel_clip))

        vx = (1.0 - world.friction) * vx + ax * world.dt
        vy = (1.0 - world.friction) * vy + ay * world.dt
        x = x + vx * world.dt + world.drift_x
        y = y + vy * world.dt + world.drift_y

        traj.append((x, y))
        obs_buf.pop(0)
        obs_buf.append(np.asarray([x, y], dtype=np.float32))

    return np.asarray(traj, dtype=np.float32)


@torch.no_grad()
def rollout_act_policy(
    policy: ACTPolicy,
    start_xy: Tuple[float, float] = (5.0, 5.0),
    steps: int = 240,
    history_len: int = 20,
    world: ACTWorldConfig = DEFAULT_ACT_WORLD,
) -> np.ndarray:
    device = get_device()
    x, y = float(start_xy[0]), float(start_xy[1])
    vx, vy = 0.0, 0.0
    traj = [(x, y)]

    obs_buf = [np.asarray([x, y], dtype=np.float32) for _ in range(world.obs_delay + 1)]
    delayed = obs_buf[0]
    hist = np.tile(delayed[None, :], (history_len, 1)).astype(np.float32)

    for _ in range(steps):
        inp = torch.from_numpy(hist[None, :, :]).to(device)
        chunk = policy(inp).detach().cpu().numpy()[0]
        ax, ay = chunk[0]
        ax = float(np.clip(ax, -world.accel_clip, world.accel_clip))
        ay = float(np.clip(ay, -world.accel_clip, world.accel_clip))

        vx = (1.0 - world.friction) * vx + ax * world.dt
        vy = (1.0 - world.friction) * vy + ay * world.dt
        x = x + vx * world.dt + world.drift_x
        y = y + vy * world.dt + world.drift_y

        traj.append((x, y))
        obs_buf.pop(0)
        obs_buf.append(np.asarray([x, y], dtype=np.float32))
        delayed = obs_buf[0]

        hist = np.roll(hist, -1, axis=0)
        hist[-1] = delayed

    return np.asarray(traj, dtype=np.float32)


def get_act_test_starts() -> List[Tuple[float, float]]:
    return [(5.0, 5.0), (7.0, -6.0), (-8.0, 8.0)]


# -----------------------------
# Section C: Visualization utils
# -----------------------------

def plot_expert_state_action_data(
    X: np.ndarray,
    Y: np.ndarray,
    title: str = "Expert data (state->action)",
    max_points: int = 2500,
) -> None:
    n = min(len(X), max_points)
    idx = np.random.choice(len(X), size=n, replace=False) if len(X) > n else np.arange(len(X))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(X[idx, 0], X[idx, 1], s=8, alpha=0.5)
    ax[0].set_title("states (x,y)")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].grid(alpha=0.3)

    ax[1].scatter(Y[idx, 0], Y[idx, 1], s=8, alpha=0.5, color="tab:orange")
    ax[1].set_title("actions")
    ax[1].set_xlabel("a_x")
    ax[1].set_ylabel("a_y")
    ax[1].grid(alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_trajectories(
    trajectories: Sequence[np.ndarray],
    labels: Sequence[str],
    title: str,
    goal: Tuple[float, float] = (0.0, 0.0),
) -> None:
    plt.figure(figsize=(6, 6))
    for traj, label in zip(trajectories, labels):
        plt.plot(traj[:, 0], traj[:, 1], "-o", ms=2.5, label=label, alpha=0.9)
        plt.scatter(traj[0, 0], traj[0, 1], s=55)

    plt.scatter(goal[0], goal[1], c="red", s=120, marker="x", label="goal")
    plt.axis("equal")
    plt.grid(alpha=0.3)
    plt.title(title)
    plt.legend()
    plt.show()


def compare_two_policies(
    traj_a: np.ndarray,
    traj_b: np.ndarray,
    label_a: str,
    label_b: str,
    title: str,
) -> None:
    d_a = final_distance_to_goal(traj_a)
    d_b = final_distance_to_goal(traj_b)
    plot_trajectories(
        [traj_a, traj_b],
        [f"{label_a} (final={d_a:.2f})", f"{label_b} (final={d_b:.2f})"],
        title,
    )


__all__ = [
    "ACTPolicy",
    "ACTWorldConfig",
    "BCPolicy",
    "BCWorldConfig",
    "DEFAULT_ACT_WORLD",
    "DEFAULT_BC_WORLD",
    "DEFAULT_DRIFT",
    "DriftConfig",
    "build_act_dataset_delayed",
    "build_bc_dataset_delayed",
    "compare_two_policies",
    "expert_action_act",
    "expert_action_bc",
    "final_distance_to_goal",
    "generate_act_expert_trajectories",
    "generate_bc_training_data",
    "get_act_test_starts",
    "get_bc_test_starts",
    "get_drift_test_starts",
    "get_device",
    "plot_expert_state_action_data",
    "plot_trajectories",
    "rollout_act_policy",
    "rollout_bc_on_act_world",
    "rollout_bc_policy",
    "set_seed",
    "train_bc_model",
    "train_dagger_model",
    "train_regression_model",
]
