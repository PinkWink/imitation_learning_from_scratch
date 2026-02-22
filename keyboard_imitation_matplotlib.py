from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import animation
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Button
from torch.utils.data import DataLoader, TensorDataset

from data_view_panel import DataViewPanel


@dataclass
class RobotState:
    x: float
    y: float
    yaw: float


class ACTPolicy(nn.Module):
    """Transformer-based temporal policy over image sequences."""

    def __init__(self, hidden: int = 128, max_seq_len: int = 32) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 18 * 18, 256),
            nn.ReLU(),
            nn.Linear(256, hidden),
            nn.ReLU(),
        )
        self.register_buffer(
            "pos_embed",
            self._build_sinusoidal_pos_embed(max_seq_len=max_seq_len, d_model=hidden),
            persistent=False,
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=3)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

    @staticmethod
    def _build_sinusoidal_pos_embed(max_seq_len: int, d_model: int) -> torch.Tensor:
        pos = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        i = torch.arange(d_model, dtype=torch.float32).unsqueeze(0)
        angle_rates = 1.0 / torch.pow(10000.0, (2.0 * torch.floor(i / 2.0)) / d_model)
        angles = pos * angle_rates

        pe = torch.zeros(max_seq_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(angles[:, 0::2])
        pe[:, 1::2] = torch.cos(angles[:, 1::2])
        return pe.unsqueeze(0)  # [1, T, D]

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: [B, T, C, H, W]
        bsz, seq_len, c, h, w = x_seq.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}; "
                "increase max_seq_len in ACTPolicy."
            )
        x = x_seq.reshape(bsz * seq_len, c, h, w)
        feats = self.encoder(x).reshape(bsz, seq_len, -1)
        pos = self.pos_embed[:, :seq_len, :].to(device=feats.device, dtype=feats.dtype)
        tokens = self.norm(feats + pos)
        out = self.transformer(tokens)
        return self.head(out)  # [B, T, 2]


class ImitationRoadDemo:
    def __init__(self) -> None:
        self.dataset_image_source = "ax_local_patch_v9_multiset"
        self.dt = 0.05
        self.v_scale = 1.8
        self.w_scale = 2.2
        self.arrow_len = 0.45

        self.road_cx = 0.0
        self.road_cy = 4.0
        self.road_r = 8.0
        self.road_x_min = 0.3
        self.road_x_max = 7.7

        self.mode = "idle"  # idle | collect | infer
        self.motion_blocked = False
        self.keys_down: set[str] = set()
        self.collect_frame_idx = 0

        self.images: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self._set_lengths: List[int] = []

        self.act_seq_len = 8
        self.act_frame_buffer: List[torch.Tensor] = []
        self.obs_patch_size = 240

        self.dataset_dir = Path("data/keyboard_imitation_sets")
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_files: List[Path] = []

        self.data_view = DataViewPanel(
            dt=self.dt,
            list_dataset_files=self._list_dataset_files,
            load_dataset_set_file=self._load_dataset_set_file,
            on_dataset_deleted=self._on_data_view_dataset_deleted,
            refresh_main_datasets=self._refresh_dataset_count,
            set_main_status=self._set_status,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.act_model: Optional[ACTPolicy] = None
        self.act_ready = False
        self._last_data_warning: Optional[str] = None

        self.state = self._initial_state()
        self.infer_action = np.zeros(2, dtype=np.float32)

        self.fig, self.ax = plt.subplots(figsize=(12.0, 6.8))
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title("Matplotlib Imitation Learning Demo")
        self.fig.subplots_adjust(right=0.72)

        self._setup_axes()
        self._build_scene()
        self._build_buttons()
        self._build_status_text()
        self._connect_events()
        self._refresh_dataset_count()

        self.anim = animation.FuncAnimation(
            self.fig,
            self._update,
            interval=int(self.dt * 1000),
            blit=False,
        )

    def _road_center(self, x: np.ndarray) -> np.ndarray:
        xx = np.clip(x, self.road_x_min, self.road_x_max)
        dx = xx - self.road_cx
        inside = np.maximum(self.road_r**2 - dx**2, 1e-6)
        return self.road_cy - np.sqrt(inside)

    def _road_slope(self, x: float) -> float:
        xx = float(np.clip(x, self.road_x_min, self.road_x_max))
        dx = xx - self.road_cx
        den = max(math.sqrt(max(self.road_r**2 - dx**2, 1e-6)), 1e-6)
        return float(dx / den)

    def _initial_state(self) -> RobotState:
        x0 = self.road_x_min + 0.25
        y0 = float(self._road_center(np.array([x0]))[0])
        yaw0 = float(np.arctan(self._road_slope(x0)))
        return RobotState(x=x0, y=y0, yaw=yaw0)

    def _setup_axes(self) -> None:
        self.ax.set_xlim(-0.8, 9.2)
        self.ax.set_ylim(-4.8, 4.8)
        self.ax.set_aspect("equal")
        self.ax.set_title("Quarter-circle Road Imitation Learning")
        self.ax.set_facecolor("#e5e7eb")
        self.fig.patch.set_facecolor("#cfd8e3")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")

    def _build_scene(self) -> None:
        x = np.linspace(self.road_x_min, self.road_x_max, 500)
        y = self._road_center(x)
        self.ax.plot(x, y, "--", color="#f59e0b", lw=6.0, alpha=0.95, zorder=2)

        self.robot_circle = Circle((self.state.x, self.state.y), radius=0.22, color="#ef4444", zorder=6)
        self.ax.add_patch(self.robot_circle)
        self.robot_arrow, = self.ax.plot([], [], color="#111827", lw=3.0, zorder=7)

        self.obs_box = Rectangle(
            (self.state.x - 0.5, self.state.y - 0.5),
            1.0,
            1.0,
            fill=False,
            ec="#22c55e",
            lw=1.8,
            ls="--",
            zorder=8,
        )
        self.ax.add_patch(self.obs_box)

        self.ax_obs = self.fig.add_axes([0.04, 0.79, 0.22, 0.12])
        self.ax_obs.set_facecolor("#0f172a")
        self.ax_obs.set_xticks([])
        self.ax_obs.set_yticks([])
        self.ax_obs.set_title("Current Input", fontsize=8, color="#e5e7eb")
        blank = np.zeros((84, 84, 3), dtype=np.float32)
        self.obs_im = self.ax_obs.imshow(blank, interpolation="nearest")

    def _build_buttons(self) -> None:
        x0 = 0.75
        w = 0.22
        h = 0.085
        gap = 0.018
        y0 = 0.83

        ax_collect = self.fig.add_axes([x0, y0 - 0 * (h + gap), w, h])
        ax_act_train = self.fig.add_axes([x0, y0 - 1 * (h + gap), w, h])
        ax_act_infer = self.fig.add_axes([x0, y0 - 2 * (h + gap), w, h])
        ax_stop = self.fig.add_axes([x0, y0 - 3 * (h + gap), w, h])
        ax_data_view = self.fig.add_axes([x0, y0 - 4 * (h + gap), w, h])
        ax_clear = self.fig.add_axes([x0, y0 - 5 * (h + gap), w, h])

        self.btn_collect = Button(ax_collect, "Collect Data", color="#dbeafe", hovercolor="#bfdbfe")
        self.btn_act_train = Button(ax_act_train, "ACT Train", color="#fef3c7", hovercolor="#fde68a")
        self.btn_act_infer = Button(ax_act_infer, "ACT Inference", color="#ede9fe", hovercolor="#ddd6fe")
        self.btn_stop = Button(ax_stop, "Stop Inference", color="#f3f4f6", hovercolor="#e5e7eb")
        self.btn_data_view = Button(ax_data_view, "Data View", color="#dcfce7", hovercolor="#bbf7d0")
        self.btn_clear = Button(ax_clear, "Clear Datasets", color="#fee2e2", hovercolor="#fecaca")

        self.btn_collect.on_clicked(self._on_collect_clicked)
        self.btn_act_train.on_clicked(self._on_act_train_clicked)
        self.btn_act_infer.on_clicked(self._on_act_infer_clicked)
        self.btn_stop.on_clicked(self._on_stop_inference_clicked)
        self.btn_data_view.on_clicked(self._on_data_view_clicked)
        self.btn_clear.on_clicked(self._on_clear_datasets_clicked)

    def _build_status_text(self) -> None:
        self.dataset_count_text = self.fig.text(0.75, 0.22, "Data Sets: 0", fontsize=10, va="top")
        self.status_text = self.fig.text(
            0.75,
            0.17,
            "Status: Idle\nControls: W/S speed, A/D turn\nSpace: stop collect/inference",
            fontsize=10,
            va="top",
        )

    def _connect_events(self) -> None:
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("key_release_event", self._on_key_release)
        self.fig.canvas.mpl_connect("button_press_event", self._on_any_mouse_press)

    def _set_status(self, msg: str) -> None:
        self.status_text.set_text(msg)

    def _focus_canvas(self) -> None:
        # Backend-agnostic best-effort focus restore for reliable key events.
        canvas = self.fig.canvas

        def _try_focus(obj) -> bool:
            if obj is None:
                return False
            for name in ("setFocus", "focus_set", "activateWindow", "raise_"):
                fn = getattr(obj, name, None)
                if callable(fn):
                    try:
                        fn()
                        return True
                    except Exception:
                        pass
            return False

        if _try_focus(canvas):
            return

        get_tk_widget = getattr(canvas, "get_tk_widget", None)
        if callable(get_tk_widget) and _try_focus(get_tk_widget()):
            return

        manager = getattr(canvas, "manager", None)
        if _try_focus(getattr(manager, "window", None)):
            return
        _try_focus(manager)

    def _on_any_mouse_press(self, _event) -> None:
        self._focus_canvas()

    def _list_dataset_files(self) -> List[Path]:
        return sorted(self.dataset_dir.glob("set_*.npz"), key=lambda p: p.name)

    @staticmethod
    def _extract_set_index(path: Path) -> int:
        stem = path.stem
        # set_0001 -> 1
        if "_" not in stem:
            return 0
        try:
            return int(stem.split("_")[-1])
        except ValueError:
            return 0

    def _next_dataset_path(self) -> Path:
        files = self._list_dataset_files()
        if not files:
            return self.dataset_dir / "set_0001.npz"
        idx = max(self._extract_set_index(p) for p in files) + 1
        return self.dataset_dir / f"set_{idx:04d}.npz"

    def _refresh_dataset_count(self) -> None:
        self.dataset_files = self._list_dataset_files()
        self.dataset_count_text.set_text(f"Data Sets: {len(self.dataset_files)}")

    def _clear_runtime_dataset_cache(self, *, reset_collect_counter: bool = False) -> None:
        self.images.clear()
        self.actions.clear()
        self._set_lengths.clear()
        if reset_collect_counter:
            self.collect_frame_idx = 0

    @staticmethod
    def _normalize_dataset_arrays(
        images: np.ndarray, actions: np.ndarray
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if images.ndim != 4 or images.shape[-1] != 3:
            return None
        if actions.ndim != 2 or actions.shape[1] < 2:
            return None
        n = min(len(images), len(actions))
        if n == 0:
            return None
        return images[:n], actions[:n, :2].astype(np.float32, copy=False)

    def _load_dataset_set_file(self, path: Path) -> Optional[tuple[np.ndarray, np.ndarray, str]]:
        try:
            with np.load(path) as d:
                src = str(d["image_source"]) if "image_source" in d else "legacy"
                if "images" not in d or "actions" not in d:
                    return None
                images = np.asarray(d["images"])
                actions = np.asarray(d["actions"])
        except Exception:
            return None

        normalized = self._normalize_dataset_arrays(images, actions)
        if normalized is None:
            return None
        images_n, actions_n = normalized
        return images_n, actions_n, src

    def _on_data_view_clicked(self, _event) -> None:
        self._focus_canvas()
        self.data_view.show_or_focus()

    def _on_data_view_dataset_deleted(self, _path: Path) -> None:
        self._clear_runtime_dataset_cache(reset_collect_counter=True)
        self._refresh_dataset_count()

    def _refresh_data_view_if_open(self) -> None:
        self.data_view.refresh_if_open()

    def _snapshot_scene_rgb(self) -> Optional[np.ndarray]:
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy()
        canvas_w, canvas_h = self.fig.canvas.get_width_height()
        buf_h, buf_w = buf.shape[0], buf.shape[1]

        bbox = self.ax.bbox
        bbox_in_display = (bbox.x1 <= canvas_w * 1.05) and (bbox.y1 <= canvas_h * 1.05)
        if bbox_in_display:
            scale_x = buf_w / max(canvas_w, 1e-6)
            scale_y = buf_h / max(canvas_h, 1e-6)
        else:
            scale_x = 1.0
            scale_y = 1.0

        x0 = int(max(0, round(bbox.x0 * scale_x)))
        y0 = int(max(0, round(bbox.y0 * scale_y)))
        x1 = int(min(buf_w, round(bbox.x1 * scale_x)))
        y1 = int(min(buf_h, round(bbox.y1 * scale_y)))
        if x1 <= x0 or y1 <= y0:
            return None

        y0_img = buf_h - y1
        y1_img = buf_h - y0
        return buf[y0_img:y1_img, x0:x1, :].copy()

    def _state_to_scene_pixel(self, scene: np.ndarray) -> tuple[int, int]:
        data_disp = self.ax.transData.transform((self.state.x, self.state.y))
        bbox = self.ax.bbox
        h, w, _ = scene.shape
        u = (data_disp[0] - bbox.x0) / max(bbox.width, 1e-6)
        v = (bbox.y1 - data_disp[1]) / max(bbox.height, 1e-6)
        px = int(np.clip(round(u * (w - 1)), 0, w - 1))
        py = int(np.clip(round(v * (h - 1)), 0, h - 1))
        return px, py

    def _scene_to_chw_tensor(self, scene: np.ndarray) -> torch.Tensor:
        obs = self._extract_local_observation(scene)
        img = obs.astype(np.float32) / 255.0
        x_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        x_t = F.interpolate(x_t, size=(84, 84), mode="bilinear", align_corners=False)
        return x_t[0]

    def _extract_local_observation(self, scene: np.ndarray) -> np.ndarray:
        h, w, _ = scene.shape
        px, py = self._state_to_scene_pixel(scene)

        hs = self.obs_patch_size // 2
        x0 = px - hs
        x1 = px + hs
        y0 = py - hs
        y1 = py + hs

        pad_l = max(0, -x0)
        pad_r = max(0, x1 - w)
        pad_t = max(0, -y0)
        pad_b = max(0, y1 - h)

        if pad_l or pad_r or pad_t or pad_b:
            scene = np.pad(scene, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="edge")
            x0 += pad_l
            x1 += pad_l
            y0 += pad_t
            y1 += pad_t

        return scene[y0:y1, x0:x1, :]

    def _update_observation_overlay(self, scene: Optional[np.ndarray]) -> None:
        if scene is None:
            return

        obs = self._extract_local_observation(scene)
        self.obs_im.set_data(obs.astype(np.float32) / 255.0)

        h, w, _ = scene.shape
        bbox = self.ax.bbox
        scale_x = bbox.width / max(w, 1e-6)
        scale_y = bbox.height / max(h, 1e-6)
        hs = self.obs_patch_size / 2.0

        data_disp = self.ax.transData.transform((self.state.x, self.state.y))
        x0_disp = data_disp[0] - hs * scale_x
        x1_disp = data_disp[0] + hs * scale_x
        y0_disp = data_disp[1] - hs * scale_y
        y1_disp = data_disp[1] + hs * scale_y

        inv = self.ax.transData.inverted()
        x0_data, y0_data = inv.transform((x0_disp, y0_disp))
        x1_data, y1_data = inv.transform((x1_disp, y1_disp))
        x_lo, x_hi = sorted((float(x0_data), float(x1_data)))
        y_lo, y_hi = sorted((float(y0_data), float(y1_data)))
        self.obs_box.set_xy((x_lo, y_lo))
        self.obs_box.set_width(x_hi - x_lo)
        self.obs_box.set_height(y_hi - y_lo)

    def _prepare_frame_dataset(self) -> tuple[torch.Tensor, torch.Tensor]:
        images = np.stack(self.images).astype(np.float32) / 255.0
        actions = np.stack(self.actions).astype(np.float32)

        x_t = torch.from_numpy(images).permute(0, 3, 1, 2)
        x_t = F.interpolate(x_t, size=(84, 84), mode="bilinear", align_corners=False)
        y_t = torch.from_numpy(actions)
        return x_t, y_t

    def _prepare_sequence_dataset(self, x_t: torch.Tensor, y_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.act_seq_len
        xs: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        start = 0
        for length in self._set_lengths:
            if length < t:
                start += length
                continue
            for i in range(start, start + length - t + 1):
                xs.append(x_t[i : i + t])
                ys.append(y_t[i : i + t])
            start += length

        if not xs:
            raise ValueError(f"Need at least {t} contiguous samples in a set for ACT training")
        return torch.stack(xs), torch.stack(ys)

    def _reset_robot(self) -> None:
        self.state = self._initial_state()
        self.infer_action[:] = 0.0
        self.act_frame_buffer.clear()

    def _on_collect_clicked(self, _event) -> None:
        if self.mode == "collect":
            self._finish_collection()
            self._focus_canvas()
            return

        self.mode = "collect"
        self.motion_blocked = False
        self._clear_runtime_dataset_cache(reset_collect_counter=True)
        self.keys_down.clear()
        self._reset_robot()
        self._focus_canvas()
        self._set_status(
            "Status: Collecting one set\nW/S speed, A/D turn\nPress Space (or Collect Data again) to save this set"
        )

    def _load_all_dataset_sets(self) -> bool:
        files = self._list_dataset_files()
        if not files:
            return False

        imgs_all: List[np.ndarray] = []
        acts_all: List[np.ndarray] = []
        lengths: List[int] = []
        skipped = 0

        for p in files:
            loaded = self._load_dataset_set_file(p)
            if loaded is None:
                continue
            imgs, acts, src = loaded
            if src != self.dataset_image_source:
                skipped += 1
                continue

            imgs_all.append(imgs)
            acts_all.append(acts)
            lengths.append(int(len(imgs)))

        if not imgs_all:
            if skipped > 0:
                self._set_status(
                    "Status: ACT Train blocked\nDataset format mismatch\nPlease recollect new sets"
                )
            return False

        imgs_cat = np.concatenate(imgs_all, axis=0)
        acts_cat = np.concatenate(acts_all, axis=0)
        self.images = list(imgs_cat)
        self.actions = list(acts_cat)
        self._set_lengths = lengths

        if skipped > 0:
            self._set_status(
                f"Status: Loaded with warnings\nLoaded sets: {len(lengths)} | Skipped: {skipped}"
            )
        return True

    def _print_action_stats(self, actions: np.ndarray) -> None:
        turn_abs_mean = float(np.mean(np.abs(actions[:, 1])))
        vel_abs_mean = float(np.mean(np.abs(actions[:, 0])))
        zero_ratio = float(np.mean(np.linalg.norm(actions, axis=1) < 1e-6))
        left_ratio = float(np.mean(actions[:, 1] > 0.2))
        right_ratio = float(np.mean(actions[:, 1] < -0.2))
        straight_ratio = float(np.mean(np.abs(actions[:, 1]) < 0.05))

        self._last_data_warning = None
        if turn_abs_mean < 0.25 or (left_ratio < 0.12 or right_ratio < 0.12):
            self._last_data_warning = "Data warning: steering diversity is low."

        print(
            f"[train] action stats | mean|v|={vel_abs_mean:.3f} mean|w|={turn_abs_mean:.3f} "
            f"zero-action ratio={zero_ratio:.3f} left={left_ratio:.3f} right={right_ratio:.3f} "
            f"straight={straight_ratio:.3f}"
        )
        if self._last_data_warning is not None:
            print(f"[train] {self._last_data_warning}")

    def _train_model(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
        *,
        batch_size: int,
        epochs: int,
        tag: str,
    ) -> tuple[nn.Module, float]:
        ds = TensorDataset(x_t, y_t)
        dl = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        model.train()
        last_loss = 0.0
        for ep in range(1, epochs + 1):
            total = 0.0
            n = 0
            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                pred = model(xb)
                loss = F.mse_loss(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(xb)
                n += len(xb)
            last_loss = total / max(n, 1)
            print(f"[{tag}] epoch {ep:02d}/{epochs} loss={last_loss:.6f}")
        return model, last_loss

    def _set_train_done_status(self, last_loss: float) -> None:
        if self._last_data_warning is None:
            self._set_status(f"Status: ACT training done\nFinal loss: {last_loss:.5f}")
            return
        self._set_status(
            f"Status: ACT training done\nFinal loss: {last_loss:.5f}\n{self._last_data_warning}"
        )

    def _start_inference(self) -> None:
        self.mode = "infer"
        self.motion_blocked = False
        self.keys_down.clear()
        self._reset_robot()
        self._set_status("Status: ACT inference running\nPress Stop Inference or Space")

    def _on_act_train_clicked(self, _event) -> None:
        self._focus_canvas()
        if self.mode == "collect":
            self._finish_collection()

        if not self._load_all_dataset_sets():
            self._set_status("Status: ACT Train failed\nNo valid dataset sets found")
            return

        set_count = len(self._set_lengths)
        sample_count = len(self.images)
        self._set_status(f"Status: ACT training started\nSets: {set_count} | Samples: {sample_count}")
        self.fig.canvas.draw_idle()

        x_t, y_t = self._prepare_frame_dataset()
        self._print_action_stats(y_t.numpy())

        try:
            x_seq, y_seq = self._prepare_sequence_dataset(x_t, y_t)
        except ValueError as exc:
            self._set_status(f"Status: ACT Train failed\n{exc}")
            return

        model, last_loss = self._train_model(
            ACTPolicy().to(self.device),
            x_seq,
            y_seq,
            batch_size=48,
            epochs=30,
            tag="act train",
        )

        self.act_model = model
        self.act_ready = True
        self.mode = "idle"
        self._set_train_done_status(last_loss)

    def _on_act_infer_clicked(self, _event) -> None:
        self._focus_canvas()
        if not self.act_ready or self.act_model is None:
            self._set_status("Status: ACT inference failed\nTrain ACT model first")
            return
        self._start_inference()

    def _on_stop_inference_clicked(self, _event) -> None:
        self._focus_canvas()
        self.mode = "idle"
        self.motion_blocked = True
        self.keys_down.clear()
        self.infer_action[:] = 0.0
        self.act_frame_buffer.clear()
        self._set_status("Status: Inference stopped\nRobot motion blocked")

    def _on_clear_datasets_clicked(self, _event) -> None:
        self._focus_canvas()
        count = 0
        for p in self._list_dataset_files():
            p.unlink(missing_ok=True)
            count += 1
        self._clear_runtime_dataset_cache(reset_collect_counter=True)
        self._refresh_dataset_count()
        self._refresh_data_view_if_open()
        self._set_status(f"Status: Cleared datasets\nRemoved sets: {count}")

    def _finish_collection(self) -> None:
        self.mode = "idle"
        self.keys_down.clear()

        if len(self.images) == 0:
            self._reset_robot()
            self._set_status("Status: Collection stopped\nNo samples captured")
            return

        arr_img = np.stack(self.images).astype(np.uint8)
        arr_act = np.stack(self.actions).astype(np.float32)
        out_path = self._next_dataset_path()
        np.savez_compressed(
            out_path,
            images=arr_img,
            actions=arr_act,
            image_source=self.dataset_image_source,
        )

        saved_n = len(arr_img)
        self._clear_runtime_dataset_cache(reset_collect_counter=True)
        self._refresh_dataset_count()
        self._refresh_data_view_if_open()
        self._reset_robot()
        self._set_status(
            f"Status: Collection set saved\nSet: {out_path.name} | Samples: {saved_n}\n"
            f"Total sets: {len(self.dataset_files)}"
        )

    def _on_key_press(self, event) -> None:
        key = (event.key or "").lower()
        if key in {"w", "s", "a", "d"}:
            self.keys_down.add(key)
            return

        if key in {" ", "space"}:
            if self.mode == "collect":
                self._finish_collection()
            elif self.mode == "infer":
                self._on_stop_inference_clicked(None)

    def _on_key_release(self, event) -> None:
        key = (event.key or "").lower()
        if key in self.keys_down:
            self.keys_down.remove(key)

    def _manual_action(self) -> np.ndarray:
        v = 0.0
        w = 0.0
        if "w" in self.keys_down:
            v += 1.0
        if "s" in self.keys_down:
            v -= 1.0
        if "a" in self.keys_down:
            w += 1.0
        if "d" in self.keys_down:
            w -= 1.0
        return np.array([v, w], dtype=np.float32)

    @torch.no_grad()
    def _predict_act_action(self, scene: np.ndarray) -> np.ndarray:
        if self.act_model is None:
            return np.zeros(2, dtype=np.float32)

        cur = self._scene_to_chw_tensor(scene)
        self.act_frame_buffer.append(cur)
        if len(self.act_frame_buffer) > self.act_seq_len:
            self.act_frame_buffer = self.act_frame_buffer[-self.act_seq_len :]

        if len(self.act_frame_buffer) < self.act_seq_len:
            pad = [self.act_frame_buffer[0]] * (self.act_seq_len - len(self.act_frame_buffer))
            seq = pad + self.act_frame_buffer
        else:
            seq = self.act_frame_buffer

        x_seq = torch.stack(seq).unsqueeze(0).to(self.device)
        pred_seq = self.act_model(x_seq).cpu().numpy()[0]
        return pred_seq[-1].astype(np.float32)

    def _step_robot(self, action: np.ndarray) -> None:
        v = float(np.clip(action[0], -1.0, 1.0)) * self.v_scale
        w = float(np.clip(action[1], -1.0, 1.0)) * self.w_scale

        self.state.yaw += w * self.dt
        self.state.x += v * np.cos(self.state.yaw) * self.dt
        self.state.y += v * np.sin(self.state.yaw) * self.dt

        self.state.x = float(np.clip(self.state.x, -0.6, 9.0))
        self.state.y = float(np.clip(self.state.y, -4.6, 4.6))

    def _update_robot_artist(self) -> None:
        self.robot_circle.center = (self.state.x, self.state.y)
        tip_x = self.state.x + self.arrow_len * np.cos(self.state.yaw)
        tip_y = self.state.y + self.arrow_len * np.sin(self.state.yaw)
        self.robot_arrow.set_data([self.state.x, tip_x], [self.state.y, tip_y])

    def _capture_sample_if_needed(self, action: np.ndarray, scene: Optional[np.ndarray]) -> None:
        if self.mode != "collect":
            return

        self.collect_frame_idx += 1
        if scene is None:
            return

        keep = bool(np.linalg.norm(action) > 1e-6 or (self.collect_frame_idx % 5 == 0))
        if not keep:
            return

        obs = self._extract_local_observation(scene)
        self.images.append(obs)
        self.actions.append(action.astype(np.float32).copy())

    def _update(self, _frame_idx: int):
        scene = self._snapshot_scene_rgb()
        self._update_observation_overlay(scene)

        if self.mode == "infer":
            pred = self._predict_act_action(scene) if scene is not None else np.zeros(2, dtype=np.float32)
            self.infer_action = 0.15 * self.infer_action + 0.85 * pred
            action = self.infer_action
            self._set_status(
                "Status: ACT inference running\n"
                f"Model(v,w)=({pred[0]:+.2f},{pred[1]:+.2f}) "
                f"Apply=({action[0]:+.2f},{action[1]:+.2f})"
            )
        elif self.motion_blocked:
            action = np.zeros(2, dtype=np.float32)
        else:
            action = self._manual_action()

        self._capture_sample_if_needed(action, scene)
        self._step_robot(action)
        self._update_robot_artist()

        return self.robot_circle, self.robot_arrow, self.obs_im, self.obs_box

    def run(self) -> None:
        plt.show()


if __name__ == "__main__":
    app = ImitationRoadDemo()
    app.run()
