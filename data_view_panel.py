from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button


@dataclass
class DataViewState:
    fig: Optional[object] = None
    ax_v: Optional[object] = None
    ax_w: Optional[object] = None
    ax_preview: Optional[object] = None
    im: Optional[object] = None
    info_text: Optional[object] = None
    summary_text: Optional[object] = None
    msg_text: Optional[object] = None
    timer: Optional[object] = None
    btn_prev: Optional[Button] = None
    btn_next: Optional[Button] = None
    btn_play: Optional[Button] = None
    btn_delete: Optional[Button] = None
    btn_refresh: Optional[Button] = None
    files: List[Path] = field(default_factory=list)
    selected_idx: int = 0
    frame_idx: int = 0
    playing: bool = False
    images: Optional[np.ndarray] = None
    actions: Optional[np.ndarray] = None


class DataViewPanel:
    def __init__(
        self,
        *,
        dt: float,
        list_dataset_files: Callable[[], List[Path]],
        load_dataset_set_file: Callable[[Path], Optional[tuple[np.ndarray, np.ndarray, str]]],
        on_dataset_deleted: Callable[[Path], None],
        refresh_main_datasets: Callable[[], None],
        set_main_status: Callable[[str], None],
    ) -> None:
        self.dt = dt
        self._list_dataset_files = list_dataset_files
        self._load_dataset_set_file = load_dataset_set_file
        self._on_dataset_deleted = on_dataset_deleted
        self._refresh_main_datasets = refresh_main_datasets
        self._set_main_status = set_main_status
        self.state = DataViewState()

    def show_or_focus(self) -> None:
        if self.state.fig is not None and plt.fignum_exists(self.state.fig.number):
            self._refresh()
            self._raise_figure_window(self.state.fig)
            self._draw_idle()
            return

        self._build_window()
        self._refresh()
        self._draw_idle()

    def refresh_if_open(self) -> None:
        if self.state.fig is None:
            return
        if not plt.fignum_exists(self.state.fig.number):
            self._on_close(None)
            return
        self._refresh()
        self._draw_idle()

    def _set_message(self, msg: str) -> None:
        if self.state.msg_text is not None:
            self.state.msg_text.set_text(msg)

    def _set_play_label(self) -> None:
        if self.state.btn_play is not None:
            self.state.btn_play.label.set_text("Pause" if self.state.playing else "Play")

    def _draw_idle(self) -> None:
        if self.state.fig is not None:
            self.state.fig.canvas.draw_idle()

    @staticmethod
    def _raise_figure_window(fig) -> None:
        manager = getattr(fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "show"):
            manager.show()
        window = getattr(manager, "window", None)
        for name in ("raise_", "activateWindow"):
            fn = getattr(window, name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass

    def _build_window(self) -> None:
        self.state.fig = plt.figure(figsize=(10.4, 6.3))
        manager = getattr(self.state.fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title("Data View")
        self.state.fig.patch.set_facecolor("#cfd8e3")

        self.state.ax_v = self.state.fig.add_axes([0.07, 0.58, 0.39, 0.34])
        self.state.ax_w = self.state.fig.add_axes([0.54, 0.58, 0.39, 0.34])
        self.state.ax_preview = self.state.fig.add_axes([0.07, 0.08, 0.39, 0.42])
        self.state.ax_preview.set_facecolor("#0f172a")
        self.state.ax_preview.set_xticks([])
        self.state.ax_preview.set_yticks([])
        self.state.ax_preview.set_title("Selected Set Preview", fontsize=10, color="#e5e7eb")
        blank = np.zeros((84, 84, 3), dtype=np.float32)
        self.state.im = self.state.ax_preview.imshow(blank, interpolation="nearest")

        self.state.summary_text = self.state.fig.text(0.54, 0.50, "Sets: 0 | Samples: 0", fontsize=10, va="top")
        self.state.info_text = self.state.fig.text(0.54, 0.45, "Selected: -", fontsize=10, va="top")
        self.state.msg_text = self.state.fig.text(0.54, 0.11, "Data View ready", fontsize=10, va="top")

        ax_prev = self.state.fig.add_axes([0.54, 0.34, 0.12, 0.08])
        ax_next = self.state.fig.add_axes([0.68, 0.34, 0.12, 0.08])
        ax_play = self.state.fig.add_axes([0.82, 0.34, 0.12, 0.08])
        ax_delete = self.state.fig.add_axes([0.54, 0.24, 0.40, 0.08])
        ax_refresh = self.state.fig.add_axes([0.54, 0.14, 0.40, 0.08])

        self.state.btn_prev = Button(ax_prev, "Prev Set", color="#e2e8f0", hovercolor="#cbd5e1")
        self.state.btn_next = Button(ax_next, "Next Set", color="#e2e8f0", hovercolor="#cbd5e1")
        self.state.btn_play = Button(ax_play, "Play", color="#dbeafe", hovercolor="#bfdbfe")
        self.state.btn_delete = Button(ax_delete, "Delete Selected Set", color="#fee2e2", hovercolor="#fecaca")
        self.state.btn_refresh = Button(ax_refresh, "Refresh", color="#dcfce7", hovercolor="#bbf7d0")

        self.state.btn_prev.on_clicked(self._on_prev_clicked)
        self.state.btn_next.on_clicked(self._on_next_clicked)
        self.state.btn_play.on_clicked(self._on_play_clicked)
        self.state.btn_delete.on_clicked(self._on_delete_clicked)
        self.state.btn_refresh.on_clicked(self._on_refresh_clicked)

        self.state.fig.canvas.mpl_connect("close_event", self._on_close)
        self.state.timer = self.state.fig.canvas.new_timer(interval=int(self.dt * 1000))
        self.state.timer.add_callback(self._on_timer)
        self.state.timer.start()

    def _on_close(self, _event) -> None:
        if self.state.timer is not None:
            try:
                self.state.timer.stop()
            except Exception:
                pass
        self.state = DataViewState()

    @staticmethod
    def _draw_hist(ax, values: np.ndarray, title: str, color: str) -> None:
        if ax is None:
            return
        ax.clear()
        ax.set_facecolor("#f8fafc")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        if len(values) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            return
        ax.hist(values, bins=30, color=color, alpha=0.85, edgecolor="#0f172a", linewidth=0.35)
        ax.grid(alpha=0.2)

    def _refresh(self) -> None:
        if self.state.fig is None:
            return

        valid_files: List[Path] = []
        all_v: List[np.ndarray] = []
        all_w: List[np.ndarray] = []
        total_samples = 0

        for p in self._list_dataset_files():
            loaded = self._load_dataset_set_file(p)
            if loaded is None:
                continue
            _, acts, _ = loaded
            valid_files.append(p)
            total_samples += len(acts)
            all_v.append(acts[:, 0])
            all_w.append(acts[:, 1])

        self.state.files = valid_files
        v_values = np.concatenate(all_v) if all_v else np.array([], dtype=np.float32)
        w_values = np.concatenate(all_w) if all_w else np.array([], dtype=np.float32)
        self._draw_hist(self.state.ax_v, v_values, "Linear Speed Distribution", "#2563eb")
        self._draw_hist(self.state.ax_w, w_values, "Angular Speed Distribution", "#7c3aed")

        if self.state.summary_text is not None:
            self.state.summary_text.set_text(f"Sets: {len(valid_files)} | Samples: {total_samples}")

        if not valid_files:
            self.state.selected_idx = 0
            self.state.frame_idx = 0
            self.state.playing = False
            self._set_play_label()
            self.state.images = None
            self.state.actions = None
            self._update_preview()
            self._set_message("No datasets found")
            return

        self.state.selected_idx = int(np.clip(self.state.selected_idx, 0, len(valid_files) - 1))
        self._load_selected_dataset()
        self._update_preview()

    def _load_selected_dataset(self) -> None:
        self.state.frame_idx = 0
        self.state.playing = False
        self._set_play_label()

        if not self.state.files:
            self.state.images = None
            self.state.actions = None
            return

        path = self.state.files[self.state.selected_idx]
        loaded = self._load_dataset_set_file(path)
        if loaded is None:
            self.state.images = None
            self.state.actions = None
            self._set_message(f"Load failed: {path.name}")
            return

        imgs, acts, _ = loaded
        self.state.images = imgs
        self.state.actions = acts
        self._set_message(f"Loaded set: {path.name}")

    def _update_preview(self) -> None:
        if self.state.ax_preview is None or self.state.im is None:
            return

        if not self.state.files:
            blank = np.zeros((84, 84, 3), dtype=np.float32)
            self.state.im.set_data(blank)
            self.state.ax_preview.set_title("Selected Set Preview", fontsize=10, color="#e5e7eb")
            if self.state.info_text is not None:
                self.state.info_text.set_text("Selected: -")
            return

        path = self.state.files[self.state.selected_idx]
        if self.state.images is None or self.state.actions is None or len(self.state.images) == 0:
            blank = np.zeros((84, 84, 3), dtype=np.float32)
            self.state.im.set_data(blank)
            self.state.ax_preview.set_title("Selected Set Preview (empty)", fontsize=10, color="#e5e7eb")
            if self.state.info_text is not None:
                self.state.info_text.set_text(f"Selected: {path.name}\nSamples: 0")
            return

        self.state.frame_idx %= len(self.state.images)
        frame = self.state.images[self.state.frame_idx].astype(np.float32) / 255.0
        self.state.im.set_data(frame)
        v, w = self.state.actions[self.state.frame_idx]
        self.state.ax_preview.set_title(
            f"Selected Set Preview  Frame {self.state.frame_idx + 1}/{len(self.state.images)}",
            fontsize=10,
            color="#e5e7eb",
        )
        if self.state.info_text is not None:
            self.state.info_text.set_text(f"Selected: {path.name}\nAction(v,w)=({v:+.2f}, {w:+.2f})")

    def _on_prev_clicked(self, _event) -> None:
        if not self.state.files:
            self._set_message("No dataset to select")
            return
        self.state.selected_idx = (self.state.selected_idx - 1) % len(self.state.files)
        self._load_selected_dataset()
        self._update_preview()
        self._draw_idle()

    def _on_next_clicked(self, _event) -> None:
        if not self.state.files:
            self._set_message("No dataset to select")
            return
        self.state.selected_idx = (self.state.selected_idx + 1) % len(self.state.files)
        self._load_selected_dataset()
        self._update_preview()
        self._draw_idle()

    def _on_play_clicked(self, _event) -> None:
        if self.state.images is None or len(self.state.images) == 0:
            self._set_message("Selected dataset has no playable frames")
            return
        self.state.playing = not self.state.playing
        self._set_play_label()
        self._set_message("Playing selected set" if self.state.playing else "Playback paused")
        self._draw_idle()

    def _on_delete_clicked(self, _event) -> None:
        if not self.state.files:
            self._set_message("No dataset to delete")
            return

        target = self.state.files[self.state.selected_idx]
        try:
            target.unlink(missing_ok=True)
        except Exception as exc:
            self._set_message(f"Delete failed: {target.name}\n{exc}")
            return

        self._on_dataset_deleted(target)
        self.state.playing = False
        self._set_play_label()
        self._refresh()
        self._set_main_status(f"Status: Dataset deleted\nDeleted: {target.name}")
        self._set_message(f"Deleted set: {target.name}")
        self._draw_idle()

    def _on_refresh_clicked(self, _event) -> None:
        self._refresh_main_datasets()
        self._refresh()
        self._set_message("Data View refreshed")
        self._draw_idle()

    def _on_timer(self) -> None:
        if self.state.fig is None:
            return
        if not plt.fignum_exists(self.state.fig.number):
            self._on_close(None)
            return
        if not self.state.playing:
            return
        if self.state.images is None or len(self.state.images) == 0:
            self.state.playing = False
            self._set_play_label()
            return
        self.state.frame_idx = (self.state.frame_idx + 1) % len(self.state.images)
        self._update_preview()
        self._draw_idle()
