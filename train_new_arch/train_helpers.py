"""Training helpers for DGRN-X-v0.2 on Google Colab T4.

All training logic: config, dataset, trainer, checkpointing, resume.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset


# ========================= CONFIG =========================

@dataclass
class TrainConfig:
    """All hyperparameters. Edit in notebook before training."""

    # --- Run identity ---
    run_name: str = "dgrn_x_v02_run1"

    # --- Paths (Colab Google Drive) ---
    data_root: str = "/content/drive/MyDrive/chess_engine/data/process"
    repo_root: str = "/content/drive/MyDrive/chess_engine"
    runs_root: str = "/content/drive/MyDrive/chess_engine/runs"

    # --- Model (must match architecture_v3 defaults) ---
    input_channels: int = 23
    width: int = 192
    board_size: int = 8
    grid_blocks: int = 12
    relation_blocks: int = 4

    # --- Training ---
    epochs: int = 30
    batch_size: int = 1024          # T4 14GB handles 1024 with AMP easily
    grad_accum_steps: int = 1       # effective batch = 1024 (no accum = fastest)
    num_workers: int = 4            # Colab typically has 2-4 CPU cores
    pin_memory: bool = True
    persistent_workers: bool = True # keep workers alive between epochs

    # --- Optimizer ---
    learning_rate: float = 1e-3
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0

    # --- Scheduler ---
    warmup_epochs: int = 1
    scheduler_T0: int = 5           # CosineAnnealingWarmRestarts T_0 (epochs)
    scheduler_T_mult: int = 2       # cycle length multiplier

    # --- Loss weights ---
    lambda_phase: float = 0.1
    lambda_material: float = 0.1

    # --- AMP ---
    use_amp: bool = True

    # --- Logging & checkpointing ---
    log_every_steps: int = 100
    eval_every_epoch: int = 1
    save_every_epoch: int = 1

    # --- Early stopping (0 = disabled) ---
    patience: int = 10

    # --- Resume ---
    resume_from: Optional[str] = None

    # --- Reproducibility ---
    seed: int = 42


# ========================= UTILS =========================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True  # auto-tune conv kernels for T4


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"[GPU] {name} | {mem:.1f} GB VRAM")
    else:
        dev = torch.device("cpu")
        print("[CPU] No GPU detected")
    return dev


def build_model(cfg: TrainConfig) -> nn.Module:
    """Build DGRN-X-v0.2 model from config."""
    import sys
    repo = cfg.repo_root
    if repo not in sys.path:
        sys.path.insert(0, repo)
    model_dir = str(Path(repo) / "model")
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    from architecture_v3.model import DGRNXv0Model
    return DGRNXv0Model(
        input_channels=cfg.input_channels,
        width=cfg.width,
        board_size=cfg.board_size,
        grid_blocks=cfg.grid_blocks,
        relation_blocks=cfg.relation_blocks,
    )


# ========================= DATASET =========================

class ShardDataset(Dataset):
    """Loads sharded .npy files with mmap for memory efficiency.

    Expects files like: shard_0000_X.npy, shard_0000_y.npy
    Or any pattern: *_X.npy paired with *_y.npy

    Auxiliary targets (phase, material) are extracted from X tensor:
      - phase = X[19, 0, 0]        (game phase, range [0, 1])
      - material = X[22, 0, 0]     (material delta, range [-1, 1])
    """

    def __init__(self, data_dir: str, max_samples: Optional[int] = None) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Discover shard pairs
        x_files = sorted(self.data_dir.glob("*_X.npy"))
        if not x_files:
            x_files = sorted(self.data_dir.glob("*_x.npy"))
        if not x_files:
            raise FileNotFoundError(f"No X shard files found in {data_dir}")

        self.x_arrays: List[np.ndarray] = []
        self.y_arrays: List[np.ndarray] = []
        lengths: List[int] = []

        for xf in x_files:
            # Find matching y file
            yf = xf.parent / xf.name.replace("_X.npy", "_y.npy").replace("_x.npy", "_y.npy")
            if not yf.exists():
                print(f"[WARN] No matching y file for {xf.name}, skipping")
                continue

            x_arr = np.load(str(xf), mmap_mode="r")
            y_arr = np.load(str(yf), mmap_mode="r")

            if x_arr.shape[0] != y_arr.shape[0]:
                print(f"[WARN] Shape mismatch {xf.name}: X={x_arr.shape[0]} vs y={y_arr.shape[0]}, skipping")
                continue

            self.x_arrays.append(x_arr)
            self.y_arrays.append(y_arr)
            lengths.append(x_arr.shape[0])

        if not lengths:
            raise RuntimeError(f"No valid shard pairs found in {data_dir}")

        self.cumulative = np.cumsum([0] + lengths)
        self.total = int(self.cumulative[-1])

        if max_samples and max_samples < self.total:
            self.total = max_samples

        print(f"[DATA] {data_dir}: {len(lengths)} shards, {self.total:,} samples")

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Find shard via binary search on cumulative index
        shard_idx = int(np.searchsorted(self.cumulative[1:], idx, side="right"))
        local_idx = idx - int(self.cumulative[shard_idx])

        x = self.x_arrays[shard_idx][local_idx]     # float16, (23, 8, 8)
        y_val = float(self.y_arrays[shard_idx][local_idx])

        # Extract auxiliary targets from scalar planes (uniform across spatial dims)
        # Phase = plane 19, Material delta = plane 22 (already normalized in encode_v2)
        y_phase = float(x[19, 0, 0])
        y_material = float(x[22, 0, 0])

        # Convert to float32 tensor (from float16 mmap)
        x_tensor = torch.from_numpy(x.astype(np.float32, copy=True))
        # Pack scalar targets into a single tensor to reduce overhead
        targets = torch.tensor([y_val, y_phase, y_material], dtype=torch.float32)
        return x_tensor, targets


def build_dataloaders(
    cfg: TrainConfig,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Build train/val/test DataLoaders from sharded data."""
    train_dir = os.path.join(cfg.data_root, "train")
    val_dir = os.path.join(cfg.data_root, "val")
    test_dir = os.path.join(cfg.data_root, "test")

    train_ds = ShardDataset(train_dir)
    val_ds = ShardDataset(val_dir)
    test_ds = ShardDataset(test_dir) if os.path.exists(test_dir) else None

    # persistent_workers keeps worker processes alive between epochs (avoids fork overhead)
    pw = cfg.persistent_workers and cfg.num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
        persistent_workers=pw,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=pw,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.batch_size * 2,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=pw,
            prefetch_factor=2 if cfg.num_workers > 0 else None,
        )

    return train_loader, val_loader, test_loader


# ========================= TRAINER =========================

class Trainer:
    """Handles training loop, AMP, grad accumulation, checkpointing, resume."""

    def __init__(
        self,
        config: TrainConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ) -> None:
        self.cfg = config
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler: step-level CosineAnnealingWarmRestarts
        self.steps_per_epoch = max(1, len(train_loader) // config.grad_accum_steps)
        T0_steps = config.scheduler_T0 * self.steps_per_epoch
        self.warmup_steps = config.warmup_epochs * self.steps_per_epoch
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=max(1, T0_steps),
            T_mult=config.scheduler_T_mult,
            eta_min=config.min_lr,
        )

        # AMP — use device-aware GradScaler (PyTorch 2.1+)
        self.scaler = GradScaler("cuda", enabled=config.use_amp)

        # State
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.train_history: List[Dict[str, float]] = []
        self.val_history: List[Dict[str, float]] = []

        # Paths
        self.run_dir = Path(config.runs_root) / config.run_name
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Resume
        if config.resume_from:
            self.load_checkpoint(config.resume_from)

    # --- LR with warmup ---

    def _get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def _update_lr(self) -> None:
        """Linear warmup then hand off to scheduler."""
        if self.global_step < self.warmup_steps:
            warmup_factor = (self.global_step + 1) / max(self.warmup_steps, 1)
            lr = self.cfg.learning_rate * warmup_factor
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        else:
            # Scheduler uses its own internal step counter
            adjusted_step = self.global_step - self.warmup_steps
            self.scheduler.step(adjusted_step)

    # --- Loss ---

    def compute_loss(
        self,
        output: Dict[str, torch.Tensor],
        y_value: torch.Tensor,
        y_phase: torch.Tensor,
        y_material: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_v = F.mse_loss(output["value"].view(-1), y_value)
        loss_p = F.mse_loss(output["phase"].view(-1), y_phase)
        loss_m = F.mse_loss(output["material"].view(-1), y_material)

        total = loss_v + self.cfg.lambda_phase * loss_p + self.cfg.lambda_material * loss_m

        details = {
            "loss_total": total.item(),
            "loss_value": loss_v.item(),
            "loss_phase": loss_p.item(),
            "loss_material": loss_m.item(),
        }
        return total, details

    # --- Training loop ---

    def train(self) -> Dict[str, List]:
        """Main training loop. Returns history dict."""
        print(f"\n{'='*60}")
        print(f"  DGRN-X-v0.2 Training | {self.cfg.run_name}")
        print(f"  Epochs: {self.cfg.epochs} | Batch: {self.cfg.batch_size}x{self.cfg.grad_accum_steps}")
        print(f"  LR: {self.cfg.learning_rate} → {self.cfg.min_lr}")
        print(f"  AMP: {self.cfg.use_amp} | Device: {self.device}")
        print(f"  Steps/epoch: {self.steps_per_epoch}")
        if self.start_epoch > 0:
            print(f"  *** Resumed from epoch {self.start_epoch} ***")
        print(f"{'='*60}\n")

        for epoch in range(self.start_epoch, self.cfg.epochs):
            t0 = time.time()
            train_metrics = self._train_one_epoch(epoch)
            train_time = time.time() - t0
            train_metrics["epoch"] = epoch
            train_metrics["time_s"] = train_time
            self.train_history.append(train_metrics)

            # Evaluate
            val_metrics = None
            if (epoch + 1) % self.cfg.eval_every_epoch == 0:
                val_metrics = self._evaluate(self.val_loader, prefix="val")
                val_metrics["epoch"] = epoch
                self.val_history.append(val_metrics)

                is_best = val_metrics["val_loss_total"] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics["val_loss_total"]
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                # Checkpoint
                if (epoch + 1) % self.cfg.save_every_epoch == 0 or is_best:
                    self.save_checkpoint(epoch, is_best)

                # Print summary
                print(
                    f"Epoch {epoch+1:3d}/{self.cfg.epochs} | "
                    f"Train: {train_metrics['train_loss_total']:.6f} "
                    f"(v={train_metrics['train_loss_value']:.4f} "
                    f"p={train_metrics['train_loss_phase']:.4f} "
                    f"m={train_metrics['train_loss_material']:.4f}) | "
                    f"Val: {val_metrics['val_loss_total']:.6f} | "
                    f"LR: {self._get_lr():.2e} | "
                    f"{'★ BEST' if is_best else ''} | "
                    f"{train_time:.0f}s"
                )

                # Early stopping
                if self.cfg.patience > 0 and self.patience_counter >= self.cfg.patience:
                    print(f"\n[EARLY STOP] No improvement for {self.cfg.patience} epochs.")
                    break
            else:
                print(
                    f"Epoch {epoch+1:3d}/{self.cfg.epochs} | "
                    f"Train: {train_metrics['train_loss_total']:.6f} | "
                    f"LR: {self._get_lr():.2e} | {train_time:.0f}s"
                )

        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.6f}")
        return {"train": self.train_history, "val": self.val_history}

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running = {"loss_total": 0.0, "loss_value": 0.0, "loss_phase": 0.0, "loss_material": 0.0}
        n_accum = 0
        n_optimizer_steps = 0

        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, (x, targets) in enumerate(self.train_loader):
            x = x.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            y_v = targets[:, 0]
            y_p = targets[:, 1]
            y_m = targets[:, 2]

            # Forward with AMP
            with autocast(device_type=self.device.type, enabled=self.cfg.use_amp):
                output = self.model(x)
                loss, details = self.compute_loss(output, y_v, y_p, y_m)
                loss = loss / self.cfg.grad_accum_steps

            # Backward
            self.scaler.scale(loss).backward()
            n_accum += 1

            # Accumulate running stats
            for k in running:
                running[k] += details[k]

            # Optimizer step
            if n_accum >= self.cfg.grad_accum_steps:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                self._update_lr()
                self.global_step += 1
                n_optimizer_steps += 1
                n_accum = 0

                # Logging
                if self.global_step % self.cfg.log_every_steps == 0:
                    avg_loss = running["loss_total"] / max(batch_idx + 1, 1)
                    print(
                        f"  step {self.global_step:6d} | "
                        f"loss: {avg_loss:.6f} | "
                        f"lr: {self._get_lr():.2e}",
                        end="\r",
                    )

        # Epoch averages
        n_batches = max(len(self.train_loader), 1)
        return {
            f"train_{k}": running[k] / n_batches for k in running
        }

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader, prefix: str = "val") -> Dict[str, float]:
        self.model.eval()
        running = {"loss_total": 0.0, "loss_value": 0.0, "loss_phase": 0.0, "loss_material": 0.0}
        n_batches = 0

        for x, targets in loader:
            x = x.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            y_v, y_p, y_m = targets[:, 0], targets[:, 1], targets[:, 2]

            with autocast(device_type=self.device.type, enabled=self.cfg.use_amp):
                output = self.model(x)
                _, details = self.compute_loss(output, y_v, y_p, y_m)

            for k in running:
                running[k] += details[k]
            n_batches += 1

        self.model.train()
        return {f"{prefix}_{k}": running[k] / max(n_batches, 1) for k in running}

    # --- Checkpointing ---

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        payload = {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
            "config": asdict(self.cfg),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "train_history": self.train_history,
            "val_history": self.val_history,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
        }

        # Save latest
        path = self.ckpt_dir / f"epoch_{epoch+1:03d}.pt"
        torch.save(payload, str(path))
        print(f"  [SAVE] {path.name}")

        # Save best
        if is_best:
            best_path = self.ckpt_dir / "best.pt"
            torch.save(payload, str(best_path))
            print(f"  [SAVE] best.pt (val_loss={self.best_val_loss:.6f})")

        # Save history JSON (human-readable)
        history_path = self.run_dir / "history.json"
        with open(str(history_path), "w", encoding="utf-8") as f:
            json.dump(
                {"train": self.train_history, "val": self.val_history},
                f,
                indent=2,
                ensure_ascii=False,
            )

    def load_checkpoint(self, path: str) -> None:
        """Resume from checkpoint. Restores ALL state."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        print(f"\n[RESUME] Loading checkpoint: {path}")
        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)

        # Verify architecture compatibility
        ckpt_cfg = ckpt.get("config", {})
        for key in ("width", "grid_blocks", "relation_blocks", "input_channels"):
            ckpt_val = ckpt_cfg.get(key)
            cur_val = getattr(self.cfg, key, None)
            if ckpt_val is not None and cur_val is not None and ckpt_val != cur_val:
                raise ValueError(
                    f"Architecture mismatch: checkpoint {key}={ckpt_val}, "
                    f"current config {key}={cur_val}"
                )

        # Restore model
        self.model.load_state_dict(ckpt["model_state"])

        # Restore optimizer & scheduler
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.scaler.load_state_dict(ckpt["scaler_state"])

        # Restore state
        self.start_epoch = ckpt["epoch"] + 1
        self.global_step = ckpt["global_step"]
        self.best_val_loss = ckpt["best_val_loss"]
        self.patience_counter = ckpt.get("patience_counter", 0)
        self.train_history = ckpt.get("train_history", [])
        self.val_history = ckpt.get("val_history", [])

        # Restore RNG state
        rng = ckpt.get("rng_state", {})
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch" in rng:
            torch.random.set_rng_state(rng["torch"])
        if "cuda" in rng and torch.cuda.is_available() and rng["cuda"]:
            torch.cuda.set_rng_state_all(rng["cuda"])

        print(
            f"[RESUME] Restored epoch={self.start_epoch}, "
            f"step={self.global_step}, "
            f"best_val={self.best_val_loss:.6f}\n"
        )

    # --- Test evaluation ---

    @torch.no_grad()
    def evaluate_test(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on test set using best checkpoint."""
        best_path = self.ckpt_dir / "best.pt"
        if best_path.exists():
            ckpt = torch.load(str(best_path), map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state"])
            print(f"[TEST] Loaded best.pt (epoch {ckpt['epoch']+1})")
        else:
            print("[TEST] No best.pt found, using current model weights")

        return self._evaluate(test_loader, prefix="test")


# ========================= VISUALIZATION =========================

def plot_training_history(history: Dict[str, List], save_path: Optional[str] = None) -> None:
    """Plot loss curves and LR schedule."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping plots")
        return

    train_h = history.get("train", [])
    val_h = history.get("val", [])

    if not train_h:
        print("[WARN] No training history to plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Total loss
    ax = axes[0]
    epochs_t = [r.get("epoch", i) + 1 for i, r in enumerate(train_h)]
    ax.plot(epochs_t, [r["train_loss_total"] for r in train_h], "b-o", label="Train", markersize=3)
    if val_h:
        epochs_v = [r.get("epoch", i) + 1 for i, r in enumerate(val_h)]
        ax.plot(epochs_v, [r["val_loss_total"] for r in val_h], "r-o", label="Val", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Total Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Component losses (train)
    ax = axes[1]
    ax.plot(epochs_t, [r["train_loss_value"] for r in train_h], "b-", label="Value")
    ax.plot(epochs_t, [r["train_loss_phase"] for r in train_h], "g-", label="Phase")
    ax.plot(epochs_t, [r["train_loss_material"] for r in train_h], "m-", label="Material")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Component Losses (Train)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Val component losses
    ax = axes[2]
    if val_h:
        ax.plot(epochs_v, [r["val_loss_value"] for r in val_h], "b-", label="Value")
        ax.plot(epochs_v, [r["val_loss_phase"] for r in val_h], "g-", label="Phase")
        ax.plot(epochs_v, [r["val_loss_material"] for r in val_h], "m-", label="Material")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Component Losses (Val)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[PLOT] Saved to {save_path}")
    plt.show()


# ========================= DETAILED EVALUATION =========================

@torch.no_grad()
def detailed_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> Dict[str, Any]:
    """Comprehensive evaluation with per-region metrics.

    Returns:
        Dict with keys:
        - loss_*: MSE losses (value, phase, material, total)
        - mae_value: Mean Absolute Error on value prediction
        - mae_phase, mae_material: MAE on auxiliary heads
        - corr_value: Pearson correlation (pred vs target)
        - region_*: MAE broken down by value region:
            center (|y|<0.1), mid (0.1<=|y|<0.5), decisive (|y|>=0.5)
        - pred_mean, pred_std: prediction statistics
        - target_mean, target_std: target statistics
    """
    model.eval()

    all_pred_v: List[np.ndarray] = []
    all_true_v: List[np.ndarray] = []
    all_pred_p: List[np.ndarray] = []
    all_true_p: List[np.ndarray] = []
    all_pred_m: List[np.ndarray] = []
    all_true_m: List[np.ndarray] = []
    running_loss = {"total": 0.0, "value": 0.0, "phase": 0.0, "material": 0.0}
    n_batches = 0

    for x, targets in loader:
        x = x.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        y_v, y_p, y_m = targets[:, 0], targets[:, 1], targets[:, 2]

        with autocast(device_type=device.type, enabled=use_amp):
            output = model(x)

        pred_v = output["value"].view(-1).float()
        pred_p = output["phase"].view(-1).float()
        pred_m = output["material"].view(-1).float()

        # Losses
        running_loss["value"] += F.mse_loss(pred_v, y_v).item()
        running_loss["phase"] += F.mse_loss(pred_p, y_p).item()
        running_loss["material"] += F.mse_loss(pred_m, y_m).item()
        n_batches += 1

        # Collect predictions
        all_pred_v.append(pred_v.cpu().numpy())
        all_true_v.append(y_v.cpu().numpy())
        all_pred_p.append(pred_p.cpu().numpy())
        all_true_p.append(y_p.cpu().numpy())
        all_pred_m.append(pred_m.cpu().numpy())
        all_true_m.append(y_m.cpu().numpy())

    model.train()

    # Concatenate all predictions
    pred_v = np.concatenate(all_pred_v)
    true_v = np.concatenate(all_true_v)
    pred_p = np.concatenate(all_pred_p)
    true_p = np.concatenate(all_true_p)
    pred_m = np.concatenate(all_pred_m)
    true_m = np.concatenate(all_true_m)
    n = len(pred_v)

    # Base losses
    nb = max(n_batches, 1)
    running_loss["total"] = (
        running_loss["value"] + 0.1 * running_loss["phase"] + 0.1 * running_loss["material"]
    )
    metrics: Dict[str, Any] = {
        "n_samples": n,
        "loss_value": running_loss["value"] / nb,
        "loss_phase": running_loss["phase"] / nb,
        "loss_material": running_loss["material"] / nb,
        "loss_total": running_loss["total"] / nb,
    }

    # MAE
    metrics["mae_value"] = float(np.mean(np.abs(pred_v - true_v)))
    metrics["mae_phase"] = float(np.mean(np.abs(pred_p - true_p)))
    metrics["mae_material"] = float(np.mean(np.abs(pred_m - true_m)))

    # Pearson correlation for value head
    if np.std(pred_v) > 1e-8 and np.std(true_v) > 1e-8:
        metrics["corr_value"] = float(np.corrcoef(pred_v, true_v)[0, 1])
    else:
        metrics["corr_value"] = 0.0

    # Prediction statistics
    metrics["pred_mean"] = float(np.mean(pred_v))
    metrics["pred_std"] = float(np.std(pred_v))
    metrics["target_mean"] = float(np.mean(true_v))
    metrics["target_std"] = float(np.std(true_v))

    # Region-wise MAE for value head
    abs_true = np.abs(true_v)
    center_mask = abs_true < 0.1       # near-draw positions
    mid_mask = (abs_true >= 0.1) & (abs_true < 0.5)  # middlegame advantage
    decisive_mask = abs_true >= 0.5    # clearly winning/losing

    for name, mask in [("center", center_mask), ("mid", mid_mask), ("decisive", decisive_mask)]:
        count = int(mask.sum())
        metrics[f"region_{name}_n"] = count
        if count > 0:
            metrics[f"region_{name}_mae"] = float(np.mean(np.abs(pred_v[mask] - true_v[mask])))
            metrics[f"region_{name}_mean_pred"] = float(np.mean(pred_v[mask]))
            metrics[f"region_{name}_mean_true"] = float(np.mean(true_v[mask]))
        else:
            metrics[f"region_{name}_mae"] = float("nan")
            metrics[f"region_{name}_mean_pred"] = float("nan")
            metrics[f"region_{name}_mean_true"] = float("nan")

    return metrics


def print_detailed_metrics(metrics: Dict[str, Any], title: str = "Evaluation") -> None:
    """Pretty-print detailed evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"  {title} | {metrics['n_samples']:,} samples")
    print(f"{'='*60}")

    print(f"\n  Losses (MSE):")
    print(f"    Total:    {metrics['loss_total']:.6f}")
    print(f"    Value:    {metrics['loss_value']:.6f}")
    print(f"    Phase:    {metrics['loss_phase']:.6f}")
    print(f"    Material: {metrics['loss_material']:.6f}")

    print(f"\n  Value Head Metrics:")
    print(f"    MAE:         {metrics['mae_value']:.6f}")
    print(f"    Correlation: {metrics['corr_value']:.4f}")
    print(f"    Pred  μ={metrics['pred_mean']:+.4f}  σ={metrics['pred_std']:.4f}")
    print(f"    True  μ={metrics['target_mean']:+.4f}  σ={metrics['target_std']:.4f}")

    print(f"\n  Auxiliary Head MAE:")
    print(f"    Phase:    {metrics['mae_phase']:.6f}")
    print(f"    Material: {metrics['mae_material']:.6f}")

    print(f"\n  Region-wise Value MAE:")
    for name in ["center", "mid", "decisive"]:
        n = metrics[f'region_{name}_n']
        mae = metrics[f'region_{name}_mae']
        pct = n / max(metrics['n_samples'], 1) * 100
        print(f"    {name:>10s}: MAE={mae:.6f}  (n={n:,}, {pct:.1f}%)")

    print(f"{'='*60}\n")
