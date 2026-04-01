from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OLD_LAB_DIR = PROJECT_ROOT / "experiments" / "teacher_root_cause_lab"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(OLD_LAB_DIR) not in sys.path:
    sys.path.insert(0, str(OLD_LAB_DIR))

import teacher_root_cause_helpers as base_lab  # noqa: E402


CANONICAL_Y600_BANDS: Tuple[float, ...] = (0.0, 0.05, 0.20, 0.50, 0.70, 1.00)
CANONICAL_MONITOR_THRS_Y600: Tuple[float, ...] = (0.10, 0.20, 0.50, 0.70)
SOURCE_TARGET_SCALE = 600.0


@dataclass
class ExperimentPaths:
    project_root: str
    run_dir: str
    data_root: str
    experiment_dir: str
    output_dir: str
    plots_dir: str
    reports_dir: str
    checkpoints_dir: str
    cache_dir: str
    runs_dir: str


@dataclass
class OracleEvalConfig:
    subset_csv_path: str = str(
        PROJECT_ROOT
        / "experiments"
        / "oracle_root_cause_diagnostic"
        / "outputs"
        / "reports"
        / "oracle_subset_rows.csv"
    )
    stable_label: str = "stable"
    center_thr_y600: float = 0.05
    pred_thr_small_y600: float = 0.10
    pred_thr_medium_y600: float = 0.20
    oracle_batch_size: int = 1024
    mapping_validation_rows: int = 24


@dataclass
class TrainConfig:
    batch_size: int = 640
    epochs: int = 1
    learning_rate: float = 3e-6
    min_lr: float = 1e-6
    weight_decay: float = 2e-4
    grad_clip_norm: float = 1.0
    seed: int = 123
    log_every_steps: int = 200
    train_num_shards: Optional[int] = None
    val_max_samples: int = 100_000
    test_max_samples: int = 200_000
    val_num_shards: int = 2
    test_num_shards: int = 4
    benchmark_num_shards: int = 1


@dataclass
class AblationVariant:
    name: str
    description: str
    target_scale: float = SOURCE_TARGET_SCALE
    sampler_mode: str = "random"
    loss_mode: str = "baseline_hybrid"
    lambda_y: float = 0.99
    z_loss_beta: float = 1.0
    z_huber_delta: float = 0.5
    target_clamp_eps: float = 1e-3
    y_loss_alpha: float = 0.65
    y_reweight_clip_max: float = 4.0
    y_reweight_eps: float = 1e-4
    balance_band_edges_y600: Tuple[float, ...] = CANONICAL_Y600_BANDS
    center_penalty_weight: float = 0.0
    center_penalty_tau_y600: float = 0.05
    center_penalty_margin_y600: float = 0.10


def build_default_paths(
    run_dir: str | Path = r"C:\Users\USER\Downloads\dgrn_5m_v3_stage2_polish_run1",
    data_root: str | Path = r"C:\Users\USER\Desktop\chess_engine\data\process",
    experiment_dir: str | Path | None = None,
) -> Dict[str, Path]:
    if experiment_dir is None:
        experiment_dir = PROJECT_ROOT / "experiments" / "root_cause_ablation_suite"
    paths = {
        "project_root": PROJECT_ROOT,
        "run_dir": Path(run_dir),
        "data_root": Path(data_root),
        "experiment_dir": Path(experiment_dir),
    }
    paths["output_dir"] = paths["experiment_dir"] / "outputs"
    paths["plots_dir"] = paths["output_dir"] / "plots"
    paths["reports_dir"] = paths["output_dir"] / "reports"
    paths["checkpoints_dir"] = paths["output_dir"] / "checkpoints"
    paths["cache_dir"] = paths["output_dir"] / "cache"
    paths["runs_dir"] = paths["output_dir"] / "runs"
    for key in ("experiment_dir", "output_dir", "plots_dir", "reports_dir", "checkpoints_dir", "cache_dir", "runs_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def export_paths_json(paths: Dict[str, Path], out_path: Path) -> None:
    payload = ExperimentPaths(**{key: str(value) for key, value in paths.items()})
    out_path.write_text(json.dumps(asdict(payload), indent=2), encoding="utf-8")


save_json = base_lab.save_json
save_dataframe = base_lab.save_dataframe
set_global_seed = base_lab.set_global_seed
choose_device = base_lab.choose_device


def atomic_torch_save(payload: dict, path: Path, retries: int = 3, pause_sec: float = 0.25) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Optional[BaseException] = None
    for attempt in range(int(max(retries, 1))):
        tmp_path = path.with_name(f"{path.name}.tmp-{time.time_ns()}")
        try:
            torch.save(payload, tmp_path)
            tmp_path.replace(path)
            return
        except BaseException as exc:  # pragma: no cover
            last_error = exc
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            if attempt + 1 < int(max(retries, 1)):
                time.sleep(float(pause_sec))
    raise RuntimeError(f"atomic_torch_save failed for {path}: {last_error}") from last_error


def remap_target_np(y_source: np.ndarray, to_scale: float, from_scale: float = SOURCE_TARGET_SCALE, eps: float = 1e-6) -> np.ndarray:
    y_arr = np.asarray(y_source, dtype=np.float64)
    if abs(float(to_scale) - float(from_scale)) < 1e-12:
        return y_arr.astype(np.float32, copy=False)
    z = np.arctanh(np.clip(y_arr, -1.0 + eps, 1.0 - eps))
    out = np.tanh((float(from_scale) / float(to_scale)) * z)
    return out.astype(np.float32, copy=False)


def remap_target_torch(y_source: torch.Tensor, to_scale: float, from_scale: float = SOURCE_TARGET_SCALE, eps: float = 1e-6) -> torch.Tensor:
    if abs(float(to_scale) - float(from_scale)) < 1e-12:
        return y_source
    z = torch.atanh(torch.clamp(y_source, -1.0 + eps, 1.0 - eps))
    return torch.tanh((float(from_scale) / float(to_scale)) * z)


def canonical_y600_to_scale_value(y_thr_y600: float, target_scale: float) -> float:
    return float(remap_target_np(np.asarray([y_thr_y600], dtype=np.float64), to_scale=target_scale)[0])


def parse_shard_id(x_path: Path) -> int:
    return int(x_path.stem.split("_")[1])


def resolve_split_shards(data_root: str | Path, split: str, num_shards: Optional[int] = None) -> List[Tuple[int, Path, Path]]:
    pairs = base_lab.resolve_split_pairs(data_root, split)
    pairs = base_lab.select_pairs_evenly(pairs, num_shards)
    return [(parse_shard_id(x_path), x_path, y_path) for x_path, y_path in pairs]


def validate_train_config(cfg: TrainConfig) -> Dict[str, object]:
    issues: List[str] = []
    if cfg.batch_size <= 0:
        issues.append("batch_size must be positive")
    if cfg.epochs <= 0:
        issues.append("epochs must be positive")
    if cfg.learning_rate <= 0.0:
        issues.append("learning_rate must be positive")
    if cfg.min_lr <= 0.0:
        issues.append("min_lr must be positive")
    if cfg.min_lr > cfg.learning_rate:
        issues.append("min_lr must be <= learning_rate")
    if cfg.weight_decay < 0.0:
        issues.append("weight_decay must be non-negative")
    if cfg.grad_clip_norm is not None and cfg.grad_clip_norm <= 0.0:
        issues.append("grad_clip_norm must be positive when provided")
    if cfg.log_every_steps <= 0:
        issues.append("log_every_steps must be positive")
    for name in ("val_max_samples", "test_max_samples", "val_num_shards", "test_num_shards", "benchmark_num_shards"):
        if getattr(cfg, name) <= 0:
            issues.append(f"{name} must be positive")
    if cfg.train_num_shards is not None and cfg.train_num_shards <= 0:
        issues.append("train_num_shards must be positive when provided")
    if issues:
        raise ValueError("Invalid TrainConfig: " + "; ".join(issues))
    return {
        "is_valid": True,
        "batch_size": int(cfg.batch_size),
        "epochs": int(cfg.epochs),
        "learning_rate": float(cfg.learning_rate),
        "min_lr": float(cfg.min_lr),
    }


def validate_variant_config(variant: AblationVariant) -> Dict[str, object]:
    issues: List[str] = []
    if not variant.name:
        issues.append("name must be non-empty")
    if variant.sampler_mode not in {"random", "band_balanced"}:
        issues.append("sampler_mode must be one of {'random', 'band_balanced'}")
    if variant.loss_mode not in {"baseline_hybrid", "curvature_compensated"}:
        issues.append("loss_mode must be one of {'baseline_hybrid', 'curvature_compensated'}")
    if variant.target_scale <= 0.0:
        issues.append("target_scale must be positive")
    if not (0.0 <= variant.lambda_y <= 1.0):
        issues.append("lambda_y must be in [0, 1]")
    if variant.z_loss_beta < 0.0:
        issues.append("z_loss_beta must be non-negative")
    if variant.z_huber_delta <= 0.0:
        issues.append("z_huber_delta must be positive")
    if not (0.0 < variant.target_clamp_eps < 0.5):
        issues.append("target_clamp_eps must be in (0, 0.5)")
    if not (0.0 <= variant.y_loss_alpha <= 1.0):
        issues.append("y_loss_alpha must be in [0, 1]")
    if variant.y_reweight_clip_max < 1.0:
        issues.append("y_reweight_clip_max must be >= 1")
    if variant.y_reweight_eps <= 0.0:
        issues.append("y_reweight_eps must be positive")
    if len(variant.balance_band_edges_y600) < 2:
        issues.append("balance_band_edges_y600 must have at least two edges")
    if any(
        float(variant.balance_band_edges_y600[idx]) >= float(variant.balance_band_edges_y600[idx + 1])
        for idx in range(len(variant.balance_band_edges_y600) - 1)
    ):
        issues.append("balance_band_edges_y600 must be strictly increasing")
    if variant.center_penalty_weight < 0.0:
        issues.append("center_penalty_weight must be non-negative")
    if variant.center_penalty_tau_y600 <= 0.0:
        issues.append("center_penalty_tau_y600 must be positive")
    if variant.center_penalty_margin_y600 <= 0.0:
        issues.append("center_penalty_margin_y600 must be positive")
    if issues:
        raise ValueError(f"Invalid AblationVariant[{variant.name}]: " + "; ".join(issues))
    return {
        "is_valid": True,
        "name": variant.name,
        "target_scale": float(variant.target_scale),
        "sampler_mode": variant.sampler_mode,
        "loss_mode": variant.loss_mode,
    }


def validate_oracle_eval_config(cfg: OracleEvalConfig) -> Dict[str, object]:
    issues: List[str] = []
    subset_path = Path(cfg.subset_csv_path)
    if not subset_path.exists():
        issues.append(f"subset_csv_path does not exist: {subset_path}")
    if cfg.center_thr_y600 <= 0.0:
        issues.append("center_thr_y600 must be positive")
    if cfg.pred_thr_small_y600 <= 0.0 or cfg.pred_thr_medium_y600 <= 0.0:
        issues.append("prediction thresholds must be positive")
    if cfg.pred_thr_medium_y600 <= cfg.pred_thr_small_y600:
        issues.append("pred_thr_medium_y600 must be > pred_thr_small_y600")
    if cfg.oracle_batch_size <= 0:
        issues.append("oracle_batch_size must be positive")
    if cfg.mapping_validation_rows <= 0:
        issues.append("mapping_validation_rows must be positive")
    required_columns = [
        "split",
        "shard_id",
        "local_index",
        "target_y",
        "teacher_pred",
        "oracle_reference_cp",
        "stability_group",
    ]
    missing: List[str] = []
    if subset_path.exists():
        frame = pd.read_csv(subset_path, nrows=4)
        missing = [name for name in required_columns if name not in frame.columns]
        if missing:
            issues.append("missing required oracle subset columns: " + ", ".join(missing))
    if issues:
        raise ValueError("Invalid OracleEvalConfig: " + "; ".join(issues))
    return {
        "is_valid": True,
        "subset_csv_path": str(subset_path),
        "required_columns": required_columns,
        "missing_columns": missing,
    }


def validate_checkpoint_model(init_ckpt_path: str | Path, device: torch.device) -> Dict[str, object]:
    model, ckpt = base_lab.load_model_from_checkpoint(init_ckpt_path, device=device)
    if not hasattr(model, "forward_logits"):
        raise RuntimeError("Current checkpoint model does not expose forward_logits().")
    output_mode = getattr(getattr(model, "head", object()), "output_mode", None)
    if output_mode != "tanh":
        raise RuntimeError(f"Expected model.head.output_mode == 'tanh', found: {output_mode}")
    return {
        "checkpoint": str(Path(init_ckpt_path)),
        "model_class": type(model).__name__,
        "head_output_mode": output_mode,
        "has_forward_logits": True,
        "checkpoint_epoch": None if ckpt.get("epoch") is None else int(ckpt.get("epoch")),
    }


def validate_target_remap_logic(scales: Sequence[float]) -> Dict[str, object]:
    probe = np.asarray([-0.95, -0.70, -0.20, -0.05, 0.0, 0.05, 0.20, 0.70, 0.95], dtype=np.float64)
    monotonic = {}
    antisymmetric = {}
    identities = {}
    for scale in scales:
        mapped = remap_target_np(probe, to_scale=float(scale)).astype(np.float64)
        monotonic[str(scale)] = bool(np.all(np.diff(mapped) > 0.0))
        antisymmetric[str(scale)] = bool(np.max(np.abs(mapped + mapped[::-1])) < 1e-6)
        identities[str(scale)] = bool(
            (abs(float(scale) - SOURCE_TARGET_SCALE) > 1e-12)
            or np.max(np.abs(mapped - probe)) < 1e-6
        )
    report = {
        "probe": [float(x) for x in probe],
        "scales": [float(x) for x in scales],
        "monotonic": monotonic,
        "antisymmetric": antisymmetric,
        "identity_at_600": identities.get(str(SOURCE_TARGET_SCALE), True),
    }
    if not all(monotonic.values()):
        raise RuntimeError("Target remap monotonicity check failed.")
    if not all(antisymmetric.values()):
        raise RuntimeError("Target remap antisymmetry check failed.")
    if not report["identity_at_600"]:
        raise RuntimeError("Target remap identity at 600 failed.")
    return report


def load_oracle_subset_frame(cfg: OracleEvalConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.subset_csv_path)
    df["split"] = df["split"].astype(str)
    df["shard_id"] = df["shard_id"].astype(int)
    df["local_index"] = df["local_index"].astype(int)
    return df


def remap_predictions_to_y600(pred: np.ndarray, from_scale: float) -> np.ndarray:
    return remap_target_np(
        np.asarray(pred, dtype=np.float64),
        to_scale=SOURCE_TARGET_SCALE,
        from_scale=float(from_scale),
    ).astype(np.float64, copy=False)


def validate_oracle_subset_mapping(
    oracle_cfg: OracleEvalConfig,
    data_root: str | Path,
    sample_rows: Optional[int] = None,
) -> Dict[str, object]:
    df = load_oracle_subset_frame(oracle_cfg)
    if sample_rows is None:
        sample_rows = oracle_cfg.mapping_validation_rows
    sample = df.sample(n=min(sample_rows, int(df.shape[0])), random_state=123, replace=False).copy()
    mismatches = 0
    mismatch_rows: List[int] = []
    shard_cache: Dict[Tuple[str, int], np.ndarray] = {}
    for rank, row in sample.iterrows():
        key = (str(row["split"]), int(row["shard_id"]))
        if key not in shard_cache:
            y_path = Path(data_root) / key[0] / f"y_{key[1]:05d}.npy"
            shard_cache[key] = np.load(y_path, mmap_mode="r").astype(np.float32, copy=False)
        y_arr = shard_cache[key]
        local_index = int(row["local_index"])
        if local_index < 0 or local_index >= y_arr.shape[0]:
            mismatches += 1
            mismatch_rows.append(int(rank))
            continue
        if abs(float(y_arr[local_index]) - float(row["target_y"])) > 1e-4:
            mismatches += 1
            mismatch_rows.append(int(rank))
    report = {
        "checked_rows": int(sample.shape[0]),
        "mismatches": int(mismatches),
        "mismatch_rows": mismatch_rows[:10],
    }
    if mismatches > 0:
        raise RuntimeError(f"Oracle subset mapping validation failed: {report}")
    return report


def build_band_balanced_order(
    abs_y: np.ndarray,
    batch_size: int,
    band_edges_y600: Sequence[float],
    rng: np.random.Generator,
    target_scale: float,
) -> np.ndarray:
    abs_y = np.asarray(abs_y, dtype=np.float64)
    band_edges = np.asarray([canonical_y600_to_scale_value(v, target_scale) for v in band_edges_y600], dtype=np.float64)
    bins = np.clip(np.digitize(abs_y, band_edges[1:-1], right=False), 0, len(band_edges) - 2)
    band_indices = [rng.permutation(np.flatnonzero(bins == idx)).astype(np.int64) for idx in range(len(band_edges) - 1)]
    pointers = [0 for _ in band_indices]
    quotas = [batch_size // len(band_indices) for _ in band_indices]
    for idx in range(batch_size % len(band_indices)):
        quotas[idx] += 1
    batches: List[np.ndarray] = []
    while True:
        active = [idx for idx, arr in enumerate(band_indices) if pointers[idx] < arr.size]
        if not active:
            break
        total_remaining = int(sum(band_indices[idx].size - pointers[idx] for idx in active))
        target_slots = min(int(batch_size), total_remaining)
        alloc = [0 for _ in band_indices]
        leftover = target_slots
        for idx in active:
            take = min(quotas[idx], int(band_indices[idx].size - pointers[idx]), leftover)
            alloc[idx] = take
            leftover -= take
        while leftover > 0:
            progressed = False
            for idx in active:
                cap = int(band_indices[idx].size - pointers[idx] - alloc[idx])
                if cap > 0 and leftover > 0:
                    alloc[idx] += 1
                    leftover -= 1
                    progressed = True
            if not progressed:
                break
        parts = []
        for idx in active:
            take = alloc[idx]
            if take <= 0:
                continue
            arr = band_indices[idx]
            parts.append(arr[pointers[idx] : pointers[idx] + take])
            pointers[idx] += take
        if not parts:
            break
        batch = np.concatenate(parts, axis=0)
        rng.shuffle(batch)
        batches.append(batch)
    if not batches:
        return np.arange(abs_y.shape[0], dtype=np.int64)
    order = np.concatenate(batches, axis=0).astype(np.int64)
    if order.size != abs_y.shape[0]:
        remaining_mask = np.ones(abs_y.shape[0], dtype=bool)
        remaining_mask[order] = False
        remaining = np.flatnonzero(remaining_mask).astype(np.int64)
        if remaining.size > 0:
            rng.shuffle(remaining)
            order = np.concatenate([order, remaining], axis=0)
    return order


def validate_band_balanced_sampler(batch_size: int = 640) -> Dict[str, object]:
    rng = np.random.default_rng(123)
    synthetic = np.concatenate(
        [
            np.full(11, 0.02, dtype=np.float32),
            np.full(17, 0.08, dtype=np.float32),
            np.full(13, 0.35, dtype=np.float32),
            np.full(19, 0.62, dtype=np.float32),
            np.full(7, 0.88, dtype=np.float32),
        ]
    )
    order = build_band_balanced_order(
        np.abs(synthetic),
        batch_size=min(batch_size, 16),
        band_edges_y600=CANONICAL_Y600_BANDS,
        rng=rng,
        target_scale=SOURCE_TARGET_SCALE,
    )
    unique_ok = bool(np.array_equal(np.sort(order), np.arange(synthetic.shape[0], dtype=np.int64)))
    report = {"n": int(synthetic.shape[0]), "unique_ok": unique_ok, "first_indices": [int(x) for x in order[:16]]}
    if not unique_ok:
        raise RuntimeError("Band-balanced sampler validation failed.")
    return report


def benchmark_single_train_step(
    init_ckpt_path: str | Path,
    data_root: str | Path,
    device: torch.device,
    batch_size: int = 640,
    num_shards: Optional[int] = 1,
) -> Dict[str, float]:
    model, _ = base_lab.load_model_from_checkpoint(init_ckpt_path, device=device)
    model.train()
    shard_rows = resolve_split_shards(data_root, "train", num_shards=num_shards)
    _, x_path, y_path = shard_rows[0]
    full_train_rows = resolve_split_shards(data_root, "train", num_shards=None)
    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    xb = torch.from_numpy(np.array(X[:batch_size], dtype=np.float32, copy=True)).to(device)
    yb = torch.from_numpy(np.array(y[:batch_size], dtype=np.float32, copy=True)).to(device).view(-1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-6)
    opt.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        pred = model(xb).view(-1)
        loss = torch.mean((pred - yb) ** 2)
    loss.backward()
    opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_samples = int(sum(int(np.load(path, mmap_mode="r").shape[0]) for _, _, path in full_train_rows))
    steps = int(math.ceil(total_samples / batch_size))
    elapsed = float(time.time() - t0)
    return {
        "batch_size": int(batch_size),
        "step_time_sec": elapsed,
        "peak_mem_gb": float(torch.cuda.max_memory_allocated() / 1024**3) if device.type == "cuda" else 0.0,
        "steps_per_epoch_estimate": steps,
        "epoch_hours_estimate": float(elapsed * steps / 3600.0),
    }


def compute_gradient_mass_profile(
    data_root: str | Path,
    split: str,
    variant: AblationVariant,
    num_shards: Optional[int] = None,
    band_edges_y600: Sequence[float] = CANONICAL_Y600_BANDS,
) -> pd.DataFrame:
    band_edges_y600 = tuple(float(x) for x in band_edges_y600)
    band_edges_variant = np.asarray([canonical_y600_to_scale_value(x, variant.target_scale) for x in band_edges_y600], dtype=np.float64)
    stats = {
        idx: {
            "left_y600": band_edges_y600[idx],
            "right_y600": band_edges_y600[idx + 1],
            "n": 0,
            "sum_abs_y": 0.0,
            "sum_y_curv": 0.0,
            "sum_z_weight": 0.0,
            "sum_effective_y_factor": 0.0,
            "sum_effective_total": 0.0,
        }
        for idx in range(len(band_edges_y600) - 1)
    }
    for _, _, y_path in resolve_split_shards(data_root, split, num_shards=num_shards):
        y_source = np.load(y_path, mmap_mode="r").astype(np.float64, copy=False)
        y = remap_target_np(y_source, to_scale=variant.target_scale).astype(np.float64, copy=False)
        abs_y = np.abs(y)
        y_curv = np.square(1.0 - y * y)
        z_weight = np.power(np.clip(1.0 - y * y, variant.target_clamp_eps, None), variant.z_loss_beta)
        if variant.loss_mode == "curvature_compensated":
            y_factor = np.clip(1.0 / (y_curv + variant.y_reweight_eps), 1.0, variant.y_reweight_clip_max) * y_curv
            y_factor = y_factor / np.maximum(y_factor.mean(), 1e-8)
            total_factor = variant.y_loss_alpha * y_factor + (1.0 - variant.y_loss_alpha) * z_weight
        else:
            y_factor = y_curv
            total_factor = variant.lambda_y * y_curv + (1.0 - variant.lambda_y) * z_weight
        for idx in range(len(band_edges_variant) - 1):
            left = band_edges_variant[idx]
            right = band_edges_variant[idx + 1]
            if idx == 0:
                mask = (abs_y >= left) & (abs_y <= right)
            else:
                mask = (abs_y > left) & (abs_y <= right)
            if not np.any(mask):
                continue
            stats[idx]["n"] += int(mask.sum())
            stats[idx]["sum_abs_y"] += float(abs_y[mask].sum())
            stats[idx]["sum_y_curv"] += float(y_curv[mask].sum())
            stats[idx]["sum_z_weight"] += float(z_weight[mask].sum())
            stats[idx]["sum_effective_y_factor"] += float(y_factor[mask].sum())
            stats[idx]["sum_effective_total"] += float(total_factor[mask].sum())
    total_n = float(sum(row["n"] for row in stats.values()))
    total_effective = float(sum(row["sum_effective_total"] for row in stats.values()))
    rows: List[dict] = []
    for idx, row in stats.items():
        n = row["n"]
        rows.append(
            {
                "band_idx": int(idx),
                "band_label_y600": f"[{row['left_y600']:.3f},{row['right_y600']:.3f}]",
                "sample_count": int(n),
                "sample_share": float(n / total_n) if total_n > 0 else float("nan"),
                "mean_abs_y": float(row["sum_abs_y"] / max(n, 1)),
                "mean_y_curvature": float(row["sum_y_curv"] / max(n, 1)),
                "mean_z_weight": float(row["sum_z_weight"] / max(n, 1)),
                "mean_effective_y_factor": float(row["sum_effective_y_factor"] / max(n, 1)),
                "effective_gradient_mass_share": float(row["sum_effective_total"] / total_effective) if total_effective > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def summarize_split_layout(data_root: str | Path, split: str, num_shards: Optional[int] = None) -> Dict[str, int]:
    shard_rows = resolve_split_shards(data_root, split, num_shards=num_shards)
    total_samples = int(sum(int(np.load(y_path, mmap_mode="r").shape[0]) for _, _, y_path in shard_rows))
    return {"split": split, "num_shards": int(len(shard_rows)), "samples": total_samples}


def summarize_scale_aware_metrics(
    y: np.ndarray,
    p: np.ndarray,
    target_scale: float,
    center_thr_y600: float,
    pred_thr_small_y600: float,
    pred_thr_medium_y600: float,
) -> Dict[str, object]:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    band_metrics = {}
    for thr_y600 in CANONICAL_MONITOR_THRS_Y600:
        thr_variant = canonical_y600_to_scale_value(thr_y600, target_scale)
        band_metrics[f"{thr_y600:.2f}"] = base_lab.compute_band_metrics(y, p, thr_variant)
    center_thr = canonical_y600_to_scale_value(center_thr_y600, target_scale)
    pred_small = canonical_y600_to_scale_value(pred_thr_small_y600, target_scale)
    pred_medium = canonical_y600_to_scale_value(pred_thr_medium_y600, target_scale)
    bucket_table = base_lab.compute_bucket_table(y, p)
    midband = bucket_table[(bucket_table["center"] >= -0.7) & (bucket_table["center"] <= 0.7) & (bucket_table["count"] > 0)]
    worst_midband = {} if midband.empty else midband.sort_values("mse", ascending=False).head(1).to_dict(orient="records")[0]
    return {
        "overall": {
            "mse": float(np.mean((p - y) ** 2)),
            "mae": float(np.mean(np.abs(p - y))),
            "bias": float(np.mean(p - y)),
            "mean_abs_pred": float(np.mean(np.abs(p))),
            "r2": base_lab.r2_score(y, p),
            "pearson": base_lab.pearsonr(y, p),
        },
        "target_scale": float(target_scale),
        "bands": band_metrics,
        "center_false_decisive": {
            f"|oracle|<={center_thr_y600:.2f},|pred|>={pred_thr_small_y600:.2f}": base_lab.false_decisive_rate(y, p, center_thr, pred_small),
            f"|oracle|<={center_thr_y600:.2f},|pred|>={pred_thr_medium_y600:.2f}": base_lab.false_decisive_rate(y, p, center_thr, pred_medium),
        },
        "center_spread_ratio": base_lab.center_spread_ratio(y, p, thr=center_thr),
        "max_midband_abs_cal_gap": float(midband["abs_cal_gap"].max()) if not midband.empty else float("nan"),
        "worst_midband_bucket": worst_midband,
    }


def _load_rows_to_tensor_bundle(df: pd.DataFrame, data_root: str | Path) -> Tuple[np.ndarray, pd.DataFrame]:
    shard_cache: Dict[Tuple[str, int], np.ndarray] = {}
    xs: List[np.ndarray] = []
    ordered = df.reset_index(drop=True).copy()
    for _, row in ordered.iterrows():
        key = (str(row["split"]), int(row["shard_id"]))
        if key not in shard_cache:
            x_path = Path(data_root) / key[0] / f"X_{key[1]:05d}.npy"
            shard_cache[key] = np.load(x_path, mmap_mode="r")
        xs.append(np.asarray(shard_cache[key][int(row["local_index"])], dtype=np.uint8))
    X = np.stack(xs, axis=0).astype(np.uint8, copy=False)
    return X, ordered


def load_oracle_subset_bundle(cfg: OracleEvalConfig, data_root: str | Path) -> Dict[str, object]:
    df = load_oracle_subset_frame(cfg)
    X, ordered = _load_rows_to_tensor_bundle(df, data_root)
    return {"rows": ordered, "X": X}


def build_oracle_band_summary(
    oracle_y: np.ndarray,
    train_y: np.ndarray,
    pred: np.ndarray,
    stable_mask: np.ndarray,
    target_scale: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows_all: List[dict] = []
    rows_stable: List[dict] = []
    band_edges = [canonical_y600_to_scale_value(x, target_scale) for x in CANONICAL_Y600_BANDS]
    for idx in range(len(band_edges) - 1):
        left = float(band_edges[idx])
        right = float(band_edges[idx + 1])
        if idx == 0:
            mask = (np.abs(oracle_y) >= left) & (np.abs(oracle_y) <= right)
        else:
            mask = (np.abs(oracle_y) > left) & (np.abs(oracle_y) <= right)
        for label, submask, rows in (("all", mask, rows_all), ("stable", mask & stable_mask, rows_stable)):
            if not np.any(submask):
                continue
            yy = oracle_y[submask]
            tt = train_y[submask]
            pp = pred[submask]
            slope, intercept = base_lab.fit_line(yy, pp)
            rows.append(
                {
                    "subset": label,
                    "band_idx": int(idx),
                    "band_label_y600": f"[{CANONICAL_Y600_BANDS[idx]:.3f},{CANONICAL_Y600_BANDS[idx + 1]:.3f}]",
                    "n": int(np.sum(submask)),
                    "teacher_vs_oracle_mae": float(np.mean(np.abs(pp - yy))),
                    "train_vs_oracle_mae": float(np.mean(np.abs(tt - yy))),
                    "teacher_closer_to_oracle_rate": float(np.mean(np.abs(pp - yy) < np.abs(tt - yy))),
                    "teacher_oracle_sign_match_rate": float(np.mean(np.sign(pp) == np.sign(yy))),
                    "train_oracle_sign_match_rate": float(np.mean(np.sign(tt) == np.sign(yy))),
                    "mean_abs_oracle": float(np.mean(np.abs(yy))),
                    "mean_abs_pred": float(np.mean(np.abs(pp))),
                    "amplitude_ratio": float(np.mean(np.abs(pp)) / max(np.mean(np.abs(yy)), 1e-12)),
                    "slope": float(slope),
                    "intercept": float(intercept),
                }
            )
    return pd.DataFrame(rows_all), pd.DataFrame(rows_stable)


def summarize_oracle_eval(
    oracle_rows: pd.DataFrame,
    pred: np.ndarray,
    target_scale: float,
    oracle_cfg: OracleEvalConfig,
) -> Dict[str, object]:
    train_y_y600 = oracle_rows["target_y"].to_numpy(dtype=np.float64)
    oracle_cp = oracle_rows["oracle_reference_cp"].to_numpy(dtype=np.float64)
    oracle_y_y600 = np.tanh(oracle_cp / float(SOURCE_TARGET_SCALE))
    pred_variant = np.asarray(pred, dtype=np.float64)
    pred_y600 = remap_predictions_to_y600(pred_variant, from_scale=target_scale)
    train_y_variant = remap_target_np(train_y_y600, to_scale=target_scale).astype(np.float64)
    oracle_y_variant = np.tanh(oracle_cp / float(target_scale))
    stable_mask = oracle_rows["stability_group"].astype(str).eq(oracle_cfg.stable_label).to_numpy(dtype=bool)

    standard_metrics = summarize_scale_aware_metrics(
        y=oracle_y_y600,
        p=pred_y600,
        target_scale=SOURCE_TARGET_SCALE,
        center_thr_y600=oracle_cfg.center_thr_y600,
        pred_thr_small_y600=oracle_cfg.pred_thr_small_y600,
        pred_thr_medium_y600=oracle_cfg.pred_thr_medium_y600,
    )
    variant_standard_metrics = summarize_scale_aware_metrics(
        y=oracle_y_variant,
        p=pred_variant,
        target_scale=target_scale,
        center_thr_y600=oracle_cfg.center_thr_y600,
        pred_thr_small_y600=oracle_cfg.pred_thr_small_y600,
        pred_thr_medium_y600=oracle_cfg.pred_thr_medium_y600,
    )
    stable_thr_07 = canonical_y600_to_scale_value(0.70, SOURCE_TARGET_SCALE)
    stable_mask_07 = stable_mask & (np.abs(oracle_y_y600) <= stable_thr_07)
    if np.any(stable_mask_07):
        stable_slope, stable_intercept = base_lab.fit_line(oracle_y_y600[stable_mask_07], pred_y600[stable_mask_07])
    else:
        stable_slope, stable_intercept = float("nan"), float("nan")
    center_thr = canonical_y600_to_scale_value(oracle_cfg.center_thr_y600, SOURCE_TARGET_SCALE)
    pred_small = canonical_y600_to_scale_value(oracle_cfg.pred_thr_small_y600, SOURCE_TARGET_SCALE)
    pred_medium = canonical_y600_to_scale_value(oracle_cfg.pred_thr_medium_y600, SOURCE_TARGET_SCALE)
    center_mask = np.abs(oracle_y_y600) <= center_thr
    if np.any(center_mask):
        center_amp_ratio = float(
            np.mean(np.abs(pred_y600[center_mask])) / max(np.mean(np.abs(oracle_y_y600[center_mask])), 1e-12)
        )
        center_false_small = float(np.mean(np.abs(pred_y600[center_mask]) >= pred_small))
        center_false_medium = float(np.mean(np.abs(pred_y600[center_mask]) >= pred_medium))
    else:
        center_amp_ratio = float("nan")
        center_false_small = float("nan")
        center_false_medium = float("nan")

    band_all_df, band_stable_df = build_oracle_band_summary(
        oracle_y=oracle_y_y600,
        train_y=train_y_y600,
        pred=pred_y600,
        stable_mask=stable_mask,
        target_scale=SOURCE_TARGET_SCALE,
    )
    midband_labels = {"[0.050,0.200]", "[0.200,0.500]", "[0.500,0.700]"}
    midband_mae_sum = float(band_stable_df[band_stable_df["band_label_y600"].isin(midband_labels)]["teacher_vs_oracle_mae"].sum())
    summary = {
        "target_scale": float(target_scale),
        "metric_scale": float(SOURCE_TARGET_SCALE),
        "overall_teacher_vs_oracle_mae": float(np.mean(np.abs(pred_y600 - oracle_y_y600))),
        "overall_train_vs_oracle_mae": float(np.mean(np.abs(train_y_y600 - oracle_y_y600))),
        "teacher_closer_to_oracle_rate": float(np.mean(np.abs(pred_y600 - oracle_y_y600) < np.abs(train_y_y600 - oracle_y_y600))),
        "stable_0.7_slope": float(stable_slope),
        "stable_0.7_intercept": float(stable_intercept),
        "center_amp_ratio_eq_0.05": center_amp_ratio,
        "center_false_pred0.1eq": center_false_small,
        "center_false_pred0.2eq": center_false_medium,
        "center_n_eq_0.05": int(np.sum(center_mask)),
        "midband_teacher_vs_oracle_mae_sum_stable": midband_mae_sum,
    }
    return {
        "summary": summary,
        "standard_metrics": standard_metrics,
        "variant_standard_metrics": variant_standard_metrics,
        "oracle_band_summary": band_all_df,
        "oracle_stable_band_summary": band_stable_df,
        "oracle_bucket_table": base_lab.compute_bucket_table(oracle_y_y600, pred_y600),
        "oracle_stable_bucket_table": base_lab.compute_bucket_table(oracle_y_y600, pred_y600, include_mask=stable_mask),
        "oracle_targets": oracle_y_y600,
        "train_targets": train_y_y600,
        "preds": pred_y600,
        "oracle_targets_variant": oracle_y_variant,
        "train_targets_variant": train_y_variant,
        "preds_variant": pred_variant,
    }


def save_oracle_eval_outputs(result: Dict[str, object], output_dir: Path, prefix: str) -> None:
    reports_dir = output_dir / "reports"
    plots_dir = output_dir / "plots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_json(result["summary"], reports_dir / f"{prefix}_oracle_summary.json")
    save_json(result["standard_metrics"], reports_dir / f"{prefix}_oracle_scale_metrics.json")
    save_json(result["variant_standard_metrics"], reports_dir / f"{prefix}_oracle_scale_metrics_variant_space.json")
    save_dataframe(result["oracle_band_summary"], reports_dir / f"{prefix}_oracle_band_summary.csv")
    save_dataframe(result["oracle_stable_band_summary"], reports_dir / f"{prefix}_oracle_stable_band_summary.csv")
    save_dataframe(result["oracle_bucket_table"], reports_dir / f"{prefix}_oracle_bucket_table.csv")
    save_dataframe(result["oracle_stable_bucket_table"], reports_dir / f"{prefix}_oracle_stable_bucket_table.csv")
    base_lab.plot_scatter(result["oracle_targets"], result["preds"], plots_dir / f"{prefix}_oracle_scatter.png", title=f"{prefix} oracle scatter", max_points=20_000)
    base_lab.plot_bucket_calibration(result["oracle_bucket_table"], plots_dir / f"{prefix}_oracle_bucket_calibration.png", title=f"{prefix} oracle bucket calibration")


def evaluate_model_on_split_scale_aware(
    model: nn.Module,
    data_root: str | Path,
    split: str,
    device: torch.device,
    max_samples: int,
    num_shards: Optional[int],
    batch_size: int,
    target_scale: float,
    oracle_cfg: OracleEvalConfig,
) -> Dict[str, object]:
    X, y_source = base_lab.load_split_arrays(data_root, split, max_samples=max_samples, num_shards=num_shards)
    y_source = y_source.astype(np.float64)
    y_variant = remap_target_np(y_source, to_scale=target_scale).astype(np.float64)
    pred = base_lab.predict_array(model, X, device=device, batch_size=batch_size, use_amp=True, progress_name=f"{split}_{target_scale:g}")
    pred_variant = pred.astype(np.float64)
    pred_y600 = remap_predictions_to_y600(pred_variant, from_scale=target_scale)
    metrics_y600 = summarize_scale_aware_metrics(
        y=y_source,
        p=pred_y600,
        target_scale=SOURCE_TARGET_SCALE,
        center_thr_y600=oracle_cfg.center_thr_y600,
        pred_thr_small_y600=oracle_cfg.pred_thr_small_y600,
        pred_thr_medium_y600=oracle_cfg.pred_thr_medium_y600,
    )
    metrics_variant = summarize_scale_aware_metrics(
        y=y_variant,
        p=pred_variant,
        target_scale=target_scale,
        center_thr_y600=oracle_cfg.center_thr_y600,
        pred_thr_small_y600=oracle_cfg.pred_thr_small_y600,
        pred_thr_medium_y600=oracle_cfg.pred_thr_medium_y600,
    )
    return {
        "targets": y_source,
        "preds": pred_y600,
        "metrics": metrics_y600,
        "targets_variant": y_variant,
        "preds_variant": pred_variant,
        "metrics_variant": metrics_variant,
    }


def evaluate_model_on_oracle_subset(
    model: nn.Module,
    oracle_bundle: Dict[str, object],
    device: torch.device,
    target_scale: float,
    oracle_cfg: OracleEvalConfig,
) -> Dict[str, object]:
    pred = base_lab.predict_array(
        model,
        oracle_bundle["X"],
        device=device,
        batch_size=oracle_cfg.oracle_batch_size,
        use_amp=True,
        progress_name=f"oracle_subset_{target_scale:g}",
    )
    return summarize_oracle_eval(oracle_bundle["rows"], pred.astype(np.float64), target_scale=target_scale, oracle_cfg=oracle_cfg)


def variant_run_dir(paths: Dict[str, Path], variant_name: str) -> Path:
    run_dir = paths["runs_dir"] / variant_name
    for sub in ("reports", "plots", "checkpoints"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def target_to_logits(y: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.atanh(torch.clamp(y, -1.0 + eps, 1.0 - eps))


def huber_per_sample(residual: torch.Tensor, delta: float) -> torch.Tensor:
    abs_res = residual.abs()
    return torch.where(abs_res <= delta, 0.5 * residual * residual, delta * (abs_res - 0.5 * delta))


def compute_variant_terms(logits: torch.Tensor, y: torch.Tensor, variant: AblationVariant) -> Dict[str, torch.Tensor]:
    pred = torch.tanh(logits)
    mse_per = (pred - y) ** 2
    y_logits = target_to_logits(y, eps=variant.target_clamp_eps)
    residual = logits - y_logits
    y_clamped = torch.clamp(y, -1.0 + variant.target_clamp_eps, 1.0 - variant.target_clamp_eps)
    z_weight = torch.pow(torch.clamp(1.0 - y_clamped * y_clamped, min=variant.target_clamp_eps), variant.z_loss_beta)
    z_huber_per = z_weight * huber_per_sample(residual, variant.z_huber_delta)

    if variant.loss_mode == "baseline_hybrid":
        y_term = mse_per
        main_per = variant.lambda_y * y_term + (1.0 - variant.lambda_y) * z_huber_per
    elif variant.loss_mode == "curvature_compensated":
        y_curv = torch.square(torch.clamp(1.0 - y_clamped * y_clamped, min=variant.target_clamp_eps))
        y_comp = torch.clamp(1.0 / (y_curv + variant.y_reweight_eps), min=1.0, max=variant.y_reweight_clip_max)
        y_comp = y_comp / torch.clamp(y_comp.mean(), min=1e-6)
        y_term = y_comp * mse_per
        main_per = variant.y_loss_alpha * y_term + (1.0 - variant.y_loss_alpha) * z_huber_per
    else:  # pragma: no cover
        raise ValueError(f"Unsupported loss_mode: {variant.loss_mode}")

    objective = torch.mean(main_per)
    center_penalty = pred.new_tensor(0.0)
    if variant.center_penalty_weight > 0.0:
        tau = canonical_y600_to_scale_value(variant.center_penalty_tau_y600, variant.target_scale)
        margin = canonical_y600_to_scale_value(variant.center_penalty_margin_y600, variant.target_scale)
        center_mask = torch.abs(y) <= tau
        if torch.any(center_mask):
            center_penalty = torch.mean(torch.relu(torch.abs(pred[center_mask]) - margin) ** 2)
            objective = objective + float(variant.center_penalty_weight) * center_penalty
    return {
        "objective": objective,
        "main_term": torch.mean(main_per),
        "y_term": torch.mean(y_term),
        "z_term": torch.mean(z_huber_per),
        "center_penalty": center_penalty,
        "pred": pred,
    }


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay_params: List[torch.nn.Parameter] = []
    no_decay_params: List[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lower = name.lower()
        if param.ndim <= 1 or name.endswith(".bias") or ".bn" in lower or "norm" in lower:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def build_recommended_variants() -> List[AblationVariant]:
    return [
        AblationVariant(
            name="A1_curvature_compensated",
            description="Curvature-compensated loss to test Failure A directly.",
            target_scale=600.0,
            sampler_mode="random",
            loss_mode="curvature_compensated",
            y_loss_alpha=0.65,
            z_loss_beta=0.0,
            z_huber_delta=1.0,
            y_reweight_clip_max=4.0,
            center_penalty_weight=0.0,
        ),
        AblationVariant(
            name="A2_band_balanced",
            description="Band-balanced sampling with baseline late-stage objective.",
            target_scale=600.0,
            sampler_mode="band_balanced",
            loss_mode="baseline_hybrid",
            lambda_y=0.99,
            z_loss_beta=1.0,
            z_huber_delta=0.5,
        ),
        AblationVariant(
            name="C1_scale800",
            description="Target remap 600 -> 800 while keeping objective otherwise unchanged.",
            target_scale=800.0,
            sampler_mode="random",
            loss_mode="baseline_hybrid",
            lambda_y=0.99,
            z_loss_beta=1.0,
            z_huber_delta=0.5,
        ),
        AblationVariant(
            name="C2_scale1200",
            description="Target remap 600 -> 1200 while keeping objective otherwise unchanged.",
            target_scale=1200.0,
            sampler_mode="random",
            loss_mode="baseline_hybrid",
            lambda_y=0.99,
            z_loss_beta=1.0,
            z_huber_delta=0.5,
        ),
        AblationVariant(
            name="B1_center_penalty",
            description="Center confidence penalty to address ultra-center over-amplification after a winner is known.",
            target_scale=600.0,
            sampler_mode="random",
            loss_mode="baseline_hybrid",
            lambda_y=0.99,
            z_loss_beta=1.0,
            z_huber_delta=0.5,
            center_penalty_weight=0.25,
            center_penalty_tau_y600=0.05,
            center_penalty_margin_y600=0.10,
        ),
    ]


def oracle_gate_score(summary: Dict[str, object]) -> float:
    required = [
        float(summary["midband_teacher_vs_oracle_mae_sum_stable"]),
        float(summary["stable_0.7_slope"]),
        float(summary["center_false_pred0.1eq"]),
        float(summary["center_false_pred0.2eq"]),
    ]
    if not all(np.isfinite(v) for v in required):
        return float("inf")
    return float(
        summary["midband_teacher_vs_oracle_mae_sum_stable"]
        + 0.5 * max(0.0, 0.80 - summary["stable_0.7_slope"])
        + 0.5 * summary["center_false_pred0.1eq"]
        + 0.25 * summary["center_false_pred0.2eq"]
    )


def run_variant_finetune(
    init_ckpt_path: str | Path,
    data_root: str | Path,
    variant: AblationVariant,
    train_cfg: TrainConfig,
    oracle_cfg: OracleEvalConfig,
    oracle_bundle: Dict[str, object],
    paths: Dict[str, Path],
    device: torch.device,
) -> Dict[str, object]:
    run_dir = variant_run_dir(paths, variant.name)
    reports_dir = run_dir / "reports"
    checkpoints_dir = run_dir / "checkpoints"

    set_global_seed(train_cfg.seed)
    model, init_ckpt = base_lab.load_model_from_checkpoint(init_ckpt_path, device=device)
    optimizer = build_optimizer(model, lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    shard_rows = resolve_split_shards(data_root, "train", num_shards=train_cfg.train_num_shards)
    total_samples = int(sum(int(np.load(y_path, mmap_mode="r").shape[0]) for _, _, y_path in shard_rows))
    total_steps = int(math.ceil(total_samples / train_cfg.batch_size) * max(train_cfg.epochs, 1))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=train_cfg.min_lr)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    history_rows: List[dict] = []
    best_gate = float("inf")
    best_ckpt = checkpoints_dir / f"{variant.name}_best.pt"
    latest_ckpt = checkpoints_dir / f"{variant.name}_latest.pt"
    rng = np.random.default_rng(train_cfg.seed)
    global_step = 0

    for epoch in range(train_cfg.epochs):
        model.train()
        t0 = time.time()
        running = {"objective": 0.0, "main_term": 0.0, "y_term": 0.0, "z_term": 0.0, "center_penalty": 0.0, "n": 0}
        for shard_rank, (_, x_path, y_path) in enumerate(shard_rows, start=1):
            X = np.load(x_path, mmap_mode="r")
            y_source = np.load(y_path, mmap_mode="r").astype(np.float32, copy=False)
            y = remap_target_np(y_source, to_scale=variant.target_scale).astype(np.float32, copy=False)
            if variant.sampler_mode == "band_balanced":
                order = build_band_balanced_order(
                    abs_y=np.abs(y.astype(np.float64)),
                    batch_size=train_cfg.batch_size,
                    band_edges_y600=variant.balance_band_edges_y600,
                    rng=rng,
                    target_scale=variant.target_scale,
                )
            else:
                order = rng.permutation(y.shape[0]).astype(np.int64)
            for start in range(0, y.shape[0], train_cfg.batch_size):
                idx = order[start : start + train_cfg.batch_size]
                xb = torch.from_numpy(np.array(X[idx], dtype=np.float32, copy=True)).to(device, non_blocking=True)
                yb = torch.from_numpy(np.array(y[idx], dtype=np.float32, copy=True)).to(device, non_blocking=True).view(-1)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    logits = model.forward_logits(xb).view(-1)
                    terms = compute_variant_terms(logits, yb, variant)
                scaler.scale(terms["objective"]).backward()
                if train_cfg.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                bs = int(yb.numel())
                for key in ("objective", "main_term", "y_term", "z_term", "center_penalty"):
                    running[key] += float(terms[key].item()) * bs
                running["n"] += bs
                global_step += 1
                if global_step % train_cfg.log_every_steps == 0:
                    print(
                        f"[{variant.name}][epoch={epoch}] step={global_step}/{total_steps} "
                        f"obj={running['objective'] / running['n']:.6f} "
                        f"y_term={running['y_term'] / running['n']:.6f} "
                        f"z_term={running['z_term'] / running['n']:.6f} "
                        f"center_pen={running['center_penalty'] / running['n']:.6f}"
                    )
            print(f"[{variant.name}] finished shard {shard_rank}/{len(shard_rows)}")

        val_eval = evaluate_model_on_split_scale_aware(
            model,
            data_root=data_root,
            split="val",
            device=device,
            max_samples=train_cfg.val_max_samples,
            num_shards=train_cfg.val_num_shards,
            batch_size=max(train_cfg.batch_size, 1024),
            target_scale=variant.target_scale,
            oracle_cfg=oracle_cfg,
        )
        oracle_eval = evaluate_model_on_oracle_subset(
            model,
            oracle_bundle=oracle_bundle,
            device=device,
            target_scale=variant.target_scale,
            oracle_cfg=oracle_cfg,
        )
        gate = oracle_gate_score(oracle_eval["summary"])
        row = {
            "epoch": int(epoch),
            "train_objective": running["objective"] / running["n"],
            "train_main_term": running["main_term"] / running["n"],
            "train_y_term": running["y_term"] / running["n"],
            "train_z_term": running["z_term"] / running["n"],
            "train_center_penalty": running["center_penalty"] / running["n"],
            "val_mse_0.7eq": float(val_eval["metrics"]["bands"]["0.70"]["mse"]),
            "val_slope_0.7eq": float(val_eval["metrics"]["bands"]["0.70"]["slope"]),
            "oracle_stable_0.7_slope": float(oracle_eval["summary"]["stable_0.7_slope"]),
            "oracle_center_amp_ratio": float(oracle_eval["summary"]["center_amp_ratio_eq_0.05"]),
            "oracle_center_false_0.1eq": float(oracle_eval["summary"]["center_false_pred0.1eq"]),
            "oracle_center_false_0.2eq": float(oracle_eval["summary"]["center_false_pred0.2eq"]),
            "oracle_midband_mae_sum_stable": float(oracle_eval["summary"]["midband_teacher_vs_oracle_mae_sum_stable"]),
            "oracle_gate_score": float(gate),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_time_sec": float(time.time() - t0),
        }
        history_rows.append(row)
        print(json.dumps(row, indent=2))

        payload = {
            "epoch": int(epoch),
            "variant": asdict(variant),
            "train_cfg": asdict(train_cfg),
            "oracle_cfg": asdict(oracle_cfg),
            "history": history_rows,
            "config": init_ckpt.get("config"),
            "model_state": model.state_dict(),
            "val_metrics": val_eval["metrics"],
            "oracle_summary": oracle_eval["summary"],
        }
        atomic_torch_save(payload, latest_ckpt)
        if gate < best_gate:
            best_gate = gate
            atomic_torch_save(payload, best_ckpt)
            save_oracle_eval_outputs(oracle_eval, run_dir, prefix="best_val_oracle")
            save_json(val_eval["metrics"], reports_dir / "best_val_scale_metrics.json")
            save_json(val_eval["metrics_variant"], reports_dir / "best_val_scale_metrics_variant_space.json")

    history_df = pd.DataFrame(history_rows)
    save_dataframe(history_df, reports_dir / "history.csv")
    save_json({"history": history_rows, "best_gate_score": best_gate, "variant": asdict(variant)}, reports_dir / "history.json")

    best_model, _ = base_lab.load_model_from_checkpoint(best_ckpt, device=device)
    test_eval = evaluate_model_on_split_scale_aware(
        best_model,
        data_root=data_root,
        split="test",
        device=device,
        max_samples=train_cfg.test_max_samples,
        num_shards=train_cfg.test_num_shards,
        batch_size=max(train_cfg.batch_size, 1024),
        target_scale=variant.target_scale,
        oracle_cfg=oracle_cfg,
    )
    oracle_test_eval = evaluate_model_on_oracle_subset(
        best_model,
        oracle_bundle=oracle_bundle,
        device=device,
        target_scale=variant.target_scale,
        oracle_cfg=oracle_cfg,
    )
    save_json(test_eval["metrics"], reports_dir / "best_test_scale_metrics.json")
    save_json(test_eval["metrics_variant"], reports_dir / "best_test_scale_metrics_variant_space.json")
    save_oracle_eval_outputs(oracle_test_eval, run_dir, prefix="best_test_oracle")
    return {
        "run_dir": run_dir,
        "best_checkpoint": best_ckpt,
        "history": history_df,
        "test_eval": test_eval,
        "oracle_test_eval": oracle_test_eval,
        "variant": variant,
    }


def baseline_snapshot(
    init_ckpt_path: str | Path,
    data_root: str | Path,
    oracle_bundle: Dict[str, object],
    oracle_cfg: OracleEvalConfig,
    train_cfg: TrainConfig,
    paths: Dict[str, Path],
    device: torch.device,
    target_scale: float = SOURCE_TARGET_SCALE,
) -> Dict[str, object]:
    model, _ = base_lab.load_model_from_checkpoint(init_ckpt_path, device=device)
    val_eval = evaluate_model_on_split_scale_aware(
        model,
        data_root=data_root,
        split="val",
        device=device,
        max_samples=train_cfg.val_max_samples,
        num_shards=train_cfg.val_num_shards,
        batch_size=max(train_cfg.batch_size, 1024),
        target_scale=target_scale,
        oracle_cfg=oracle_cfg,
    )
    test_eval = evaluate_model_on_split_scale_aware(
        model,
        data_root=data_root,
        split="test",
        device=device,
        max_samples=train_cfg.test_max_samples,
        num_shards=train_cfg.test_num_shards,
        batch_size=max(train_cfg.batch_size, 1024),
        target_scale=target_scale,
        oracle_cfg=oracle_cfg,
    )
    oracle_eval = evaluate_model_on_oracle_subset(
        model,
        oracle_bundle=oracle_bundle,
        device=device,
        target_scale=target_scale,
        oracle_cfg=oracle_cfg,
    )
    save_json(val_eval["metrics"], paths["reports_dir"] / "baseline_val_scale_metrics.json")
    save_json(val_eval["metrics_variant"], paths["reports_dir"] / "baseline_val_scale_metrics_variant_space.json")
    save_json(test_eval["metrics"], paths["reports_dir"] / "baseline_test_scale_metrics.json")
    save_json(test_eval["metrics_variant"], paths["reports_dir"] / "baseline_test_scale_metrics_variant_space.json")
    save_oracle_eval_outputs(oracle_eval, paths["output_dir"], prefix="baseline")
    return {"val_eval": val_eval, "test_eval": test_eval, "oracle_eval": oracle_eval}


def compare_runs_table(baseline: Dict[str, object], variant_runs: Sequence[Dict[str, object]]) -> pd.DataFrame:
    rows = [
        {
            "label": "baseline",
            "target_scale": float(baseline["oracle_eval"]["summary"]["target_scale"]),
            "test_mse_0.7eq": float(baseline["test_eval"]["metrics"]["bands"]["0.70"]["mse"]),
            "test_slope_0.7eq": float(baseline["test_eval"]["metrics"]["bands"]["0.70"]["slope"]),
            "oracle_stable_0.7_slope": float(baseline["oracle_eval"]["summary"]["stable_0.7_slope"]),
            "oracle_midband_mae_sum_stable": float(baseline["oracle_eval"]["summary"]["midband_teacher_vs_oracle_mae_sum_stable"]),
            "oracle_center_amp_ratio": float(baseline["oracle_eval"]["summary"]["center_amp_ratio_eq_0.05"]),
            "oracle_center_false_0.1eq": float(baseline["oracle_eval"]["summary"]["center_false_pred0.1eq"]),
            "oracle_center_false_0.2eq": float(baseline["oracle_eval"]["summary"]["center_false_pred0.2eq"]),
            "oracle_gate_score": float(oracle_gate_score(baseline["oracle_eval"]["summary"])),
        }
    ]
    for run in variant_runs:
        rows.append(
            {
                "label": run["variant"].name,
                "target_scale": float(run["oracle_test_eval"]["summary"]["target_scale"]),
                "test_mse_0.7eq": float(run["test_eval"]["metrics"]["bands"]["0.70"]["mse"]),
                "test_slope_0.7eq": float(run["test_eval"]["metrics"]["bands"]["0.70"]["slope"]),
                "oracle_stable_0.7_slope": float(run["oracle_test_eval"]["summary"]["stable_0.7_slope"]),
                "oracle_midband_mae_sum_stable": float(run["oracle_test_eval"]["summary"]["midband_teacher_vs_oracle_mae_sum_stable"]),
                "oracle_center_amp_ratio": float(run["oracle_test_eval"]["summary"]["center_amp_ratio_eq_0.05"]),
                "oracle_center_false_0.1eq": float(run["oracle_test_eval"]["summary"]["center_false_pred0.1eq"]),
                "oracle_center_false_0.2eq": float(run["oracle_test_eval"]["summary"]["center_false_pred0.2eq"]),
                "oracle_gate_score": float(oracle_gate_score(run["oracle_test_eval"]["summary"])),
            }
        )
    df = pd.DataFrame(rows)
    baseline_gate = float(df.loc[df["label"] == "baseline", "oracle_gate_score"].iloc[0])
    df["delta_gate_vs_baseline"] = df["oracle_gate_score"] - baseline_gate
    return df
