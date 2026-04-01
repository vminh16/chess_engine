from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "model") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "model"))

from architecture_v2.model import DGRNChessNetV2  # noqa: E402
from core.board import Board  # noqa: E402
from core.constants import Color, Piece, PieceType, Square  # noqa: E402
from evaluation.static_eval import material_evaluate  # noqa: E402
from representation.encode import encode_board  # noqa: E402


PIECE_TYPES = [
    PieceType.PAWN,
    PieceType.KNIGHT,
    PieceType.BISHOP,
    PieceType.ROOK,
    PieceType.QUEEN,
    PieceType.KING,
]
BUCKET_EDGES = np.linspace(-1.0, 1.0, 21)


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


@dataclass
class PilotTrainConfig:
    lambda_y: float = 0.99
    z_loss_beta: float = 1.0
    z_huber_delta: float = 0.5
    target_clamp_eps: float = 1e-3
    center_tau: float = 0.10
    center_margin: float = 0.20
    center_weight: float = 0.25
    learning_rate: float = 5e-6
    min_lr: float = 1e-6
    weight_decay: float = 2e-4
    grad_clip_norm: float = 1.0
    batch_size: int = 1024
    epochs: int = 3
    train_max_samples: int = 200_000
    val_max_samples: int = 50_000
    train_num_shards: int = 8
    val_num_shards: int = 2
    log_every: int = 50
    seed: int = 123


def build_default_paths(
    run_dir: str | Path = r"C:\Users\USER\Downloads\dgrn_5m_v3_stage2_polish_run1",
    data_root: str | Path = r"C:\Users\USER\Desktop\chess_engine\data\process",
    experiment_dir: str | Path | None = None,
) -> Dict[str, Path]:
    if experiment_dir is None:
        experiment_dir = PROJECT_ROOT / "experiments" / "teacher_root_cause_lab"
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
    for key in ("experiment_dir", "output_dir", "plots_dir", "reports_dir", "checkpoints_dir", "cache_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def export_paths_json(paths: Dict[str, Path], out_path: Path) -> None:
    payload = ExperimentPaths(**{key: str(value) for key, value in paths.items()})
    out_path.write_text(json.dumps(asdict(payload), indent=2), encoding="utf-8")


def set_global_seed(seed: int = 123) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def resolve_split_pairs(data_root: str | Path, split: str) -> List[Tuple[Path, Path]]:
    split_dir = Path(data_root) / split
    x_files = sorted(split_dir.glob("X_*.npy"))
    y_files = sorted(split_dir.glob("y_*.npy"))
    if not x_files or not y_files:
        raise FileNotFoundError(f"Missing shards for split={split} at {split_dir}")
    if len(x_files) != len(y_files):
        raise ValueError(f"Shard count mismatch for split={split}: X={len(x_files)} y={len(y_files)}")
    return list(zip(x_files, y_files))


def select_pairs_evenly(pairs: Sequence[Tuple[Path, Path]], num_pairs: Optional[int]) -> List[Tuple[Path, Path]]:
    if num_pairs is None or num_pairs >= len(pairs):
        return list(pairs)
    idx = np.unique(np.linspace(0, len(pairs) - 1, num_pairs, dtype=int))
    return [pairs[int(i)] for i in idx]


def iter_xy_batches(
    data_root: str | Path,
    split: str,
    batch_size: int = 2048,
    max_samples: Optional[int] = None,
    num_shards: Optional[int] = None,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    total = 0
    for x_path, y_path in select_pairs_evenly(resolve_split_pairs(data_root, split), num_shards):
        X = np.load(x_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r").astype(np.float32, copy=False)
        n = X.shape[0]
        if max_samples is not None:
            remain = max_samples - total
            if remain <= 0:
                break
            n = min(n, remain)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            yield np.asarray(X[start:end]), np.asarray(y[start:end], dtype=np.float32)
            total += end - start
            if max_samples is not None and total >= max_samples:
                break
        if max_samples is not None and total >= max_samples:
            break


def load_split_arrays(
    data_root: str | Path,
    split: str,
    max_samples: int,
    num_shards: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    total = 0
    for x_batch, y_batch in iter_xy_batches(data_root, split, batch_size=50_000, max_samples=max_samples, num_shards=num_shards):
        x_parts.append(x_batch.astype(np.uint8, copy=False))
        y_parts.append(y_batch.astype(np.float32, copy=False))
        total += y_batch.shape[0]
        if total >= max_samples:
            break
    if not x_parts:
        raise RuntimeError(f"No samples collected for split={split}")
    X = np.concatenate(x_parts, axis=0)[:max_samples]
    y = np.concatenate(y_parts, axis=0)[:max_samples]
    return X, y


def load_model_from_checkpoint(ckpt_path: str | Path, device: torch.device) -> Tuple[nn.Module, dict]:
    ckpt = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    model_cfg = (ckpt.get("config") or {}).get("model_cfg") or {}
    model = DGRNChessNetV2(**model_cfg)
    state_dict = ckpt.get("model")
    if state_dict is None:
        state_dict = ckpt.get("model_state")
    if state_dict is None:
        raise KeyError(f"Checkpoint does not contain 'model' or 'model_state': {ckpt_path}")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, ckpt


def predict_array(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
    use_amp: bool = True,
    progress_name: str = "predict",
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    start_time = time.time()
    with torch.no_grad():
        for offset in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(np.array(X[offset : offset + batch_size], dtype=np.float32, copy=True)).to(device=device)
            with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
                pred = model(xb).view(-1)
            outputs.append(pred.detach().cpu().numpy().astype(np.float64))
            if offset == 0 or ((offset // batch_size) % 25 == 0):
                print(f"[{progress_name}] offset={offset} / {X.shape[0]} elapsed={time.time() - start_time:.1f}s")
    if device.type == "cuda":
        torch.cuda.synchronize()
    return np.clip(np.concatenate(outputs, axis=0), -1.0, 1.0)


def collect_predictions_for_checkpoint(
    ckpt_path: str | Path,
    data_root: str | Path,
    split: str,
    device: torch.device,
    max_samples: int,
    batch_size: int = 2048,
    num_shards: Optional[int] = None,
    cache_prefix: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cache_key = {
        "ckpt_path": str(ckpt_path.resolve()),
        "split": split,
        "max_samples": int(max_samples),
        "num_shards": -1 if num_shards is None else int(num_shards),
    }
    if cache_prefix is not None and cache_prefix.exists():
        cached = np.load(cache_prefix, allow_pickle=False)
        cached_ckpt_path = str(cached["ckpt_path"].tolist())
        cached_split = str(cached["split"].tolist())
        cached_max_samples = int(cached["max_samples"].tolist())
        cached_num_shards = int(cached["num_shards"].tolist())
        if (
            cached_ckpt_path == cache_key["ckpt_path"]
            and cached_split == cache_key["split"]
            and cached_max_samples == cache_key["max_samples"]
            and cached_num_shards == cache_key["num_shards"]
        ):
            return {
                "targets": cached["targets"].astype(np.float64),
                "preds": cached["preds"].astype(np.float64),
                "ckpt": ckpt,
            }

    X, y = load_split_arrays(data_root, split, max_samples=max_samples, num_shards=num_shards)
    model, _ = load_model_from_checkpoint(ckpt_path, device=device)
    pred = predict_array(
        model,
        X,
        device=device,
        batch_size=batch_size,
        use_amp=True,
        progress_name=f"{Path(ckpt_path).stem}:{split}",
    )
    payload = {"targets": y.astype(np.float64), "preds": pred.astype(np.float64), "ckpt": ckpt}
    if cache_prefix is not None:
        cache_prefix.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_prefix,
            targets=payload["targets"].astype(np.float32),
            preds=payload["preds"].astype(np.float32),
            ckpt_path=np.asarray(cache_key["ckpt_path"]),
            split=np.asarray(cache_key["split"]),
            max_samples=np.asarray(cache_key["max_samples"], dtype=np.int64),
            num_shards=np.asarray(cache_key["num_shards"], dtype=np.int64),
        )
    return payload


def r2_score(y: np.ndarray, p: np.ndarray) -> float:
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def pearsonr(y: np.ndarray, p: np.ndarray) -> float:
    y0 = y - np.mean(y)
    p0 = p - np.mean(p)
    den = math.sqrt(float(np.sum(y0 * y0) * np.sum(p0 * p0))) + 1e-12
    return float(np.sum(y0 * p0) / den)


def fit_line(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    y_mean = float(np.mean(y))
    p_mean = float(np.mean(p))
    var = float(np.mean((y - y_mean) ** 2))
    if var < 1e-12:
        return float("nan"), float("nan")
    slope = float(np.mean((y - y_mean) * (p - p_mean)) / var)
    intercept = float(p_mean - slope * y_mean)
    return slope, intercept


def bucket_id(v: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bid = np.searchsorted(edges, v, side="right") - 1
    return np.clip(bid, 0, len(edges) - 2)


def compute_bucket_table(
    y: np.ndarray,
    p: np.ndarray,
    edges: np.ndarray = BUCKET_EDGES,
    include_mask: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    if include_mask is None:
        include_mask = np.ones_like(y, dtype=bool)
    yy = y[include_mask]
    pp = p[include_mask]
    bid = bucket_id(yy, edges)
    rows = []
    for b in range(len(edges) - 1):
        mask = bid == b
        if not np.any(mask):
            rows.append(
                {
                    "bucket": b,
                    "left": float(edges[b]),
                    "right": float(edges[b + 1]),
                    "center": float(0.5 * (edges[b] + edges[b + 1])),
                    "count": 0,
                    "mse": float("nan"),
                    "mae": float("nan"),
                    "mean_y": float("nan"),
                    "mean_p": float("nan"),
                    "bias": float("nan"),
                    "abs_cal_gap": float("nan"),
                }
            )
            continue
        yb = yy[mask]
        pb = pp[mask]
        rows.append(
            {
                "bucket": b,
                "left": float(edges[b]),
                "right": float(edges[b + 1]),
                "center": float(0.5 * (edges[b] + edges[b + 1])),
                "count": int(mask.sum()),
                "mse": float(np.mean((pb - yb) ** 2)),
                "mae": float(np.mean(np.abs(pb - yb))),
                "mean_y": float(np.mean(yb)),
                "mean_p": float(np.mean(pb)),
                "bias": float(np.mean(pb - yb)),
                "abs_cal_gap": float(abs(np.mean(pb) - np.mean(yb))),
            }
        )
    return pd.DataFrame(rows)


def compute_band_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    mask = np.abs(y) <= thr
    yy = y[mask]
    pp = p[mask]
    slope, intercept = fit_line(yy, pp)
    return {
        "n": int(mask.sum()),
        "mse": float(np.mean((pp - yy) ** 2)),
        "mae": float(np.mean(np.abs(pp - yy))),
        "bias": float(np.mean(pp - yy)),
        "mean_abs_y": float(np.mean(np.abs(yy))),
        "mean_abs_p": float(np.mean(np.abs(pp))),
        "r2": r2_score(yy, pp),
        "pearson": pearsonr(yy, pp),
        "slope": slope,
        "intercept": intercept,
    }


def false_decisive_rate(y: np.ndarray, p: np.ndarray, y_thr: float, p_thr: float) -> Dict[str, float]:
    mask = np.abs(y) <= y_thr
    yy = y[mask]
    pp = p[mask]
    decisive = np.abs(pp) >= p_thr
    wrong_sign = decisive & (np.sign(pp) != np.sign(yy)) & (np.abs(yy) > 0.02)
    return {
        "n": int(mask.sum()),
        "rate": float(np.mean(decisive)),
        "wrong_sign_rate": float(np.mean(wrong_sign)),
    }


def center_spread_ratio(y: np.ndarray, p: np.ndarray, thr: float = 0.05) -> Dict[str, float]:
    mask = np.abs(y) <= thr
    yy = y[mask]
    pp = p[mask]
    return {
        "n": int(mask.sum()),
        "std_y": float(np.std(yy)),
        "std_p": float(np.std(pp)),
        "ratio": float(np.std(pp) / (np.std(yy) + 1e-12)),
    }


def summarize_teacher_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, object]:
    bucket_table = compute_bucket_table(y, p)
    midband_table = bucket_table[(bucket_table["center"] >= -0.7) & (bucket_table["center"] <= 0.7) & (bucket_table["count"] > 0)]
    worst_midband = midband_table.sort_values("mse", ascending=False).head(1).to_dict(orient="records")[0]
    return {
        "overall": {
            "mse": float(np.mean((p - y) ** 2)),
            "mae": float(np.mean(np.abs(p - y))),
            "bias": float(np.mean(p - y)),
            "mean_abs_pred": float(np.mean(np.abs(p))),
            "r2": r2_score(y, p),
            "pearson": pearsonr(y, p),
        },
        "bands": {str(thr): compute_band_metrics(y, p, thr) for thr in (0.1, 0.2, 0.5, 0.7)},
        "false_decisive": {
            "y<=0.1,p>=0.3": false_decisive_rate(y, p, 0.1, 0.3),
            "y<=0.2,p>=0.4": false_decisive_rate(y, p, 0.2, 0.4),
            "y<=0.2,p>=0.5": false_decisive_rate(y, p, 0.2, 0.5),
        },
        "center_spread_ratio_0.05": center_spread_ratio(y, p, 0.05),
        "max_midband_abs_cal_gap": float(midband_table["abs_cal_gap"].max()),
        "worst_midband_bucket": worst_midband,
    }


def plot_scatter(y: np.ndarray, p: np.ndarray, path: Path, title: str, max_points: int = 20_000) -> None:
    rng = np.random.default_rng(123)
    if y.size > max_points:
        idx = rng.choice(y.size, size=max_points, replace=False)
        yy = y[idx]
        pp = p[idx]
    else:
        yy = y
        pp = p
    plt.figure(figsize=(6, 6))
    plt.scatter(yy, pp, s=3, alpha=0.25)
    plt.plot([-1, 1], [-1, 1], linestyle="--", color="black", linewidth=1)
    plt.xlabel("target")
    plt.ylabel("prediction")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_bucket_calibration(bucket_table: pd.DataFrame, path: Path, title: str) -> None:
    df = bucket_table[bucket_table["count"] > 0].copy()
    plt.figure(figsize=(7, 4))
    plt.plot(df["center"], df["mean_y"], marker="o", label="mean target")
    plt.plot(df["center"], df["mean_p"], marker="o", label="mean prediction")
    plt.xlabel("target bucket center")
    plt.ylabel("value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_bucket_mse(bucket_table: pd.DataFrame, path: Path, title: str) -> None:
    df = bucket_table[bucket_table["count"] > 0].copy()
    plt.figure(figsize=(7, 4))
    plt.plot(df["center"], df["mse"], marker="o")
    plt.xlabel("target bucket center")
    plt.ylabel("MSE")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_center_prediction_hist(y: np.ndarray, p: np.ndarray, path: Path, center_thr: float = 0.1) -> None:
    mask = np.abs(y) <= center_thr
    plt.figure(figsize=(7, 4))
    plt.hist(p[mask], bins=60, alpha=0.85)
    plt.xlabel("prediction")
    plt.ylabel("count")
    plt.title(f"Prediction histogram on |y| <= {center_thr}")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def run_teacher_eval_suite(
    ckpt_path: str | Path,
    data_root: str | Path,
    split: str,
    device: torch.device,
    output_dir: str | Path,
    max_samples: int = 200_000,
    batch_size: int = 2048,
    num_shards: Optional[int] = None,
    prefix: Optional[str] = None,
) -> Dict[str, object]:
    prefix = prefix or f"{Path(ckpt_path).stem}_{split}"
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    reports_dir = output_dir / "reports"
    cache_dir = output_dir / "cache"
    for d in (plots_dir, reports_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)
    payload = collect_predictions_for_checkpoint(
        ckpt_path=ckpt_path,
        data_root=data_root,
        split=split,
        device=device,
        max_samples=max_samples,
        batch_size=batch_size,
        num_shards=num_shards,
        cache_prefix=cache_dir / f"{prefix}_predictions.npz",
    )
    y = payload["targets"]
    p = payload["preds"]
    metrics = summarize_teacher_metrics(y, p)
    bucket_table = compute_bucket_table(y, p)
    save_json(metrics, reports_dir / f"{prefix}_metrics.json")
    save_dataframe(bucket_table, reports_dir / f"{prefix}_bucket_table.csv")
    plot_scatter(y, p, plots_dir / f"{prefix}_scatter_all.png", f"{prefix}: prediction vs target")
    center_mask = np.abs(y) <= 0.2
    plot_scatter(y[center_mask], p[center_mask], plots_dir / f"{prefix}_scatter_center.png", f"{prefix}: |y| <= 0.2")
    plot_bucket_calibration(bucket_table, plots_dir / f"{prefix}_bucket_calibration.png", f"{prefix}: bucket calibration")
    plot_bucket_mse(bucket_table, plots_dir / f"{prefix}_bucket_mse.png", f"{prefix}: bucket MSE")
    plot_center_prediction_hist(y, p, plots_dir / f"{prefix}_center_hist.png", center_thr=0.1)
    return {"metrics": metrics, "bucket_table": bucket_table, "targets": y, "preds": p, "ckpt": payload["ckpt"]}


class SymmetricIsotonicCalibrator:
    def __init__(self) -> None:
        self.x_thresholds: Optional[np.ndarray] = None
        self.y_thresholds: Optional[np.ndarray] = None

    def fit(self, preds: np.ndarray, targets: np.ndarray) -> "SymmetricIsotonicCalibrator":
        x = np.abs(preds).astype(np.float64)
        y = np.clip(np.abs(targets).astype(np.float64), 0.0, 1.0)
        order = np.argsort(x, kind="mergesort")
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        iso.fit(x[order], y[order])
        self.x_thresholds = np.asarray(iso.X_thresholds_, dtype=np.float64)
        self.y_thresholds = np.asarray(iso.y_thresholds_, dtype=np.float64)
        return self

    def transform(self, preds: np.ndarray) -> np.ndarray:
        if self.x_thresholds is None or self.y_thresholds is None:
            raise RuntimeError("Calibrator is not fitted.")
        abs_preds = np.abs(preds).astype(np.float64)
        if self.x_thresholds.size == 1:
            mag = np.full_like(abs_preds, fill_value=float(self.y_thresholds[0]), dtype=np.float64)
        else:
            mag = np.interp(abs_preds, self.x_thresholds, self.y_thresholds, left=self.y_thresholds[0], right=self.y_thresholds[-1])
        return np.sign(preds) * mag

    def to_dict(self) -> Dict[str, List[float]]:
        if self.x_thresholds is None or self.y_thresholds is None:
            raise RuntimeError("Calibrator is not fitted.")
        return {"x_thresholds": self.x_thresholds.tolist(), "y_thresholds": self.y_thresholds.tolist()}


def run_posthoc_calibration_experiment(
    val_y: np.ndarray,
    val_p: np.ndarray,
    test_y: np.ndarray,
    test_p: np.ndarray,
    output_dir: str | Path,
    prefix: str = "posthoc_calibration",
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    reports_dir = output_dir / "reports"
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    calibrator = SymmetricIsotonicCalibrator().fit(val_p, val_y)
    calibrated_test = np.clip(calibrator.transform(test_p), -1.0, 1.0)

    before = summarize_teacher_metrics(test_y, test_p)
    after = summarize_teacher_metrics(test_y, calibrated_test)
    report = {
        "before": before,
        "after": after,
        "delta": {
            "mse_0.7": after["bands"]["0.7"]["mse"] - before["bands"]["0.7"]["mse"],
            "slope_0.7": after["bands"]["0.7"]["slope"] - before["bands"]["0.7"]["slope"],
            "false_decisive_0.1_0.3": after["false_decisive"]["y<=0.1,p>=0.3"]["rate"]
            - before["false_decisive"]["y<=0.1,p>=0.3"]["rate"],
            "max_midband_abs_cal_gap": after["max_midband_abs_cal_gap"] - before["max_midband_abs_cal_gap"],
        },
        "calibrator": calibrator.to_dict(),
    }
    save_json(report, reports_dir / f"{prefix}_report.json")
    bucket_before = compute_bucket_table(test_y, test_p)
    bucket_after = compute_bucket_table(test_y, calibrated_test)
    save_dataframe(bucket_before, reports_dir / f"{prefix}_bucket_before.csv")
    save_dataframe(bucket_after, reports_dir / f"{prefix}_bucket_after.csv")

    plt.figure(figsize=(7, 4))
    plt.plot(bucket_before["center"], bucket_before["mean_p"], marker="o", label="before")
    plt.plot(bucket_after["center"], bucket_after["mean_p"], marker="o", label="after")
    plt.plot(bucket_before["center"], bucket_before["mean_y"], marker="o", label="target")
    plt.xlabel("target bucket center")
    plt.ylabel("value")
    plt.title("Post-hoc calibration on test buckets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / f"{prefix}_bucket_compare.png", dpi=180)
    plt.close()

    plot_scatter(test_y, calibrated_test, plots_dir / f"{prefix}_scatter_after.png", "Calibrated test predictions")
    return {"report": report, "calibrated_test_preds": calibrated_test}


def ensure_channels_first(x: np.ndarray) -> np.ndarray:
    if x.shape == (18, 8, 8):
        return x
    if x.shape == (8, 8, 18):
        return np.transpose(x, (2, 0, 1))
    raise ValueError(f"Unexpected tensor shape: {x.shape}")


def decode_tensor_to_board(x: np.ndarray) -> Board:
    arr = ensure_channels_first(x)
    stm = Color.WHITE if float(arr[12].mean()) >= 0.5 else Color.BLACK
    opp = stm.opponent()
    board = Board("8/8/8/8/8/8/8/8 w - - 0 1")
    board.current_turn = stm
    board.castling_rights = {
        Color.WHITE: {"kingside": False, "queenside": False},
        Color.BLACK: {"kingside": False, "queenside": False},
    }
    board.en_passant_target = None
    board.halfmove_clock = 0
    board.fullmove_number = 1

    for plane in range(12):
        coords = np.argwhere(arr[plane] > 0.5)
        piece_type = PIECE_TYPES[plane % 6]
        color = stm if plane < 6 else opp
        for r_idx, f_idx in coords:
            rank = int(7 - r_idx) if stm == Color.BLACK else int(r_idx)
            file = int(f_idx)
            square = Square.from_rank_file(rank, file)
            board.set_piece(square, Piece(color, piece_type))

    if stm == Color.WHITE:
        board.castling_rights[Color.WHITE]["kingside"] = bool(arr[13].any())
        board.castling_rights[Color.WHITE]["queenside"] = bool(arr[14].any())
        board.castling_rights[Color.BLACK]["kingside"] = bool(arr[15].any())
        board.castling_rights[Color.BLACK]["queenside"] = bool(arr[16].any())
    else:
        board.castling_rights[Color.BLACK]["kingside"] = bool(arr[13].any())
        board.castling_rights[Color.BLACK]["queenside"] = bool(arr[14].any())
        board.castling_rights[Color.WHITE]["kingside"] = bool(arr[15].any())
        board.castling_rights[Color.WHITE]["queenside"] = bool(arr[16].any())

    ep_coords = np.argwhere(arr[17] > 0.5)
    if ep_coords.size > 0:
        r_idx, f_idx = ep_coords[0]
        rank = int(7 - r_idx) if stm == Color.BLACK else int(r_idx)
        board.en_passant_target = Square.from_rank_file(rank, int(f_idx))
    return board


def validate_encode_decode_roundtrip(
    data_root: str | Path,
    split: str,
    sample_count: int = 32,
) -> Dict[str, object]:
    X, _ = load_split_arrays(data_root, split, max_samples=sample_count, num_shards=1)
    mismatches = 0
    mismatch_examples: List[int] = []
    for idx, x in enumerate(X):
        board = decode_tensor_to_board(x)
        x_rt = ensure_channels_first(encode_board(board))
        if not np.array_equal(x.astype(np.uint8), x_rt.astype(np.uint8)):
            mismatches += 1
            mismatch_examples.append(int(idx))
    return {"sample_count": int(sample_count), "mismatches": int(mismatches), "mismatch_examples": mismatch_examples[:10]}


class ZeroPredictor:
    def predict(self, _x) -> float:
        return 0.0


def import_negamax_module():
    arch_module = import_module("model.architecture.model")
    if not hasattr(arch_module, "PhantomChessNet"):
        setattr(arch_module, "PhantomChessNet", arch_module.DGRNChessNet)
    return import_module("search.negamax")


def side_to_move_material_score(board: Board) -> float:
    score = float(material_evaluate(board))
    return score if board.current_turn == Color.WHITE else -score


def search_depth_curve(board: Board, depths: Sequence[int]) -> Dict[str, object]:
    negamax_mod = import_negamax_module()
    zero_model = ZeroPredictor()
    search_scores: List[float] = []
    best_moves: List[Optional[str]] = []
    static_score = side_to_move_material_score(board)
    for depth in depths:
        if hasattr(negamax_mod.TT, "new_search"):
            negamax_mod.TT.new_search()
        cloned = Board(board.to_fen())
        hash_key = negamax_mod.TT.compute_hash(cloned)
        score = float(
            negamax_mod.negamax(
                cloned,
                depth=int(depth),
                alpha=-float("inf"),
                beta=float("inf"),
                epsilon=0.0,
                model=zero_model,
                ply=0,
            )
        )
        tt_entry = negamax_mod.TT.lookup(hash_key)
        best_move = None
        if tt_entry is not None and tt_entry.get("best_move") is not None:
            best_move = str(tt_entry["best_move"])
        search_scores.append(score)
        best_moves.append(best_move)
    return {
        "static_score": static_score,
        "search_scores": search_scores,
        "best_moves": best_moves,
    }


def run_near_zero_stability_experiment(
    ckpt_path: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    device: torch.device,
    split: str = "test",
    near_zero_thr: float = 0.2,
    sample_positions: int = 128,
    search_depths: Sequence[int] = (1, 2, 3, 4),
    sample_seed: int = 123,
    max_source_samples: int = 50_000,
    source_num_shards: int = 4,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    reports_dir = output_dir / "reports"
    plots_dir = output_dir / "plots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_split_arrays(data_root, split, max_samples=max_source_samples, num_shards=source_num_shards)
    model, _ = load_model_from_checkpoint(ckpt_path, device=device)
    pred = predict_array(model, X, device=device, batch_size=2048, use_amp=True, progress_name="stability_teacher")

    near_mask = np.abs(y) <= near_zero_thr
    near_idx = np.flatnonzero(near_mask)
    if near_idx.size == 0:
        raise RuntimeError("No near-zero samples found for stability experiment.")
    rng = np.random.default_rng(sample_seed)
    chosen = np.sort(rng.choice(near_idx, size=min(sample_positions, near_idx.size), replace=False))

    rows: List[dict] = []
    for rank, idx in enumerate(chosen):
        board = decode_tensor_to_board(X[idx])
        curve = search_depth_curve(board, depths=search_depths)
        search_scores = np.asarray(curve["search_scores"], dtype=np.float64)
        best_moves = list(curve["best_moves"])
        best_move_changes = int(
            sum(
                1
                for prev, curr in zip(best_moves[:-1], best_moves[1:])
                if prev is not None and curr is not None and prev != curr
            )
        )
        sign_flips = int(np.sum(np.sign(search_scores[1:]) != np.sign(search_scores[:-1])))
        static_score = float(curve["static_score"])
        static_gap = float(abs(search_scores[-1] - static_score))
        row = {
            "index": int(idx),
            "fen": board.to_fen(),
            "target_y": float(y[idx]),
            "teacher_pred": float(pred[idx]),
            "teacher_abs_err": float(abs(pred[idx] - y[idx])),
            "search_static_score": static_score,
            "proxy_final_score": float(search_scores[-1]),
            "proxy_abs_final_score": float(abs(search_scores[-1])),
            "proxy_vol_range": float(np.max(search_scores) - np.min(search_scores)),
            "proxy_vol_std": float(np.std(search_scores)),
            "proxy_static_gap": static_gap,
            "proxy_sign_flips": sign_flips,
            "proxy_best_move_changes": best_move_changes,
        }
        for depth, score, best_move in zip(search_depths, curve["search_scores"], curve["best_moves"]):
            row[f"search_score_d{depth}"] = float(score)
            row[f"best_move_d{depth}"] = best_move
        rows.append(row)
        if rank % 16 == 0:
            print(f"[stability] processed={rank + 1}/{len(chosen)}")

    df = pd.DataFrame(rows).sort_values(["proxy_vol_range", "proxy_static_gap"], ascending=True).reset_index(drop=True)
    q33_vol = float(df["proxy_vol_range"].quantile(0.33))
    q67_vol = float(df["proxy_vol_range"].quantile(0.67))
    q33_gap = float(df["proxy_static_gap"].quantile(0.33))
    q67_gap = float(df["proxy_static_gap"].quantile(0.67))
    ordered = df.sort_values(
        ["proxy_best_move_changes", "proxy_sign_flips", "proxy_vol_range", "proxy_static_gap"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    subset_size = max(1, ordered.shape[0] // 3)
    stable_df = ordered.head(subset_size).copy()
    unstable_df = ordered.tail(subset_size).copy()

    def subset_metrics(frame: pd.DataFrame) -> Dict[str, float]:
        y_sub = frame["target_y"].to_numpy(dtype=np.float64)
        p_sub = frame["teacher_pred"].to_numpy(dtype=np.float64)
        return {
            "n": int(frame.shape[0]),
            "mse": float(np.mean((p_sub - y_sub) ** 2)),
            "mae": float(np.mean(np.abs(p_sub - y_sub))),
            "mean_proxy_vol_range": float(frame["proxy_vol_range"].mean()),
            "mean_proxy_static_gap": float(frame["proxy_static_gap"].mean()),
            "mean_proxy_best_move_changes": float(frame["proxy_best_move_changes"].mean()),
            "mean_proxy_sign_flips": float(frame["proxy_sign_flips"].mean()),
            "false_decisive_0.3": float(np.mean(np.abs(p_sub) >= 0.3)),
            "false_decisive_0.4": float(np.mean(np.abs(p_sub) >= 0.4)),
        }

    proxy_diagnostics = {
        "score_unit": "search_native_normalized_score",
        "stable_unstable_rule": "lexicographic order on (best_move_changes, sign_flips, vol_range, static_gap)",
        "q33_vol_range": q33_vol,
        "q67_vol_range": q67_vol,
        "q33_static_gap": q33_gap,
        "q67_static_gap": q67_gap,
        "vol_range_min": float(df["proxy_vol_range"].min()),
        "vol_range_max": float(df["proxy_vol_range"].max()),
        "vol_range_mean": float(df["proxy_vol_range"].mean()),
        "vol_range_std": float(df["proxy_vol_range"].std()),
        "vol_range_unique_count": int(df["proxy_vol_range"].nunique()),
        "static_gap_min": float(df["proxy_static_gap"].min()),
        "static_gap_max": float(df["proxy_static_gap"].max()),
        "static_gap_mean": float(df["proxy_static_gap"].mean()),
        "static_gap_std": float(df["proxy_static_gap"].std()),
        "static_gap_unique_count": int(df["proxy_static_gap"].nunique()),
        "move_change_nonzero_count": int(np.sum(df["proxy_best_move_changes"] > 0)),
        "sign_flip_nonzero_count": int(np.sum(df["proxy_sign_flips"] > 0)),
        "is_informative": bool(
            (q67_vol > q33_vol)
            or (q67_gap > q33_gap)
            or np.any(df["proxy_best_move_changes"] > 0)
            or np.any(df["proxy_sign_flips"] > 0)
        ),
    }
    report = {
        "config": {
            "split": split,
            "near_zero_thr": near_zero_thr,
            "sample_positions": int(sample_positions),
            "search_depths": [int(x) for x in search_depths],
            "max_source_samples": int(max_source_samples),
            "source_num_shards": int(source_num_shards),
        },
        "proxy_diagnostics": proxy_diagnostics,
        "stable": subset_metrics(stable_df),
        "unstable": subset_metrics(unstable_df),
    }
    save_dataframe(df, reports_dir / "near_zero_stability_rows.csv")
    save_json(report, reports_dir / "near_zero_stability_report.json")

    plt.figure(figsize=(7, 4))
    plt.hist(df["proxy_vol_range"], bins=30, alpha=0.9)
    plt.axvline(q33_vol, color="green", linestyle="--", label="q33")
    plt.axvline(q67_vol, color="red", linestyle="--", label="q67")
    plt.xlabel("search-proxy volatility range")
    plt.ylabel("count")
    plt.title("Near-zero search-proxy volatility distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "near_zero_volatility_hist.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.scatter(df["proxy_vol_range"], df["teacher_abs_err"], s=8, alpha=0.5)
    plt.xlabel("search-proxy volatility range")
    plt.ylabel("teacher absolute error")
    plt.title("Teacher error vs search-proxy volatility on near-zero positions")
    plt.tight_layout()
    plt.savefig(plots_dir / "near_zero_error_vs_volatility.png", dpi=180)
    plt.close()
    return {"rows": df, "report": report}


def target_to_logits(y: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.atanh(torch.clamp(y, -1.0 + eps, 1.0 - eps))


def huber_per_sample(residual: torch.Tensor, delta: float) -> torch.Tensor:
    abs_res = residual.abs()
    return torch.where(abs_res <= delta, 0.5 * residual * residual, delta * (abs_res - 0.5 * delta))


def center_safe_penalty(pred: torch.Tensor, y: torch.Tensor, tau: float, margin: float) -> torch.Tensor:
    mask = torch.abs(y) <= tau
    if not torch.any(mask):
        return pred.new_tensor(0.0)
    excess = torch.relu(torch.abs(pred[mask]) - margin)
    return torch.mean(excess * excess)


def compute_center_safe_objective(logits: torch.Tensor, y: torch.Tensor, cfg: PilotTrainConfig) -> Dict[str, torch.Tensor]:
    pred = torch.tanh(logits)
    metric = F.mse_loss(pred, y)
    y_logits = target_to_logits(y, eps=cfg.target_clamp_eps)
    residual = logits - y_logits
    y_clamped = torch.clamp(y, -1.0 + cfg.target_clamp_eps, 1.0 - cfg.target_clamp_eps)
    z_weights = torch.pow(torch.clamp(1.0 - y_clamped * y_clamped, min=cfg.target_clamp_eps), cfg.z_loss_beta)
    z_huber = torch.mean(z_weights * huber_per_sample(residual, cfg.z_huber_delta))
    center_pen = center_safe_penalty(pred, y, tau=cfg.center_tau, margin=cfg.center_margin)
    objective = cfg.lambda_y * metric + (1.0 - cfg.lambda_y) * z_huber + cfg.center_weight * center_pen
    return {"objective": objective, "metric": metric, "z_huber": z_huber, "center_penalty": center_pen, "pred": pred}


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
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


def evaluate_model_on_arrays(model: nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device, batch_size: int = 2048) -> Dict[str, object]:
    pred = predict_array(model, X, device=device, batch_size=batch_size, use_amp=True, progress_name="pilot_eval")
    return {"preds": pred, "metrics": summarize_teacher_metrics(y.astype(np.float64), pred.astype(np.float64))}


def run_center_safe_finetune_pilot(
    init_ckpt_path: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    device: torch.device,
    cfg: PilotTrainConfig,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    reports_dir = output_dir / "reports"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(cfg.seed)
    train_X, train_y = load_split_arrays(data_root, "train", cfg.train_max_samples, num_shards=cfg.train_num_shards)
    val_X, val_y = load_split_arrays(data_root, "val", cfg.val_max_samples, num_shards=cfg.val_num_shards)
    train_ds = TensorDataset(torch.from_numpy(train_X.astype(np.float32)), torch.from_numpy(train_y.astype(np.float32)))
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model, init_ckpt = load_model_from_checkpoint(init_ckpt_path, device=device)
    if not hasattr(model, "forward_logits"):
        raise RuntimeError("Current model does not expose forward_logits().")
    optimizer = build_optimizer(model, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs, 1), eta_min=cfg.min_lr)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    history: List[dict] = []
    best_score = float("inf")
    best_ckpt_path = checkpoints_dir / "center_safe_pilot_best.pt"
    latest_ckpt_path = checkpoints_dir / "center_safe_pilot_latest.pt"

    for epoch in range(cfg.epochs):
        model.train()
        running = {"objective": 0.0, "metric": 0.0, "z_huber": 0.0, "center_penalty": 0.0, "n": 0}
        t0 = time.time()
        for step, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).view(-1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model.forward_logits(xb).view(-1)
                terms = compute_center_safe_objective(logits, yb, cfg)
            scaler.scale(terms["objective"]).backward()
            if cfg.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            bs = yb.numel()
            running["objective"] += float(terms["objective"].item()) * bs
            running["metric"] += float(terms["metric"].item()) * bs
            running["z_huber"] += float(terms["z_huber"].item()) * bs
            running["center_penalty"] += float(terms["center_penalty"].item()) * bs
            running["n"] += bs
            if step % cfg.log_every == 0:
                print(
                    f"[pilot][epoch={epoch}] step={step} "
                    f"obj={running['objective'] / running['n']:.6f} "
                    f"metric={running['metric'] / running['n']:.6f} "
                    f"center_pen={running['center_penalty'] / running['n']:.6f}"
                )

        scheduler.step()
        val_eval = evaluate_model_on_arrays(model, val_X, val_y, device=device, batch_size=max(cfg.batch_size, 1024))
        val_metrics = val_eval["metrics"]
        gate_score = (
            val_metrics["bands"]["0.7"]["mse"]
            + 0.5 * val_metrics["false_decisive"]["y<=0.1,p>=0.3"]["rate"]
            + 0.5 * max(0.0, 0.8 - val_metrics["bands"]["0.7"]["slope"])
        )
        row = {
            "epoch": int(epoch),
            "train_objective": running["objective"] / running["n"],
            "train_metric": running["metric"] / running["n"],
            "train_z_huber": running["z_huber"] / running["n"],
            "train_center_penalty": running["center_penalty"] / running["n"],
            "val_mse_0.7": val_metrics["bands"]["0.7"]["mse"],
            "val_slope_0.7": val_metrics["bands"]["0.7"]["slope"],
            "val_false_decisive_0.1_0.3": val_metrics["false_decisive"]["y<=0.1,p>=0.3"]["rate"],
            "val_center_spread_ratio_0.05": val_metrics["center_spread_ratio_0.05"]["ratio"],
            "gate_score": float(gate_score),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_time_sec": float(time.time() - t0),
        }
        history.append(row)
        print(json.dumps(row, indent=2))

        ckpt_payload = {
            "epoch": int(epoch),
            "history": history,
            "pilot_train_config": asdict(cfg),
            "init_checkpoint": str(init_ckpt_path),
            "base_checkpoint_epoch": init_ckpt.get("epoch"),
            "config": init_ckpt.get("config"),
            "model": model.state_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_metrics": val_metrics,
        }
        torch.save(ckpt_payload, latest_ckpt_path)
        if gate_score < best_score:
            best_score = float(gate_score)
            torch.save(ckpt_payload, best_ckpt_path)

    history_df = pd.DataFrame(history)
    save_dataframe(history_df, reports_dir / "center_safe_pilot_history.csv")
    save_json({"history": history, "best_gate_score": best_score}, reports_dir / "center_safe_pilot_history.json")
    return {
        "history": history_df,
        "best_checkpoint": best_ckpt_path,
        "latest_checkpoint": latest_ckpt_path,
        "best_gate_score": best_score,
    }
