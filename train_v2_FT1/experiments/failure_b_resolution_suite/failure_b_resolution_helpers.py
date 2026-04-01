from __future__ import annotations

import gc
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRC_DIR = PROJECT_ROOT / "experiments" / "teacher_root_cause_lab"
AB_DIR = PROJECT_ROOT / "experiments" / "root_cause_ablation_suite"
OBJ_DIR = PROJECT_ROOT / "experiments" / "objective_resolution_suite"
OR_DIR = PROJECT_ROOT / "experiments" / "oracle_root_cause_diagnostic"
for path in (PROJECT_ROOT, TRC_DIR, AB_DIR, OBJ_DIR, OR_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import teacher_root_cause_helpers as base_lab  # noqa: E402
import root_cause_ablation_helpers as ab_lab  # noqa: E402
import objective_resolution_helpers as obj_lab  # noqa: E402


SOURCE_TARGET_SCALE = ab_lab.SOURCE_TARGET_SCALE
CANONICAL_BANDS = ab_lab.CANONICAL_Y600_BANDS
CENTER_FALSE_SMALL_KEY = obj_lab.CENTER_FALSE_SMALL_KEY
CENTER_FALSE_MEDIUM_KEY = obj_lab.CENTER_FALSE_MEDIUM_KEY

save_json = base_lab.save_json
save_dataframe = base_lab.save_dataframe
set_global_seed = base_lab.set_global_seed
choose_device = base_lab.choose_device


@dataclass
class ExperimentPaths:
    project_root: str
    experiment_dir: str
    output_dir: str
    reports_dir: str
    plots_dir: str
    cache_dir: str
    pilots_dir: str
    replicates_dir: str
    data_root: str
    run_dir: str
    objective_output_dir: str
    ablation_output_dir: str
    oracle_output_dir: str


@dataclass
class CenterPurityConfig:
    oracle_center_thr: float = 0.05
    raw_center_thresholds: Tuple[float, ...] = (0.02, 0.05, 0.10)
    oracle_center_thresholds: Tuple[float, ...] = (0.05, 0.10)
    lookup_train_max_abs_y: float = 0.10
    lookup_abs_y_edges: Tuple[float, ...] = (0.0, 0.02, 0.05, 0.10, 0.20)
    lookup_abs_pred_edges: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 1.01)
    lookup_err_quantiles: Tuple[float, ...] = (1.0 / 3.0, 2.0 / 3.0)
    lookup_smoothing_strength: float = 8.0
    seed: int = 123


@dataclass
class GradientAuditConfig:
    train_split: str = "train"
    train_num_shards: int = 8
    band_edges_y600: Tuple[float, ...] = (0.0, 0.05, 0.20, 0.50, 0.70)
    batch_size: int = 256
    batches_per_band: int = 3
    sample_seed: int = 123
    influence_step_l2: float = 5e-4
    probe_center_max: int = 128
    probe_midband_max: int = 128


@dataclass
class PilotTrainConfig:
    batch_size: int = 576
    epochs: int = 1
    learning_rate: float = 3e-6
    min_lr: float = 1e-6
    weight_decay: float = 2e-4
    grad_clip_norm: float = 1.0
    seed: int = 123
    train_num_shards: int = 8
    test_max_samples: int = 200_000
    test_num_shards: int = 4
    log_every_steps: int = 200
    max_mem_ratio: float = 0.85
    min_batch_size: int = 128
    batch_step: int = 64


@dataclass
class FailureBPilotVariant:
    name: str
    description: str
    init_ckpt_path: str
    main_variant: ab_lab.AblationVariant
    center_mode: str = "raw"
    center_penalty_weight: float = 0.40
    center_penalty_tau_y600: float = 0.05
    center_penalty_margin_y600: float = 0.12
    proxy_min_score: float = 0.0
    proxy_power: float = 1.0


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return isinstance(exc, RuntimeError) and ("out of memory" in text or "cuda error" in text)


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def build_default_paths(
    run_dir: str | Path = r"C:\Users\USER\Downloads\dgrn_5m_v3_stage2_polish_run1",
    data_root: str | Path = r"C:\Users\USER\Desktop\chess_engine\data\process",
    experiment_dir: str | Path | None = None,
    objective_output_dir: str | Path | None = None,
    ablation_output_dir: str | Path | None = None,
    oracle_output_dir: str | Path | None = None,
) -> Dict[str, Path]:
    if experiment_dir is None:
        experiment_dir = PROJECT_ROOT / "experiments" / "failure_b_resolution_suite"
    if objective_output_dir is None:
        objective_output_dir = OBJ_DIR / "outputs"
    if ablation_output_dir is None:
        ablation_output_dir = AB_DIR / "outputs"
    if oracle_output_dir is None:
        oracle_output_dir = OR_DIR / "outputs"
    paths = {
        "project_root": PROJECT_ROOT,
        "experiment_dir": Path(experiment_dir),
        "output_dir": Path(experiment_dir) / "outputs",
        "reports_dir": Path(experiment_dir) / "outputs" / "reports",
        "plots_dir": Path(experiment_dir) / "outputs" / "plots",
        "cache_dir": Path(experiment_dir) / "outputs" / "cache",
        "pilots_dir": Path(experiment_dir) / "outputs" / "pilots",
        "replicates_dir": Path(experiment_dir) / "outputs" / "replicates",
        "data_root": Path(data_root),
        "run_dir": Path(run_dir),
        "objective_output_dir": Path(objective_output_dir),
        "ablation_output_dir": Path(ablation_output_dir),
        "oracle_output_dir": Path(oracle_output_dir),
    }
    for key in ("experiment_dir", "output_dir", "reports_dir", "plots_dir", "cache_dir", "pilots_dir", "replicates_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def export_paths_json(paths: Dict[str, Path], out_path: Path) -> None:
    payload = ExperimentPaths(**{key: str(value) for key, value in paths.items()})
    out_path.write_text(json.dumps(asdict(payload), indent=2), encoding="utf-8")


def validate_runtime_paths(
    paths: Dict[str, Path],
    stockfish_path: str | Path = r"D:\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe",
) -> Dict[str, object]:
    required_dirs = {
        "data_root": paths["data_root"],
        "train_dir": paths["data_root"] / "train",
        "test_dir": paths["data_root"] / "test",
        "objective_output_dir": paths["objective_output_dir"],
        "ablation_output_dir": paths["ablation_output_dir"],
        "oracle_output_dir": paths["oracle_output_dir"],
    }
    required_files = [
        paths["run_dir"] / "ckpt_best.pt",
        paths["objective_output_dir"] / "runs" / "L0_control_hybrid" / "checkpoints" / "L0_control_hybrid_best.pt",
        paths["objective_output_dir"] / "runs" / "L1_z_strong_hybrid" / "checkpoints" / "L1_z_strong_hybrid_best.pt",
        paths["objective_output_dir"] / "runs" / "L3_full_A1" / "checkpoints" / "L3_full_A1_best.pt",
        paths["objective_output_dir"] / "runs" / "L4_A1_plus_A2" / "checkpoints" / "L4_A1_plus_A2_best.pt",
        paths["objective_output_dir"] / "runs" / "S1_A1_center_w020_m010" / "checkpoints" / "S1_A1_center_w020_m010_best.pt",
        paths["ablation_output_dir"] / "runs" / "A2_band_balanced" / "checkpoints" / "A2_band_balanced_best.pt",
        paths["oracle_output_dir"] / "reports" / "oracle_subset_rows.csv",
        Path(stockfish_path),
    ]
    missing_dirs = [name for name, path in required_dirs.items() if not Path(path).exists()]
    missing_files = [str(path) for path in required_files if not Path(path).exists()]
    if missing_dirs or missing_files:
        parts: List[str] = []
        if missing_dirs:
            parts.append("missing dirs: " + ", ".join(missing_dirs))
        if missing_files:
            parts.append("missing files: " + ", ".join(missing_files))
        raise FileNotFoundError("Runtime path validation failed: " + " | ".join(parts))
    return {
        "is_valid": True,
        "stockfish_path": str(Path(stockfish_path)),
        "baseline_ckpt": str(paths["run_dir"] / "ckpt_best.pt"),
        "l3_ckpt": str(paths["objective_output_dir"] / "runs" / "L3_full_A1" / "checkpoints" / "L3_full_A1_best.pt"),
    }


def resolve_split_shards(data_root: str | Path, split: str, num_shards: Optional[int] = None) -> List[Tuple[int, Path, Path]]:
    pairs = base_lab.resolve_split_pairs(data_root, split)
    pairs = base_lab.select_pairs_evenly(pairs, num_shards)
    rows = []
    for x_path, y_path in pairs:
        rows.append((int(x_path.stem.split("_")[1]), x_path, y_path))
    return rows


def load_checkpoint_registry(paths: Dict[str, Path]) -> pd.DataFrame:
    rows = [
        {
            "label": "baseline",
            "checkpoint": str(paths["run_dir"] / "ckpt_best.pt"),
            "target_scale": 600.0,
            "source": "stage2_polish",
        },
        {
            "label": "A2_band_balanced",
            "checkpoint": str(paths["ablation_output_dir"] / "runs" / "A2_band_balanced" / "checkpoints" / "A2_band_balanced_best.pt"),
            "target_scale": 600.0,
            "source": "root_cause_ablation_suite",
        },
        {
            "label": "L0_control_hybrid",
            "checkpoint": str(paths["objective_output_dir"] / "runs" / "L0_control_hybrid" / "checkpoints" / "L0_control_hybrid_best.pt"),
            "target_scale": 600.0,
            "source": "objective_resolution_suite",
        },
        {
            "label": "L1_z_strong_hybrid",
            "checkpoint": str(paths["objective_output_dir"] / "runs" / "L1_z_strong_hybrid" / "checkpoints" / "L1_z_strong_hybrid_best.pt"),
            "target_scale": 600.0,
            "source": "objective_resolution_suite",
        },
        {
            "label": "L3_full_A1",
            "checkpoint": str(paths["objective_output_dir"] / "runs" / "L3_full_A1" / "checkpoints" / "L3_full_A1_best.pt"),
            "target_scale": 600.0,
            "source": "objective_resolution_suite",
        },
        {
            "label": "L4_A1_plus_A2",
            "checkpoint": str(paths["objective_output_dir"] / "runs" / "L4_A1_plus_A2" / "checkpoints" / "L4_A1_plus_A2_best.pt"),
            "target_scale": 600.0,
            "source": "objective_resolution_suite",
        },
        {
            "label": "S1_A1_center_w020_m010",
            "checkpoint": str(paths["objective_output_dir"] / "runs" / "S1_A1_center_w020_m010" / "checkpoints" / "S1_A1_center_w020_m010_best.pt"),
            "target_scale": 600.0,
            "source": "objective_resolution_suite",
        },
    ]
    return pd.DataFrame(rows)


def _normalize_oracle_rows(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    out = df.copy()
    out["source_name"] = str(source_name)
    int_cols = ["shard_id", "local_index", "band_idx", "err_bin_id", "oracle_reference_node_budget", "bestmove_changes", "sign_flips"]
    float_cols = [
        "target_y",
        "teacher_pred",
        "teacher_abs_err",
        "oracle_reference_cp",
        "oracle_reference_y_600",
        "teacher_vs_oracle_abs_600",
        "train_vs_oracle_abs_600",
        "oracle_target_range_600",
        "oracle_target_std_600",
        "oracle_cp_range",
    ]
    for col in int_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    for col in float_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    out["split"] = out["split"].astype(str)
    out["stability_group"] = out["stability_group"].astype(str)
    out["band_label"] = out["band_label"].astype(str)
    return out


def build_oracle_center_pool(
    paths: Dict[str, Path],
    center_cfg: CenterPurityConfig,
    refresh: bool = False,
) -> Dict[str, object]:
    reports_dir = paths["reports_dir"]
    pooled_all_path = reports_dir / "oracle_center_pool_all.csv"
    pooled_unique_path = reports_dir / "oracle_center_pool_unique.csv"
    report_path = reports_dir / "oracle_center_pool_report.json"
    if not refresh and pooled_all_path.exists() and pooled_unique_path.exists() and report_path.exists():
        return {
            "all_rows": pd.read_csv(pooled_all_path),
            "unique_rows": pd.read_csv(pooled_unique_path),
            "report": json.loads(report_path.read_text(encoding="utf-8")),
        }

    source_paths: List[Tuple[str, Path]] = []
    source_paths.append(("oracle_root_cause_primary", paths["oracle_output_dir"] / "reports" / "oracle_subset_rows.csv"))
    for csv_path in sorted((paths["objective_output_dir"] / "replicates").glob("replicate_*\\reports\\oracle_subset_rows.csv")):
        source_paths.append((f"objective_replicate_{csv_path.parents[1].name}", csv_path))
    for csv_path in sorted((paths["objective_output_dir"] / "final_replicates" / "replicates").glob("replicate_*\\reports\\oracle_subset_rows.csv")):
        source_paths.append((f"final_replicate_{csv_path.parents[1].name}", csv_path))

    frames: List[pd.DataFrame] = []
    for source_name, csv_path in source_paths:
        frame = pd.read_csv(csv_path)
        frames.append(_normalize_oracle_rows(frame, source_name))
    pooled_all = pd.concat(frames, ignore_index=True)
    pooled_all["row_key"] = (
        pooled_all["split"].astype(str)
        + "|"
        + pooled_all["shard_id"].astype(str)
        + "|"
        + pooled_all["local_index"].astype(str)
    )
    pooled_all["oracle_center_clean_005"] = (
        pooled_all["stability_group"].eq("stable")
        & (pooled_all["oracle_reference_y_600"].abs() <= float(center_cfg.oracle_center_thr))
    )
    conflict_rows: List[str] = []
    for row_key, group in pooled_all.groupby("row_key"):
        if group["oracle_reference_cp"].nunique(dropna=False) > 1 or group["target_y"].nunique(dropna=False) > 1:
            conflict_rows.append(str(row_key))
    pooled_unique = (
        pooled_all.sort_values(["row_key", "source_name"])
        .groupby("row_key", as_index=False)
        .first()
        .copy()
    )
    occurrence = pooled_all.groupby("row_key").size().rename("occurrence_count").reset_index()
    pooled_unique = pooled_unique.merge(occurrence, on="row_key", how="left")

    report = {
        "num_sources": int(len(source_paths)),
        "num_rows_all": int(pooled_all.shape[0]),
        "num_rows_unique": int(pooled_unique.shape[0]),
        "num_conflict_rows": int(len(conflict_rows)),
        "conflict_rows_preview": conflict_rows[:10],
        "source_counts": {name: int((pooled_all["source_name"] == name).sum()) for name, _ in source_paths},
        "clean_center_rate_unique": float(pooled_unique["oracle_center_clean_005"].mean()),
    }
    save_dataframe(pooled_all, pooled_all_path)
    save_dataframe(pooled_unique, pooled_unique_path)
    save_json(report, report_path)
    return {"all_rows": pooled_all, "unique_rows": pooled_unique, "report": report}


def run_center_label_purity_audit(
    pooled_unique: pd.DataFrame,
    cfg: CenterPurityConfig,
    paths: Dict[str, Path],
) -> Dict[str, object]:
    summary_rows: List[dict] = []
    train_abs = pooled_unique["target_y"].abs().to_numpy(dtype=np.float64)
    oracle_abs = pooled_unique["oracle_reference_y_600"].abs().to_numpy(dtype=np.float64)
    stable = pooled_unique["stability_group"].astype(str).eq("stable").to_numpy(dtype=bool)
    for raw_thr in cfg.raw_center_thresholds:
        raw_mask = train_abs <= float(raw_thr)
        for oracle_thr in cfg.oracle_center_thresholds:
            oracle_mask = stable & (oracle_abs <= float(oracle_thr))
            tp = int(np.sum(raw_mask & oracle_mask))
            fp = int(np.sum(raw_mask & ~oracle_mask))
            fn = int(np.sum(~raw_mask & oracle_mask))
            tn = int(np.sum(~raw_mask & ~oracle_mask))
            precision = float(tp / max(tp + fp, 1))
            recall = float(tp / max(tp + fn, 1))
            specificity = float(tn / max(tn + fp, 1))
            summary_rows.append(
                {
                    "raw_center_thr": float(raw_thr),
                    "oracle_center_thr": float(oracle_thr),
                    "n": int(pooled_unique.shape[0]),
                    "raw_center_count": int(raw_mask.sum()),
                    "oracle_center_clean_count": int(oracle_mask.sum()),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "precision": precision,
                    "recall": recall,
                    "specificity": specificity,
                    "oracle_clean_rate_inside_raw_center": precision,
                    "mean_abs_oracle_inside_raw_center": float(oracle_abs[raw_mask].mean()) if np.any(raw_mask) else float("nan"),
                    "stable_rate_inside_raw_center": float(stable[raw_mask].mean()) if np.any(raw_mask) else float("nan"),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    save_dataframe(summary_df, paths["reports_dir"] / "center_label_purity_summary.csv")

    band_rows: List[dict] = []
    raw_edges = [0.0, 0.02, 0.05, 0.10, 0.20, 1.01]
    for idx in range(len(raw_edges) - 1):
        left = raw_edges[idx]
        right = raw_edges[idx + 1]
        if idx == 0:
            mask = (train_abs >= left) & (train_abs <= right)
        else:
            mask = (train_abs > left) & (train_abs <= right)
        if not np.any(mask):
            continue
        band_rows.append(
            {
                "raw_band": f"[{left:.3f},{right:.3f}]",
                "n": int(mask.sum()),
                "mean_abs_oracle": float(oracle_abs[mask].mean()),
                "stable_rate": float(stable[mask].mean()),
                "oracle_clean_rate_005": float((stable & (oracle_abs <= 0.05))[mask].mean()),
                "oracle_clean_rate_010": float((stable & (oracle_abs <= 0.10))[mask].mean()),
            }
        )
    band_df = pd.DataFrame(band_rows)
    save_dataframe(band_df, paths["reports_dir"] / "center_label_purity_by_train_band.csv")
    report = {
        "rows_unique": int(pooled_unique.shape[0]),
        "raw_center_count_005": int((train_abs <= 0.05).sum()),
        "oracle_clean_center_count_005": int((stable & (oracle_abs <= 0.05)).sum()),
    }
    save_json(report, paths["reports_dir"] / "center_label_purity_report.json")
    return {"summary": summary_df, "band_summary": band_df, "report": report}


def build_center_purity_lookup(
    pooled_unique: pd.DataFrame,
    cfg: CenterPurityConfig,
    paths: Dict[str, Path],
) -> Dict[str, object]:
    subset = pooled_unique[pooled_unique["target_y"].abs() <= float(cfg.lookup_train_max_abs_y)].copy()
    if subset.empty:
        raise RuntimeError("Center purity lookup subset is empty.")
    base_rate = float(subset["oracle_center_clean_005"].mean())
    err_edges = [0.0]
    err_edges.extend(float(subset["teacher_abs_err"].quantile(q)) for q in cfg.lookup_err_quantiles)
    err_edges.append(float(subset["teacher_abs_err"].max()) + 1e-6)
    err_edges = sorted(set(err_edges))

    def assign_bin(values: np.ndarray, edges: Sequence[float]) -> np.ndarray:
        return np.clip(np.digitize(values, list(edges)[1:-1], right=False), 0, len(edges) - 2)

    subset["abs_y"] = subset["target_y"].abs()
    subset["abs_pred"] = subset["teacher_pred"].abs()
    subset["abs_err"] = subset["teacher_abs_err"].abs()
    subset["bin_abs_y"] = assign_bin(subset["abs_y"].to_numpy(dtype=np.float64), cfg.lookup_abs_y_edges)
    subset["bin_abs_pred"] = assign_bin(subset["abs_pred"].to_numpy(dtype=np.float64), cfg.lookup_abs_pred_edges)
    subset["bin_abs_err"] = assign_bin(subset["abs_err"].to_numpy(dtype=np.float64), err_edges)

    rows: List[dict] = []
    grouped = subset.groupby(["bin_abs_y", "bin_abs_pred", "bin_abs_err"], as_index=False)
    for _, group in grouped:
        count = int(group.shape[0])
        clean_count = int(group["oracle_center_clean_005"].sum())
        smoothed = float((clean_count + cfg.lookup_smoothing_strength * base_rate) / (count + cfg.lookup_smoothing_strength))
        rows.append(
            {
                "bin_abs_y": int(group["bin_abs_y"].iloc[0]),
                "bin_abs_pred": int(group["bin_abs_pred"].iloc[0]),
                "bin_abs_err": int(group["bin_abs_err"].iloc[0]),
                "count": count,
                "clean_count": clean_count,
                "clean_rate": float(clean_count / max(count, 1)),
                "smoothed_clean_rate": smoothed,
            }
        )
    lookup_df = pd.DataFrame(rows).sort_values(["bin_abs_y", "bin_abs_pred", "bin_abs_err"]).reset_index(drop=True)
    report = {
        "base_rate": base_rate,
        "lookup_train_max_abs_y": float(cfg.lookup_train_max_abs_y),
        "abs_y_edges": [float(x) for x in cfg.lookup_abs_y_edges],
        "abs_pred_edges": [float(x) for x in cfg.lookup_abs_pred_edges],
        "abs_err_edges": [float(x) for x in err_edges],
        "smoothing_strength": float(cfg.lookup_smoothing_strength),
        "smoothed_clean_rate_min": float(lookup_df["smoothed_clean_rate"].min()),
        "smoothed_clean_rate_max": float(lookup_df["smoothed_clean_rate"].max()),
        "proxy_score_mode": "normalized_above_base_rate",
    }
    save_dataframe(lookup_df, paths["reports_dir"] / "center_purity_lookup.csv")
    save_json(report, paths["reports_dir"] / "center_purity_lookup_report.json")
    return {"lookup": lookup_df, "report": report}


def _assign_bins_np(values: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    return np.clip(np.digitize(values, list(edges)[1:-1], right=False), 0, len(edges) - 2)


def score_center_purity(
    abs_y: np.ndarray,
    abs_pred: np.ndarray,
    abs_err: np.ndarray,
    lookup_df: pd.DataFrame,
    lookup_report: Dict[str, object],
) -> np.ndarray:
    by = _assign_bins_np(np.asarray(abs_y, dtype=np.float64), lookup_report["abs_y_edges"])
    bp = _assign_bins_np(np.asarray(abs_pred, dtype=np.float64), lookup_report["abs_pred_edges"])
    be = _assign_bins_np(np.asarray(abs_err, dtype=np.float64), lookup_report["abs_err_edges"])
    table = {
        (int(row["bin_abs_y"]), int(row["bin_abs_pred"]), int(row["bin_abs_err"])): float(row["smoothed_clean_rate"])
        for _, row in lookup_df.iterrows()
    }
    default = float(lookup_report["base_rate"])
    score_max = float(lookup_report.get("smoothed_clean_rate_max", default))
    denom = max(score_max - default, 1e-6)
    out = np.empty_like(np.asarray(abs_y, dtype=np.float64), dtype=np.float64)
    for idx in range(out.shape[0]):
        raw_score = table.get((int(by[idx]), int(bp[idx]), int(be[idx])), default)
        out[idx] = float(np.clip((raw_score - default) / denom, 0.0, 1.0))
    return out.astype(np.float32, copy=False)


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


def build_probe_sets(
    pooled_unique: pd.DataFrame,
    data_root: str | Path,
    cfg: GradientAuditConfig,
    paths: Dict[str, Path],
    refresh: bool = False,
) -> Dict[str, object]:
    cache_dir = paths["cache_dir"] / "probe_sets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "probe_manifest.json"
    if not refresh and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bundles = {}
        for name, info in manifest["bundles"].items():
            npz = np.load(cache_dir / info["npz"], allow_pickle=False)
            bundles[name] = {
                "name": name,
                "rows": pd.read_csv(cache_dir / info["rows_csv"]),
                "X": npz["X"].astype(np.uint8),
                "oracle_y600": npz["oracle_y600"].astype(np.float64),
                "train_y": npz["train_y"].astype(np.float64),
            }
        return {"bundles": bundles, "manifest": manifest}

    masks = {
        "center_clean_005": pooled_unique["oracle_center_clean_005"].astype(bool),
        "center_raw_mismatch": (pooled_unique["target_y"].abs() <= 0.05) & (pooled_unique["oracle_reference_y_600"].abs() > 0.10),
        "midband_stable_02_07": pooled_unique["stability_group"].eq("stable") & (pooled_unique["oracle_reference_y_600"].abs() > 0.20) & (pooled_unique["oracle_reference_y_600"].abs() <= 0.70),
    }
    limits = {
        "center_clean_005": int(cfg.probe_center_max),
        "center_raw_mismatch": int(cfg.probe_center_max),
        "midband_stable_02_07": int(cfg.probe_midband_max),
    }
    bundles: Dict[str, object] = {}
    manifest = {"bundles": {}}
    for name, mask in masks.items():
        frame = pooled_unique.loc[mask].copy()
        if frame.empty:
            continue
        if frame.shape[0] > limits[name]:
            frame = frame.sample(n=limits[name], random_state=cfg.sample_seed, replace=False).copy()
        X, rows = _load_rows_to_tensor_bundle(frame, data_root)
        oracle_y = rows["oracle_reference_y_600"].to_numpy(dtype=np.float64)
        train_y = rows["target_y"].to_numpy(dtype=np.float64)
        rows_csv = f"{name}_rows.csv"
        npz_name = f"{name}.npz"
        save_dataframe(rows, cache_dir / rows_csv)
        np.savez_compressed(cache_dir / npz_name, X=X.astype(np.uint8), oracle_y600=oracle_y.astype(np.float32), train_y=train_y.astype(np.float32))
        bundles[name] = {"name": name, "rows": rows, "X": X, "oracle_y600": oracle_y, "train_y": train_y}
        manifest["bundles"][name] = {"rows_csv": rows_csv, "npz": npz_name, "n": int(rows.shape[0])}
    save_json(manifest, manifest_path)
    return {"bundles": bundles, "manifest": manifest}


def sample_gradient_batches(
    data_root: str | Path,
    cfg: GradientAuditConfig,
    paths: Dict[str, Path],
    refresh: bool = False,
) -> Dict[str, object]:
    cache_dir = paths["cache_dir"] / "gradient_batches"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    if not refresh and manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    rng = np.random.default_rng(cfg.sample_seed)
    band_specs = [
        ("center_raw_0_005", 0.0, 0.05),
        ("near_center_005_02", 0.05, 0.20),
        ("mid_02_05", 0.20, 0.50),
        ("mid_05_07", 0.50, 0.70),
    ]
    need_per_band = int(cfg.batch_size * cfg.batches_per_band)
    candidates: Dict[str, List[Tuple[int, int, float]]] = {name: [] for name, _, _ in band_specs}
    for shard_id, _, y_path in resolve_split_shards(data_root, cfg.train_split, num_shards=cfg.train_num_shards):
        y = np.load(y_path, mmap_mode="r").astype(np.float64, copy=False)
        abs_y = np.abs(y)
        for name, left, right in band_specs:
            if left == 0.0:
                idx = np.flatnonzero((abs_y >= left) & (abs_y <= right))
            else:
                idx = np.flatnonzero((abs_y > left) & (abs_y <= right))
            for local_index in idx.tolist():
                candidates[name].append((int(shard_id), int(local_index), float(y[local_index])))
    manifest = {"bands": {}, "config": asdict(cfg)}
    for name, _, _ in band_specs:
        rows = candidates[name]
        if len(rows) < need_per_band:
            raise RuntimeError(f"Not enough candidates for band {name}: need {need_per_band}, got {len(rows)}")
        picked = rng.choice(len(rows), size=need_per_band, replace=False)
        sample_rows = [rows[int(i)] for i in picked]
        frame = pd.DataFrame(sample_rows, columns=["shard_id", "local_index", "target_y"])
        frame["split"] = cfg.train_split
        save_dataframe(frame, cache_dir / f"{name}_rows.csv")
        manifest["bands"][name] = {
            "rows_csv": f"{name}_rows.csv",
            "n": int(frame.shape[0]),
            "batch_size": int(cfg.batch_size),
            "batches_per_band": int(cfg.batches_per_band),
        }
    save_json(manifest, manifest_path)
    return manifest


def _load_row_frame_to_arrays(rows_df: pd.DataFrame, data_root: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    shard_cache_x: Dict[Tuple[str, int], np.ndarray] = {}
    shard_cache_y: Dict[Tuple[str, int], np.ndarray] = {}
    Xs: List[np.ndarray] = []
    ys: List[float] = []
    for _, row in rows_df.iterrows():
        key = (str(row["split"]), int(row["shard_id"]))
        if key not in shard_cache_x:
            x_path = Path(data_root) / key[0] / f"X_{key[1]:05d}.npy"
            y_path = Path(data_root) / key[0] / f"y_{key[1]:05d}.npy"
            shard_cache_x[key] = np.load(x_path, mmap_mode="r")
            shard_cache_y[key] = np.load(y_path, mmap_mode="r")
        local_index = int(row["local_index"])
        Xs.append(np.asarray(shard_cache_x[key][local_index], dtype=np.uint8))
        ys.append(float(shard_cache_y[key][local_index]))
    return np.stack(Xs, axis=0).astype(np.uint8, copy=False), np.asarray(ys, dtype=np.float32)


def load_gradient_batches(data_root: str | Path, manifest: Dict[str, object], paths: Dict[str, Path]) -> Dict[str, Dict[str, np.ndarray]]:
    cache_dir = paths["cache_dir"] / "gradient_batches"
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for band_name, info in manifest["bands"].items():
        rows_df = pd.read_csv(cache_dir / info["rows_csv"])
        X, y = _load_row_frame_to_arrays(rows_df, data_root)
        out[band_name] = {"X": X, "y": y, "rows": rows_df}
    return out


def build_failure_b_objective_specs() -> Dict[str, ab_lab.AblationVariant]:
    return {
        "baseline_obj": ab_lab.AblationVariant(
            name="baseline_obj",
            description="Stage2 near-pure y-space objective.",
            target_scale=600.0,
            loss_mode="baseline_hybrid",
            lambda_y=0.99,
            z_loss_beta=1.0,
            z_huber_delta=0.5,
            center_penalty_weight=0.0,
        ),
        "a1_obj": ab_lab.AblationVariant(
            name="a1_obj",
            description="A1 full objective.",
            target_scale=600.0,
            loss_mode="curvature_compensated",
            y_loss_alpha=0.65,
            z_loss_beta=0.0,
            z_huber_delta=1.0,
            y_reweight_clip_max=4.0,
            center_penalty_weight=0.0,
        ),
    }


def _predict_logits_array(model: torch.nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for offset in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(np.array(X[offset : offset + batch_size], dtype=np.float32, copy=True)).to(device)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model.forward_logits(xb).view(-1)
            outputs.append(logits.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(outputs, axis=0)


def _iter_named_trainable_params(model: torch.nn.Module) -> Iterable[Tuple[str, torch.nn.Parameter]]:
    for name, param in model.named_parameters():
        if param.requires_grad:
            yield name, param


def _grad_group_name(param_name: str) -> str:
    lower = param_name.lower()
    if "head" in lower:
        return "head"
    if "block" in lower:
        return "backbone"
    return "stem"


def compute_average_gradient_snapshot(
    checkpoint_path: str | Path,
    objective_variant: ab_lab.AblationVariant,
    band_name: str,
    batch_payload: Dict[str, np.ndarray],
    cfg: GradientAuditConfig,
    device: torch.device,
    paths: Dict[str, Path],
    refresh: bool = False,
) -> Dict[str, object]:
    out_dir = paths["cache_dir"] / "gradient_snapshots" / objective_variant.name
    out_dir.mkdir(parents=True, exist_ok=True)
    grad_pt = out_dir / f"{band_name}_avg_grad.pt"
    summary_json = out_dir / f"{band_name}_summary.json"
    if not refresh and grad_pt.exists() and summary_json.exists():
        return json.loads(summary_json.read_text(encoding="utf-8"))

    model, _ = base_lab.load_model_from_checkpoint(checkpoint_path, device=device)
    model.train()
    avg_grads: Dict[str, torch.Tensor] = {}
    group_sq: Dict[str, float] = {"stem": 0.0, "backbone": 0.0, "head": 0.0, "all": 0.0}
    batches = int(cfg.batches_per_band)
    X = batch_payload["X"]
    y = batch_payload["y"]
    for batch_idx in range(batches):
        start = batch_idx * cfg.batch_size
        end = start + cfg.batch_size
        xb = torch.from_numpy(np.array(X[start:end], dtype=np.float32, copy=True)).to(device)
        yb = torch.from_numpy(np.array(y[start:end], dtype=np.float32, copy=True)).to(device).view(-1)
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model.forward_logits(xb).view(-1)
            terms = ab_lab.compute_variant_terms(logits, yb, objective_variant)
        terms["objective"].backward()
        for name, param in _iter_named_trainable_params(model):
            if param.grad is None:
                continue
            grad_cpu = param.grad.detach().float().cpu()
            if name not in avg_grads:
                avg_grads[name] = torch.zeros_like(grad_cpu)
            avg_grads[name].add_(grad_cpu)
    for name in list(avg_grads.keys()):
        avg_grads[name].div_(float(batches))
        sq = float(torch.sum(avg_grads[name] * avg_grads[name]).item())
        group = _grad_group_name(name)
        group_sq[group] += sq
        group_sq["all"] += sq
    norms = {f"grad_norm_{group}": float(math.sqrt(max(value, 0.0))) for group, value in group_sq.items()}
    payload = {"avg_grads": avg_grads, "norms": norms, "band_name": band_name, "objective_name": objective_variant.name}
    torch.save(payload, grad_pt)
    summary = {"grad_path": str(grad_pt), "band_name": band_name, "objective_name": objective_variant.name, **norms}
    save_json(summary, summary_json)
    del model
    _cleanup_cuda()
    return summary


def _load_grad_snapshot(grad_path: str | Path) -> Dict[str, torch.Tensor]:
    payload = torch.load(Path(grad_path), map_location="cpu", weights_only=False)
    grads = payload.get("avg_grads")
    if not isinstance(grads, dict):
        raise RuntimeError(f"Invalid gradient snapshot payload: {grad_path}")
    return {str(name): tensor.detach().float().cpu() for name, tensor in grads.items()}


def _flatten_group_vector(grad_dict: Dict[str, torch.Tensor], group: str) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    for name, tensor in grad_dict.items():
        tensor = tensor.detach().float().view(-1).cpu()
        if group == "all" or _grad_group_name(name) == group:
            parts.append(tensor)
    if not parts:
        return torch.zeros(1, dtype=torch.float32)
    return torch.cat(parts, dim=0)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().view(-1)
    b = b.detach().float().view(-1)
    denom = float(torch.linalg.norm(a).item() * torch.linalg.norm(b).item())
    if denom <= 0.0:
        return float("nan")
    return float(torch.dot(a, b).item() / denom)


def apply_normalized_gradient_step(
    checkpoint_path: str | Path,
    grad_path: str | Path,
    device: torch.device,
    step_l2: float,
) -> torch.nn.Module:
    model, _ = base_lab.load_model_from_checkpoint(checkpoint_path, device=device)
    grad_dict = _load_grad_snapshot(grad_path)
    total_sq = 0.0
    for tensor in grad_dict.values():
        total_sq += float(torch.sum(tensor * tensor).item())
    total_norm = math.sqrt(max(total_sq, 0.0))
    scale = 0.0 if total_norm <= 0.0 else float(step_l2) / float(total_norm)
    with torch.no_grad():
        for name, param in model.named_parameters():
            grad_cpu = grad_dict.get(name)
            if grad_cpu is None:
                continue
            param.add_((-scale) * grad_cpu.to(device=device, dtype=param.dtype))
    model.eval()
    return model


def _probe_metrics(
    model: torch.nn.Module,
    bundle: Dict[str, object],
    device: torch.device,
    batch_size: int = 1024,
) -> Dict[str, float]:
    logits = _predict_logits_array(model, bundle["X"], device=device, batch_size=batch_size)
    pred = np.tanh(logits.astype(np.float64))
    oracle = np.asarray(bundle["oracle_y600"], dtype=np.float64)
    out = {
        "n": int(pred.shape[0]),
        "mean_abs_pred": float(np.mean(np.abs(pred))),
        "mean_abs_oracle": float(np.mean(np.abs(oracle))),
        "mean_signed_pred": float(np.mean(pred)),
        "mae_vs_oracle": float(np.mean(np.abs(pred - oracle))),
    }
    if np.any(np.abs(oracle) <= 0.05):
        center_mask = np.abs(oracle) <= 0.05
        out["center_false_0.1"] = float(np.mean(np.abs(pred[center_mask]) >= 0.10))
        out["center_false_0.2"] = float(np.mean(np.abs(pred[center_mask]) >= 0.20))
    return out


def run_gradient_interference_audit(
    checkpoint_path: str | Path,
    gradient_batches: Dict[str, Dict[str, np.ndarray]],
    probe_sets: Dict[str, object],
    cfg: GradientAuditConfig,
    paths: Dict[str, Path],
    device: torch.device,
    refresh: bool = False,
) -> Dict[str, object]:
    reports_dir = paths["reports_dir"]
    summary_json = reports_dir / "gradient_interference_summary.json"
    norms_csv = reports_dir / "gradient_interference_norms.csv"
    cosine_csv = reports_dir / "gradient_interference_cosines.csv"
    influence_csv = reports_dir / "gradient_interference_influence.csv"
    if (
        not refresh
        and summary_json.exists()
        and norms_csv.exists()
        and cosine_csv.exists()
        and influence_csv.exists()
    ):
        return {
            "summary": json.loads(summary_json.read_text(encoding="utf-8")),
            "norms": pd.read_csv(norms_csv),
            "cosines": pd.read_csv(cosine_csv),
            "influence": pd.read_csv(influence_csv),
        }

    objective_specs = build_failure_b_objective_specs()
    grad_info: Dict[Tuple[str, str], Dict[str, object]] = {}
    norm_rows: List[dict] = []
    for objective_name, variant in objective_specs.items():
        for band_name, payload in gradient_batches.items():
            info = compute_average_gradient_snapshot(
                checkpoint_path=checkpoint_path,
                objective_variant=variant,
                band_name=band_name,
                batch_payload=payload,
                cfg=cfg,
                device=device,
                paths=paths,
                refresh=refresh,
            )
            grad_info[(objective_name, band_name)] = info
            norm_rows.append(info)
    norms_df = pd.DataFrame(norm_rows)
    save_dataframe(norms_df, norms_csv)

    cosine_rows: List[dict] = []
    band_names = list(gradient_batches.keys())
    for objective_name in objective_specs:
        snapshots = {
            band_name: _load_grad_snapshot(grad_info[(objective_name, band_name)]["grad_path"])
            for band_name in band_names
        }
        for left in band_names:
            for right in band_names:
                row = {
                    "objective_name": objective_name,
                    "band_left": left,
                    "band_right": right,
                }
                for group in ("all", "backbone", "head"):
                    row[f"cosine_{group}"] = _cosine(
                        _flatten_group_vector(snapshots[left], group),
                        _flatten_group_vector(snapshots[right], group),
                    )
                cosine_rows.append(row)
    cosines_df = pd.DataFrame(cosine_rows)
    save_dataframe(cosines_df, cosine_csv)

    base_model, _ = base_lab.load_model_from_checkpoint(checkpoint_path, device=device)
    baseline_probe = {
        name: _probe_metrics(base_model, bundle, device=device)
        for name, bundle in probe_sets["bundles"].items()
    }
    del base_model
    _cleanup_cuda()

    influence_rows: List[dict] = []
    for objective_name in objective_specs:
        for source_band in band_names:
            stepped_model = apply_normalized_gradient_step(
                checkpoint_path=checkpoint_path,
                grad_path=grad_info[(objective_name, source_band)]["grad_path"],
                device=device,
                step_l2=cfg.influence_step_l2,
            )
            for probe_name, bundle in probe_sets["bundles"].items():
                after = _probe_metrics(stepped_model, bundle, device=device)
                before = baseline_probe[probe_name]
                influence_rows.append(
                    {
                        "objective_name": objective_name,
                        "source_band": source_band,
                        "probe_name": probe_name,
                        "step_l2": float(cfg.influence_step_l2),
                        "before_mean_abs_pred": float(before["mean_abs_pred"]),
                        "after_mean_abs_pred": float(after["mean_abs_pred"]),
                        "delta_mean_abs_pred": float(after["mean_abs_pred"] - before["mean_abs_pred"]),
                        "before_mae_vs_oracle": float(before["mae_vs_oracle"]),
                        "after_mae_vs_oracle": float(after["mae_vs_oracle"]),
                        "delta_mae_vs_oracle": float(after["mae_vs_oracle"] - before["mae_vs_oracle"]),
                        "before_center_false_0.1": float(before.get("center_false_0.1", float("nan"))),
                        "after_center_false_0.1": float(after.get("center_false_0.1", float("nan"))),
                        "delta_center_false_0.1": float(after.get("center_false_0.1", float("nan")) - before.get("center_false_0.1", float("nan"))),
                    }
                )
            del stepped_model
            _cleanup_cuda()
    influence_df = pd.DataFrame(influence_rows)
    save_dataframe(influence_df, influence_csv)

    summary = {
        "baseline_checkpoint": str(Path(checkpoint_path)),
        "objectives": list(objective_specs.keys()),
        "bands": band_names,
        "probe_names": list(probe_sets["bundles"].keys()),
        "influence_step_l2": float(cfg.influence_step_l2),
        "center_probe_delta_mean_abs_pred_from_midbands": {
            objective_name: float(
                influence_df.loc[
                    (influence_df["objective_name"] == objective_name)
                    & (influence_df["source_band"].isin(["mid_02_05", "mid_05_07"]))
                    & (influence_df["probe_name"] == "center_clean_005"),
                    "delta_mean_abs_pred",
                ].mean()
            )
            for objective_name in objective_specs
        },
    }
    save_json(summary, summary_json)
    return {"summary": summary, "norms": norms_df, "cosines": cosines_df, "influence": influence_df}


def run_control_replicate_l0_l1(
    baseline_ckpt: str | Path,
    data_root: str | Path,
    cfg: obj_lab.ReplicateOracleConfig,
    paths: Dict[str, Path],
    device: torch.device,
    refresh: bool = False,
) -> Dict[str, object]:
    output_dir = paths["replicates_dir"] / "l0_l1_controls"
    reports_dir = output_dir / "reports"
    aggregate_csv = reports_dir / "replicate_oracle_aggregate.csv"
    per_csv = reports_dir / "replicate_oracle_metrics.csv"
    if not refresh and aggregate_csv.exists() and per_csv.exists():
        return {
            "per_replicate": pd.read_csv(per_csv),
            "aggregate": pd.read_csv(aggregate_csv),
        }

    registry = [
        {"label": "baseline", "checkpoint": str(Path(baseline_ckpt)), "target_scale": 600.0},
        {
            "label": "L0_control_hybrid",
            "checkpoint": str(paths["objective_output_dir"] / "runs" / "L0_control_hybrid" / "checkpoints" / "L0_control_hybrid_best.pt"),
            "target_scale": 600.0,
        },
        {
            "label": "L1_z_strong_hybrid",
            "checkpoint": str(paths["objective_output_dir"] / "runs" / "L1_z_strong_hybrid" / "checkpoints" / "L1_z_strong_hybrid_best.pt"),
            "target_scale": 600.0,
        },
    ]
    per_df, agg_df = obj_lab.run_oracle_replicates(
        registry=registry,
        baseline_ckpt=baseline_ckpt,
        data_root=data_root,
        cfg=cfg,
        device=device,
        output_dir=output_dir,
    )
    return {"per_replicate": per_df, "aggregate": agg_df}


def precompute_train_prediction_cache(
    checkpoint_path: str | Path,
    data_root: str | Path,
    split: str,
    num_shards: int,
    paths: Dict[str, Path],
    device: torch.device,
    batch_size: int = 1024,
    refresh: bool = False,
) -> Dict[str, object]:
    cache_dir = paths["cache_dir"] / "train_pred_cache" / Path(checkpoint_path).stem
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    shard_rows = resolve_split_shards(data_root, split, num_shards=num_shards)
    existing = None
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = None
    if (
        not refresh
        and existing is not None
        and existing.get("checkpoint_path") == str(Path(checkpoint_path).resolve())
        and existing.get("split") == split
        and int(existing.get("num_shards", -1)) == int(num_shards)
        and all((cache_dir / item["pred_file"]).exists() for item in existing.get("shards", []))
    ):
        return existing

    model, _ = base_lab.load_model_from_checkpoint(checkpoint_path, device=device)
    model.eval()
    manifest = {
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "split": str(split),
        "num_shards": int(num_shards),
        "batch_size": int(batch_size),
        "shards": [],
    }
    with torch.no_grad():
        for shard_id, x_path, _ in shard_rows:
            pred_file = cache_dir / f"pred_{shard_id:05d}.npy"
            X = np.load(x_path, mmap_mode="r")
            if not refresh and pred_file.exists():
                preds = np.load(pred_file, mmap_mode="r")
                n = int(preds.shape[0])
            else:
                preds = base_lab.predict_array(model, X, device=device, batch_size=batch_size)
                np.save(pred_file, preds.astype(np.float32, copy=False))
                n = int(preds.shape[0])
            manifest["shards"].append({"shard_id": int(shard_id), "pred_file": pred_file.name, "n": n})
    save_json(manifest, manifest_path)
    del model
    _cleanup_cuda()
    return manifest


def build_pilot_variants(paths: Dict[str, Path]) -> Dict[str, FailureBPilotVariant]:
    init_ckpt = paths["objective_output_dir"] / "runs" / "L3_full_A1" / "checkpoints" / "L3_full_A1_best.pt"
    base_main = ab_lab.AblationVariant(
        name="P_base_A1",
        description="Base A1 main loss without built-in center penalty.",
        target_scale=600.0,
        sampler_mode="random",
        loss_mode="curvature_compensated",
        y_loss_alpha=0.65,
        z_loss_beta=0.0,
        z_huber_delta=1.0,
        y_reweight_clip_max=4.0,
        center_penalty_weight=0.0,
    )
    return {
        "P_B2_raw_center_strong": FailureBPilotVariant(
            name="P_B2_raw_center_strong",
            description="Test B2 path: stronger raw center penalty on top of A1.",
            init_ckpt_path=str(init_ckpt),
            main_variant=base_main,
            center_mode="raw",
            center_penalty_weight=0.50,
            center_penalty_tau_y600=0.05,
            center_penalty_margin_y600=0.12,
        ),
        "P_B1_proxy_center_weighted": FailureBPilotVariant(
            name="P_B1_proxy_center_weighted",
            description="Test B1 path: center penalty weighted by trusted-center proxy.",
            init_ckpt_path=str(init_ckpt),
            main_variant=base_main,
            center_mode="proxy_weighted",
            center_penalty_weight=0.50,
            center_penalty_tau_y600=0.05,
            center_penalty_margin_y600=0.12,
            proxy_min_score=0.55,
            proxy_power=1.0,
        ),
    }


def _pilot_gate(primary: Dict[str, object]) -> float:
    values = [
        float(primary["oracle_midband_mae_sum_stable"]),
        float(primary["oracle_center_amp_ratio"]),
        float(primary["oracle_center_false_0.1eq"]),
        float(primary["oracle_center_false_0.2eq"]),
    ]
    if not all(np.isfinite(v) for v in values):
        return float("inf")
    return float(
        primary["oracle_midband_mae_sum_stable"]
        + 0.35 * max(0.0, 0.80 - float(primary["oracle_stable_0.7_slope"]))
        + 0.35 * float(primary["oracle_center_false_0.1eq"])
        + 0.20 * float(primary["oracle_center_false_0.2eq"])
        + 0.10 * max(0.0, float(primary["oracle_center_amp_ratio"]) - 2.5)
    )


def _compute_center_penalty(
    pred: torch.Tensor,
    y_source: torch.Tensor,
    batch_proxy_weight: Optional[torch.Tensor],
    variant: FailureBPilotVariant,
) -> torch.Tensor:
    tau = float(variant.center_penalty_tau_y600)
    margin = float(variant.center_penalty_margin_y600)
    center_mask = torch.abs(y_source) <= tau
    if not torch.any(center_mask):
        return pred.new_tensor(0.0)
    excess = torch.relu(torch.abs(pred[center_mask]) - margin) ** 2
    if variant.center_mode == "proxy_weighted" and batch_proxy_weight is not None:
        weights = batch_proxy_weight[center_mask]
        weights = torch.where(weights >= float(variant.proxy_min_score), weights, torch.zeros_like(weights))
        weights = torch.pow(torch.clamp(weights, min=0.0), float(max(variant.proxy_power, 0.0)))
        denom = torch.clamp(weights.sum(), min=1e-6)
        return torch.sum(excess * weights) / denom
    return torch.mean(excess)


def run_failure_b_pilot(
    variant: FailureBPilotVariant,
    pilot_cfg: PilotTrainConfig,
    data_root: str | Path,
    oracle_cfg: ab_lab.OracleEvalConfig,
    oracle_bundle: Dict[str, object],
    pooled_center_bundle: Dict[str, object],
    center_lookup: Dict[str, object],
    pred_cache_manifest: Dict[str, object],
    paths: Dict[str, Path],
    device: torch.device,
    refresh: bool = False,
) -> Dict[str, object]:
    run_dir = paths["pilots_dir"] / variant.name
    reports_dir = run_dir / "reports"
    checkpoints_dir = run_dir / "checkpoints"
    reports_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = checkpoints_dir / f"{variant.name}_best.pt"
    latest_ckpt = checkpoints_dir / f"{variant.name}_latest.pt"
    resume_ckpt = checkpoints_dir / f"{variant.name}_resume.pt"
    history_csv = reports_dir / "history.csv"
    history_json = reports_dir / "history.json"

    if not refresh and best_ckpt.exists() and history_csv.exists():
        best_model, _ = base_lab.load_model_from_checkpoint(best_ckpt, device=device)
        try:
            test_eval = ab_lab.evaluate_model_on_split_scale_aware(
                model=best_model,
                data_root=data_root,
                split="test",
                device=device,
                max_samples=pilot_cfg.test_max_samples,
                num_shards=pilot_cfg.test_num_shards,
                batch_size=max(pilot_cfg.batch_size, 1024),
                target_scale=variant.main_variant.target_scale,
                oracle_cfg=oracle_cfg,
            )
            oracle_eval = ab_lab.evaluate_model_on_oracle_subset(
                model=best_model,
                oracle_bundle=oracle_bundle,
                device=device,
                target_scale=variant.main_variant.target_scale,
                oracle_cfg=oracle_cfg,
            )
            pooled_center_eval = evaluate_model_on_center_bundle(best_model, pooled_center_bundle, device=device)
        finally:
            del best_model
            _cleanup_cuda()
        return {
            "variant": variant,
            "best_checkpoint": best_ckpt,
            "history": pd.read_csv(history_csv),
            "test_eval": test_eval,
            "oracle_eval": oracle_eval,
            "pooled_center_eval": pooled_center_eval,
        }

    set_global_seed(pilot_cfg.seed)
    model, init_ckpt = base_lab.load_model_from_checkpoint(variant.init_ckpt_path, device=device)
    optimizer = ab_lab.build_optimizer(model, lr=pilot_cfg.learning_rate, weight_decay=pilot_cfg.weight_decay)
    shard_rows = resolve_split_shards(data_root, "train", num_shards=pilot_cfg.train_num_shards)
    total_samples = int(sum(int(np.load(y_path, mmap_mode="r").shape[0]) for _, _, y_path in shard_rows))
    total_steps = int(math.ceil(total_samples / pilot_cfg.batch_size) * max(pilot_cfg.epochs, 1))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=pilot_cfg.min_lr)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    cache_dir = paths["cache_dir"] / "train_pred_cache" / Path(variant.init_ckpt_path).stem
    lookup_df = center_lookup["lookup"]
    lookup_report = center_lookup["report"]
    rng = np.random.default_rng(pilot_cfg.seed)
    history_rows: List[dict] = []
    best_gate = float("inf")
    global_step = 0
    resume_epoch = 0
    resume_shard_index = 0

    if not refresh and resume_ckpt.exists():
        resume_payload = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_payload["model_state"])
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        _optimizer_to_device(optimizer, device)
        scheduler.load_state_dict(resume_payload["scheduler_state"])
        scaler.load_state_dict(resume_payload["scaler_state"])
        history_rows = list(resume_payload.get("history_rows", []))
        best_gate = float(resume_payload.get("best_gate", best_gate))
        global_step = int(resume_payload.get("global_step", 0))
        resume_epoch = int(resume_payload.get("resume_epoch", 0))
        resume_shard_index = int(resume_payload.get("resume_shard_index", 0))

    for epoch in range(resume_epoch, pilot_cfg.epochs):
        t0 = time.time()
        running = {
            "objective": 0.0,
            "main_term": 0.0,
            "center_penalty": 0.0,
            "n": 0,
            "center_samples": 0,
            "proxy_active_samples": 0,
            "proxy_weight_sum": 0.0,
        }
        model.train()
        epoch_shards = shard_rows[resume_shard_index:] if epoch == resume_epoch and resume_shard_index > 0 else shard_rows
        for local_shard_rank, (shard_id, x_path, y_path) in enumerate(epoch_shards, start=1):
            shard_rank = (resume_shard_index if epoch == resume_epoch else 0) + local_shard_rank
            X = np.load(x_path, mmap_mode="r")
            y_source = np.load(y_path, mmap_mode="r").astype(np.float32, copy=False)
            y_train = ab_lab.remap_target_np(y_source, to_scale=variant.main_variant.target_scale).astype(np.float32, copy=False)
            pred_file = cache_dir / f"pred_{shard_id:05d}.npy"
            if not pred_file.exists():
                raise FileNotFoundError(f"Missing cached prediction shard for pilot: {pred_file}")
            init_pred = np.load(pred_file, mmap_mode="r").astype(np.float32, copy=False)
            abs_y_source = np.abs(y_source.astype(np.float64))
            abs_pred = np.abs(init_pred.astype(np.float64))
            abs_err = np.abs((init_pred - y_source).astype(np.float64))
            proxy_score = score_center_purity(
                abs_y=abs_y_source,
                abs_pred=abs_pred,
                abs_err=abs_err,
                lookup_df=lookup_df,
                lookup_report=lookup_report,
            )
            order = rng.permutation(y_train.shape[0]).astype(np.int64)
            for start in range(0, y_train.shape[0], pilot_cfg.batch_size):
                idx = order[start : start + pilot_cfg.batch_size]
                xb = torch.from_numpy(np.array(X[idx], dtype=np.float32, copy=True)).to(device, non_blocking=True)
                yb = torch.from_numpy(np.array(y_train[idx], dtype=np.float32, copy=True)).to(device, non_blocking=True).view(-1)
                yb_source = torch.from_numpy(np.array(y_source[idx], dtype=np.float32, copy=True)).to(device, non_blocking=True).view(-1)
                proxy_w = torch.from_numpy(np.array(proxy_score[idx], dtype=np.float32, copy=True)).to(device, non_blocking=True).view(-1)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    logits = model.forward_logits(xb).view(-1)
                    terms = ab_lab.compute_variant_terms(logits, yb, variant.main_variant)
                    center_pen = _compute_center_penalty(
                        pred=terms["pred"],
                        y_source=yb_source,
                        batch_proxy_weight=proxy_w,
                        variant=variant,
                    )
                    objective = terms["main_term"] + float(variant.center_penalty_weight) * center_pen
                scaler.scale(objective).backward()
                if pilot_cfg.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), pilot_cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                bs = int(yb.numel())
                running["objective"] += float(objective.item()) * bs
                running["main_term"] += float(terms["main_term"].item()) * bs
                running["center_penalty"] += float(center_pen.item()) * bs
                running["n"] += bs
                center_mask_np = np.abs(np.asarray(y_source[idx], dtype=np.float32)) <= float(variant.center_penalty_tau_y600)
                running["center_samples"] += int(center_mask_np.sum())
                if variant.center_mode == "proxy_weighted":
                    proxy_np = np.asarray(proxy_score[idx], dtype=np.float32)
                    active_np = center_mask_np & (proxy_np >= float(variant.proxy_min_score))
                    running["proxy_active_samples"] += int(active_np.sum())
                    running["proxy_weight_sum"] += float(proxy_np[active_np].sum())
                global_step += 1
                if global_step % pilot_cfg.log_every_steps == 0:
                    print(
                        f"[{variant.name}] step={global_step}/{total_steps} "
                        f"obj={running['objective'] / running['n']:.6f} "
                        f"main={running['main_term'] / running['n']:.6f} "
                        f"center_pen={running['center_penalty'] / running['n']:.6f}"
                    )
            print(f"[{variant.name}] finished shard {shard_rank}/{len(shard_rows)}")
            resume_payload = {
                "resume_epoch": int(epoch),
                "resume_shard_index": int(shard_rank),
                "global_step": int(global_step),
                "best_gate": float(best_gate),
                "history_rows": history_rows,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "pilot_variant": asdict(variant),
                "pilot_train_cfg": asdict(pilot_cfg),
            }
            ab_lab.atomic_torch_save(resume_payload, resume_ckpt)

        test_eval = ab_lab.evaluate_model_on_split_scale_aware(
            model=model,
            data_root=data_root,
            split="test",
            device=device,
            max_samples=pilot_cfg.test_max_samples,
            num_shards=pilot_cfg.test_num_shards,
            batch_size=max(pilot_cfg.batch_size, 1024),
            target_scale=variant.main_variant.target_scale,
            oracle_cfg=oracle_cfg,
        )
        oracle_eval = ab_lab.evaluate_model_on_oracle_subset(
            model=model,
            oracle_bundle=oracle_bundle,
            device=device,
            target_scale=variant.main_variant.target_scale,
            oracle_cfg=oracle_cfg,
        )
        pooled_center_eval = evaluate_model_on_center_bundle(model, pooled_center_bundle, device=device)
        primary = obj_lab.extract_primary_metrics(
            label=variant.name,
            split_eval=test_eval,
            oracle_eval=oracle_eval,
        )
        primary["pooled_center_mae"] = float(pooled_center_eval["mae_vs_oracle"])
        primary["pooled_center_false_0.1eq"] = float(pooled_center_eval["false_decisive_0.1"])
        primary["pooled_center_false_0.2eq"] = float(pooled_center_eval["false_decisive_0.2"])
        primary["pooled_center_amp_ratio"] = float(pooled_center_eval["amp_ratio"])
        gate = _pilot_gate(primary)
        row = {
            "epoch": int(epoch),
            "train_objective": running["objective"] / running["n"],
            "train_main_term": running["main_term"] / running["n"],
            "train_center_penalty": running["center_penalty"] / running["n"],
            "oracle_gate_score": float(gate),
            "oracle_stable_0.7_slope": float(primary["oracle_stable_0.7_slope"]),
            "oracle_midband_mae_sum_stable": float(primary["oracle_midband_mae_sum_stable"]),
            "oracle_center_amp_ratio": float(primary["oracle_center_amp_ratio"]),
            "oracle_center_false_0.1eq": float(primary["oracle_center_false_0.1eq"]),
            "oracle_center_false_0.2eq": float(primary["oracle_center_false_0.2eq"]),
            "pooled_center_amp_ratio": float(primary["pooled_center_amp_ratio"]),
            "pooled_center_false_0.1eq": float(primary["pooled_center_false_0.1eq"]),
            "pooled_center_false_0.2eq": float(primary["pooled_center_false_0.2eq"]),
            "center_proxy_active_rate": float(running["proxy_active_samples"] / max(running["center_samples"], 1)),
            "center_proxy_active_weight_mean": float(running["proxy_weight_sum"] / max(running["proxy_active_samples"], 1)),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_time_sec": float(time.time() - t0),
        }
        history_rows.append(row)
        print(json.dumps(row, indent=2))

        payload = {
            "epoch": int(epoch),
            "pilot_variant": asdict(variant),
            "pilot_train_cfg": asdict(pilot_cfg),
            "config": init_ckpt.get("config"),
            "history": history_rows,
            "model_state": model.state_dict(),
            "oracle_summary": oracle_eval["summary"],
            "pooled_center_eval": pooled_center_eval,
        }
        ab_lab.atomic_torch_save(payload, latest_ckpt)
        if gate < best_gate:
            best_gate = gate
            ab_lab.atomic_torch_save(payload, best_ckpt)
            save_json(test_eval["metrics"], reports_dir / "best_test_scale_metrics.json")
            save_json(oracle_eval["summary"], reports_dir / "best_test_oracle_summary.json")
            save_json(pooled_center_eval, reports_dir / "best_pooled_center_eval.json")
            ab_lab.save_oracle_eval_outputs(oracle_eval, run_dir, prefix="best_test_oracle")
        resume_shard_index = 0

    if resume_ckpt.exists():
        try:
            resume_ckpt.unlink()
        except OSError:
            pass

    history_df = pd.DataFrame(history_rows)
    save_dataframe(history_df, history_csv)
    save_json({"history": history_rows, "best_gate_score": best_gate, "pilot_variant": asdict(variant)}, history_json)

    best_model, _ = base_lab.load_model_from_checkpoint(best_ckpt, device=device)
    try:
        test_eval = ab_lab.evaluate_model_on_split_scale_aware(
            model=best_model,
            data_root=data_root,
            split="test",
            device=device,
            max_samples=pilot_cfg.test_max_samples,
            num_shards=pilot_cfg.test_num_shards,
            batch_size=max(pilot_cfg.batch_size, 1024),
            target_scale=variant.main_variant.target_scale,
            oracle_cfg=oracle_cfg,
        )
        oracle_eval = ab_lab.evaluate_model_on_oracle_subset(
            model=best_model,
            oracle_bundle=oracle_bundle,
            device=device,
            target_scale=variant.main_variant.target_scale,
            oracle_cfg=oracle_cfg,
        )
        pooled_center_eval = evaluate_model_on_center_bundle(best_model, pooled_center_bundle, device=device)
    finally:
        del best_model
        del model
        _cleanup_cuda()
    return {
        "variant": variant,
        "best_checkpoint": best_ckpt,
        "history": history_df,
        "test_eval": test_eval,
        "oracle_eval": oracle_eval,
        "pooled_center_eval": pooled_center_eval,
        "pred_cache_manifest": pred_cache_manifest,
    }


def build_pooled_center_bundle(
    pooled_unique: pd.DataFrame,
    data_root: str | Path,
    center_thr: float = 0.05,
    refresh: bool = False,
    paths: Optional[Dict[str, Path]] = None,
) -> Dict[str, object]:
    if paths is not None:
        cache_dir = paths["cache_dir"] / "pooled_center_bundle"
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = cache_dir / "manifest.json"
        if not refresh and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            npz = np.load(cache_dir / manifest["npz"], allow_pickle=False)
            return {
                "rows": pd.read_csv(cache_dir / manifest["rows_csv"]),
                "X": npz["X"].astype(np.uint8),
                "oracle_y600": npz["oracle_y600"].astype(np.float64),
                "train_y": npz["train_y"].astype(np.float64),
                "manifest": manifest,
            }
    frame = pooled_unique[
        pooled_unique["stability_group"].eq("stable")
        & (pooled_unique["oracle_reference_y_600"].abs() <= float(center_thr))
    ].copy()
    if frame.empty:
        raise RuntimeError(
            f"Pooled center bundle is empty for center_thr={center_thr}. "
            "Check oracle pool construction before running Failure B pilots."
        )
    X, rows = _load_rows_to_tensor_bundle(frame, data_root)
    bundle = {
        "rows": rows.reset_index(drop=True).copy(),
        "X": X.astype(np.uint8, copy=False),
        "oracle_y600": rows["oracle_reference_y_600"].to_numpy(dtype=np.float64),
        "train_y": rows["target_y"].to_numpy(dtype=np.float64),
    }
    if paths is not None:
        rows_csv = "pooled_center_rows.csv"
        npz_name = "pooled_center_bundle.npz"
        save_dataframe(bundle["rows"], cache_dir / rows_csv)
        np.savez_compressed(
            cache_dir / npz_name,
            X=bundle["X"].astype(np.uint8),
            oracle_y600=bundle["oracle_y600"].astype(np.float32),
            train_y=bundle["train_y"].astype(np.float32),
        )
        manifest = {
            "rows_csv": rows_csv,
            "npz": npz_name,
            "center_thr": float(center_thr),
            "n": int(bundle["rows"].shape[0]),
        }
        save_json(manifest, manifest_path)
        bundle["manifest"] = manifest
    return bundle


def evaluate_model_on_center_bundle(
    model: torch.nn.Module,
    bundle: Dict[str, object],
    device: torch.device,
    batch_size: int = 1024,
) -> Dict[str, float]:
    logits = _predict_logits_array(model, bundle["X"], device=device, batch_size=batch_size)
    pred = np.tanh(logits.astype(np.float64))
    oracle = np.asarray(bundle["oracle_y600"], dtype=np.float64)
    out = {
        "n": int(pred.shape[0]),
        "mae_vs_oracle": float(np.mean(np.abs(pred - oracle))),
        "amp_ratio": float(np.mean(np.abs(pred)) / max(np.mean(np.abs(oracle)), 1e-12)),
        "false_decisive_0.1": float(np.mean(np.abs(pred) >= 0.10)),
        "false_decisive_0.2": float(np.mean(np.abs(pred) >= 0.20)),
        "wrong_sign_0.1": float(np.mean((np.abs(pred) >= 0.10) & (np.sign(pred) != np.sign(oracle)) & (np.abs(oracle) > 0.02))),
        "wrong_sign_0.2": float(np.mean((np.abs(pred) >= 0.20) & (np.sign(pred) != np.sign(oracle)) & (np.abs(oracle) > 0.02))),
        "spread_ratio": float(np.std(pred) / (np.std(oracle) + 1e-12)),
        "mean_abs_pred": float(np.mean(np.abs(pred))),
        "mean_abs_oracle": float(np.mean(np.abs(oracle))),
    }
    return out


def evaluate_on_pooled_center_bundle(
    checkpoint_path: str | Path,
    label: str,
    pooled_center_bundle: Dict[str, object],
    device: torch.device,
) -> Dict[str, object]:
    model, _ = base_lab.load_model_from_checkpoint(checkpoint_path, device=device)
    try:
        metrics = evaluate_model_on_center_bundle(model, pooled_center_bundle, device=device)
    finally:
        del model
        _cleanup_cuda()
    return {"label": str(label), "checkpoint": str(Path(checkpoint_path)), **metrics}


def evaluate_failure_b_registry(
    registry: Sequence[Dict[str, object]],
    pooled_center_bundle: Dict[str, object],
    oracle_bundle: Dict[str, object],
    data_root: str | Path,
    train_cfg: ab_lab.TrainConfig,
    oracle_cfg: ab_lab.OracleEvalConfig,
    paths: Dict[str, Path],
    device: torch.device,
    prefix: str,
) -> Dict[str, object]:
    rows_center: List[dict] = []
    rows_primary: List[dict] = []
    results: Dict[str, object] = {}
    for item in registry:
        label = str(item["label"])
        ckpt_path = item["checkpoint"]
        center_eval = evaluate_on_pooled_center_bundle(
            checkpoint_path=ckpt_path,
            label=label,
            pooled_center_bundle=pooled_center_bundle,
            device=device,
        )
        result = obj_lab.evaluate_checkpoint(
            ckpt_path=ckpt_path,
            label=label,
            data_root=data_root,
            oracle_bundle=oracle_bundle,
            oracle_cfg=oracle_cfg,
            train_cfg=train_cfg,
            device=device,
            target_scale=float(item.get("target_scale", 600.0)),
        )
        row = dict(result["primary"])
        row.update(
            {
                "pooled_center_mae": float(center_eval["mae_vs_oracle"]),
                "pooled_center_amp_ratio": float(center_eval["amp_ratio"]),
                "pooled_center_false_0.1eq": float(center_eval["false_decisive_0.1"]),
                "pooled_center_false_0.2eq": float(center_eval["false_decisive_0.2"]),
                "pooled_center_wrong_sign_0.1eq": float(center_eval["wrong_sign_0.1"]),
                "pooled_center_wrong_sign_0.2eq": float(center_eval["wrong_sign_0.2"]),
                "pooled_center_spread_ratio": float(center_eval["spread_ratio"]),
            }
        )
        row["failure_b_score"] = float(
            row["pooled_center_mae"]
            + 0.30 * row["pooled_center_false_0.1eq"]
            + 0.20 * row["pooled_center_false_0.2eq"]
            + 0.10 * max(0.0, row["pooled_center_amp_ratio"] - 2.5)
            + 0.30 * row["oracle_midband_mae_sum_stable"]
            + 0.20 * max(0.0, 0.80 - row["oracle_stable_0.7_slope"])
        )
        rows_center.append(center_eval)
        rows_primary.append(row)
        results[label] = {"primary": row, "center_eval": center_eval, "full_eval": result}
    center_df = pd.DataFrame(rows_center)
    primary_df = pd.DataFrame(rows_primary).sort_values("failure_b_score", ascending=True).reset_index(drop=True)
    save_dataframe(center_df, paths["reports_dir"] / f"{prefix}_pooled_center_metrics.csv")
    save_dataframe(primary_df, paths["reports_dir"] / f"{prefix}_primary_metrics.csv")
    return {"pooled_center": center_df, "primary": primary_df, "results": results}
