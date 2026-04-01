from __future__ import annotations

import gc
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT_CAUSE_DIR = PROJECT_ROOT / "experiments" / "root_cause_ablation_suite"
OBJECTIVE_DIR = PROJECT_ROOT / "experiments" / "objective_resolution_suite"
FAILURE_B_DIR = PROJECT_ROOT / "experiments" / "failure_b_resolution_suite"
STABILITY_DIR = PROJECT_ROOT / "experiments" / "stability_weighted_near_zero_finetune"
TEACHER_DIR = PROJECT_ROOT / "experiments" / "teacher_root_cause_lab"

for path in (PROJECT_ROOT, ROOT_CAUSE_DIR, OBJECTIVE_DIR, FAILURE_B_DIR, STABILITY_DIR, TEACHER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import teacher_root_cause_helpers as base_lab  # noqa: E402
import root_cause_ablation_helpers as ab_lab  # noqa: E402
import objective_resolution_helpers as obj_lab  # noqa: E402
import failure_b_resolution_helpers as fb_lab  # noqa: E402
import stability_weighted_helpers as sw_lab  # noqa: E402


save_json = base_lab.save_json
save_dataframe = base_lab.save_dataframe
set_global_seed = base_lab.set_global_seed
choose_device = base_lab.choose_device

CACHE_SCHEMA_VERSION = 1
ROLE_CENTER_ANCHOR = 0
ROLE_CENTER_HARD = 1
ROLE_CENTER_AMBIGUOUS = 2
ROLE_NAME_BY_CODE = {
    ROLE_CENTER_ANCHOR: "center_anchor",
    ROLE_CENTER_HARD: "center_hard",
    ROLE_CENTER_AMBIGUOUS: "center_ambiguous",
}


@dataclass
class ExperimentPaths:
    project_root: str
    run_dir: str
    data_root: str
    experiment_dir: str
    output_dir: str
    reports_dir: str
    checkpoints_dir: str
    cache_dir: str
    plots_dir: str
    objective_output_dir: str
    failure_b_output_dir: str
    failure_b_reports_dir: str


@dataclass
class OracleMineConfig:
    raw_abs_y_edges: Tuple[float, ...] = (0.0, 0.02, 0.05, 0.10)
    pred_abs_pred_edges: Tuple[float, ...] = (0.0, 0.10, 0.30, 0.60, 1.01)
    sample_per_cell: int = 16
    train_num_shards: int = 8
    prediction_cache_batch_size: int = 2048
    selection_seed: int = 123
    stockfish_path: str = r"D:\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
    stockfish_threads: int = 1
    stockfish_hash_mb: int = 32
    stockfish_node_budgets: Tuple[int, ...] = (4_000, 16_000, 64_000)
    stockfish_command_pause_ms: int = 50
    stockfish_timeout_sec: float = 20.0
    stable_target_range_max: float = 0.06
    stable_target_std_max: float = 0.025
    stable_bestmove_changes_max: int = 1
    stable_sign_flips_max: int = 0
    trusted_center_thr: float = 0.05
    aux_oracle_abs_max: float = 0.25
    center_anchor_pred_abs_max: float = 0.20
    min_center_anchor_count: int = 6
    min_center_hard_count: int = 6
    min_center_ambiguous_count: int = 12


@dataclass
class PilotTrainConfig:
    epochs: int = 1
    main_batch_size: int = 384
    anchor_batch_size: int = 24
    hard_batch_size: int = 24
    ambiguous_batch_size: int = 48
    learning_rate: float = 2.0e-6
    min_lr: float = 8.0e-7
    weight_decay: float = 2.0e-4
    grad_clip_norm: float = 1.0
    seed: int = 123
    main_train_num_shards: int = 8
    eval_test_samples: int = 200_000
    eval_test_num_shards: int = 4
    log_every_steps: int = 200
    max_mem_ratio: float = 0.82
    min_batch_size: int = 128
    batch_step: int = 64
    main_center_tau_y600: float = 0.10
    main_center_min_weight: float = 0.35
    main_center_weight_power: float = 1.0
    lambda_anchor: float = 0.20
    lambda_hard: float = 0.15
    lambda_ambiguous: float = 0.10
    aux_margin_y600: float = 0.08
    aux_margin_weight: float = 0.40
    aux_huber_delta: float = 0.05


@dataclass
class MidbandGateConfig:
    midband_mae_rel_tol: float = 0.05
    stable_slope_abs_tol: float = 0.02


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


def _is_strictly_increasing(values: Sequence[float]) -> bool:
    return all(float(values[idx]) < float(values[idx + 1]) for idx in range(len(values) - 1))


def validate_oracle_mine_config(cfg: OracleMineConfig) -> Dict[str, object]:
    issues: List[str] = []
    if len(cfg.raw_abs_y_edges) < 2 or not _is_strictly_increasing(cfg.raw_abs_y_edges):
        issues.append("raw_abs_y_edges must be strictly increasing with at least 2 values")
    if len(cfg.pred_abs_pred_edges) < 2 or not _is_strictly_increasing(cfg.pred_abs_pred_edges):
        issues.append("pred_abs_pred_edges must be strictly increasing with at least 2 values")
    if cfg.sample_per_cell <= 0:
        issues.append("sample_per_cell must be positive")
    if cfg.train_num_shards <= 0:
        issues.append("train_num_shards must be positive")
    if cfg.prediction_cache_batch_size <= 0:
        issues.append("prediction_cache_batch_size must be positive")
    if cfg.stockfish_threads <= 0 or cfg.stockfish_hash_mb <= 0:
        issues.append("stockfish_threads and stockfish_hash_mb must be positive")
    if any(int(x) <= 0 for x in cfg.stockfish_node_budgets):
        issues.append("stockfish_node_budgets must all be positive")
    if cfg.stable_target_range_max <= 0.0 or cfg.stable_target_std_max <= 0.0:
        issues.append("stable target thresholds must be positive")
    if cfg.trusted_center_thr <= 0.0 or cfg.aux_oracle_abs_max <= 0.0:
        issues.append("trusted_center_thr and aux_oracle_abs_max must be positive")
    if cfg.center_anchor_pred_abs_max < 0.0:
        issues.append("center_anchor_pred_abs_max must be non-negative")
    if cfg.aux_oracle_abs_max < cfg.trusted_center_thr:
        issues.append("aux_oracle_abs_max must be >= trusted_center_thr")
    if cfg.min_center_anchor_count <= 0 or cfg.min_center_hard_count <= 0 or cfg.min_center_ambiguous_count <= 0:
        issues.append("all minimum role counts must be positive")
    if issues:
        raise ValueError("Invalid OracleMineConfig: " + "; ".join(issues))
    return {"ok": True}


def validate_pilot_train_config(cfg: PilotTrainConfig) -> Dict[str, object]:
    issues: List[str] = []
    if cfg.epochs <= 0:
        issues.append("epochs must be positive")
    if cfg.main_batch_size <= 0:
        issues.append("main_batch_size must be positive")
    if cfg.anchor_batch_size < 0 or cfg.hard_batch_size < 0 or cfg.ambiguous_batch_size < 0:
        issues.append("aux batch sizes must be non-negative")
    if (cfg.anchor_batch_size + cfg.hard_batch_size + cfg.ambiguous_batch_size) <= 0:
        issues.append("at least one aux batch size must be positive")
    if cfg.learning_rate <= 0.0 or cfg.min_lr <= 0.0:
        issues.append("learning_rate and min_lr must be positive")
    if cfg.weight_decay < 0.0:
        issues.append("weight_decay must be non-negative")
    if cfg.grad_clip_norm is not None and cfg.grad_clip_norm <= 0.0:
        issues.append("grad_clip_norm must be positive when set")
    if cfg.main_train_num_shards <= 0 or cfg.eval_test_num_shards <= 0:
        issues.append("main_train_num_shards and eval_test_num_shards must be positive")
    if cfg.eval_test_samples <= 0:
        issues.append("eval_test_samples must be positive")
    if cfg.max_mem_ratio <= 0.0 or cfg.max_mem_ratio > 1.0:
        issues.append("max_mem_ratio must be in (0, 1]")
    if cfg.min_batch_size <= 0 or cfg.batch_step <= 0:
        issues.append("min_batch_size and batch_step must be positive")
    if cfg.main_center_tau_y600 <= 0.0:
        issues.append("main_center_tau_y600 must be positive")
    if not (0.0 < cfg.main_center_min_weight <= 1.0):
        issues.append("main_center_min_weight must be in (0, 1]")
    if cfg.main_center_weight_power <= 0.0:
        issues.append("main_center_weight_power must be positive")
    if cfg.lambda_anchor < 0.0 or cfg.lambda_hard < 0.0 or cfg.lambda_ambiguous < 0.0:
        issues.append("aux lambda weights must be non-negative")
    if cfg.aux_margin_y600 < 0.0 or cfg.aux_margin_weight < 0.0 or cfg.aux_huber_delta <= 0.0:
        issues.append("aux margin/huber hyperparameters must be valid non-negative values")
    if issues:
        raise ValueError("Invalid PilotTrainConfig: " + "; ".join(issues))
    return {"ok": True}


def validate_gate_config(cfg: MidbandGateConfig) -> Dict[str, object]:
    issues: List[str] = []
    if cfg.midband_mae_rel_tol < 0.0:
        issues.append("midband_mae_rel_tol must be non-negative")
    if cfg.stable_slope_abs_tol < 0.0:
        issues.append("stable_slope_abs_tol must be non-negative")
    if issues:
        raise ValueError("Invalid MidbandGateConfig: " + "; ".join(issues))
    return {"ok": True}


def build_default_paths(
    run_dir: str | Path = r"C:\Users\USER\Downloads\dgrn_5m_v3_stage2_polish_run1",
    data_root: str | Path = r"C:\Users\USER\Desktop\chess_engine\data\process",
    experiment_dir: str | Path | None = None,
) -> Dict[str, Path]:
    if experiment_dir is None:
        experiment_dir = PROJECT_ROOT / "experiments" / "oc2_joint_oracle_full_model_pilot"
    paths = {
        "project_root": PROJECT_ROOT,
        "run_dir": Path(run_dir),
        "data_root": Path(data_root),
        "experiment_dir": Path(experiment_dir),
        "objective_output_dir": OBJECTIVE_DIR / "outputs",
        "failure_b_output_dir": FAILURE_B_DIR / "outputs",
        "failure_b_reports_dir": FAILURE_B_DIR / "outputs" / "reports",
    }
    paths["output_dir"] = paths["experiment_dir"] / "outputs"
    paths["reports_dir"] = paths["output_dir"] / "reports"
    paths["checkpoints_dir"] = paths["output_dir"] / "checkpoints"
    paths["cache_dir"] = paths["output_dir"] / "cache"
    paths["plots_dir"] = paths["output_dir"] / "plots"
    for key in ("experiment_dir", "output_dir", "reports_dir", "checkpoints_dir", "cache_dir", "plots_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def export_paths_json(paths: Dict[str, Path], out_path: Path) -> None:
    payload = ExperimentPaths(**{key: str(value) for key, value in paths.items()})
    out_path.write_text(json.dumps(asdict(payload), indent=2), encoding="utf-8")


def validate_runtime_paths(paths: Dict[str, Path], oracle_cfg: OracleMineConfig) -> Dict[str, object]:
    issues: List[str] = []
    required_files = [
        paths["run_dir"] / "ckpt_best.pt",
        paths["objective_output_dir"] / "runs" / "L0_control_hybrid" / "checkpoints" / "L0_control_hybrid_best.pt",
        ROOT_CAUSE_DIR / "outputs" / "runs" / "A2_band_balanced" / "checkpoints" / "A2_band_balanced_best.pt",
        paths["objective_output_dir"] / "runs" / "L4_A1_plus_A2" / "checkpoints" / "L4_A1_plus_A2_best.pt",
        paths["objective_output_dir"] / "reports" / "full_primary_metrics.csv",
        paths["failure_b_reports_dir"] / "combined_failure_b_primary_metrics.csv",
        Path(oracle_cfg.stockfish_path),
    ]
    required_dirs = [
        paths["data_root"],
        paths["data_root"] / "train",
        paths["data_root"] / "test",
        paths["objective_output_dir"],
        paths["failure_b_output_dir"],
    ]
    for path in required_dirs:
        if not Path(path).exists():
            issues.append(f"Missing directory: {path}")
    for path in required_files:
        if not Path(path).exists():
            issues.append(f"Missing file: {path}")
    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "baseline_checkpoint": str(paths["run_dir"] / "ckpt_best.pt"),
        "a2_checkpoint": str(ROOT_CAUSE_DIR / "outputs" / "runs" / "A2_band_balanced" / "checkpoints" / "A2_band_balanced_best.pt"),
        "l0_checkpoint": str(paths["objective_output_dir"] / "runs" / "L0_control_hybrid" / "checkpoints" / "L0_control_hybrid_best.pt"),
        "l4_checkpoint": str(paths["objective_output_dir"] / "runs" / "L4_A1_plus_A2" / "checkpoints" / "L4_A1_plus_A2_best.pt"),
        "stockfish_path": str(oracle_cfg.stockfish_path),
    }


def build_reference_registry(paths: Dict[str, Path]) -> List[Dict[str, object]]:
    return [
        {
            "label": "baseline",
            "checkpoint": str(paths["run_dir"] / "ckpt_best.pt"),
            "target_scale": 600.0,
        },
        {
            "label": "A2_band_balanced",
            "checkpoint": str(ROOT_CAUSE_DIR / "outputs" / "runs" / "A2_band_balanced" / "checkpoints" / "A2_band_balanced_best.pt"),
            "target_scale": 600.0,
        },
        {
            "label": "L0_control_hybrid",
            "checkpoint": str(paths["objective_output_dir"] / "runs" / "L0_control_hybrid" / "checkpoints" / "L0_control_hybrid_best.pt"),
            "target_scale": 600.0,
        },
        {
            "label": "L4_A1_plus_A2",
            "checkpoint": str(paths["objective_output_dir"] / "runs" / "L4_A1_plus_A2" / "checkpoints" / "L4_A1_plus_A2_best.pt"),
            "target_scale": 600.0,
        },
    ]


def build_reference_context(
    data_root: str | Path,
    paths: Dict[str, Path],
    refresh: bool = False,
) -> Dict[str, object]:
    failure_b_paths = fb_lab.build_default_paths()
    center_cfg = fb_lab.CenterPurityConfig()
    center_pool = fb_lab.build_oracle_center_pool(failure_b_paths, center_cfg, refresh=refresh)
    pooled_center_bundle = fb_lab.build_pooled_center_bundle(
        pooled_unique=center_pool["unique_rows"],
        data_root=data_root,
        center_thr=center_cfg.oracle_center_thr,
        refresh=refresh,
        paths=failure_b_paths,
    )
    oracle_bundle = ab_lab.load_oracle_subset_bundle(ab_lab.OracleEvalConfig(), data_root=data_root)
    eval_cfg = ab_lab.TrainConfig(
        batch_size=576,
        epochs=1,
        learning_rate=3e-6,
        min_lr=1e-6,
        weight_decay=2e-4,
        grad_clip_norm=1.0,
        seed=123,
        log_every_steps=200,
        train_num_shards=None,
        val_max_samples=100_000,
        test_max_samples=200_000,
        val_num_shards=2,
        test_num_shards=4,
        benchmark_num_shards=1,
    )
    return {
        "failure_b_paths": failure_b_paths,
        "center_pool": center_pool,
        "pooled_center_bundle": pooled_center_bundle,
        "oracle_bundle": oracle_bundle,
        "registry": build_reference_registry(paths),
        "eval_cfg": eval_cfg,
    }


def build_l4_variant() -> ab_lab.AblationVariant:
    return obj_lab.build_variant_catalog()["L4_A1_plus_A2"]


def _stockfish_cfg(cfg: OracleMineConfig) -> sw_lab.StabilityWeightConfig:
    return sw_lab.StabilityWeightConfig(
        near_zero_thr=float(cfg.raw_abs_y_edges[-1]),
        calibration_abs_y_sample_edges=cfg.raw_abs_y_edges,
        sample_per_abs_y_band=max(1, int(cfg.sample_per_cell)),
        stockfish_path=str(cfg.stockfish_path),
        stockfish_threads=int(cfg.stockfish_threads),
        stockfish_hash_mb=int(cfg.stockfish_hash_mb),
        stockfish_node_budgets=tuple(int(x) for x in cfg.stockfish_node_budgets),
        stockfish_command_pause_ms=int(cfg.stockfish_command_pause_ms),
        stockfish_timeout_sec=float(cfg.stockfish_timeout_sec),
        calibration_seed=int(cfg.selection_seed),
        prediction_batch_size=int(cfg.prediction_cache_batch_size),
    )


def _candidate_cache_dir(paths: Dict[str, Path]) -> Path:
    out = paths["cache_dir"] / "candidate_bundle_2d"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _oracle_probe_cache_dir(paths: Dict[str, Path]) -> Path:
    out = paths["cache_dir"] / "oracle_probe_2d"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _role_bundle_cache_dir(paths: Dict[str, Path]) -> Path:
    out = paths["cache_dir"] / "oracle_role_bundle"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _candidate_manifest_payload(checkpoint_path: str | Path, cfg: OracleMineConfig) -> Dict[str, object]:
    return {
        "schema_version": int(CACHE_SCHEMA_VERSION),
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "train_num_shards": int(cfg.train_num_shards),
        "raw_abs_y_edges": [float(x) for x in cfg.raw_abs_y_edges],
        "pred_abs_pred_edges": [float(x) for x in cfg.pred_abs_pred_edges],
        "sample_per_cell": int(cfg.sample_per_cell),
    }


def _oracle_audit_manifest_payload(candidate_manifest: Dict[str, object], cfg: OracleMineConfig) -> Dict[str, object]:
    return {
        "schema_version": int(CACHE_SCHEMA_VERSION),
        "candidate_manifest": candidate_manifest,
        "oracle_cfg": asdict(cfg),
    }


def _role_manifest_payload(candidate_manifest: Dict[str, object], oracle_report: Dict[str, object], cfg: OracleMineConfig) -> Dict[str, object]:
    return {
        "schema_version": int(CACHE_SCHEMA_VERSION),
        "candidate_manifest": candidate_manifest,
        "oracle_report": oracle_report,
        "trusted_center_thr": float(cfg.trusted_center_thr),
        "aux_oracle_abs_max": float(cfg.aux_oracle_abs_max),
        "center_anchor_pred_abs_max": float(cfg.center_anchor_pred_abs_max),
    }


def _digitize_band(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(values, edges[1:-1], right=False), 0, len(edges) - 2)


def _manifest_matches(existing: Dict[str, object], expected: Dict[str, object]) -> bool:
    existing_norm = json.loads(json.dumps(existing, sort_keys=True))
    expected_norm = json.loads(json.dumps(expected, sort_keys=True))
    return all(existing_norm.get(key) == value for key, value in expected_norm.items())


def build_2d_candidate_bundle(
    checkpoint_path: str | Path,
    pred_cache_manifest: Dict[str, object],
    data_root: str | Path,
    cfg: OracleMineConfig,
    paths: Dict[str, Path],
    refresh: bool = False,
) -> Dict[str, object]:
    cache_dir = _candidate_cache_dir(paths)
    manifest_path = cache_dir / "manifest.json"
    rows_csv = cache_dir / "candidate_rows.csv"
    npz_path = cache_dir / "candidate_bundle.npz"
    quota_summary_csv = cache_dir / "candidate_quota_summary.csv"
    expected_manifest = _candidate_manifest_payload(checkpoint_path, cfg)
    if not refresh and manifest_path.exists() and rows_csv.exists() and npz_path.exists() and quota_summary_csv.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _manifest_matches(manifest, expected_manifest):
            npz = np.load(npz_path, allow_pickle=False)
            return {
                "rows": pd.read_csv(rows_csv),
                "X": npz["X"].astype(np.uint8, copy=False),
                "manifest": manifest,
                "quota_summary": pd.read_csv(quota_summary_csv),
            }
        print("[oc2-candidates] cache signature mismatch; rebuilding.")

    raw_edges = np.asarray(cfg.raw_abs_y_edges, dtype=np.float64)
    pred_edges = np.asarray(cfg.pred_abs_pred_edges, dtype=np.float64)
    raw_labels = sw_lab.band_labels_from_edges(raw_edges)
    pred_labels = sw_lab.band_labels_from_edges(pred_edges)
    shard_rows = ab_lab.resolve_split_shards(data_root, "train", num_shards=cfg.train_num_shards)
    pred_files = {
        int(row["shard_id"]): Path(paths["cache_dir"]) / "train_pred_cache" / Path(checkpoint_path).stem / row["pred_file"]
        for row in pred_cache_manifest["shards"]
    }

    quota_rows: List[dict] = []
    for raw_idx, raw_label in enumerate(raw_labels):
        for pred_idx, pred_label in enumerate(pred_labels):
            counts: List[int] = []
            shard_ids: List[int] = []
            for shard_id, _, y_path in shard_rows:
                pred_path = pred_files.get(int(shard_id))
                if pred_path is None or not pred_path.exists():
                    raise FileNotFoundError(f"Missing cached prediction shard for shard_id={shard_id}")
                y = np.load(y_path, mmap_mode="r").astype(np.float32, copy=False)
                pred = np.load(pred_path, mmap_mode="r").astype(np.float32, copy=False)
                raw_bins = _digitize_band(np.abs(y.astype(np.float64)), raw_edges)
                pred_bins = _digitize_band(np.abs(pred.astype(np.float64)), pred_edges)
                count = int(np.sum((raw_bins == raw_idx) & (pred_bins == pred_idx)))
                counts.append(count)
                shard_ids.append(int(shard_id))
            quotas = sw_lab.proportional_allocation(counts, int(cfg.sample_per_cell))
            for shard_id, count, quota in zip(shard_ids, counts, quotas):
                quota_rows.append(
                    {
                        "shard_id": int(shard_id),
                        "raw_band_idx": int(raw_idx),
                        "raw_band_label": str(raw_label),
                        "pred_band_idx": int(pred_idx),
                        "pred_band_label": str(pred_label),
                        "count": int(count),
                        "quota": int(quota),
                    }
                )
    quota_df = pd.DataFrame(quota_rows)
    quota_summary = quota_df.groupby(
        ["raw_band_idx", "raw_band_label", "pred_band_idx", "pred_band_label"],
        as_index=False,
    )[["count", "quota"]].sum()
    save_dataframe(quota_summary, quota_summary_csv)

    rng = np.random.default_rng(cfg.selection_seed)
    rows: List[dict] = []
    x_rows: List[np.ndarray] = []
    next_candidate_id = 0
    for shard_id, x_path, y_path in shard_rows:
        shard_quota = quota_df[(quota_df["shard_id"] == int(shard_id)) & (quota_df["quota"] > 0)].copy()
        if shard_quota.empty:
            continue
        X = np.load(x_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r").astype(np.float32, copy=False)
        pred = np.load(pred_files[int(shard_id)], mmap_mode="r").astype(np.float32, copy=False)
        abs_y = np.abs(y.astype(np.float64))
        abs_pred = np.abs(pred.astype(np.float64))
        abs_err = np.abs((pred - y).astype(np.float64))
        raw_bins = _digitize_band(abs_y, raw_edges)
        pred_bins = _digitize_band(abs_pred, pred_edges)
        for _, quota_row in shard_quota.iterrows():
            raw_idx = int(quota_row["raw_band_idx"])
            pred_idx = int(quota_row["pred_band_idx"])
            quota = int(quota_row["quota"])
            candidate_idx = np.flatnonzero((raw_bins == raw_idx) & (pred_bins == pred_idx))
            if candidate_idx.size == 0 or quota <= 0:
                continue
            take = min(quota, int(candidate_idx.size))
            chosen = rng.choice(candidate_idx, size=take, replace=False)
            for local_index in np.sort(chosen.astype(np.int64)):
                rows.append(
                    {
                        "candidate_id": int(next_candidate_id),
                        "shard_id": int(shard_id),
                        "local_index": int(local_index),
                        "raw_band_idx": int(raw_idx),
                        "raw_band_label": str(quota_row["raw_band_label"]),
                        "pred_band_idx": int(pred_idx),
                        "pred_band_label": str(quota_row["pred_band_label"]),
                        "raw_target_y": float(y[local_index]),
                        "raw_abs_y": float(abs_y[local_index]),
                        "init_pred": float(pred[local_index]),
                        "init_abs_pred": float(abs_pred[local_index]),
                        "init_abs_err": float(abs_err[local_index]),
                    }
                )
                x_rows.append(np.array(X[local_index], dtype=np.uint8, copy=True))
                next_candidate_id += 1

    rows_df = pd.DataFrame(rows)
    if rows_df.empty:
        raise RuntimeError("2D candidate bundle is empty. Check stratification edges and shard selection.")
    X_bundle = np.stack(x_rows, axis=0) if x_rows else np.empty((0, 18, 8, 8), dtype=np.uint8)
    rows_df["_tensor_pos"] = np.arange(rows_df.shape[0], dtype=np.int64)
    rows_df = rows_df.sort_values(
        ["raw_band_idx", "pred_band_idx", "init_abs_pred"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    tensor_order = rows_df["_tensor_pos"].to_numpy(dtype=np.int64, copy=False)
    X_bundle = X_bundle[tensor_order]
    rows_df = rows_df.drop(columns="_tensor_pos")
    manifest = dict(expected_manifest)
    manifest["num_candidates"] = int(rows_df.shape[0])
    save_dataframe(rows_df, rows_csv)
    np.savez_compressed(npz_path, X=X_bundle.astype(np.uint8))
    save_json(manifest, manifest_path)
    return {"rows": rows_df, "X": X_bundle, "manifest": manifest, "quota_summary": quota_summary}


def run_oracle_candidate_audit(
    candidate_bundle: Dict[str, object],
    cfg: OracleMineConfig,
    paths: Dict[str, Path],
    refresh: bool = False,
) -> Dict[str, object]:
    cache_dir = _oracle_probe_cache_dir(paths)
    manifest_path = cache_dir / "manifest.json"
    rows_csv = cache_dir / "oracle_candidate_rows.csv"
    summary_csv = cache_dir / "oracle_candidate_summary.csv"
    summary_json = cache_dir / "oracle_candidate_report.json"
    expected_manifest = _oracle_audit_manifest_payload(candidate_bundle["manifest"], cfg)
    if not refresh and manifest_path.exists() and rows_csv.exists() and summary_json.exists() and summary_csv.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _manifest_matches(manifest, expected_manifest):
            return {
                "rows": pd.read_csv(rows_csv),
                "summary": pd.read_csv(summary_csv),
                "report": json.loads(summary_json.read_text(encoding="utf-8")),
            }
        print("[oc2-oracle-audit] cache signature mismatch; rebuilding.")

    rows = candidate_bundle["rows"].copy().reset_index(drop=True)
    X = np.asarray(candidate_bundle["X"], dtype=np.uint8)
    if rows.shape[0] != X.shape[0]:
        raise RuntimeError("Candidate rows and tensors are misaligned.")

    stockfish_cfg = _stockfish_cfg(cfg)
    out_rows: List[dict] = []
    for pos, (_, row) in enumerate(rows.iterrows()):
        board = sw_lab.sanitize_board_for_stockfish(base_lab.decode_tensor_to_board(X[pos]))
        curve = sw_lab.run_stockfish_probe_curve(board.to_fen(), stockfish_cfg)
        oracle_final_y = float(curve["target_values"][-1])
        oracle_target_range = float(curve["target_range"])
        oracle_target_std = float(curve["target_std"])
        bestmove_changes = int(curve["bestmove_changes"])
        sign_flips = int(curve["sign_flips"])
        is_stable = bool(
            oracle_target_range <= float(cfg.stable_target_range_max)
            and oracle_target_std <= float(cfg.stable_target_std_max)
            and bestmove_changes <= int(cfg.stable_bestmove_changes_max)
            and sign_flips <= int(cfg.stable_sign_flips_max)
        )
        oracle_abs_y = float(abs(oracle_final_y))
        role_keep = bool(is_stable and oracle_abs_y <= float(cfg.aux_oracle_abs_max))
        row_out = row.to_dict()
        row_out.update(
            {
                "fen": board.to_fen(),
                "oracle_final_y": oracle_final_y,
                "oracle_abs_y": oracle_abs_y,
                "oracle_target_range": oracle_target_range,
                "oracle_target_std": oracle_target_std,
                "oracle_cp_range": float(curve["cp_range"]),
                "oracle_bestmove_changes": bestmove_changes,
                "oracle_sign_flips": sign_flips,
                "oracle_bestmove_final": str(curve["bestmoves"][-1]),
                "is_stable": is_stable,
                "is_center_clean": bool(is_stable and oracle_abs_y <= float(cfg.trusted_center_thr)),
                "aux_keep": role_keep,
            }
        )
        for curve_row in curve["rows"]:
            node_budget = int(curve_row["node_budget"])
            row_out[f"oracle_y_n{node_budget}"] = float(curve_row["target_value"])
            row_out[f"oracle_cp_n{node_budget}"] = float(curve_row["cp_equivalent"])
            row_out[f"oracle_bestmove_n{node_budget}"] = str(curve_row["bestmove"])
        out_rows.append(row_out)
        if pos == 0 or ((pos + 1) % 16 == 0):
            print(f"[oc2-oracle-audit] processed={pos + 1}/{rows.shape[0]}")

    out_df = pd.DataFrame(out_rows)
    summary_df = out_df.groupby(
        ["raw_band_idx", "raw_band_label", "pred_band_idx", "pred_band_label"],
        as_index=False,
    ).agg(
        n=("candidate_id", "size"),
        stable_count=("is_stable", "sum"),
        center_clean_count=("is_center_clean", "sum"),
        aux_keep_count=("aux_keep", "sum"),
        mean_init_abs_pred=("init_abs_pred", "mean"),
        mean_oracle_abs_y=("oracle_abs_y", "mean"),
        mean_oracle_target_range=("oracle_target_range", "mean"),
    )
    report = {
        "num_candidates": int(out_df.shape[0]),
        "stable_count": int(out_df["is_stable"].sum()),
        "center_clean_count": int(out_df["is_center_clean"].sum()),
        "aux_keep_count": int(out_df["aux_keep"].sum()),
        "stable_rate": float(out_df["is_stable"].mean()) if out_df.shape[0] else 0.0,
        "center_clean_rate": float(out_df["is_center_clean"].mean()) if out_df.shape[0] else 0.0,
        "aux_keep_rate": float(out_df["aux_keep"].mean()) if out_df.shape[0] else 0.0,
        "trusted_center_thr": float(cfg.trusted_center_thr),
        "aux_oracle_abs_max": float(cfg.aux_oracle_abs_max),
    }
    save_dataframe(out_df, rows_csv)
    save_dataframe(summary_df, summary_csv)
    save_json(report, summary_json)
    save_json(expected_manifest, manifest_path)
    return {"rows": out_df, "summary": summary_df, "report": report}


def build_role_bundle(
    candidate_bundle: Dict[str, object],
    oracle_audit: Dict[str, object],
    cfg: OracleMineConfig,
    paths: Dict[str, Path],
    refresh: bool = False,
) -> Dict[str, object]:
    cache_dir = _role_bundle_cache_dir(paths)
    manifest_path = cache_dir / "manifest.json"
    rows_csv = cache_dir / "oracle_role_rows.csv"
    npz_path = cache_dir / "oracle_role_bundle.npz"
    summary_csv = cache_dir / "oracle_role_summary.csv"
    expected_manifest = _role_manifest_payload(candidate_bundle["manifest"], oracle_audit["report"], cfg)
    if not refresh and manifest_path.exists() and rows_csv.exists() and npz_path.exists() and summary_csv.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _manifest_matches(manifest, expected_manifest):
            npz = np.load(npz_path, allow_pickle=False)
            rows = pd.read_csv(rows_csv)
            _validate_role_counts(rows, cfg)
            bundle = {
                "rows": rows,
                "X": npz["X"].astype(np.uint8, copy=False),
                "oracle_y": npz["oracle_y"].astype(np.float32, copy=False),
                "role_code": npz["role_code"].astype(np.int64, copy=False),
                "manifest": manifest,
                "summary": pd.read_csv(summary_csv),
            }
            return _attach_role_indices(bundle)
        print("[oc2-role-bundle] cache signature mismatch; rebuilding.")

    rows = oracle_audit["rows"].copy().reset_index(drop=True)
    X = np.asarray(candidate_bundle["X"], dtype=np.uint8)
    keep = rows["aux_keep"].astype(bool).to_numpy()
    kept_rows = rows.loc[keep].copy().reset_index(drop=True)
    kept_X = X[keep]
    if kept_rows.empty:
        raise RuntimeError("Role bundle is empty. Oracle audit did not retain any stable rows.")

    oracle_abs = kept_rows["oracle_abs_y"].to_numpy(dtype=np.float64)
    init_abs_pred = kept_rows["init_abs_pred"].to_numpy(dtype=np.float64)
    center_clean = oracle_abs <= float(cfg.trusted_center_thr)
    anchor_mask = center_clean & (init_abs_pred <= float(cfg.center_anchor_pred_abs_max))
    hard_mask = center_clean & ~anchor_mask
    ambiguous_mask = (~center_clean) & (oracle_abs <= float(cfg.aux_oracle_abs_max))
    role_code = np.full(kept_rows.shape[0], -1, dtype=np.int64)
    role_code[anchor_mask] = ROLE_CENTER_ANCHOR
    role_code[hard_mask] = ROLE_CENTER_HARD
    role_code[ambiguous_mask] = ROLE_CENTER_AMBIGUOUS
    valid = role_code >= 0
    if not np.any(valid):
        raise RuntimeError("Role bundle has no valid rows after role assignment.")
    kept_rows = kept_rows.loc[valid].copy().reset_index(drop=True)
    kept_X = kept_X[valid]
    role_code = role_code[valid]
    kept_rows["role_code"] = role_code.astype(np.int64)
    kept_rows["role_name"] = [ROLE_NAME_BY_CODE[int(code)] for code in role_code]
    kept_rows["oracle_role_target_y"] = kept_rows["oracle_final_y"].astype(np.float32)

    role_counts = _validate_role_counts(kept_rows, cfg)

    summary = kept_rows.groupby(
        ["role_code", "role_name", "raw_band_label", "pred_band_label"],
        as_index=False,
    ).agg(
        n=("candidate_id", "size"),
        mean_init_abs_pred=("init_abs_pred", "mean"),
        mean_oracle_abs_y=("oracle_abs_y", "mean"),
    )
    manifest = dict(expected_manifest)
    manifest.update(
        {
            "num_rows": int(kept_rows.shape[0]),
            "center_anchor_count": int(role_counts["center_anchor"]),
            "center_hard_count": int(role_counts["center_hard"]),
            "center_ambiguous_count": int(role_counts["center_ambiguous"]),
        }
    )
    save_dataframe(kept_rows, rows_csv)
    save_dataframe(summary, summary_csv)
    np.savez_compressed(
        npz_path,
        X=kept_X.astype(np.uint8),
        oracle_y=kept_rows["oracle_role_target_y"].to_numpy(dtype=np.float32),
        role_code=role_code.astype(np.int64),
    )
    save_json(manifest, manifest_path)
    bundle = {
        "rows": kept_rows,
        "X": kept_X.astype(np.uint8, copy=False),
        "oracle_y": kept_rows["oracle_role_target_y"].to_numpy(dtype=np.float32),
        "role_code": role_code.astype(np.int64),
        "manifest": manifest,
        "summary": summary,
    }
    return _attach_role_indices(bundle)


def _attach_role_indices(bundle: Dict[str, object]) -> Dict[str, object]:
    role_code = np.asarray(bundle["role_code"], dtype=np.int64)
    bundle["indices_by_role"] = {
        role_name: np.flatnonzero(role_code == code).astype(np.int64)
        for code, role_name in ROLE_NAME_BY_CODE.items()
    }
    return bundle


def _validate_role_counts(rows: pd.DataFrame, cfg: OracleMineConfig) -> Dict[str, int]:
    role_counts = rows["role_name"].value_counts().to_dict()
    anchor_count = int(role_counts.get("center_anchor", 0))
    hard_count = int(role_counts.get("center_hard", 0))
    ambiguous_count = int(role_counts.get("center_ambiguous", 0))
    if anchor_count < int(cfg.min_center_anchor_count):
        raise RuntimeError(f"center_anchor_count too small: {anchor_count}")
    if hard_count < int(cfg.min_center_hard_count):
        raise RuntimeError(f"center_hard_count too small: {hard_count}")
    if ambiguous_count < int(cfg.min_center_ambiguous_count):
        raise RuntimeError(f"center_ambiguous_count too small: {ambiguous_count}")
    return {
        "center_anchor": anchor_count,
        "center_hard": hard_count,
        "center_ambiguous": ambiguous_count,
    }


def configure_full_model_trainable(model: nn.Module) -> pd.DataFrame:
    for param in model.parameters():
        param.requires_grad = True
    rows = []
    modules = [("stem", model.stem)] + [(f"blocks.{idx}", model.blocks[idx]) for idx in range(len(model.blocks))] + [("head", model.head)]
    for name, module in modules:
        total = sum(param.numel() for param in module.parameters())
        trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
        rows.append(
            {
                "module": name,
                "total_params": int(total),
                "trainable_params": int(trainable),
                "trainable": bool(trainable > 0),
            }
        )
    summary = pd.DataFrame(rows)
    return summary


def _main_center_weights(y_source: torch.Tensor, cfg: PilotTrainConfig) -> torch.Tensor:
    tau = max(float(cfg.main_center_tau_y600), 1e-6)
    ratio = torch.clamp(torch.abs(y_source).float() / tau, 0.0, 1.0)
    smooth = torch.pow(ratio, float(cfg.main_center_weight_power))
    return float(cfg.main_center_min_weight) + (1.0 - float(cfg.main_center_min_weight)) * smooth


def compute_l4_main_terms(
    logits: torch.Tensor,
    y_source: torch.Tensor,
    variant: ab_lab.AblationVariant,
    cfg: PilotTrainConfig,
) -> Dict[str, torch.Tensor]:
    y = ab_lab.remap_target_torch(y_source.view(-1), to_scale=variant.target_scale)
    pred = torch.tanh(logits.view(-1))
    mse_per = (pred - y) ** 2
    y_logits = ab_lab.target_to_logits(y, eps=variant.target_clamp_eps)
    residual = logits.view(-1) - y_logits
    y_clamped = torch.clamp(y, -1.0 + variant.target_clamp_eps, 1.0 - variant.target_clamp_eps)
    z_weight = torch.pow(torch.clamp(1.0 - y_clamped * y_clamped, min=variant.target_clamp_eps), variant.z_loss_beta)
    z_huber_per = z_weight * ab_lab.huber_per_sample(residual, variant.z_huber_delta)
    y_curv = torch.square(torch.clamp(1.0 - y_clamped * y_clamped, min=variant.target_clamp_eps))
    y_comp = torch.clamp(1.0 / (y_curv + variant.y_reweight_eps), min=1.0, max=variant.y_reweight_clip_max)
    y_comp = y_comp / torch.clamp(y_comp.mean(), min=1e-6)
    y_term_per = y_comp * mse_per
    main_per = variant.y_loss_alpha * y_term_per + (1.0 - variant.y_loss_alpha) * z_huber_per
    sample_weight = _main_center_weights(y_source.view(-1), cfg)
    objective = torch.sum(sample_weight * main_per) / torch.clamp(sample_weight.sum(), min=1e-6)
    return {
        "objective": objective,
        "main_term": torch.mean(main_per),
        "pred": pred,
        "mean_main_weight": torch.mean(sample_weight),
        "downweighted_frac": torch.mean((sample_weight < 0.999).float()),
    }


def _huber_mean(residual: torch.Tensor, delta: float) -> torch.Tensor:
    delta = float(delta)
    abs_residual = residual.abs()
    value = torch.where(abs_residual <= delta, 0.5 * residual * residual, delta * (abs_residual - 0.5 * delta))
    return torch.mean(value)


def sample_aux_batch_indices(role_bundle: Dict[str, object], cfg: PilotTrainConfig, rng: np.random.Generator) -> np.ndarray:
    role_sizes = {
        "center_anchor": int(cfg.anchor_batch_size),
        "center_hard": int(cfg.hard_batch_size),
        "center_ambiguous": int(cfg.ambiguous_batch_size),
    }
    parts: List[np.ndarray] = []
    for role_name, size in role_sizes.items():
        if size <= 0:
            continue
        pool = np.asarray(role_bundle["indices_by_role"][role_name], dtype=np.int64)
        if pool.size == 0:
            raise RuntimeError(f"Role pool is empty for role={role_name}")
        replace = pool.size < size
        parts.append(rng.choice(pool, size=size, replace=replace).astype(np.int64))
    if not parts:
        raise RuntimeError("Aux role sampling produced no indices.")
    out = np.concatenate(parts, axis=0).astype(np.int64)
    rng.shuffle(out)
    return out


def compute_oc2_aux_terms(
    logits: torch.Tensor,
    oracle_y: torch.Tensor,
    role_code: torch.Tensor,
    cfg: PilotTrainConfig,
) -> Dict[str, torch.Tensor]:
    pred = torch.tanh(logits.view(-1))
    oracle_y = oracle_y.view(-1)
    role_code = role_code.view(-1).long()

    anchor_mask = role_code == ROLE_CENTER_ANCHOR
    hard_mask = role_code == ROLE_CENTER_HARD
    ambiguous_mask = role_code == ROLE_CENTER_AMBIGUOUS
    center_mask = anchor_mask | hard_mask

    anchor_loss = _huber_mean(pred[anchor_mask] - oracle_y[anchor_mask], cfg.aux_huber_delta) if torch.any(anchor_mask) else pred.new_tensor(0.0)
    hard_loss = _huber_mean(pred[hard_mask] - oracle_y[hard_mask], cfg.aux_huber_delta) if torch.any(hard_mask) else pred.new_tensor(0.0)
    ambiguous_loss = _huber_mean(pred[ambiguous_mask] - oracle_y[ambiguous_mask], cfg.aux_huber_delta) if torch.any(ambiguous_mask) else pred.new_tensor(0.0)
    margin_penalty = torch.mean(torch.relu(torch.abs(pred[center_mask]) - float(cfg.aux_margin_y600)) ** 2) if torch.any(center_mask) else pred.new_tensor(0.0)

    objective = (
        float(cfg.lambda_anchor) * anchor_loss
        + float(cfg.lambda_hard) * hard_loss
        + float(cfg.lambda_ambiguous) * ambiguous_loss
        + float(cfg.aux_margin_weight) * margin_penalty
    )
    return {
        "objective": objective,
        "anchor_loss": anchor_loss,
        "hard_loss": hard_loss,
        "ambiguous_loss": ambiguous_loss,
        "margin_penalty": margin_penalty,
        "pred": pred,
        "anchor_frac": torch.mean(anchor_mask.float()),
        "hard_frac": torch.mean(hard_mask.float()),
        "ambiguous_frac": torch.mean(ambiguous_mask.float()),
    }


def center_only_score(pooled_center_eval: Dict[str, object]) -> float:
    return float(
        float(pooled_center_eval["mae_vs_oracle"])
        + 0.30 * float(pooled_center_eval["false_decisive_0.1"])
        + 0.20 * float(pooled_center_eval["false_decisive_0.2"])
        + 0.10 * max(0.0, float(pooled_center_eval["amp_ratio"]) - 2.5)
    )


def legacy_failure_b_score(primary: Dict[str, object], pooled_center_eval: Dict[str, object]) -> float:
    return float(
        center_only_score(pooled_center_eval)
        + 0.30 * float(primary["oracle_midband_mae_sum_stable"])
        + 0.20 * max(0.0, 0.80 - float(primary["oracle_stable_0.7_slope"]))
    )


def passes_midband_gate(primary: Dict[str, object], l4_primary: Dict[str, object], gate_cfg: MidbandGateConfig) -> bool:
    midband_limit = float(l4_primary["oracle_midband_mae_sum_stable"]) * (1.0 + float(gate_cfg.midband_mae_rel_tol))
    slope_limit = float(l4_primary["oracle_stable_0.7_slope"]) - float(gate_cfg.stable_slope_abs_tol)
    return bool(
        float(primary["oracle_midband_mae_sum_stable"]) <= midband_limit
        and float(primary["oracle_stable_0.7_slope"]) >= slope_limit
    )


def evaluate_l4_reference(
    checkpoint_path: str | Path,
    data_root: str | Path,
    pooled_center_bundle: Dict[str, object],
    oracle_bundle: Dict[str, object],
    eval_cfg: ab_lab.TrainConfig,
    paths: Dict[str, Path],
    device: torch.device,
    prefix: str = "l4_reference",
) -> Dict[str, object]:
    result = obj_lab.evaluate_checkpoint(
        ckpt_path=checkpoint_path,
        label="L4_A1_plus_A2",
        data_root=data_root,
        oracle_bundle=oracle_bundle,
        oracle_cfg=ab_lab.OracleEvalConfig(),
        train_cfg=eval_cfg,
        device=device,
        target_scale=600.0,
    )
    center_eval = fb_lab.evaluate_on_pooled_center_bundle(
        checkpoint_path=checkpoint_path,
        label="L4_A1_plus_A2",
        pooled_center_bundle=pooled_center_bundle,
        device=device,
    )
    payload = {
        "primary": result["primary"],
        "center_eval": center_eval,
        "center_score": float(center_only_score(center_eval)),
        "legacy_failure_b_score": float(legacy_failure_b_score(result["primary"], center_eval)),
    }
    save_json(payload, paths["reports_dir"] / f"{prefix}.json")
    return payload


def benchmark_joint_train_step(
    init_ckpt_path: str | Path,
    data_root: str | Path,
    role_bundle: Dict[str, object],
    pilot_cfg: PilotTrainConfig,
    device: torch.device,
    batch_size: int,
    num_shards: int = 1,
) -> Dict[str, float]:
    variant = build_l4_variant()
    model, _ = base_lab.load_model_from_checkpoint(init_ckpt_path, device=device)
    model.train()
    configure_full_model_trainable(model)
    optimizer = ab_lab.build_optimizer(model, lr=pilot_cfg.learning_rate, weight_decay=pilot_cfg.weight_decay)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    rng = np.random.default_rng(pilot_cfg.seed)

    shard_rows = ab_lab.resolve_split_shards(data_root, "train", num_shards=num_shards)
    _, x_path, y_path = shard_rows[0]
    X = np.load(x_path, mmap_mode="r")
    y_source = np.load(y_path, mmap_mode="r").astype(np.float32, copy=False)
    main_take = min(int(batch_size), int(X.shape[0]))
    main_idx = np.arange(main_take, dtype=np.int64)
    aux_idx = sample_aux_batch_indices(role_bundle, pilot_cfg, rng)

    xb = torch.from_numpy(np.array(X[main_idx], dtype=np.float32, copy=True)).to(device, non_blocking=True)
    yb = torch.from_numpy(np.array(y_source[main_idx], dtype=np.float32, copy=True)).to(device, non_blocking=True)
    aux_x = torch.from_numpy(np.array(role_bundle["X"][aux_idx], dtype=np.float32, copy=True)).to(device, non_blocking=True)
    aux_y = torch.from_numpy(np.array(role_bundle["oracle_y"][aux_idx], dtype=np.float32, copy=True)).to(device, non_blocking=True)
    aux_role = torch.from_numpy(np.array(role_bundle["role_code"][aux_idx], dtype=np.int64, copy=True)).to(device, non_blocking=True)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        main_logits = model.forward_logits(xb).view(-1)
        main_terms = compute_l4_main_terms(main_logits, yb, variant, pilot_cfg)
    scaler.scale(main_terms["objective"]).backward()
    with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        aux_logits = model.forward_logits(aux_x).view(-1)
        aux_terms = compute_oc2_aux_terms(aux_logits, aux_y, aux_role, pilot_cfg)
    scaler.scale(aux_terms["objective"]).backward()
    if pilot_cfg.grad_clip_norm is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), pilot_cfg.grad_clip_norm)
    scaler.step(optimizer)
    scaler.update()
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_mem_gb = float(torch.cuda.max_memory_allocated(device) / 1024**3)
    else:
        peak_mem_gb = float("nan")
    step_time_sec = float(time.time() - t0)
    total_main_samples = int(sum(int(np.load(item[2], mmap_mode="r").shape[0]) for item in ab_lab.resolve_split_shards(data_root, "train", num_shards=pilot_cfg.main_train_num_shards)))
    steps_per_epoch = int(math.ceil(total_main_samples / max(int(batch_size), 1)))
    report = {
        "batch_size": int(batch_size),
        "peak_mem_gb": float(peak_mem_gb),
        "step_time_sec": float(step_time_sec),
        "epoch_hours_estimate": float(step_time_sec * steps_per_epoch / 3600.0),
    }
    del model, optimizer, scaler, xb, yb, aux_x, aux_y, aux_role
    _cleanup_cuda()
    return report


def autotune_joint_batch_size(
    init_ckpt_path: str | Path,
    data_root: str | Path,
    role_bundle: Dict[str, object],
    pilot_cfg: PilotTrainConfig,
    device: torch.device,
    preferred_batch_size: int,
    min_batch_size: int = 128,
    step: int = 64,
    max_mem_ratio: float = 0.82,
) -> Dict[str, object]:
    candidates: List[int] = []
    current = int(preferred_batch_size)
    while current >= int(min_batch_size):
        candidates.append(int(current))
        current -= int(step)
    if not candidates or candidates[-1] != int(min_batch_size):
        candidates.append(int(min_batch_size))
    candidates = sorted(set(candidates), reverse=True)

    attempts: List[dict] = []
    total_mem_gb = None
    if device.type == "cuda":
        total_mem_gb = float(torch.cuda.get_device_properties(device).total_memory / 1024**3)

    selected_batch = None
    selected_report: Optional[Dict[str, float]] = None
    for batch_size in candidates:
        _cleanup_cuda()
        try:
            report = benchmark_joint_train_step(
                init_ckpt_path=init_ckpt_path,
                data_root=data_root,
                role_bundle=role_bundle,
                pilot_cfg=pilot_cfg,
                device=device,
                batch_size=batch_size,
                num_shards=1,
            )
            mem_ratio = None if total_mem_gb is None else float(report["peak_mem_gb"] / total_mem_gb)
            attempts.append(
                {
                    "batch_size": int(batch_size),
                    "ok": True,
                    "peak_mem_gb": float(report["peak_mem_gb"]),
                    "mem_ratio": mem_ratio,
                    "epoch_hours_estimate": float(report["epoch_hours_estimate"]),
                }
            )
            if mem_ratio is None or mem_ratio <= float(max_mem_ratio):
                selected_batch = int(batch_size)
                selected_report = report
                break
        except RuntimeError as exc:
            if not _is_cuda_oom(exc):
                raise
            attempts.append({"batch_size": int(batch_size), "ok": False, "error": str(exc)})
            _cleanup_cuda()
    if selected_batch is None or selected_report is None:
        raise RuntimeError("Unable to find a safe joint train batch size: " + json.dumps(attempts, ensure_ascii=False))
    return {
        "selected_batch_size": int(selected_batch),
        "preferred_batch_size": int(preferred_batch_size),
        "min_batch_size": int(min_batch_size),
        "step": int(step),
        "max_mem_ratio": float(max_mem_ratio),
        "device": str(device),
        "total_mem_gb": total_mem_gb,
        "selected_report": selected_report,
        "attempts": attempts,
    }


def _extract_checkpoint_payload(ckpt: Dict[str, object]) -> Dict[str, object]:
    return {
        "config": ckpt.get("config"),
    }


def _clear_previous_pilot_outputs(paths: Dict[str, Path]) -> None:
    stale_files = [
        paths["checkpoints_dir"] / "OC2_joint_oracle_full_model_latest.pt",
        paths["checkpoints_dir"] / "OC2_joint_oracle_full_model_best_any_center.pt",
        paths["checkpoints_dir"] / "OC2_joint_oracle_full_model_best_gate.pt",
        paths["reports_dir"] / "best_any_primary_metrics.json",
        paths["reports_dir"] / "best_any_pooled_center_eval.json",
        paths["reports_dir"] / "best_gate_primary_metrics.json",
        paths["reports_dir"] / "best_gate_pooled_center_eval.json",
        paths["reports_dir"] / "oc2_pilot_history.csv",
        paths["reports_dir"] / "oc2_pilot_history.json",
        paths["reports_dir"] / "decision_summary.json",
    ]
    for path in stale_files:
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass


def run_oc2_joint_oracle_full_model_pilot(
    init_ckpt_path: str | Path,
    data_root: str | Path,
    pilot_cfg: PilotTrainConfig,
    gate_cfg: MidbandGateConfig,
    role_bundle: Dict[str, object],
    oracle_bundle: Dict[str, object],
    pooled_center_bundle: Dict[str, object],
    l4_reference: Dict[str, object],
    paths: Dict[str, Path],
    device: torch.device,
) -> Dict[str, object]:
    checkpoint_dir = paths["checkpoints_dir"]
    reports_dir = paths["reports_dir"]
    latest_ckpt = checkpoint_dir / "OC2_joint_oracle_full_model_latest.pt"
    best_any_ckpt = checkpoint_dir / "OC2_joint_oracle_full_model_best_any_center.pt"
    best_gate_ckpt = checkpoint_dir / "OC2_joint_oracle_full_model_best_gate.pt"
    history_csv = reports_dir / "oc2_pilot_history.csv"
    history_json = reports_dir / "oc2_pilot_history.json"

    set_global_seed(pilot_cfg.seed)
    _clear_previous_pilot_outputs(paths)
    variant = build_l4_variant()
    model, init_ckpt = base_lab.load_model_from_checkpoint(init_ckpt_path, device=device)
    trainable_scope = configure_full_model_trainable(model)
    save_dataframe(trainable_scope, reports_dir / "trainable_scope.csv")
    optimizer = ab_lab.build_optimizer(model, lr=pilot_cfg.learning_rate, weight_decay=pilot_cfg.weight_decay)
    scheduler = None
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    rng = np.random.default_rng(pilot_cfg.seed)

    shard_rows = ab_lab.resolve_split_shards(data_root, "train", num_shards=pilot_cfg.main_train_num_shards)
    total_main_samples = int(sum(int(np.load(y_path, mmap_mode="r").shape[0]) for _, _, y_path in shard_rows))
    steps_per_epoch = int(math.ceil(total_main_samples / max(int(pilot_cfg.main_batch_size), 1)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(steps_per_epoch * max(pilot_cfg.epochs, 1)), 1),
        eta_min=float(pilot_cfg.min_lr),
    )

    best_any_score = float("inf")
    best_gate_score = float("inf")
    history_rows: List[dict] = []
    global_step = 0
    for epoch in range(int(pilot_cfg.epochs)):
        model.train()
        epoch_start = time.time()
        running = {
            "main_objective": 0.0,
            "main_term": 0.0,
            "mean_main_weight": 0.0,
            "downweighted_frac": 0.0,
            "main_n": 0,
            "aux_objective": 0.0,
            "aux_anchor_loss": 0.0,
            "aux_hard_loss": 0.0,
            "aux_ambiguous_loss": 0.0,
            "aux_margin": 0.0,
            "aux_anchor_frac": 0.0,
            "aux_hard_frac": 0.0,
            "aux_ambiguous_frac": 0.0,
            "aux_steps": 0,
        }

        for shard_rank, (_, x_path, y_path) in enumerate(shard_rows, start=1):
            X = np.load(x_path, mmap_mode="r")
            y_source = np.load(y_path, mmap_mode="r").astype(np.float32, copy=False)
            y_scaled = ab_lab.remap_target_np(y_source, to_scale=variant.target_scale).astype(np.float32, copy=False)
            order = ab_lab.build_band_balanced_order(
                abs_y=np.abs(y_scaled.astype(np.float64)),
                batch_size=pilot_cfg.main_batch_size,
                band_edges_y600=variant.balance_band_edges_y600,
                rng=rng,
                target_scale=variant.target_scale,
            )
            for start in range(0, y_source.shape[0], pilot_cfg.main_batch_size):
                idx = order[start : start + pilot_cfg.main_batch_size]
                xb = torch.from_numpy(np.array(X[idx], dtype=np.float32, copy=True)).to(device, non_blocking=True)
                yb = torch.from_numpy(np.array(y_source[idx], dtype=np.float32, copy=True)).to(device, non_blocking=True).view(-1)
                aux_idx = sample_aux_batch_indices(role_bundle, pilot_cfg, rng)
                aux_x = torch.from_numpy(np.array(role_bundle["X"][aux_idx], dtype=np.float32, copy=True)).to(device, non_blocking=True)
                aux_y = torch.from_numpy(np.array(role_bundle["oracle_y"][aux_idx], dtype=np.float32, copy=True)).to(device, non_blocking=True).view(-1)
                aux_role = torch.from_numpy(np.array(role_bundle["role_code"][aux_idx], dtype=np.int64, copy=True)).to(device, non_blocking=True).view(-1)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    main_logits = model.forward_logits(xb).view(-1)
                    main_terms = compute_l4_main_terms(main_logits, yb, variant, pilot_cfg)
                scaler.scale(main_terms["objective"]).backward()
                with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    aux_logits = model.forward_logits(aux_x).view(-1)
                    aux_terms = compute_oc2_aux_terms(aux_logits, aux_y, aux_role, pilot_cfg)
                scaler.scale(aux_terms["objective"]).backward()
                if pilot_cfg.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), pilot_cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                bs = int(yb.numel())
                running["main_objective"] += float(main_terms["objective"].item()) * bs
                running["main_term"] += float(main_terms["main_term"].item()) * bs
                running["mean_main_weight"] += float(main_terms["mean_main_weight"].item()) * bs
                running["downweighted_frac"] += float(main_terms["downweighted_frac"].item()) * bs
                running["main_n"] += bs
                running["aux_objective"] += float(aux_terms["objective"].item())
                running["aux_anchor_loss"] += float(aux_terms["anchor_loss"].item())
                running["aux_hard_loss"] += float(aux_terms["hard_loss"].item())
                running["aux_ambiguous_loss"] += float(aux_terms["ambiguous_loss"].item())
                running["aux_margin"] += float(aux_terms["margin_penalty"].item())
                running["aux_anchor_frac"] += float(aux_terms["anchor_frac"].item())
                running["aux_hard_frac"] += float(aux_terms["hard_frac"].item())
                running["aux_ambiguous_frac"] += float(aux_terms["ambiguous_frac"].item())
                running["aux_steps"] += 1
                global_step += 1

                if global_step % int(pilot_cfg.log_every_steps) == 0:
                    print(
                        f"[oc2-pilot] step={global_step}/{steps_per_epoch * max(int(pilot_cfg.epochs), 1)} "
                        f"main_obj={running['main_objective'] / max(running['main_n'], 1):.6f} "
                        f"aux_obj={running['aux_objective'] / max(running['aux_steps'], 1):.6f}"
                    )
            print(f"[oc2-pilot] finished shard {shard_rank}/{len(shard_rows)}")

        model.eval()
        test_eval = ab_lab.evaluate_model_on_split_scale_aware(
            model=model,
            data_root=data_root,
            split="test",
            device=device,
            max_samples=pilot_cfg.eval_test_samples,
            num_shards=pilot_cfg.eval_test_num_shards,
            batch_size=max(int(pilot_cfg.main_batch_size), 1024),
            target_scale=variant.target_scale,
            oracle_cfg=ab_lab.OracleEvalConfig(),
        )
        oracle_eval = ab_lab.evaluate_model_on_oracle_subset(
            model=model,
            oracle_bundle=oracle_bundle,
            device=device,
            target_scale=variant.target_scale,
            oracle_cfg=ab_lab.OracleEvalConfig(),
        )
        pooled_center_eval = fb_lab.evaluate_model_on_center_bundle(model, pooled_center_bundle, device=device)
        primary = obj_lab.extract_primary_metrics("OC2_joint_oracle_full_model", test_eval, oracle_eval)
        current_center_score = float(center_only_score(pooled_center_eval))
        current_legacy_score = float(legacy_failure_b_score(primary, pooled_center_eval))
        gate_pass = bool(passes_midband_gate(primary, l4_reference["primary"], gate_cfg))
        row = {
            "epoch": int(epoch),
            "train_main_objective": running["main_objective"] / max(running["main_n"], 1),
            "train_main_term": running["main_term"] / max(running["main_n"], 1),
            "train_mean_main_weight": running["mean_main_weight"] / max(running["main_n"], 1),
            "train_downweighted_frac": running["downweighted_frac"] / max(running["main_n"], 1),
            "train_aux_objective": running["aux_objective"] / max(running["aux_steps"], 1),
            "train_aux_anchor_loss": running["aux_anchor_loss"] / max(running["aux_steps"], 1),
            "train_aux_hard_loss": running["aux_hard_loss"] / max(running["aux_steps"], 1),
            "train_aux_ambiguous_loss": running["aux_ambiguous_loss"] / max(running["aux_steps"], 1),
            "train_aux_margin": running["aux_margin"] / max(running["aux_steps"], 1),
            "train_aux_anchor_frac": running["aux_anchor_frac"] / max(running["aux_steps"], 1),
            "train_aux_hard_frac": running["aux_hard_frac"] / max(running["aux_steps"], 1),
            "train_aux_ambiguous_frac": running["aux_ambiguous_frac"] / max(running["aux_steps"], 1),
            "oracle_midband_mae_sum_stable": float(primary["oracle_midband_mae_sum_stable"]),
            "oracle_stable_0.7_slope": float(primary["oracle_stable_0.7_slope"]),
            "pooled_center_mae": float(pooled_center_eval["mae_vs_oracle"]),
            "pooled_center_amp_ratio": float(pooled_center_eval["amp_ratio"]),
            "pooled_center_false_0.1eq": float(pooled_center_eval["false_decisive_0.1"]),
            "pooled_center_false_0.2eq": float(pooled_center_eval["false_decisive_0.2"]),
            "center_score": current_center_score,
            "legacy_failure_b_score": current_legacy_score,
            "midband_gate_pass": gate_pass,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_time_sec": float(time.time() - epoch_start),
        }
        history_rows.append(row)
        print(json.dumps(row, indent=2))

        payload = {
            "epoch": int(epoch),
            "history": history_rows,
            "pilot_cfg": asdict(pilot_cfg),
            "gate_cfg": asdict(gate_cfg),
            "trainable_scope": trainable_scope.to_dict("records"),
            "config": _extract_checkpoint_payload(init_ckpt)["config"],
            "model_state": model.state_dict(),
            "primary": primary,
            "pooled_center_eval": pooled_center_eval,
            "oracle_summary": oracle_eval["summary"],
            "center_score": current_center_score,
            "legacy_failure_b_score": current_legacy_score,
            "midband_gate_pass": gate_pass,
        }
        ab_lab.atomic_torch_save(payload, latest_ckpt)
        if current_center_score < best_any_score:
            best_any_score = current_center_score
            ab_lab.atomic_torch_save(payload, best_any_ckpt)
            save_json(primary, reports_dir / "best_any_primary_metrics.json")
            save_json(pooled_center_eval, reports_dir / "best_any_pooled_center_eval.json")
        if gate_pass and current_center_score < best_gate_score:
            best_gate_score = current_center_score
            ab_lab.atomic_torch_save(payload, best_gate_ckpt)
            save_json(primary, reports_dir / "best_gate_primary_metrics.json")
            save_json(pooled_center_eval, reports_dir / "best_gate_pooled_center_eval.json")
        model.train()

    history_df = pd.DataFrame(history_rows)
    save_dataframe(history_df, history_csv)
    decision_summary = {
        "best_any_center_score": None if not math.isfinite(best_any_score) else float(best_any_score),
        "best_gate_center_score": None if not math.isfinite(best_gate_score) else float(best_gate_score),
        "has_gate_checkpoint": bool(best_gate_ckpt.exists()),
        "l4_center_score": float(l4_reference["center_score"]),
        "l4_legacy_failure_b_score": float(l4_reference["legacy_failure_b_score"]),
        "l4_midband_mae": float(l4_reference["primary"]["oracle_midband_mae_sum_stable"]),
        "l4_stable_slope": float(l4_reference["primary"]["oracle_stable_0.7_slope"]),
    }
    save_json(
        {
            "history": history_rows,
            "decision_summary": decision_summary,
            "pilot_cfg": asdict(pilot_cfg),
            "gate_cfg": asdict(gate_cfg),
        },
        history_json,
    )
    save_json(decision_summary, reports_dir / "decision_summary.json")
    return {
        "latest_checkpoint": latest_ckpt,
        "best_any_checkpoint": best_any_ckpt if best_any_ckpt.exists() else None,
        "best_gate_checkpoint": best_gate_ckpt if best_gate_ckpt.exists() else None,
        "history": history_df,
        "trainable_scope": trainable_scope,
        "decision_summary": decision_summary,
    }


def evaluate_registry_with_oc2(
    registry: Sequence[Dict[str, object]],
    oc2_best_any_checkpoint: Optional[str | Path],
    oc2_best_gate_checkpoint: Optional[str | Path],
    data_root: str | Path,
    pooled_center_bundle: Dict[str, object],
    oracle_bundle: Dict[str, object],
    eval_cfg: ab_lab.TrainConfig,
    l4_reference: Dict[str, object],
    gate_cfg: MidbandGateConfig,
    paths: Dict[str, Path],
    device: torch.device,
    prefix: str = "combined_oc2_final_pilot",
) -> Dict[str, object]:
    combined = list(registry)
    if oc2_best_any_checkpoint is not None:
        combined.append(
            {
                "label": "OC2_best_any_center",
                "checkpoint": str(Path(oc2_best_any_checkpoint)),
                "target_scale": 600.0,
            }
        )
    if oc2_best_gate_checkpoint is not None:
        combined.append(
            {
                "label": "OC2_best_gate",
                "checkpoint": str(Path(oc2_best_gate_checkpoint)),
                "target_scale": 600.0,
            }
        )

    rows_primary: List[dict] = []
    rows_center: List[dict] = []
    results: Dict[str, object] = {}
    for item in combined:
        label = str(item["label"])
        ckpt_path = item["checkpoint"]
        center_eval = fb_lab.evaluate_on_pooled_center_bundle(
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
            oracle_cfg=ab_lab.OracleEvalConfig(),
            train_cfg=eval_cfg,
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
                "center_score": float(center_only_score(center_eval)),
                "legacy_failure_b_score": float(legacy_failure_b_score(result["primary"], center_eval)),
                "midband_gate_pass": bool(passes_midband_gate(result["primary"], l4_reference["primary"], gate_cfg)),
            }
        )
        rows_primary.append(row)
        rows_center.append(center_eval)
        results[label] = {"primary": row, "center_eval": center_eval, "full_eval": result}
    primary_df = pd.DataFrame(rows_primary).sort_values(["center_score", "oracle_midband_mae_sum_stable"], ascending=[True, True]).reset_index(drop=True)
    center_df = pd.DataFrame(rows_center)
    save_dataframe(primary_df, paths["reports_dir"] / f"{prefix}_primary_metrics.csv")
    save_dataframe(center_df, paths["reports_dir"] / f"{prefix}_pooled_center_metrics.csv")
    return {"primary": primary_df, "pooled_center": center_df, "results": results}
