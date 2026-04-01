from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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

CACHE_SCHEMA_VERSION = 2


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
class OracleSubsetConfig:
    raw_abs_y_edges: Tuple[float, ...] = (0.0, 0.02, 0.05, 0.10)
    sample_per_abs_y_band: int = 64
    top_pred_frac: float = 0.50
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


@dataclass
class PilotTrainConfig:
    epochs: int = 2
    main_batch_size: int = 576
    aux_batch_size: int = 192
    learning_rate: float = 2.5e-6
    min_lr: float = 1.0e-6
    weight_decay: float = 2.0e-4
    grad_clip_norm: float = 1.0
    seed: int = 123
    main_train_num_shards: int = 8
    eval_test_samples: int = 200_000
    eval_test_num_shards: int = 4
    log_every_steps: int = 200
    max_mem_ratio: float = 0.85
    min_batch_size: int = 128
    batch_step: int = 64
    freeze_last_blocks: int = 2
    main_center_downweight_tau_y600: float = 0.10
    main_center_downweight_factor: float = 0.25
    aux_updates_per_main_step: int = 1
    aux_center_boost: float = 3.0
    aux_margin_y600: float = 0.08
    aux_margin_weight: float = 0.50


def build_default_paths(
    run_dir: str | Path = r"C:\Users\USER\Downloads\dgrn_5m_v3_stage2_polish_run1",
    data_root: str | Path = r"C:\Users\USER\Desktop\chess_engine\data\process",
    experiment_dir: str | Path | None = None,
) -> Dict[str, Path]:
    if experiment_dir is None:
        experiment_dir = PROJECT_ROOT / "experiments" / "l4_oracle_center_correction_pilot"
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


def _candidate_manifest_payload(checkpoint_path: str | Path, cfg: OracleSubsetConfig) -> Dict[str, object]:
    return {
        "schema_version": int(CACHE_SCHEMA_VERSION),
        "checkpoint_path": str(Path(checkpoint_path)),
        "train_num_shards": int(cfg.train_num_shards),
        "raw_abs_y_edges": [float(x) for x in cfg.raw_abs_y_edges],
        "sample_per_abs_y_band": int(cfg.sample_per_abs_y_band),
        "top_pred_frac": float(cfg.top_pred_frac),
    }


def _oracle_audit_manifest_payload(candidate_manifest: Dict[str, object], cfg: OracleSubsetConfig) -> Dict[str, object]:
    return {
        "schema_version": int(CACHE_SCHEMA_VERSION),
        "candidate_manifest": candidate_manifest,
        "oracle_cfg": asdict(cfg),
    }


def _aux_manifest_payload(
    candidate_manifest: Dict[str, object],
    oracle_report: Dict[str, object],
    pilot_cfg: PilotTrainConfig,
) -> Dict[str, object]:
    return {
        "schema_version": int(CACHE_SCHEMA_VERSION),
        "candidate_manifest": candidate_manifest,
        "oracle_report": oracle_report,
        "aux_center_boost": float(pilot_cfg.aux_center_boost),
    }


def validate_runtime_paths(paths: Dict[str, Path], oracle_cfg: OracleSubsetConfig) -> Dict[str, object]:
    l4_ckpt = paths["objective_output_dir"] / "runs" / "L4_A1_plus_A2" / "checkpoints" / "L4_A1_plus_A2_best.pt"
    issues: List[str] = []
    if not l4_ckpt.exists():
        issues.append(f"Missing L4 checkpoint: {l4_ckpt}")
    if not (paths["run_dir"] / "ckpt_best.pt").exists():
        issues.append(f"Missing baseline checkpoint: {paths['run_dir'] / 'ckpt_best.pt'}")
    if not paths["data_root"].exists():
        issues.append(f"Missing data_root: {paths['data_root']}")
    if not Path(oracle_cfg.stockfish_path).exists():
        issues.append(f"Missing Stockfish binary: {oracle_cfg.stockfish_path}")
    reference_paths = [
        paths["run_dir"] / "ckpt_best.pt",
        ROOT_CAUSE_DIR / "outputs" / "runs" / "A2_band_balanced" / "checkpoints" / "A2_band_balanced_best.pt",
        paths["objective_output_dir"] / "runs" / "L0_control_hybrid" / "checkpoints" / "L0_control_hybrid_best.pt",
        l4_ckpt,
    ]
    for ref_path in reference_paths:
        if not ref_path.exists():
            issues.append(f"Missing reference checkpoint: {ref_path}")
    return {
        "l4_checkpoint": str(l4_ckpt),
        "baseline_checkpoint": str(paths["run_dir"] / "ckpt_best.pt"),
        "stockfish_path": str(oracle_cfg.stockfish_path),
        "ok": len(issues) == 0,
        "issues": issues,
    }


def build_reference_failure_b_paths() -> Dict[str, Path]:
    return fb_lab.build_default_paths()


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
    failure_b_paths = build_reference_failure_b_paths()
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
    registry = build_reference_registry(paths)
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
        "registry": registry,
        "eval_cfg": eval_cfg,
    }


def build_l4_variant() -> ab_lab.AblationVariant:
    return obj_lab.build_variant_catalog()["L4_A1_plus_A2"]


def _stockfish_cfg(cfg: OracleSubsetConfig) -> sw_lab.StabilityWeightConfig:
    return sw_lab.StabilityWeightConfig(
        near_zero_thr=float(cfg.raw_abs_y_edges[-1]),
        calibration_abs_y_sample_edges=cfg.raw_abs_y_edges,
        sample_per_abs_y_band=cfg.sample_per_abs_y_band,
        stockfish_path=cfg.stockfish_path,
        stockfish_threads=cfg.stockfish_threads,
        stockfish_hash_mb=cfg.stockfish_hash_mb,
        stockfish_node_budgets=cfg.stockfish_node_budgets,
        stockfish_command_pause_ms=cfg.stockfish_command_pause_ms,
        stockfish_timeout_sec=cfg.stockfish_timeout_sec,
        calibration_seed=cfg.selection_seed,
        prediction_batch_size=cfg.prediction_cache_batch_size,
    )


def _candidate_cache_dir(paths: Dict[str, Path]) -> Path:
    out = paths["cache_dir"] / "oracle_candidates"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _oracle_probe_cache_dir(paths: Dict[str, Path]) -> Path:
    out = paths["cache_dir"] / "oracle_probe"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _aux_bundle_cache_dir(paths: Dict[str, Path]) -> Path:
    out = paths["cache_dir"] / "oracle_aux_bundle"
    out.mkdir(parents=True, exist_ok=True)
    return out


def build_oracle_candidate_bundle(
    checkpoint_path: str | Path,
    pred_cache_manifest: Dict[str, object],
    data_root: str | Path,
    cfg: OracleSubsetConfig,
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
        if manifest == expected_manifest:
            npz = np.load(npz_path, allow_pickle=False)
            return {
                "rows": pd.read_csv(rows_csv),
                "X": npz["X"].astype(np.uint8, copy=False),
                "manifest": manifest,
                "quota_summary": pd.read_csv(quota_summary_csv),
            }
        print("[candidate-bundle] cache signature mismatch; rebuilding candidate bundle.")

    shard_rows = ab_lab.resolve_split_shards(data_root, "train", num_shards=cfg.train_num_shards)
    pred_files = {
        int(row["shard_id"]): Path(paths["cache_dir"]) / "train_pred_cache" / Path(checkpoint_path).stem / row["pred_file"]
        for row in pred_cache_manifest["shards"]
    }
    abs_y_edges = np.asarray(cfg.raw_abs_y_edges, dtype=np.float64)
    band_labels = sw_lab.band_labels_from_edges(abs_y_edges)
    count_rows: List[dict] = []
    for shard_id, _, y_path in shard_rows:
        y = np.load(y_path, mmap_mode="r").astype(np.float32, copy=False)
        abs_y = np.abs(y.astype(np.float64))
        for band_idx, (left, right) in enumerate(zip(abs_y_edges[:-1], abs_y_edges[1:])):
            if band_idx == len(band_labels) - 1:
                mask = (abs_y >= left) & (abs_y <= right)
            else:
                mask = (abs_y >= left) & (abs_y < right)
            count_rows.append(
                {
                    "shard_id": int(shard_id),
                    "band_idx": int(band_idx),
                    "band_label": band_labels[band_idx],
                    "count": int(np.sum(mask)),
                }
            )
    counts_df = pd.DataFrame(count_rows)
    quota_rows: List[pd.DataFrame] = []
    for band_idx, band_label in enumerate(band_labels):
        band_df = counts_df[counts_df["band_idx"] == band_idx].copy().sort_values("shard_id")
        quotas = sw_lab.proportional_allocation(
            band_df["count"].to_numpy(dtype=np.int64),
            int(cfg.sample_per_abs_y_band),
        )
        band_df["quota"] = quotas
        quota_rows.append(band_df)
    quota_df = pd.concat(quota_rows, ignore_index=True)
    quota_summary = quota_df.groupby(["band_idx", "band_label"], as_index=False)[["count", "quota"]].sum()
    save_dataframe(quota_summary, quota_summary_csv)

    rng = np.random.default_rng(cfg.selection_seed)
    rows: List[dict] = []
    x_rows: List[np.ndarray] = []
    next_candidate_id = 0
    for shard_id, x_path, y_path in shard_rows:
        quota_slice = quota_df[quota_df["shard_id"] == shard_id]
        if quota_slice["quota"].sum() <= 0:
            continue
        X = np.load(x_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r").astype(np.float32, copy=False)
        pred_path = pred_files.get(int(shard_id))
        if pred_path is None or not pred_path.exists():
            raise FileNotFoundError(f"Missing cached L4 prediction shard for candidate build: shard={shard_id}")
        pred = np.load(pred_path, mmap_mode="r").astype(np.float32, copy=False)
        abs_y = np.abs(y.astype(np.float64))
        abs_pred = np.abs(pred.astype(np.float64))
        abs_err = np.abs((pred - y).astype(np.float64))
        for _, row in quota_slice.iterrows():
            quota = int(row["quota"])
            if quota <= 0:
                continue
            band_idx = int(row["band_idx"])
            left = float(abs_y_edges[band_idx])
            right = float(abs_y_edges[band_idx + 1])
            if band_idx == len(band_labels) - 1:
                idx = np.flatnonzero((abs_y >= left) & (abs_y <= right))
            else:
                idx = np.flatnonzero((abs_y >= left) & (abs_y < right))
            if idx.size == 0:
                continue
            scores = abs_pred[idx]
            ranked = idx[np.argsort(-scores, kind="mergesort")]
            top_quota = min(int(round(quota * float(cfg.top_pred_frac))), ranked.size)
            chosen_top = ranked[:top_quota]
            remaining_idx = ranked[top_quota:]
            remaining_quota = max(0, min(quota, ranked.size) - chosen_top.size)
            if remaining_quota > 0 and remaining_idx.size > 0:
                chosen_rand = rng.choice(remaining_idx, size=min(remaining_quota, remaining_idx.size), replace=False)
                chosen = np.concatenate([chosen_top, np.sort(chosen_rand)])
            else:
                chosen = chosen_top
            for local_index in np.sort(np.unique(chosen.astype(np.int64))):
                rows.append(
                    {
                        "candidate_id": int(next_candidate_id),
                        "shard_id": int(shard_id),
                        "local_index": int(local_index),
                        "band_idx": int(band_idx),
                        "band_label": str(row["band_label"]),
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
        raise RuntimeError(
            "Candidate bundle is empty. Check train shard selection, raw_abs_y_edges, "
            "and whether the prediction cache was built from the intended checkpoint."
        )
    X_bundle = np.stack(x_rows, axis=0) if x_rows else np.empty((0, 18, 8, 8), dtype=np.uint8)
    rows_df["_tensor_pos"] = np.arange(rows_df.shape[0], dtype=np.int64)
    rows_df = rows_df.sort_values(["band_idx", "init_abs_pred"], ascending=[True, False]).reset_index(drop=True)
    tensor_order = rows_df["_tensor_pos"].to_numpy(dtype=np.int64, copy=False)
    rows_df = rows_df.drop(columns="_tensor_pos")
    X_bundle = X_bundle[tensor_order]
    manifest = dict(expected_manifest)
    manifest["num_candidates"] = int(rows_df.shape[0])
    save_dataframe(rows_df, rows_csv)
    np.savez_compressed(npz_path, X=X_bundle.astype(np.uint8))
    save_json(manifest, manifest_path)
    return {"rows": rows_df, "X": X_bundle, "manifest": manifest, "quota_summary": quota_summary}


def run_oracle_candidate_audit(
    candidate_bundle: Dict[str, object],
    cfg: OracleSubsetConfig,
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
        if manifest == expected_manifest:
            return {
                "rows": pd.read_csv(rows_csv),
                "report": json.loads(summary_json.read_text(encoding="utf-8")),
                "summary": pd.read_csv(summary_csv),
            }
        print("[oracle-candidate-audit] cache signature mismatch; rebuilding oracle audit.")

    stockfish_cfg = _stockfish_cfg(cfg)
    rows = candidate_bundle["rows"].copy().reset_index(drop=True)
    X = np.asarray(candidate_bundle["X"], dtype=np.uint8)
    if rows.shape[0] != X.shape[0]:
        raise RuntimeError("Candidate rows and tensor bundle are misaligned.")

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
        is_center_clean = bool(is_stable and abs(oracle_final_y) <= float(cfg.trusted_center_thr))
        aux_keep = bool(is_stable and abs(oracle_final_y) <= float(cfg.aux_oracle_abs_max))
        record = row.to_dict()
        record.update(
            {
                "fen": board.to_fen(),
                "oracle_final_y": oracle_final_y,
                "oracle_abs_y": float(abs(oracle_final_y)),
                "oracle_target_range": oracle_target_range,
                "oracle_target_std": oracle_target_std,
                "oracle_cp_range": float(curve["cp_range"]),
                "oracle_bestmove_changes": bestmove_changes,
                "oracle_sign_flips": sign_flips,
                "oracle_bestmove_final": str(curve["bestmoves"][-1]),
                "is_stable": is_stable,
                "is_center_clean": is_center_clean,
                "aux_keep": aux_keep,
            }
        )
        for curve_row in curve["rows"]:
            node_budget = int(curve_row["node_budget"])
            record[f"oracle_y_n{node_budget}"] = float(curve_row["target_value"])
            record[f"oracle_cp_n{node_budget}"] = float(curve_row["cp_equivalent"])
            record[f"oracle_bestmove_n{node_budget}"] = str(curve_row["bestmove"])
        out_rows.append(record)
        if pos % 16 == 0:
            print(f"[oracle-candidate-audit] processed={pos + 1}/{rows.shape[0]}")

    out_df = pd.DataFrame(out_rows)
    summary_df = out_df.groupby(["band_idx", "band_label"], as_index=False).agg(
        n=("candidate_id", "size"),
        stable_count=("is_stable", "sum"),
        center_clean_count=("is_center_clean", "sum"),
        aux_keep_count=("aux_keep", "sum"),
        mean_oracle_abs_y=("oracle_abs_y", "mean"),
        mean_init_abs_pred=("init_abs_pred", "mean"),
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
        "stable_target_range_max": float(cfg.stable_target_range_max),
        "stable_target_std_max": float(cfg.stable_target_std_max),
        "stable_bestmove_changes_max": int(cfg.stable_bestmove_changes_max),
        "stable_sign_flips_max": int(cfg.stable_sign_flips_max),
    }
    save_dataframe(out_df, rows_csv)
    save_dataframe(summary_df, summary_csv)
    save_json(report, summary_json)
    manifest = dict(expected_manifest)
    manifest["report"] = report
    save_json(manifest, manifest_path)
    return {"rows": out_df, "report": report, "summary": summary_df}


def build_oracle_aux_bundle(
    candidate_bundle: Dict[str, object],
    oracle_audit: Dict[str, object],
    pilot_cfg: PilotTrainConfig,
    paths: Dict[str, Path],
    refresh: bool = False,
) -> Dict[str, object]:
    cache_dir = _aux_bundle_cache_dir(paths)
    manifest_path = cache_dir / "manifest.json"
    rows_csv = cache_dir / "oracle_aux_rows.csv"
    npz_path = cache_dir / "oracle_aux_bundle.npz"
    expected_manifest = _aux_manifest_payload(candidate_bundle["manifest"], oracle_audit["report"], pilot_cfg)
    if not refresh and manifest_path.exists() and rows_csv.exists() and npz_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest == expected_manifest:
            npz = np.load(npz_path, allow_pickle=False)
            return {
                "rows": pd.read_csv(rows_csv),
                "X": npz["X"].astype(np.uint8, copy=False),
                "oracle_y": npz["oracle_y"].astype(np.float32, copy=False),
                "is_center_clean": npz["is_center_clean"].astype(bool, copy=False),
                "sample_weight": npz["sample_weight"].astype(np.float32, copy=False),
                "manifest": manifest,
            }
        print("[oracle-aux-bundle] cache signature mismatch; rebuilding aux bundle.")

    rows = oracle_audit["rows"].copy()
    keep_mask = rows["aux_keep"].astype(bool).to_numpy()
    kept_rows = rows.loc[keep_mask].copy().reset_index(drop=True)
    if kept_rows.empty:
        raise RuntimeError(
            "Oracle auxiliary bundle is empty. Stable thresholds are too strict or "
            "the oracle audit did not keep any candidate rows."
        )
    X = np.asarray(candidate_bundle["X"], dtype=np.uint8)[keep_mask]
    oracle_y = kept_rows["oracle_final_y"].to_numpy(dtype=np.float32)
    is_center_clean = kept_rows["is_center_clean"].astype(bool).to_numpy()
    if not np.any(is_center_clean):
        raise RuntimeError(
            "Oracle auxiliary bundle has no center-clean rows. The current thresholds do not "
            "produce trusted-center supervision, so this pilot would not test the intended hypothesis."
        )
    sample_weight = np.where(is_center_clean, float(pilot_cfg.aux_center_boost), 1.0).astype(np.float32)
    kept_rows["sample_weight"] = sample_weight
    kept_rows["aux_role"] = np.where(is_center_clean, "center_clean", "stable_ambiguous")

    manifest = dict(expected_manifest)
    manifest.update(
        {
            "num_rows": int(kept_rows.shape[0]),
            "center_clean_count": int(np.sum(is_center_clean)),
            "stable_ambiguous_count": int(np.sum(~is_center_clean)),
        }
    )
    save_dataframe(kept_rows, rows_csv)
    np.savez_compressed(
        npz_path,
        X=X.astype(np.uint8),
        oracle_y=oracle_y.astype(np.float32),
        is_center_clean=is_center_clean.astype(np.uint8),
        sample_weight=sample_weight.astype(np.float32),
    )
    save_json(manifest, manifest_path)
    return {
        "rows": kept_rows,
        "X": X.astype(np.uint8, copy=False),
        "oracle_y": oracle_y,
        "is_center_clean": is_center_clean,
        "sample_weight": sample_weight,
        "manifest": manifest,
    }


def configure_trainable_scope(model: nn.Module, freeze_last_blocks: int) -> pd.DataFrame:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True
    num_blocks = len(model.blocks)
    start_idx = max(0, num_blocks - int(freeze_last_blocks))
    for idx in range(start_idx, num_blocks):
        for param in model.blocks[idx].parameters():
            param.requires_grad = True
    rows: List[dict] = []
    for name, module in [("head", model.head)] + [(f"blocks.{idx}", model.blocks[idx]) for idx in range(num_blocks)]:
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
    return pd.DataFrame(rows)


def _main_center_weights(y_source: torch.Tensor, cfg: PilotTrainConfig) -> torch.Tensor:
    weight = torch.ones_like(y_source, dtype=torch.float32)
    center_mask = torch.abs(y_source) <= float(cfg.main_center_downweight_tau_y600)
    if torch.any(center_mask):
        weight = torch.where(
            center_mask,
            torch.full_like(weight, float(cfg.main_center_downweight_factor)),
            weight,
        )
    return weight


def compute_l4_main_terms(
    logits: torch.Tensor,
    y_source: torch.Tensor,
    variant: ab_lab.AblationVariant,
    cfg: PilotTrainConfig,
) -> Dict[str, torch.Tensor]:
    y = ab_lab.remap_target_torch(y_source, to_scale=variant.target_scale).view(-1)
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
    denom = torch.clamp(sample_weight.sum(), min=1e-6)
    objective = torch.sum(sample_weight * main_per) / denom
    return {
        "objective": objective,
        "main_term": torch.mean(main_per),
        "pred": pred,
        "mean_main_weight": torch.mean(sample_weight),
        "downweighted_frac": torch.mean((sample_weight < 0.999).float()),
    }


def compute_oracle_aux_terms(
    logits: torch.Tensor,
    oracle_y: torch.Tensor,
    center_mask: torch.Tensor,
    margin_y600: float,
    margin_weight: float,
) -> Dict[str, torch.Tensor]:
    pred = torch.tanh(logits.view(-1))
    mse = F.mse_loss(pred, oracle_y.view(-1))
    if torch.any(center_mask):
        margin_penalty = torch.mean(torch.relu(torch.abs(pred[center_mask]) - float(margin_y600)) ** 2)
    else:
        margin_penalty = pred.new_tensor(0.0)
    objective = mse + float(margin_weight) * margin_penalty
    return {
        "objective": objective,
        "mse": mse,
        "margin_penalty": margin_penalty,
        "pred": pred,
    }


def _sample_aux_indices(bundle: Dict[str, object], batch_size: int, rng: np.random.Generator) -> np.ndarray:
    weights = np.asarray(bundle["sample_weight"], dtype=np.float64)
    if weights.size == 0:
        raise RuntimeError("Aux bundle is empty.")
    probs = weights / weights.sum()
    replace = weights.shape[0] < batch_size
    size = batch_size if replace else min(batch_size, weights.shape[0])
    return rng.choice(np.arange(weights.shape[0], dtype=np.int64), size=size, replace=replace, p=probs)


def _failure_b_score(primary: Dict[str, object], pooled_center_eval: Dict[str, object]) -> float:
    return float(
        float(pooled_center_eval["mae_vs_oracle"])
        + 0.30 * float(pooled_center_eval["false_decisive_0.1"])
        + 0.20 * float(pooled_center_eval["false_decisive_0.2"])
        + 0.10 * max(0.0, float(pooled_center_eval["amp_ratio"]) - 2.5)
        + 0.30 * float(primary["oracle_midband_mae_sum_stable"])
        + 0.20 * max(0.0, 0.80 - float(primary["oracle_stable_0.7_slope"]))
    )


def run_l4_oracle_center_correction_pilot(
    init_ckpt_path: str | Path,
    data_root: str | Path,
    pilot_cfg: PilotTrainConfig,
    aux_bundle: Dict[str, object],
    oracle_bundle: Dict[str, object],
    pooled_center_bundle: Dict[str, object],
    paths: Dict[str, Path],
    device: torch.device,
) -> Dict[str, object]:
    checkpoint_dir = paths["checkpoints_dir"]
    reports_dir = paths["reports_dir"]
    best_ckpt = checkpoint_dir / "OC1_L4_oracle_center_correction_best.pt"
    latest_ckpt = checkpoint_dir / "OC1_L4_oracle_center_correction_latest.pt"
    history_csv = reports_dir / "pilot_history.csv"
    history_json = reports_dir / "pilot_history.json"

    set_global_seed(pilot_cfg.seed)
    model, init_ckpt = base_lab.load_model_from_checkpoint(init_ckpt_path, device=device)
    trainable_scope = configure_trainable_scope(model, pilot_cfg.freeze_last_blocks)
    save_dataframe(trainable_scope, reports_dir / "trainable_scope.csv")
    variant = build_l4_variant()
    optimizer = ab_lab.build_optimizer(model, lr=pilot_cfg.learning_rate, weight_decay=pilot_cfg.weight_decay)
    shard_rows = ab_lab.resolve_split_shards(data_root, "train", num_shards=pilot_cfg.main_train_num_shards)
    total_main_samples = int(sum(int(np.load(y_path, mmap_mode="r").shape[0]) for _, _, y_path in shard_rows))
    total_main_steps = int(math.ceil(total_main_samples / pilot_cfg.main_batch_size) * max(pilot_cfg.epochs, 1))
    total_steps = total_main_steps * (1 + int(pilot_cfg.aux_updates_per_main_step))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=pilot_cfg.min_lr)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    rng = np.random.default_rng(pilot_cfg.seed)

    history_rows: List[dict] = []
    best_score = float("inf")
    global_step = 0
    for epoch in range(pilot_cfg.epochs):
        model.train()
        t0 = time.time()
        running = {
            "main_objective": 0.0,
            "main_term": 0.0,
            "mean_main_weight": 0.0,
            "downweighted_frac": 0.0,
            "main_n": 0,
            "aux_objective": 0.0,
            "aux_mse": 0.0,
            "aux_margin": 0.0,
            "aux_steps": 0,
            "aux_center_clean_frac": 0.0,
        }
        for shard_rank, (_, x_path, y_path) in enumerate(shard_rows, start=1):
            X = np.load(x_path, mmap_mode="r")
            y_source = np.load(y_path, mmap_mode="r").astype(np.float32, copy=False)
            y_scaled = ab_lab.remap_target_np(y_source, to_scale=variant.target_scale).astype(np.float32, copy=False)
            if variant.sampler_mode == "band_balanced":
                order = ab_lab.build_band_balanced_order(
                    abs_y=np.abs(y_scaled.astype(np.float64)),
                    batch_size=pilot_cfg.main_batch_size,
                    band_edges_y600=variant.balance_band_edges_y600,
                    rng=rng,
                    target_scale=variant.target_scale,
                )
            else:
                order = rng.permutation(y_scaled.shape[0]).astype(np.int64)

            for start in range(0, y_source.shape[0], pilot_cfg.main_batch_size):
                idx = order[start : start + pilot_cfg.main_batch_size]
                xb = torch.from_numpy(np.array(X[idx], dtype=np.float32, copy=True)).to(device, non_blocking=True)
                yb_source = torch.from_numpy(np.array(y_source[idx], dtype=np.float32, copy=True)).to(device, non_blocking=True).view(-1)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    logits = model.forward_logits(xb).view(-1)
                    main_terms = compute_l4_main_terms(logits, yb_source, variant, pilot_cfg)
                scaler.scale(main_terms["objective"]).backward()
                if pilot_cfg.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), pilot_cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                bs = int(yb_source.numel())
                running["main_objective"] += float(main_terms["objective"].item()) * bs
                running["main_term"] += float(main_terms["main_term"].item()) * bs
                running["mean_main_weight"] += float(main_terms["mean_main_weight"].item()) * bs
                running["downweighted_frac"] += float(main_terms["downweighted_frac"].item()) * bs
                running["main_n"] += bs
                global_step += 1

                for _ in range(int(pilot_cfg.aux_updates_per_main_step)):
                    aux_idx = _sample_aux_indices(aux_bundle, pilot_cfg.aux_batch_size, rng=rng)
                    aux_X = torch.from_numpy(np.array(aux_bundle["X"][aux_idx], dtype=np.float32, copy=True)).to(device, non_blocking=True)
                    aux_y = torch.from_numpy(np.array(aux_bundle["oracle_y"][aux_idx], dtype=np.float32, copy=True)).to(device, non_blocking=True).view(-1)
                    aux_center_mask = torch.from_numpy(np.array(aux_bundle["is_center_clean"][aux_idx], dtype=np.bool_, copy=True)).to(device, non_blocking=True).view(-1)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                        logits_aux = model.forward_logits(aux_X).view(-1)
                        aux_terms = compute_oracle_aux_terms(
                            logits=logits_aux,
                            oracle_y=aux_y,
                            center_mask=aux_center_mask,
                            margin_y600=float(pilot_cfg.aux_margin_y600),
                            margin_weight=float(pilot_cfg.aux_margin_weight),
                        )
                    scaler.scale(aux_terms["objective"]).backward()
                    if pilot_cfg.grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), pilot_cfg.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    running["aux_objective"] += float(aux_terms["objective"].item())
                    running["aux_mse"] += float(aux_terms["mse"].item())
                    running["aux_margin"] += float(aux_terms["margin_penalty"].item())
                    running["aux_steps"] += 1
                    running["aux_center_clean_frac"] += float(aux_center_mask.float().mean().item())

                if global_step % pilot_cfg.log_every_steps == 0:
                    print(
                        f"[oracle-center-pilot] step={global_step}/{total_main_steps} "
                        f"main_obj={running['main_objective'] / max(running['main_n'], 1):.6f} "
                        f"aux_obj={running['aux_objective'] / max(running['aux_steps'], 1):.6f}"
                    )
            print(f"[oracle-center-pilot] finished shard {shard_rank}/{len(shard_rows)}")

        test_eval = ab_lab.evaluate_model_on_split_scale_aware(
            model=model,
            data_root=data_root,
            split="test",
            device=device,
            max_samples=pilot_cfg.eval_test_samples,
            num_shards=pilot_cfg.eval_test_num_shards,
            batch_size=max(pilot_cfg.main_batch_size, 1024),
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
        primary = obj_lab.extract_primary_metrics("OC1_L4_oracle_center_correction", test_eval, oracle_eval)
        pilot_score = _failure_b_score(primary, pooled_center_eval)
        row = {
            "epoch": int(epoch),
            "train_main_objective": running["main_objective"] / max(running["main_n"], 1),
            "train_main_term": running["main_term"] / max(running["main_n"], 1),
            "train_mean_main_weight": running["mean_main_weight"] / max(running["main_n"], 1),
            "train_downweighted_frac": running["downweighted_frac"] / max(running["main_n"], 1),
            "train_aux_objective": running["aux_objective"] / max(running["aux_steps"], 1),
            "train_aux_mse": running["aux_mse"] / max(running["aux_steps"], 1),
            "train_aux_margin": running["aux_margin"] / max(running["aux_steps"], 1),
            "train_aux_center_clean_frac": running["aux_center_clean_frac"] / max(running["aux_steps"], 1),
            "oracle_midband_mae_sum_stable": float(primary["oracle_midband_mae_sum_stable"]),
            "oracle_stable_0.7_slope": float(primary["oracle_stable_0.7_slope"]),
            "oracle_center_amp_ratio": float(primary["oracle_center_amp_ratio"]),
            "oracle_center_false_0.1eq": float(primary["oracle_center_false_0.1eq"]),
            "oracle_center_false_0.2eq": float(primary["oracle_center_false_0.2eq"]),
            "pooled_center_mae": float(pooled_center_eval["mae_vs_oracle"]),
            "pooled_center_amp_ratio": float(pooled_center_eval["amp_ratio"]),
            "pooled_center_false_0.1eq": float(pooled_center_eval["false_decisive_0.1"]),
            "pooled_center_false_0.2eq": float(pooled_center_eval["false_decisive_0.2"]),
            "failure_b_score": float(pilot_score),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_time_sec": float(time.time() - t0),
        }
        history_rows.append(row)
        print(json.dumps(row, indent=2))
        payload = {
            "epoch": int(epoch),
            "history": history_rows,
            "trainable_scope": trainable_scope.to_dict("records"),
            "pilot_cfg": asdict(pilot_cfg),
            "config": init_ckpt.get("config"),
            "model_state": model.state_dict(),
            "oracle_summary": oracle_eval["summary"],
            "pooled_center_eval": pooled_center_eval,
            "primary": primary,
            "failure_b_score": pilot_score,
        }
        ab_lab.atomic_torch_save(payload, latest_ckpt)
        if pilot_score < best_score:
            best_score = pilot_score
            ab_lab.atomic_torch_save(payload, best_ckpt)
            save_json(oracle_eval["summary"], reports_dir / "best_test_oracle_summary.json")
            save_json(pooled_center_eval, reports_dir / "best_pooled_center_eval.json")
            save_json(primary, reports_dir / "best_primary_metrics.json")
            ab_lab.save_oracle_eval_outputs(oracle_eval, paths["output_dir"], prefix="best_test_oracle")

    history_df = pd.DataFrame(history_rows)
    save_dataframe(history_df, history_csv)
    save_json({"history": history_rows, "best_score": best_score, "pilot_cfg": asdict(pilot_cfg)}, history_json)
    return {
        "best_checkpoint": best_ckpt,
        "latest_checkpoint": latest_ckpt,
        "history": history_df,
        "trainable_scope": trainable_scope,
    }


def compare_registry_with_pilot(
    registry: Sequence[Dict[str, object]],
    pilot_checkpoint: str | Path,
    data_root: str | Path,
    pooled_center_bundle: Dict[str, object],
    oracle_bundle: Dict[str, object],
    eval_cfg: ab_lab.TrainConfig,
    paths: Dict[str, Path],
    device: torch.device,
    prefix: str = "combined_oracle_center_pilot",
) -> Dict[str, object]:
    combined = list(registry) + [
        {
            "label": "OC1_L4_oracle_center_correction",
            "checkpoint": str(Path(pilot_checkpoint)),
            "target_scale": 600.0,
        }
    ]
    return fb_lab.evaluate_failure_b_registry(
        registry=combined,
        pooled_center_bundle=pooled_center_bundle,
        oracle_bundle=oracle_bundle,
        data_root=data_root,
        train_cfg=eval_cfg,
        oracle_cfg=ab_lab.OracleEvalConfig(),
        paths=paths,
        device=device,
        prefix=prefix,
    )
