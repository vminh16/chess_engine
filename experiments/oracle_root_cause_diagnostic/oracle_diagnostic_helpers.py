from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SW_DIR = PROJECT_ROOT / "experiments" / "stability_weighted_near_zero_finetune"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SW_DIR) not in sys.path:
    sys.path.insert(0, str(SW_DIR))

import stability_weighted_helpers as sw_lab  # noqa: E402

base_lab = sw_lab.base_lab


@dataclass
class ExperimentPaths:
    project_root: str
    run_dir: str
    data_root: str
    experiment_dir: str
    output_dir: str
    plots_dir: str
    reports_dir: str
    cache_dir: str
    checkpoints_dir: str
    split_pred_cache_dir: str


@dataclass
class OracleDiagnosticConfig:
    split: str = "test"
    sample_abs_y_edges: Tuple[float, ...] = (0.0, 0.05, 0.20, 0.50, 0.70, 1.00)
    sample_per_band: int = 48
    err_quantiles: Tuple[float, ...] = (1.0 / 3.0, 2.0 / 3.0)
    oracle_scales: Tuple[float, ...] = (400.0, 600.0, 800.0, 1200.0)
    stockfish_path: str = r"D:\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
    stockfish_threads: int = 1
    stockfish_hash_mb: int = 32
    stockfish_node_budgets: Tuple[int, ...] = (8_000, 32_000, 128_000)
    stockfish_command_pause_ms: int = 50
    stockfish_timeout_sec: float = 20.0
    prediction_batch_size: int = 2048
    sample_seed: int = 123
    benchmark_train_batch_size: int = 640
    decode_validation_samples: int = 64
    subset_num_shards: Optional[int] = None


save_json = sw_lab.save_json
save_dataframe = sw_lab.save_dataframe
set_global_seed = sw_lab.set_global_seed
choose_device = sw_lab.choose_device


def build_default_paths(
    run_dir: str | Path = r"C:\Users\USER\Downloads\dgrn_5m_v3_stage2_polish_run1",
    data_root: str | Path = r"C:\Users\USER\Desktop\chess_engine\data\process",
    experiment_dir: str | Path | None = None,
) -> Dict[str, Path]:
    if experiment_dir is None:
        experiment_dir = PROJECT_ROOT / "experiments" / "oracle_root_cause_diagnostic"
    paths = {
        "project_root": PROJECT_ROOT,
        "run_dir": Path(run_dir),
        "data_root": Path(data_root),
        "experiment_dir": Path(experiment_dir),
    }
    paths["output_dir"] = paths["experiment_dir"] / "outputs"
    paths["plots_dir"] = paths["output_dir"] / "plots"
    paths["reports_dir"] = paths["output_dir"] / "reports"
    paths["cache_dir"] = paths["output_dir"] / "cache"
    paths["checkpoints_dir"] = paths["output_dir"] / "checkpoints"
    paths["split_pred_cache_dir"] = paths["cache_dir"] / "split_preds"
    for key in (
        "experiment_dir",
        "output_dir",
        "plots_dir",
        "reports_dir",
        "cache_dir",
        "checkpoints_dir",
        "split_pred_cache_dir",
    ):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def export_paths_json(paths: Dict[str, Path], out_path: Path) -> None:
    payload = ExperimentPaths(**{key: str(value) for key, value in paths.items()})
    out_path.write_text(json.dumps(asdict(payload), indent=2), encoding="utf-8")


def cp_to_target_scale(cp: float, scale: float, clip_cp: float = 1200.0) -> float:
    clipped = float(np.clip(cp, -clip_cp, clip_cp))
    return float(np.tanh(clipped / float(scale)))


def build_stockfish_cfg(cfg: OracleDiagnosticConfig) -> sw_lab.StabilityWeightConfig:
    return sw_lab.StabilityWeightConfig(
        near_zero_thr=float(min(0.20, cfg.sample_abs_y_edges[-1])),
        calibration_abs_y_sample_edges=tuple(float(x) for x in cfg.sample_abs_y_edges),
        sample_per_abs_y_band=int(cfg.sample_per_band),
        stockfish_path=str(cfg.stockfish_path),
        stockfish_threads=int(cfg.stockfish_threads),
        stockfish_hash_mb=int(cfg.stockfish_hash_mb),
        stockfish_node_budgets=tuple(int(x) for x in cfg.stockfish_node_budgets),
        stockfish_command_pause_ms=int(cfg.stockfish_command_pause_ms),
        stockfish_timeout_sec=float(cfg.stockfish_timeout_sec),
        calibration_seed=int(cfg.sample_seed),
        prediction_batch_size=int(cfg.prediction_batch_size),
    )


def validate_diagnostic_config(cfg: OracleDiagnosticConfig) -> Dict[str, object]:
    issues: List[str] = []
    if len(cfg.sample_abs_y_edges) < 2 or not all(
        cfg.sample_abs_y_edges[idx] < cfg.sample_abs_y_edges[idx + 1] for idx in range(len(cfg.sample_abs_y_edges) - 1)
    ):
        issues.append("sample_abs_y_edges must be strictly increasing")
    if cfg.sample_abs_y_edges[0] != 0.0 or cfg.sample_abs_y_edges[-1] != 1.0:
        issues.append("sample_abs_y_edges must span [0.0, 1.0]")
    if cfg.sample_per_band <= 0:
        issues.append("sample_per_band must be positive")
    if len(cfg.err_quantiles) < 1 or not all(0.0 < q < 1.0 for q in cfg.err_quantiles):
        issues.append("err_quantiles must lie inside (0, 1)")
    if sorted(cfg.err_quantiles) != list(cfg.err_quantiles):
        issues.append("err_quantiles must be increasing")
    if len(cfg.oracle_scales) < 2 or not all(scale > 0.0 for scale in cfg.oracle_scales):
        issues.append("oracle_scales must be positive")
    scale_keys = [int(round(scale)) for scale in cfg.oracle_scales]
    if len(set(scale_keys)) != len(scale_keys):
        issues.append("oracle_scales must remain unique after rounding to integer keys")
    sw_report = sw_lab.validate_stability_weight_config(build_stockfish_cfg(cfg))
    report = {
        "is_valid": bool(not issues and sw_report["is_valid"]),
        "issues": issues,
        "stockfish_validation": sw_report,
    }
    if issues:
        raise ValueError("Invalid OracleDiagnosticConfig: " + "; ".join(issues))
    return report


resolve_split_shards = sw_lab.resolve_split_shards
shard_prediction_path = sw_lab.shard_prediction_path


def benchmark_single_train_step(
    init_ckpt_path: str | Path,
    data_root: str | Path,
    device: torch.device,
    cfg: OracleDiagnosticConfig,
) -> Dict[str, float]:
    return sw_lab.benchmark_single_train_step(
        init_ckpt_path=init_ckpt_path,
        data_root=data_root,
        device=device,
        batch_size=cfg.benchmark_train_batch_size,
        num_shards=1,
    )


def precompute_split_prediction_cache(
    init_ckpt_path: str | Path,
    data_root: str | Path,
    split: str,
    output_dir: str | Path,
    device: torch.device,
    batch_size: int,
    num_shards: Optional[int] = None,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    cache_dir = output_dir / "cache" / "split_preds"
    reports_dir = output_dir / "reports"
    cache_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model, _ = base_lab.load_model_from_checkpoint(init_ckpt_path, device=device)
    shard_rows = resolve_split_shards(data_root, split, num_shards=num_shards)
    init_ckpt_path = str(Path(init_ckpt_path).resolve())
    manifest_path = reports_dir / f"{split}_pred_cache_manifest.json"
    existing_manifest: Dict[str, object] = {}
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_compatible = bool(
        existing_manifest.get("init_ckpt_path") == init_ckpt_path and existing_manifest.get("split") == split
    )

    manifest_rows: List[dict] = []
    t0 = time.time()
    for rank, (shard_id, x_path, y_path) in enumerate(shard_rows):
        X = np.load(x_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r")
        pred_path = shard_prediction_path(cache_dir, shard_id)
        need_write = True
        if cache_compatible and pred_path.exists():
            cached = np.load(pred_path, mmap_mode="r")
            if cached.shape[0] == y.shape[0]:
                need_write = False
        if need_write:
            preds = base_lab.predict_array(
                model,
                np.asarray(X),
                device=device,
                batch_size=batch_size,
                use_amp=True,
                progress_name=f"{split}_pred_cache_{shard_id:05d}",
            )
            np.save(pred_path, preds.astype(np.float16))
        manifest_rows.append(
            {
                "split": split,
                "shard_id": int(shard_id),
                "samples": int(y.shape[0]),
                "pred_path": str(pred_path),
                "cached": bool(not need_write),
            }
        )
        manifest = {
            "split": split,
            "init_ckpt_path": init_ckpt_path,
            "num_shards": int(-1 if num_shards is None else num_shards),
            "batch_size": int(batch_size),
            "num_cached_shards": int(len(manifest_rows)),
            "cache_compatible_with_existing_manifest": bool(cache_compatible),
            "elapsed_sec": float(time.time() - t0),
        }
        save_dataframe(pd.DataFrame(manifest_rows), reports_dir / f"{split}_pred_cache_manifest.csv")
        save_json(manifest, manifest_path)
        if rank % 4 == 0:
            print(f"[{split}-pred-cache] shards={rank + 1}/{len(shard_rows)} elapsed={time.time() - t0:.1f}s")
    return {"manifest": manifest, "rows": pd.DataFrame(manifest_rows), "cache_dir": cache_dir}


def load_candidate_table(
    data_root: str | Path,
    pred_cache_dir: str | Path,
    split: str,
    num_shards: Optional[int] = None,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for shard_id, _, y_path in resolve_split_shards(data_root, split, num_shards=num_shards):
        y = np.load(y_path, mmap_mode="r").astype(np.float32, copy=False)
        pred = np.load(shard_prediction_path(Path(pred_cache_dir), shard_id), mmap_mode="r").astype(np.float32, copy=False)
        local_df = pd.DataFrame(
            {
                "shard_id": int(shard_id),
                "local_index": np.arange(y.shape[0], dtype=np.int32),
                "target_y": y.astype(np.float64),
                "teacher_pred": pred.astype(np.float64),
            }
        )
        local_df["abs_y"] = np.abs(local_df["target_y"])
        local_df["teacher_abs_err"] = np.abs(local_df["teacher_pred"] - local_df["target_y"])
        rows.append(local_df)
    return pd.concat(rows, ignore_index=True)


def allocate_evenly_with_capacity(counts: Sequence[int], total: int) -> np.ndarray:
    counts_arr = np.asarray(counts, dtype=np.int64)
    total = int(min(total, int(counts_arr.sum())))
    out = np.zeros_like(counts_arr)
    if total <= 0 or counts_arr.sum() <= 0:
        return out
    active = counts_arr > 0
    while total > 0 and np.any(active):
        share = max(1, total // max(1, int(np.sum(active))))
        progressed = False
        for idx in np.flatnonzero(active):
            give = int(min(share, counts_arr[idx] - out[idx], total))
            if give > 0:
                out[idx] += give
                total -= give
                progressed = True
            if total <= 0:
                break
        active = out < counts_arr
        if not progressed:
            break
    if total > 0:
        spare = counts_arr - out
        for idx in np.argsort(-spare):
            if total <= 0:
                break
            give = int(min(spare[idx], total))
            if give > 0:
                out[idx] += give
                total -= give
    return out


def build_stratified_subset(
    data_root: str | Path,
    pred_cache_dir: str | Path,
    split: str,
    cfg: OracleDiagnosticConfig,
    num_shards: Optional[int] = None,
) -> Dict[str, object]:
    candidate_df = load_candidate_table(data_root, pred_cache_dir, split, num_shards=num_shards)
    edges = np.asarray(cfg.sample_abs_y_edges, dtype=np.float64)
    labels = sw_lab.band_labels_from_edges(edges)
    candidate_df["band_idx"] = np.clip(np.digitize(candidate_df["abs_y"], edges[1:-1], right=False), 0, len(labels) - 1)
    candidate_df["band_label"] = candidate_df["band_idx"].map({idx: label for idx, label in enumerate(labels)})
    count_table = (
        candidate_df.groupby(["band_idx", "band_label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("band_idx")
        .reset_index(drop=True)
    )

    rng = np.random.default_rng(cfg.sample_seed)
    sampled_parts: List[pd.DataFrame] = []
    quota_rows: List[dict] = []
    for band_idx, band_label in enumerate(labels):
        band_df = candidate_df[candidate_df["band_idx"] == band_idx].copy().reset_index(drop=True)
        if band_df.empty:
            continue
        q_edges = [0.0]
        for q in cfg.err_quantiles:
            q_edges.append(float(np.quantile(band_df["teacher_abs_err"].to_numpy(dtype=np.float64), q)))
        q_edges.append(float(band_df["teacher_abs_err"].max() + 1e-8))
        clean_edges: List[float] = []
        for value in q_edges:
            if not clean_edges or value > clean_edges[-1] + 1e-12:
                clean_edges.append(value)
        if len(clean_edges) < 2:
            clean_edges = [0.0, float(band_df["teacher_abs_err"].max() + 1e-8)]
        err_edges = np.asarray(clean_edges, dtype=np.float64)
        band_df["err_bin_id"] = np.clip(
            np.digitize(band_df["teacher_abs_err"].to_numpy(dtype=np.float64), err_edges[1:-1], right=False),
            0,
            len(err_edges) - 2,
        )
        err_counts = band_df.groupby("err_bin_id").size().reindex(range(len(err_edges) - 1), fill_value=0).to_numpy(dtype=np.int64)
        quotas = allocate_evenly_with_capacity(err_counts, cfg.sample_per_band)
        for err_bin_id, quota in enumerate(quotas):
            err_slice = band_df[band_df["err_bin_id"] == err_bin_id]
            quota_rows.append(
                {
                    "band_idx": int(band_idx),
                    "band_label": band_label,
                    "err_bin_id": int(err_bin_id),
                    "err_left": float(err_edges[err_bin_id]),
                    "err_right": float(err_edges[err_bin_id + 1]),
                    "count": int(err_slice.shape[0]),
                    "quota": int(quota),
                }
            )
            if quota <= 0 or err_slice.empty:
                continue
            chosen = rng.choice(err_slice.index.to_numpy(dtype=np.int64), size=min(int(quota), int(err_slice.shape[0])), replace=False)
            sampled_parts.append(err_slice.loc[np.sort(chosen)].copy())

    if not sampled_parts:
        raise RuntimeError("No subset samples selected")
    sampled_df = pd.concat(sampled_parts, ignore_index=True).sort_values(["band_idx", "teacher_abs_err"]).reset_index(drop=True)

    shard_rows = {shard_id: (x_path, y_path) for shard_id, x_path, y_path in resolve_split_shards(data_root, split, num_shards=num_shards)}
    samples: List[dict] = []
    for shard_id, shard_df in sampled_df.groupby("shard_id", sort=True):
        x_path, _ = shard_rows[int(shard_id)]
        X = np.load(x_path, mmap_mode="r")
        for row in shard_df.itertuples(index=False):
            samples.append(
                {
                    "split": split,
                    "shard_id": int(row.shard_id),
                    "local_index": int(row.local_index),
                    "band_idx": int(row.band_idx),
                    "band_label": str(row.band_label),
                    "err_bin_id": int(row.err_bin_id),
                    "x": np.array(X[int(row.local_index)], dtype=np.uint8, copy=True),
                    "target_y": float(row.target_y),
                    "teacher_pred": float(row.teacher_pred),
                    "teacher_abs_err": float(row.teacher_abs_err),
                }
            )
    quota_df = pd.DataFrame(quota_rows)
    sampled_summary = (
        sampled_df.groupby(["band_idx", "band_label"], as_index=False)
        .size()
        .rename(columns={"size": "selected"})
        .sort_values("band_idx")
        .reset_index(drop=True)
    )
    return {
        "candidate_table": candidate_df,
        "count_table": count_table,
        "quota_table": quota_df,
        "sampled_summary": sampled_summary,
        "samples": samples,
    }


def bandwise_normalized_rank(frame: pd.DataFrame, value_col: str) -> pd.Series:
    out = pd.Series(np.zeros(frame.shape[0], dtype=np.float64), index=frame.index)
    for _, group in frame.groupby("band_idx"):
        values = group[value_col].to_numpy(dtype=np.float64)
        out.loc[group.index] = sw_lab.normalized_rank(values)
    return out


def assign_stability_groups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("oracle_target_range_600", "oracle_target_std_600", "bestmove_changes", "sign_flips"):
        out[f"rank_{col}"] = bandwise_normalized_rank(out, col)
    out["band_instability_score"] = out[
        ["rank_oracle_target_range_600", "rank_oracle_target_std_600", "rank_bestmove_changes", "rank_sign_flips"]
    ].mean(axis=1)
    out["stability_group"] = "middle"
    for _, group in out.groupby("band_idx"):
        order = group.sort_values("band_instability_score").index.to_list()
        third = max(1, len(order) // 3)
        stable_idx = order[:third]
        unstable_idx = order[-third:]
        out.loc[stable_idx, "stability_group"] = "stable"
        out.loc[unstable_idx, "stability_group"] = "unstable"
    return out


def run_stockfish_oracle_on_subset(
    subset: Sequence[dict],
    cfg: OracleDiagnosticConfig,
    output_dir: str | Path,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    sf_cfg = build_stockfish_cfg(cfg)
    scale_keys = [int(round(scale)) for scale in cfg.oracle_scales]

    rows: List[dict] = []
    for pos, sample in enumerate(subset):
        board = sw_lab.sanitize_board_for_stockfish(base_lab.decode_tensor_to_board(sample["x"]))
        curve = sw_lab.run_stockfish_probe_curve(board.to_fen(), cfg=sf_cfg)
        ref_cp = float(curve["cp_values"][-1])
        ref_y_600 = cp_to_target_scale(ref_cp, 600.0)
        y_curve_600 = np.asarray([cp_to_target_scale(cp, 600.0) for cp in curve["cp_values"]], dtype=np.float64)
        row = {
            "split": sample["split"],
            "shard_id": int(sample["shard_id"]),
            "local_index": int(sample["local_index"]),
            "band_idx": int(sample["band_idx"]),
            "band_label": sample["band_label"],
            "err_bin_id": int(sample["err_bin_id"]),
            "fen": board.to_fen(),
            "target_y": float(sample["target_y"]),
            "teacher_pred": float(sample["teacher_pred"]),
            "teacher_abs_err": float(sample["teacher_abs_err"]),
            "oracle_reference_node_budget": int(cfg.stockfish_node_budgets[-1]),
            "oracle_reference_cp": ref_cp,
            "oracle_reference_y_600": ref_y_600,
            "teacher_vs_train_abs": float(abs(sample["teacher_pred"] - sample["target_y"])),
            "teacher_vs_oracle_abs_600": float(abs(sample["teacher_pred"] - ref_y_600)),
            "train_vs_oracle_abs_600": float(abs(sample["target_y"] - ref_y_600)),
            "oracle_target_range_600": float(np.max(y_curve_600) - np.min(y_curve_600)),
            "oracle_target_std_600": float(np.std(y_curve_600)),
            "oracle_cp_range": float(curve["cp_range"]),
            "bestmove_changes": int(curve["bestmove_changes"]),
            "sign_flips": int(curve["sign_flips"]),
        }
        for scale_key, scale in zip(scale_keys, cfg.oracle_scales):
            ref_y = cp_to_target_scale(ref_cp, scale)
            row[f"oracle_reference_y_s{scale_key}"] = ref_y
            row[f"teacher_vs_oracle_abs_s{scale_key}"] = float(abs(sample["teacher_pred"] - ref_y))
            row[f"train_vs_oracle_abs_s{scale_key}"] = float(abs(sample["target_y"] - ref_y))
        for probe_row in curve["rows"]:
            node_budget = int(probe_row["node_budget"])
            y_600 = cp_to_target_scale(float(probe_row["cp_equivalent"]), 600.0)
            row[f"oracle_cp_n{node_budget}"] = float(probe_row["cp_equivalent"])
            row[f"oracle_y600_n{node_budget}"] = y_600
            row[f"oracle_bestmove_n{node_budget}"] = str(probe_row["bestmove"])
            row[f"train_vs_oracle_abs_n{node_budget}"] = float(abs(sample["target_y"] - y_600))
        rows.append(row)
        if pos % 12 == 0:
            print(f"[oracle-subset] processed={pos + 1}/{len(subset)}")
    df = assign_stability_groups(pd.DataFrame(rows))
    df["teacher_closer_to_oracle_than_train_600"] = df["teacher_vs_oracle_abs_600"] < df["train_vs_oracle_abs_600"]
    df["teacher_oracle_sign_match"] = np.sign(df["teacher_pred"]) == np.sign(df["oracle_reference_y_600"])
    df["train_oracle_sign_match"] = np.sign(df["target_y"]) == np.sign(df["oracle_reference_y_600"])
    save_dataframe(df, reports_dir / "oracle_subset_rows.csv")
    return df


def oracle_false_decisive_rate(y_ref: np.ndarray, pred: np.ndarray, y_thr: float, p_thr: float) -> Dict[str, float]:
    mask = np.abs(y_ref) <= y_thr
    if not np.any(mask):
        return {"n": 0, "rate": float("nan"), "wrong_sign_rate": float("nan")}
    yy = y_ref[mask]
    pp = pred[mask]
    decisive = np.abs(pp) >= p_thr
    wrong_sign = decisive & (np.sign(pp) != np.sign(yy))
    return {
        "n": int(mask.sum()),
        "rate": float(np.mean(decisive)),
        "wrong_sign_rate": float(np.mean(wrong_sign)),
    }


def summarize_by_band(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for band_label, group in df.groupby("band_label", sort=False):
        y_ref = group["oracle_reference_y_600"].to_numpy(dtype=np.float64)
        pred = group["teacher_pred"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "band_label": band_label,
                "n": int(group.shape[0]),
                "teacher_vs_train_mae": float(group["teacher_vs_train_abs"].mean()),
                "teacher_vs_oracle_mae_600": float(group["teacher_vs_oracle_abs_600"].mean()),
                "train_vs_oracle_mae_600": float(group["train_vs_oracle_abs_600"].mean()),
                "teacher_closer_to_oracle_rate_600": float(group["teacher_closer_to_oracle_than_train_600"].mean()),
                "teacher_oracle_sign_match_rate": float(group["teacher_oracle_sign_match"].mean()),
                "train_oracle_sign_match_rate": float(group["train_oracle_sign_match"].mean()),
                "oracle_false_0.1_0.3": oracle_false_decisive_rate(y_ref, pred, 0.1, 0.3)["rate"],
                "oracle_false_0.2_0.4": oracle_false_decisive_rate(y_ref, pred, 0.2, 0.4)["rate"],
                "mean_band_instability_score": float(group["band_instability_score"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_budget(df: pd.DataFrame, cfg: OracleDiagnosticConfig) -> pd.DataFrame:
    rows: List[dict] = []
    for band_label, group in df.groupby("band_label", sort=False):
        for node_budget in cfg.stockfish_node_budgets:
            rows.append(
                {
                    "band_label": band_label,
                    "node_budget": int(node_budget),
                    "train_vs_oracle_budget_mae": float(group[f"train_vs_oracle_abs_n{int(node_budget)}"].mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize_stability_groups(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for (band_label, stability_group), group in df.groupby(["band_label", "stability_group"], sort=False):
        y_ref = group["oracle_reference_y_600"].to_numpy(dtype=np.float64)
        pred = group["teacher_pred"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "band_label": band_label,
                "stability_group": stability_group,
                "n": int(group.shape[0]),
                "teacher_vs_oracle_mae_600": float(group["teacher_vs_oracle_abs_600"].mean()),
                "train_vs_oracle_mae_600": float(group["train_vs_oracle_abs_600"].mean()),
                "teacher_closer_to_oracle_rate_600": float(group["teacher_closer_to_oracle_than_train_600"].mean()),
                "teacher_oracle_sign_match_rate": float(group["teacher_oracle_sign_match"].mean()),
                "train_oracle_sign_match_rate": float(group["train_oracle_sign_match"].mean()),
                "oracle_false_0.1_0.3": oracle_false_decisive_rate(y_ref, pred, 0.1, 0.3)["rate"],
                "oracle_false_0.2_0.4": oracle_false_decisive_rate(y_ref, pred, 0.2, 0.4)["rate"],
                "mean_band_instability_score": float(group["band_instability_score"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_scale_sweep(df: pd.DataFrame, cfg: OracleDiagnosticConfig) -> pd.DataFrame:
    rows: List[dict] = []
    groups = {
        "overall": np.ones(df.shape[0], dtype=bool),
        "stable": df["stability_group"].to_numpy(dtype=object) == "stable",
        "stable_0.2": (df["stability_group"].to_numpy(dtype=object) == "stable") & (np.abs(df["oracle_reference_y_600"].to_numpy(dtype=np.float64)) <= 0.2),
        "stable_0.7": (df["stability_group"].to_numpy(dtype=object) == "stable") & (np.abs(df["oracle_reference_y_600"].to_numpy(dtype=np.float64)) <= 0.7),
    }
    for scale in cfg.oracle_scales:
        scale_key = int(round(scale))
        oracle_col = f"oracle_reference_y_s{scale_key}"
        oracle_y = df[oracle_col].to_numpy(dtype=np.float64)
        pred = df["teacher_pred"].to_numpy(dtype=np.float64)
        for group_name, mask in groups.items():
            if not np.any(mask):
                continue
            yy = oracle_y[mask]
            pp = pred[mask]
            rows.append(
                {
                    "group": group_name,
                    "scale": float(scale),
                    "n": int(mask.sum()),
                    "mse": float(np.mean((pp - yy) ** 2)),
                    "mae": float(np.mean(np.abs(pp - yy))),
                }
            )
    return pd.DataFrame(rows)


def make_root_cause_summary(df: pd.DataFrame, scale_sweep: pd.DataFrame) -> Dict[str, object]:
    stable_mask = df["stability_group"].to_numpy(dtype=object) == "stable"
    unstable_mask = df["stability_group"].to_numpy(dtype=object) == "unstable"
    near_mask = np.abs(df["oracle_reference_y_600"].to_numpy(dtype=np.float64)) <= 0.2
    stable_near = stable_mask & near_mask
    unstable_near = unstable_mask & near_mask
    stable_07 = stable_mask & (np.abs(df["oracle_reference_y_600"].to_numpy(dtype=np.float64)) <= 0.7)
    best_scale_table = scale_sweep.sort_values(["group", "mae", "scale"]).groupby("group", as_index=False).first()
    best_scale_map = {
        row["group"]: {"scale": float(row["scale"]), "mae": float(row["mae"]), "mse": float(row["mse"])}
        for _, row in best_scale_table.iterrows()
    }
    slope_stable_07 = float("nan")
    if np.any(stable_07):
        slope_stable_07 = float(
            base_lab.fit_line(
                df.loc[stable_07, "oracle_reference_y_600"].to_numpy(dtype=np.float64),
                df.loc[stable_07, "teacher_pred"].to_numpy(dtype=np.float64),
            )[0]
        )
    teacher_oracle_abs = df["teacher_vs_oracle_abs_600"].to_numpy(dtype=np.float64)
    train_oracle_abs = df["train_vs_oracle_abs_600"].to_numpy(dtype=np.float64)
    instability = df["band_instability_score"].to_numpy(dtype=np.float64)

    def masked_mean(mask: np.ndarray, values: np.ndarray) -> float:
        return float(np.mean(values[mask])) if np.any(mask) else float("nan")

    def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.size <= 1 or b.size <= 1 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "n_total": int(df.shape[0]),
        "teacher_closer_to_oracle_rate_600_overall": float(df["teacher_closer_to_oracle_than_train_600"].mean()),
        "teacher_closer_to_oracle_rate_600_near_zero": masked_mean(near_mask, df["teacher_closer_to_oracle_than_train_600"].to_numpy(dtype=np.float64)),
        "train_vs_oracle_mae_600_near_zero": masked_mean(near_mask, train_oracle_abs),
        "teacher_vs_oracle_mae_600_near_zero": masked_mean(near_mask, teacher_oracle_abs),
        "stable_near_teacher_vs_oracle_mae_600": masked_mean(stable_near, teacher_oracle_abs),
        "unstable_near_teacher_vs_oracle_mae_600": masked_mean(unstable_near, teacher_oracle_abs),
        "stable_near_train_vs_oracle_mae_600": masked_mean(stable_near, train_oracle_abs),
        "unstable_near_train_vs_oracle_mae_600": masked_mean(unstable_near, train_oracle_abs),
        "stable_0.7_slope_600": slope_stable_07,
        "corr_teacher_oracle_vs_label_oracle_600": safe_corr(teacher_oracle_abs, train_oracle_abs),
        "corr_instability_vs_teacher_oracle_600": safe_corr(instability, teacher_oracle_abs),
        "corr_instability_vs_train_oracle_600": safe_corr(instability, train_oracle_abs),
        "best_scale_by_group": best_scale_map,
    }


def run_diagnostic_analysis(
    df: pd.DataFrame,
    cfg: OracleDiagnosticConfig,
    output_dir: str | Path,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    reports_dir = output_dir / "reports"
    plots_dir = output_dir / "plots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    band_summary = summarize_by_band(df)
    budget_summary = summarize_by_budget(df, cfg)
    stability_summary = summarize_stability_groups(df)
    scale_sweep = summarize_scale_sweep(df, cfg)
    overall_bucket = base_lab.compute_bucket_table(
        df["oracle_reference_y_600"].to_numpy(dtype=np.float64),
        df["teacher_pred"].to_numpy(dtype=np.float64),
    )
    stable_bucket = base_lab.compute_bucket_table(
        df.loc[df["stability_group"] == "stable", "oracle_reference_y_600"].to_numpy(dtype=np.float64),
        df.loc[df["stability_group"] == "stable", "teacher_pred"].to_numpy(dtype=np.float64),
    )
    summary = make_root_cause_summary(df, scale_sweep)

    save_dataframe(band_summary, reports_dir / "oracle_band_summary.csv")
    save_dataframe(budget_summary, reports_dir / "oracle_budget_alignment.csv")
    save_dataframe(stability_summary, reports_dir / "oracle_stability_summary.csv")
    save_dataframe(scale_sweep, reports_dir / "oracle_scale_sweep.csv")
    save_dataframe(overall_bucket, reports_dir / "oracle_teacher_bucket_table.csv")
    save_dataframe(stable_bucket, reports_dir / "oracle_teacher_stable_bucket_table.csv")
    save_json(summary, reports_dir / "oracle_root_cause_summary.json")

    plt.figure(figsize=(7, 4))
    plt.scatter(df["train_vs_oracle_abs_600"], df["teacher_vs_oracle_abs_600"], s=16, alpha=0.65)
    lim = float(max(df["train_vs_oracle_abs_600"].max(), df["teacher_vs_oracle_abs_600"].max()) * 1.05)
    plt.plot([0.0, lim], [0.0, lim], linestyle="--", color="black", linewidth=1.0)
    plt.xlabel("train label vs oracle abs error")
    plt.ylabel("teacher vs oracle abs error")
    plt.title("Teacher error vs label disagreement")
    plt.tight_layout()
    plt.savefig(plots_dir / "teacher_vs_label_disagreement.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    sweep_plot = scale_sweep[scale_sweep["group"].isin(["overall", "stable_0.2", "stable_0.7"])]
    for group_name, group in sweep_plot.groupby("group"):
        plt.plot(group["scale"], group["mae"], marker="o", label=group_name)
    plt.xlabel("oracle tanh scale")
    plt.ylabel("teacher vs oracle MAE")
    plt.title("Scale sweep against fixed-node oracle")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "oracle_scale_sweep_mae.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    stable_mask = df["stability_group"] == "stable"
    unstable_mask = df["stability_group"] == "unstable"
    plt.scatter(df.loc[stable_mask, "oracle_reference_y_600"], df.loc[stable_mask, "teacher_pred"], s=18, alpha=0.7, label="stable")
    plt.scatter(df.loc[unstable_mask, "oracle_reference_y_600"], df.loc[unstable_mask, "teacher_pred"], s=18, alpha=0.7, label="unstable")
    plt.xlabel("oracle reference y (scale=600)")
    plt.ylabel("teacher prediction")
    plt.title("Teacher vs oracle on stable / unstable subset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "teacher_vs_oracle_stable_unstable.png", dpi=180)
    plt.close()
    return {
        "band_summary": band_summary,
        "budget_summary": budget_summary,
        "stability_summary": stability_summary,
        "scale_sweep": scale_sweep,
        "overall_bucket": overall_bucket,
        "stable_bucket": stable_bucket,
        "summary": summary,
    }
