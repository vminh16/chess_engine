from __future__ import annotations

import gc
import json
import shutil
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AB_DIR = PROJECT_ROOT / "experiments" / "root_cause_ablation_suite"
OR_DIR = PROJECT_ROOT / "experiments" / "oracle_root_cause_diagnostic"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(AB_DIR) not in sys.path:
    sys.path.insert(0, str(AB_DIR))
if str(OR_DIR) not in sys.path:
    sys.path.insert(0, str(OR_DIR))

import root_cause_ablation_helpers as ab_lab  # noqa: E402
import oracle_diagnostic_helpers as or_lab  # noqa: E402


SOURCE_TARGET_SCALE = ab_lab.SOURCE_TARGET_SCALE
CANONICAL_Y600_BANDS = ab_lab.CANONICAL_Y600_BANDS
CENTER_FALSE_SMALL_KEY = "|oracle|<=0.05,|pred|>=0.10"
CENTER_FALSE_MEDIUM_KEY = "|oracle|<=0.05,|pred|>=0.20"

save_json = ab_lab.save_json
save_dataframe = ab_lab.save_dataframe
set_global_seed = ab_lab.set_global_seed
choose_device = ab_lab.choose_device
base_lab = ab_lab.base_lab


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return isinstance(exc, RuntimeError) and ("out of memory" in text or "cuda error" in text)


def _is_checkpoint_write_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return isinstance(exc, RuntimeError) and (
        "pytorchstreamwriter" in text
        or "file write failed" in text
        or "unexpected pos" in text
        or "inline_container" in text
    )


@dataclass
class ExperimentPaths:
    project_root: str
    run_dir: str
    data_root: str
    experiment_dir: str
    output_dir: str
    reports_dir: str
    plots_dir: str
    cache_dir: str
    runs_dir: str
    replicate_dir: str


@dataclass
class BootstrapConfig:
    n_bootstrap: int = 4000
    seed: int = 123
    ci_alpha: float = 0.05


@dataclass
class ReplicateOracleConfig:
    split: str = "test"
    sample_abs_y_edges: Tuple[float, ...] = CANONICAL_Y600_BANDS
    sample_per_band: int = 24
    err_quantiles: Tuple[float, ...] = (1.0 / 3.0, 2.0 / 3.0)
    num_replicates: int = 2
    base_seed: int = 777
    oracle_scales: Tuple[float, ...] = (400.0, 600.0, 800.0, 1200.0)
    stockfish_path: str = r"D:\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
    stockfish_threads: int = 1
    stockfish_hash_mb: int = 32
    stockfish_node_budgets: Tuple[int, ...] = (8_000, 32_000, 128_000)
    stockfish_command_pause_ms: int = 50
    stockfish_timeout_sec: float = 20.0
    prediction_batch_size: int = 2048
    decode_validation_samples: int = 32
    subset_num_shards: Optional[int] = None


def build_default_paths(
    run_dir: str | Path = r"C:\Users\USER\Downloads\dgrn_5m_v3_stage2_polish_run1",
    data_root: str | Path = r"C:\Users\USER\Desktop\chess_engine\data\process",
    experiment_dir: str | Path | None = None,
) -> Dict[str, Path]:
    if experiment_dir is None:
        experiment_dir = PROJECT_ROOT / "experiments" / "objective_resolution_suite"
    paths = {
        "project_root": PROJECT_ROOT,
        "run_dir": Path(run_dir),
        "data_root": Path(data_root),
        "experiment_dir": Path(experiment_dir),
    }
    paths["output_dir"] = paths["experiment_dir"] / "outputs"
    paths["reports_dir"] = paths["output_dir"] / "reports"
    paths["plots_dir"] = paths["output_dir"] / "plots"
    paths["cache_dir"] = paths["output_dir"] / "cache"
    paths["runs_dir"] = paths["output_dir"] / "runs"
    paths["replicate_dir"] = paths["output_dir"] / "replicates"
    for key in ("experiment_dir", "output_dir", "reports_dir", "plots_dir", "cache_dir", "runs_dir", "replicate_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def _free_space_gb(path: str | Path) -> float:
    usage = shutil.disk_usage(str(Path(path)))
    return float(usage.free / 1024**3)


def export_paths_json(paths: Dict[str, Path], out_path: Path) -> None:
    payload = ExperimentPaths(**{key: str(value) for key, value in paths.items()})
    out_path.write_text(json.dumps(asdict(payload), indent=2), encoding="utf-8")


def validate_runtime_paths(
    paths: Dict[str, Path],
    baseline_ckpt: str | Path,
    existing_suite_output_dir: str | Path,
    existing_labels: Sequence[str],
    stockfish_path: str | Path,
) -> Dict[str, object]:
    baseline_ckpt = Path(baseline_ckpt)
    existing_suite_output_dir = Path(existing_suite_output_dir)
    stockfish_path = Path(stockfish_path)
    required_dirs = {
        "project_root": paths["project_root"],
        "data_root": paths["data_root"],
        "train_dir": paths["data_root"] / "train",
        "val_dir": paths["data_root"] / "val",
        "test_dir": paths["data_root"] / "test",
        "existing_suite_output_dir": existing_suite_output_dir,
    }
    missing_dirs = [name for name, path in required_dirs.items() if not Path(path).exists()]
    missing_files: List[str] = []
    if not baseline_ckpt.exists():
        missing_files.append(str(baseline_ckpt))
    if not stockfish_path.exists():
        missing_files.append(str(stockfish_path))
    missing_existing_runs: List[str] = []
    for label in existing_labels:
        ckpt_path = existing_suite_output_dir / "runs" / str(label) / "checkpoints" / f"{label}_best.pt"
        if not ckpt_path.exists():
            missing_existing_runs.append(str(ckpt_path))
    oracle_subset_csv = Path(ab_lab.OracleEvalConfig().subset_csv_path)
    if not oracle_subset_csv.exists():
        missing_files.append(str(oracle_subset_csv))
    if missing_dirs or missing_files or missing_existing_runs:
        problems: List[str] = []
        if missing_dirs:
            problems.append("missing dirs: " + ", ".join(missing_dirs))
        if missing_files:
            problems.append("missing files: " + ", ".join(missing_files))
        if missing_existing_runs:
            problems.append("missing existing run checkpoints: " + ", ".join(missing_existing_runs))
        raise FileNotFoundError("Runtime path validation failed: " + " | ".join(problems))
    return {
        "is_valid": True,
        "baseline_ckpt": str(baseline_ckpt),
        "stockfish_path": str(stockfish_path),
        "existing_labels": list(existing_labels),
    }


def maybe_redirect_followup_paths(
    paths: Dict[str, Path],
    min_free_gb: float = 2.0,
    fallback_experiment_dir: str | Path = r"D:\chess_engine_experiments\objective_resolution_suite",
) -> Dict[str, object]:
    current_output_dir = Path(paths["output_dir"])
    free_gb = _free_space_gb(current_output_dir)
    if free_gb >= float(min_free_gb):
        report = {
            "redirected": False,
            "free_gb": float(free_gb),
            "output_dir": str(current_output_dir),
        }
        save_json(report, paths["reports_dir"] / "followup_output_redirect.json")
        return report

    new_paths = build_default_paths(
        run_dir=paths["run_dir"],
        data_root=paths["data_root"],
        experiment_dir=Path(fallback_experiment_dir),
    )
    moved_runs: List[str] = []
    old_runs_dir = Path(paths["runs_dir"])
    if old_runs_dir.exists():
        for child in old_runs_dir.iterdir():
            dst = new_paths["runs_dir"] / child.name
            if dst.exists():
                continue
            try:
                shutil.move(str(child), str(dst))
                moved_runs.append(child.name)
            except OSError:
                pass
    paths.clear()
    paths.update(new_paths)
    report = {
        "redirected": True,
        "reason": "low_free_space",
        "free_gb_before_redirect": float(free_gb),
        "output_dir": str(paths["output_dir"]),
        "moved_runs": moved_runs,
    }
    save_json(report, paths["reports_dir"] / "followup_output_redirect.json")
    return report


def validate_bootstrap_config(cfg: BootstrapConfig) -> Dict[str, object]:
    issues: List[str] = []
    if cfg.n_bootstrap <= 0:
        issues.append("n_bootstrap must be positive")
    if not (0.0 < cfg.ci_alpha < 1.0):
        issues.append("ci_alpha must be in (0, 1)")
    if issues:
        raise ValueError("Invalid BootstrapConfig: " + "; ".join(issues))
    return {"is_valid": True, "n_bootstrap": int(cfg.n_bootstrap), "ci_alpha": float(cfg.ci_alpha)}


def build_replicate_oracle_cfg(cfg: ReplicateOracleConfig, seed: int) -> or_lab.OracleDiagnosticConfig:
    return or_lab.OracleDiagnosticConfig(
        split=str(cfg.split),
        sample_abs_y_edges=tuple(float(x) for x in cfg.sample_abs_y_edges),
        sample_per_band=int(cfg.sample_per_band),
        err_quantiles=tuple(float(x) for x in cfg.err_quantiles),
        oracle_scales=tuple(float(x) for x in cfg.oracle_scales),
        stockfish_path=str(cfg.stockfish_path),
        stockfish_threads=int(cfg.stockfish_threads),
        stockfish_hash_mb=int(cfg.stockfish_hash_mb),
        stockfish_node_budgets=tuple(int(x) for x in cfg.stockfish_node_budgets),
        stockfish_command_pause_ms=int(cfg.stockfish_command_pause_ms),
        stockfish_timeout_sec=float(cfg.stockfish_timeout_sec),
        prediction_batch_size=int(cfg.prediction_batch_size),
        sample_seed=int(seed),
        benchmark_train_batch_size=640,
        decode_validation_samples=int(cfg.decode_validation_samples),
        subset_num_shards=cfg.subset_num_shards,
    )


def validate_replicate_config(cfg: ReplicateOracleConfig) -> Dict[str, object]:
    issues: List[str] = []
    if cfg.num_replicates <= 0:
        issues.append("num_replicates must be positive")
    if cfg.sample_per_band <= 0:
        issues.append("sample_per_band must be positive")
    inner = or_lab.validate_diagnostic_config(build_replicate_oracle_cfg(cfg, seed=cfg.base_seed))
    if issues:
        raise ValueError("Invalid ReplicateOracleConfig: " + "; ".join(issues))
    return {
        "is_valid": True,
        "num_replicates": int(cfg.num_replicates),
        "sample_per_band": int(cfg.sample_per_band),
        "stockfish_validation": inner["stockfish_validation"],
    }


def autotune_train_batch_size(
    init_ckpt_path: str | Path,
    data_root: str | Path,
    device: torch.device,
    preferred_batch_size: int,
    min_batch_size: int = 128,
    step: int = 64,
    max_mem_ratio: float = 0.85,
) -> Dict[str, object]:
    candidates = []
    current = int(preferred_batch_size)
    while current >= int(min_batch_size):
        candidates.append(int(current))
        current -= int(step)
    if candidates[-1] != int(min_batch_size):
        candidates.append(int(min_batch_size))
    candidates = sorted(set(candidates), reverse=True)

    attempts: List[dict] = []
    total_mem_gb = None
    if device.type == "cuda":
        total_mem_gb = float(torch.cuda.get_device_properties(device).total_memory / 1024**3)

    selected_report: Optional[Dict[str, float]] = None
    selected_batch = None
    for batch_size in candidates:
        _cleanup_cuda()
        try:
            report = ab_lab.benchmark_single_train_step(
                init_ckpt_path=init_ckpt_path,
                data_root=data_root,
                device=device,
                batch_size=batch_size,
                num_shards=1,
            )
            mem_ratio = None
            if total_mem_gb is not None and total_mem_gb > 0.0:
                mem_ratio = float(report["peak_mem_gb"] / total_mem_gb)
            attempts.append(
                {
                    "batch_size": int(batch_size),
                    "ok": True,
                    "peak_mem_gb": float(report["peak_mem_gb"]),
                    "mem_ratio": None if mem_ratio is None else float(mem_ratio),
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
        raise RuntimeError(
            "Unable to find a safe train batch size. Attempts: "
            + json.dumps(attempts, ensure_ascii=False)
        )
    return {
        "selected_batch_size": int(selected_batch),
        "preferred_batch_size": int(preferred_batch_size),
        "min_batch_size": int(min_batch_size),
        "step": int(step),
        "max_mem_ratio": float(max_mem_ratio),
        "device": str(device),
        "total_mem_gb": None if total_mem_gb is None else float(total_mem_gb),
        "selected_report": selected_report,
        "attempts": attempts,
    }


def build_followup_variants() -> List[ab_lab.AblationVariant]:
    full_a1 = ab_lab.AblationVariant(
        name="L3_full_A1",
        description="Full A1 objective: inverse-curvature y-loss plus stronger z-space term.",
        target_scale=600.0,
        sampler_mode="random",
        loss_mode="curvature_compensated",
        y_loss_alpha=0.65,
        z_loss_beta=0.0,
        z_huber_delta=1.0,
        y_reweight_clip_max=4.0,
    )
    return [
        ab_lab.AblationVariant(
            name="L0_control_hybrid",
            description="Control run: continue stage2 objective with random sampler.",
            target_scale=600.0,
            sampler_mode="random",
            loss_mode="baseline_hybrid",
            lambda_y=0.99,
            z_loss_beta=1.0,
            z_huber_delta=0.5,
        ),
        ab_lab.AblationVariant(
            name="L1_z_strong_hybrid",
            description="Increase z-space role without inverse-curvature weighting.",
            target_scale=600.0,
            sampler_mode="random",
            loss_mode="baseline_hybrid",
            lambda_y=0.65,
            z_loss_beta=0.0,
            z_huber_delta=1.0,
        ),
        ab_lab.AblationVariant(
            name="L2_curvature_y_only",
            description="Inverse-curvature weighting only, with z branch disabled.",
            target_scale=600.0,
            sampler_mode="random",
            loss_mode="curvature_compensated",
            y_loss_alpha=1.0,
            z_loss_beta=0.0,
            z_huber_delta=1.0,
            y_reweight_clip_max=4.0,
        ),
        full_a1,
        replace(
            full_a1,
            name="L4_A1_plus_A2",
            description="Full A1 objective with band-balanced sampler.",
            sampler_mode="band_balanced",
        ),
        replace(
            full_a1,
            name="S1_A1_center_w020_m010",
            description="A1 with medium center penalty.",
            center_penalty_weight=0.20,
            center_penalty_tau_y600=0.05,
            center_penalty_margin_y600=0.10,
        ),
        replace(
            full_a1,
            name="S2_A1_center_w030_m012",
            description="A1 with stronger center penalty.",
            center_penalty_weight=0.30,
            center_penalty_tau_y600=0.05,
            center_penalty_margin_y600=0.12,
        ),
    ]


def build_variant_catalog() -> Dict[str, ab_lab.AblationVariant]:
    return {variant.name: variant for variant in build_followup_variants()}


def build_primary_oracle_bundle(data_root: str | Path) -> Dict[str, object]:
    return ab_lab.load_oracle_subset_bundle(ab_lab.OracleEvalConfig(), data_root=data_root)


def _oracle_center_spread_ratio(oracle_eval: Dict[str, object], thr: float = 0.05) -> float:
    return float(base_lab.center_spread_ratio(oracle_eval["oracle_targets"], oracle_eval["preds"], thr=thr)["ratio"])


def _stable_band_map(oracle_eval: Dict[str, object]) -> Dict[str, dict]:
    stable_df = oracle_eval["oracle_stable_band_summary"]
    return {str(row["band_label_y600"]): row.to_dict() for _, row in stable_df.iterrows()}


def selection_score_v2(primary: Dict[str, object]) -> float:
    return float(
        primary["oracle_midband_mae_sum_stable"]
        + 0.50 * max(0.0, 0.80 - primary["oracle_stable_0.7_slope"])
        + 0.10 * max(0.0, primary["oracle_center_amp_ratio"] - 3.0)
        + 0.50 * primary["oracle_center_false_0.1eq"]
        + 0.25 * primary["oracle_center_false_0.2eq"]
        + 0.20 * primary["oracle_center_wrong_sign_0.1eq"]
        + 0.10 * primary["oracle_center_wrong_sign_0.2eq"]
        + 0.10 * max(0.0, 0.70 - primary["oracle_band_sign_0.05_0.2_stable"])
        + 0.02 * max(0.0, primary["test_center_spread_ratio"] - 2.0)
        + 0.10 * max(0.0, primary["test_max_midband_abs_cal_gap"] - 0.20)
    )


def extract_primary_metrics(label: str, split_eval: Dict[str, object], oracle_eval: Dict[str, object]) -> Dict[str, object]:
    split_metrics = split_eval["metrics"]
    oracle_metrics = oracle_eval["standard_metrics"]
    oracle_summary = oracle_eval["summary"]
    band_map = _stable_band_map(oracle_eval)
    primary = {
        "label": label,
        "target_scale": float(oracle_summary["target_scale"]),
        "metric_scale": float(oracle_summary.get("metric_scale", SOURCE_TARGET_SCALE)),
        "test_mse_0.1eq": float(split_metrics["bands"]["0.10"]["mse"]),
        "test_mse_0.2eq": float(split_metrics["bands"]["0.20"]["mse"]),
        "test_mse_0.5eq": float(split_metrics["bands"]["0.50"]["mse"]),
        "test_mse_0.7eq": float(split_metrics["bands"]["0.70"]["mse"]),
        "test_slope_0.1eq": float(split_metrics["bands"]["0.10"]["slope"]),
        "test_slope_0.2eq": float(split_metrics["bands"]["0.20"]["slope"]),
        "test_slope_0.7eq": float(split_metrics["bands"]["0.70"]["slope"]),
        "test_center_false_0.1eq": float(split_metrics["center_false_decisive"][CENTER_FALSE_SMALL_KEY]["rate"]),
        "test_center_false_0.2eq": float(split_metrics["center_false_decisive"][CENTER_FALSE_MEDIUM_KEY]["rate"]),
        "test_center_wrong_sign_0.1eq": float(split_metrics["center_false_decisive"][CENTER_FALSE_SMALL_KEY]["wrong_sign_rate"]),
        "test_center_wrong_sign_0.2eq": float(split_metrics["center_false_decisive"][CENTER_FALSE_MEDIUM_KEY]["wrong_sign_rate"]),
        "test_center_spread_ratio": float(split_metrics["center_spread_ratio"]["ratio"]),
        "test_max_midband_abs_cal_gap": float(split_metrics["max_midband_abs_cal_gap"]),
        "oracle_teacher_mae": float(oracle_summary["overall_teacher_vs_oracle_mae"]),
        "oracle_closer_rate": float(oracle_summary["teacher_closer_to_oracle_rate"]),
        "oracle_stable_0.7_slope": float(oracle_summary["stable_0.7_slope"]),
        "oracle_midband_mae_sum_stable": float(oracle_summary["midband_teacher_vs_oracle_mae_sum_stable"]),
        "oracle_center_amp_ratio": float(oracle_summary["center_amp_ratio_eq_0.05"]),
        "oracle_center_false_0.1eq": float(oracle_summary["center_false_pred0.1eq"]),
        "oracle_center_false_0.2eq": float(oracle_summary["center_false_pred0.2eq"]),
        "oracle_center_wrong_sign_0.1eq": float(oracle_metrics["center_false_decisive"][CENTER_FALSE_SMALL_KEY]["wrong_sign_rate"]),
        "oracle_center_wrong_sign_0.2eq": float(oracle_metrics["center_false_decisive"][CENTER_FALSE_MEDIUM_KEY]["wrong_sign_rate"]),
        "oracle_center_spread_ratio": _oracle_center_spread_ratio(oracle_eval),
        "oracle_band_mae_0.05_0.2_stable": float(band_map["[0.050,0.200]"]["teacher_vs_oracle_mae"]),
        "oracle_band_mae_0.2_0.5_stable": float(band_map["[0.200,0.500]"]["teacher_vs_oracle_mae"]),
        "oracle_band_mae_0.5_0.7_stable": float(band_map["[0.500,0.700]"]["teacher_vs_oracle_mae"]),
        "oracle_band_amp_0_0.05_stable": float(band_map["[0.000,0.050]"]["amplitude_ratio"]),
        "oracle_band_amp_0.2_0.5_stable": float(band_map["[0.200,0.500]"]["amplitude_ratio"]),
        "oracle_band_amp_0.5_0.7_stable": float(band_map["[0.500,0.700]"]["amplitude_ratio"]),
        "oracle_band_sign_0_0.05_stable": float(band_map["[0.000,0.050]"]["teacher_oracle_sign_match_rate"]),
        "oracle_band_sign_0.05_0.2_stable": float(band_map["[0.050,0.200]"]["teacher_oracle_sign_match_rate"]),
        "oracle_band_sign_0.2_0.5_stable": float(band_map["[0.200,0.500]"]["teacher_oracle_sign_match_rate"]),
    }
    primary["selection_score_v2"] = selection_score_v2(primary)
    return primary


def add_decision_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    base_df = out[out["label"] == "baseline"]
    if base_df.empty:
        return out
    base = base_df.iloc[0]
    out["improves_midband"] = (
        (out["oracle_stable_0.7_slope"] >= base["oracle_stable_0.7_slope"])
        & (out["oracle_midband_mae_sum_stable"] <= base["oracle_midband_mae_sum_stable"])
    )
    out["improves_center_amplitude"] = (
        (out["oracle_center_amp_ratio"] <= base["oracle_center_amp_ratio"])
        & (out["oracle_center_false_0.1eq"] <= base["oracle_center_false_0.1eq"])
        & (out["oracle_center_false_0.2eq"] <= base["oracle_center_false_0.2eq"])
        & (out["test_center_spread_ratio"] <= base["test_center_spread_ratio"])
    )
    out["improves_center_direction"] = (
        (out["oracle_center_wrong_sign_0.1eq"] <= base["oracle_center_wrong_sign_0.1eq"])
        & (out["oracle_center_wrong_sign_0.2eq"] <= base["oracle_center_wrong_sign_0.2eq"])
        & (out["oracle_band_sign_0.05_0.2_stable"] >= base["oracle_band_sign_0.05_0.2_stable"])
        & (out["test_center_wrong_sign_0.1eq"] <= base["test_center_wrong_sign_0.1eq"])
        & (out["test_center_wrong_sign_0.2eq"] <= base["test_center_wrong_sign_0.2eq"])
    )
    out["improves_center"] = out["improves_center_amplitude"] & out["improves_center_direction"]
    out["dominates_baseline"] = out["improves_midband"] & out["improves_center"]
    return out


def evaluate_checkpoint(
    ckpt_path: str | Path,
    label: str,
    data_root: str | Path,
    oracle_bundle: Dict[str, object],
    oracle_cfg: ab_lab.OracleEvalConfig,
    train_cfg: ab_lab.TrainConfig,
    device: torch.device,
    target_scale: float,
) -> Dict[str, object]:
    model, _ = base_lab.load_model_from_checkpoint(ckpt_path, device=device)
    try:
        test_eval = ab_lab.evaluate_model_on_split_scale_aware(
            model=model,
            data_root=data_root,
            split="test",
            device=device,
            max_samples=train_cfg.test_max_samples,
            num_shards=train_cfg.test_num_shards,
            batch_size=max(train_cfg.batch_size, 1024),
            target_scale=target_scale,
            oracle_cfg=oracle_cfg,
        )
        oracle_eval = ab_lab.evaluate_model_on_oracle_subset(
            model=model,
            oracle_bundle=oracle_bundle,
            device=device,
            target_scale=target_scale,
            oracle_cfg=oracle_cfg,
        )
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    primary = extract_primary_metrics(label=label, split_eval=test_eval, oracle_eval=oracle_eval)
    return {
        "label": label,
        "checkpoint": str(Path(ckpt_path)),
        "target_scale": float(target_scale),
        "test_eval": test_eval,
        "oracle_eval": oracle_eval,
        "oracle_bundle": oracle_bundle,
        "primary": primary,
    }


def _infer_target_scale_from_checkpoint(ckpt_path: str | Path) -> float:
    ckpt = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    variant = ckpt.get("variant") or {}
    return float(variant.get("target_scale", SOURCE_TARGET_SCALE))


def build_checkpoint_registry(
    baseline_ckpt: str | Path,
    existing_suite_output_dir: str | Path,
    existing_labels: Sequence[str],
    followup_output_dir: str | Path,
    followup_labels: Sequence[str],
) -> List[Dict[str, object]]:
    baseline_ckpt = Path(baseline_ckpt)
    if not baseline_ckpt.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {baseline_ckpt}")
    rows = [{"label": "baseline", "checkpoint": str(baseline_ckpt), "target_scale": SOURCE_TARGET_SCALE}]
    existing_dir = Path(existing_suite_output_dir)
    missing_existing: List[str] = []
    for label in existing_labels:
        ckpt_path = existing_dir / "runs" / label / "checkpoints" / f"{label}_best.pt"
        if ckpt_path.exists():
            rows.append({"label": label, "checkpoint": str(ckpt_path), "target_scale": _infer_target_scale_from_checkpoint(ckpt_path)})
        else:
            missing_existing.append(str(ckpt_path))
    followup_dir = Path(followup_output_dir)
    missing_followup: List[str] = []
    for label in followup_labels:
        ckpt_path = followup_dir / "runs" / label / "checkpoints" / f"{label}_best.pt"
        if ckpt_path.exists():
            rows.append({"label": label, "checkpoint": str(ckpt_path), "target_scale": _infer_target_scale_from_checkpoint(ckpt_path)})
        else:
            missing_followup.append(str(ckpt_path))
    if missing_existing or missing_followup:
        problems: List[str] = []
        if missing_existing:
            problems.append("missing existing checkpoints: " + ", ".join(missing_existing))
        if missing_followup:
            problems.append("missing followup checkpoints: " + ", ".join(missing_followup))
        raise FileNotFoundError("Checkpoint registry validation failed: " + " | ".join(problems))
    return rows


def evaluate_registry(
    registry: Sequence[Dict[str, object]],
    data_root: str | Path,
    oracle_bundle: Dict[str, object],
    oracle_cfg: ab_lab.OracleEvalConfig,
    train_cfg: ab_lab.TrainConfig,
    device: torch.device,
    output_dir: str | Path,
    prefix: str,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    output_dir = Path(output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    results: Dict[str, Dict[str, object]] = {}
    for item in registry:
        result = evaluate_checkpoint(
            ckpt_path=item["checkpoint"],
            label=str(item["label"]),
            data_root=data_root,
            oracle_bundle=oracle_bundle,
            oracle_cfg=oracle_cfg,
            train_cfg=train_cfg,
            device=device,
            target_scale=float(item["target_scale"]),
        )
        results[str(item["label"])] = result
        rows.append(result["primary"])
    frame = add_decision_flags(pd.DataFrame(rows))
    save_dataframe(frame, reports_dir / f"{prefix}_primary_metrics.csv")
    return frame, results


def _oracle_metric_summary(
    oracle_y: np.ndarray,
    train_y: np.ndarray,
    pred_y: np.ndarray,
    stable_mask: np.ndarray,
    indices: np.ndarray,
) -> Dict[str, float]:
    yy = oracle_y[indices]
    tt = train_y[indices]
    pp = pred_y[indices]
    ss = stable_mask[indices]
    stable_07 = ss & (np.abs(yy) <= 0.70)
    center = np.abs(yy) <= 0.05
    decisive_01 = center & (np.abs(pp) >= 0.10)
    decisive_02 = center & (np.abs(pp) >= 0.20)
    wrong_01 = decisive_01 & (np.sign(pp) != np.sign(yy)) & (np.abs(yy) > 0.02)
    wrong_02 = decisive_02 & (np.sign(pp) != np.sign(yy)) & (np.abs(yy) > 0.02)
    stable_sign_005_02 = ss & (np.abs(yy) > 0.05) & (np.abs(yy) <= 0.20)
    stable_sign_02_05 = ss & (np.abs(yy) > 0.20) & (np.abs(yy) <= 0.50)
    mid_ranges = [(0.05, 0.20), (0.20, 0.50), (0.50, 0.70)]
    mid_sum = 0.0
    for left, right in mid_ranges:
        mask = ss & (np.abs(yy) > left) & (np.abs(yy) <= right)
        if np.any(mask):
            mid_sum += float(np.mean(np.abs(pp[mask] - yy[mask])))
    return {
        "overall_teacher_vs_oracle_mae": float(np.mean(np.abs(pp - yy))),
        "teacher_closer_to_oracle_rate": float(np.mean(np.abs(pp - yy) < np.abs(tt - yy))),
        "stable_0.7_slope": float(base_lab.fit_line(yy[stable_07], pp[stable_07])[0]) if np.any(stable_07) else float("nan"),
        "midband_teacher_vs_oracle_mae_sum_stable": mid_sum,
        "center_amp_ratio_eq_0.05": float(np.mean(np.abs(pp[center])) / max(np.mean(np.abs(yy[center])), 1e-12)) if np.any(center) else float("nan"),
        "center_false_pred0.1eq": float(np.mean(np.abs(pp[center]) >= 0.10)) if np.any(center) else float("nan"),
        "center_false_pred0.2eq": float(np.mean(np.abs(pp[center]) >= 0.20)) if np.any(center) else float("nan"),
        "center_wrong_sign_pred0.1eq": float(np.mean(wrong_01)) if np.any(center) else float("nan"),
        "center_wrong_sign_pred0.2eq": float(np.mean(wrong_02)) if np.any(center) else float("nan"),
        "center_spread_ratio_eq_0.05": float(np.std(pp[center]) / (np.std(yy[center]) + 1e-12)) if np.any(center) else float("nan"),
        "stable_sign_match_0.05_0.2": float(np.mean(np.sign(pp[stable_sign_005_02]) == np.sign(yy[stable_sign_005_02]))) if np.any(stable_sign_005_02) else float("nan"),
        "stable_sign_match_0.2_0.5": float(np.mean(np.sign(pp[stable_sign_02_05]) == np.sign(yy[stable_sign_02_05]))) if np.any(stable_sign_02_05) else float("nan"),
    }


def bootstrap_compare_to_baseline(
    baseline_result: Dict[str, object],
    candidate_result: Dict[str, object],
    cfg: BootstrapConfig,
) -> pd.DataFrame:
    baseline_rows = baseline_result["oracle_bundle"]["rows"]
    candidate_rows = candidate_result["oracle_bundle"]["rows"]
    if not baseline_rows[["shard_id", "local_index"]].reset_index(drop=True).equals(candidate_rows[["shard_id", "local_index"]].reset_index(drop=True)):
        raise RuntimeError("Bootstrap compare requires identical oracle subset ordering")

    oracle_y = baseline_result["oracle_eval"]["oracle_targets"].astype(np.float64)
    train_y = baseline_result["oracle_eval"]["train_targets"].astype(np.float64)
    baseline_pred = baseline_result["oracle_eval"]["preds"].astype(np.float64)
    candidate_pred = candidate_result["oracle_eval"]["preds"].astype(np.float64)
    stable_mask = baseline_rows["stability_group"].astype(str).eq("stable").to_numpy(dtype=bool)
    n = int(oracle_y.shape[0])

    metric_names = [
        "overall_teacher_vs_oracle_mae",
        "teacher_closer_to_oracle_rate",
        "stable_0.7_slope",
        "midband_teacher_vs_oracle_mae_sum_stable",
        "center_amp_ratio_eq_0.05",
        "center_false_pred0.1eq",
        "center_false_pred0.2eq",
        "center_wrong_sign_pred0.1eq",
        "center_wrong_sign_pred0.2eq",
        "center_spread_ratio_eq_0.05",
        "stable_sign_match_0.05_0.2",
        "stable_sign_match_0.2_0.5",
    ]
    better_direction = {
        "overall_teacher_vs_oracle_mae": "lower",
        "teacher_closer_to_oracle_rate": "higher",
        "stable_0.7_slope": "higher",
        "midband_teacher_vs_oracle_mae_sum_stable": "lower",
        "center_amp_ratio_eq_0.05": "lower",
        "center_false_pred0.1eq": "lower",
        "center_false_pred0.2eq": "lower",
        "center_wrong_sign_pred0.1eq": "lower",
        "center_wrong_sign_pred0.2eq": "lower",
        "center_spread_ratio_eq_0.05": "lower",
        "stable_sign_match_0.05_0.2": "higher",
        "stable_sign_match_0.2_0.5": "higher",
    }

    baseline_point = _oracle_metric_summary(oracle_y, train_y, baseline_pred, stable_mask, np.arange(n, dtype=np.int64))
    candidate_point = _oracle_metric_summary(oracle_y, train_y, candidate_pred, stable_mask, np.arange(n, dtype=np.int64))

    rng = np.random.default_rng(cfg.seed)
    deltas = {name: [] for name in metric_names}
    for _ in range(cfg.n_bootstrap):
        idx = rng.integers(0, n, size=n, endpoint=False)
        boot_base = _oracle_metric_summary(oracle_y, train_y, baseline_pred, stable_mask, idx)
        boot_cand = _oracle_metric_summary(oracle_y, train_y, candidate_pred, stable_mask, idx)
        for name in metric_names:
            deltas[name].append(float(boot_cand[name] - boot_base[name]))

    q_low = cfg.ci_alpha / 2.0
    q_high = 1.0 - q_low
    rows: List[dict] = []
    for name in metric_names:
        arr = np.asarray(deltas[name], dtype=np.float64)
        ci_low = float(np.quantile(arr, q_low))
        ci_high = float(np.quantile(arr, q_high))
        if better_direction[name] == "higher":
            prob_better = float(np.mean(arr > 0.0))
            supports = bool(ci_low > 0.0)
        else:
            prob_better = float(np.mean(arr < 0.0))
            supports = bool(ci_high < 0.0)
        rows.append(
            {
                "candidate": str(candidate_result["label"]),
                "metric": name,
                "baseline_value": float(baseline_point[name]),
                "candidate_value": float(candidate_point[name]),
                "delta_candidate_minus_baseline": float(candidate_point[name] - baseline_point[name]),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "better_direction": better_direction[name],
                "prob_candidate_better": prob_better,
                "supports_improvement": supports,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_registry(results: Dict[str, Dict[str, object]], cfg: BootstrapConfig, output_dir: str | Path, prefix: str) -> pd.DataFrame:
    output_dir = Path(output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    baseline = results["baseline"]
    rows: List[pd.DataFrame] = []
    for label, result in results.items():
        if label == "baseline":
            continue
        rows.append(bootstrap_compare_to_baseline(baseline, result, cfg))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        save_dataframe(out, reports_dir / f"{prefix}_bootstrap_metrics.csv")
    return out


def summarize_bootstrap(bootstrap_df: pd.DataFrame) -> pd.DataFrame:
    if bootstrap_df.empty:
        return pd.DataFrame()
    cols = ["delta_candidate_minus_baseline", "prob_candidate_better", "supports_improvement"]
    pivot = bootstrap_df.pivot(index="candidate", columns="metric", values=cols)
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    return pivot.reset_index()


def _ensure_pred_cache_for_replicates(
    baseline_ckpt: str | Path,
    data_root: str | Path,
    cfg: ReplicateOracleConfig,
    output_dir: str | Path,
    device: torch.device,
) -> Path:
    legacy_cache = PROJECT_ROOT / "experiments" / "oracle_root_cause_diagnostic" / "outputs" / "cache" / "split_preds"
    legacy_manifest = PROJECT_ROOT / "experiments" / "oracle_root_cause_diagnostic" / "outputs" / "reports" / f"{cfg.split}_pred_cache_manifest.json"
    baseline_ckpt = str(Path(baseline_ckpt).resolve())
    if legacy_cache.exists() and legacy_manifest.exists():
        try:
            payload = json.loads(legacy_manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if (
            payload.get("init_ckpt_path") == baseline_ckpt
            and payload.get("split") == cfg.split
            and int(payload.get("num_shards", -1)) == int(-1 if cfg.subset_num_shards is None else cfg.subset_num_shards)
        ):
            return legacy_cache
    rep_cfg = build_replicate_oracle_cfg(cfg, seed=cfg.base_seed)
    cache_info = or_lab.precompute_split_prediction_cache(
        init_ckpt_path=baseline_ckpt,
        data_root=data_root,
        split=cfg.split,
        output_dir=output_dir,
        device=device,
        batch_size=rep_cfg.prediction_batch_size,
        num_shards=cfg.subset_num_shards,
    )
    return Path(cache_info["cache_dir"])


def _build_replicate_bundle(samples: Sequence[dict], oracle_df: pd.DataFrame) -> Dict[str, object]:
    X = np.stack([np.asarray(sample["x"], dtype=np.uint8) for sample in samples], axis=0).astype(np.uint8, copy=False)
    return {"rows": oracle_df.reset_index(drop=True).copy(), "X": X}


def _extract_replicate_primary(label: str, oracle_eval: Dict[str, object]) -> Dict[str, object]:
    summary = oracle_eval["summary"]
    std_metrics = oracle_eval["standard_metrics"]
    band_map = _stable_band_map(oracle_eval)
    return {
        "label": label,
        "oracle_teacher_mae": float(summary["overall_teacher_vs_oracle_mae"]),
        "oracle_closer_rate": float(summary["teacher_closer_to_oracle_rate"]),
        "oracle_stable_0.7_slope": float(summary["stable_0.7_slope"]),
        "oracle_midband_mae_sum_stable": float(summary["midband_teacher_vs_oracle_mae_sum_stable"]),
        "oracle_center_amp_ratio": float(summary["center_amp_ratio_eq_0.05"]),
        "oracle_center_false_0.1eq": float(summary["center_false_pred0.1eq"]),
        "oracle_center_false_0.2eq": float(summary["center_false_pred0.2eq"]),
        "oracle_center_wrong_sign_0.1eq": float(std_metrics["center_false_decisive"][CENTER_FALSE_SMALL_KEY]["wrong_sign_rate"]),
        "oracle_center_wrong_sign_0.2eq": float(std_metrics["center_false_decisive"][CENTER_FALSE_MEDIUM_KEY]["wrong_sign_rate"]),
        "oracle_center_spread_ratio": _oracle_center_spread_ratio(oracle_eval),
        "oracle_band_amp_0_0.05_stable": float(band_map["[0.000,0.050]"]["amplitude_ratio"]),
        "oracle_band_sign_0.05_0.2_stable": float(band_map["[0.050,0.200]"]["teacher_oracle_sign_match_rate"]),
    }


def run_oracle_replicates(
    registry: Sequence[Dict[str, object]],
    baseline_ckpt: str | Path,
    data_root: str | Path,
    cfg: ReplicateOracleConfig,
    device: torch.device,
    output_dir: str | Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    pred_cache_dir = _ensure_pred_cache_for_replicates(baseline_ckpt, data_root, cfg, output_dir, device)

    rows: List[dict] = []
    for rep_idx in range(cfg.num_replicates):
        rep_seed = int(cfg.base_seed + rep_idx)
        rep_cfg = build_replicate_oracle_cfg(cfg, seed=rep_seed)
        rep_dir = output_dir / "replicates" / f"replicate_{rep_idx:02d}"
        rep_dir.mkdir(parents=True, exist_ok=True)
        subset_info = or_lab.build_stratified_subset(
            data_root=data_root,
            pred_cache_dir=pred_cache_dir,
            split=cfg.split,
            cfg=rep_cfg,
            num_shards=cfg.subset_num_shards,
        )
        save_dataframe(subset_info["quota_table"], rep_dir / "quota_table.csv")
        save_dataframe(subset_info["sampled_summary"], rep_dir / "sampled_summary.csv")
        oracle_df = or_lab.run_stockfish_oracle_on_subset(subset_info["samples"], rep_cfg, rep_dir)
        bundle = _build_replicate_bundle(subset_info["samples"], oracle_df)
        oracle_cfg = ab_lab.OracleEvalConfig(
            subset_csv_path=str(rep_dir / "reports" / "oracle_subset_rows.csv"),
            stable_label="stable",
            center_thr_y600=0.05,
            pred_thr_small_y600=0.10,
            pred_thr_medium_y600=0.20,
            oracle_batch_size=1024,
            mapping_validation_rows=8,
        )
        for item in registry:
            label = str(item["label"])
            target_scale = float(item["target_scale"])
            model, _ = base_lab.load_model_from_checkpoint(item["checkpoint"], device=device)
            oracle_eval = ab_lab.evaluate_model_on_oracle_subset(
                model=model,
                oracle_bundle=bundle,
                device=device,
                target_scale=target_scale,
                oracle_cfg=oracle_cfg,
            )
            row = _extract_replicate_primary(label, oracle_eval)
            row["replicate_idx"] = int(rep_idx)
            row["sample_seed"] = int(rep_seed)
            rows.append(row)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    per_rep = pd.DataFrame(rows)
    agg = per_rep.groupby("label", as_index=False).agg(
        {
            "oracle_teacher_mae": ["mean", "std"],
            "oracle_closer_rate": ["mean", "std"],
            "oracle_stable_0.7_slope": ["mean", "std"],
            "oracle_midband_mae_sum_stable": ["mean", "std"],
            "oracle_center_amp_ratio": ["mean", "std"],
            "oracle_center_false_0.1eq": ["mean", "std"],
            "oracle_center_false_0.2eq": ["mean", "std"],
            "oracle_center_wrong_sign_0.1eq": ["mean", "std"],
            "oracle_center_wrong_sign_0.2eq": ["mean", "std"],
            "oracle_center_spread_ratio": ["mean", "std"],
            "oracle_band_amp_0_0.05_stable": ["mean", "std"],
            "oracle_band_sign_0.05_0.2_stable": ["mean", "std"],
        }
    )
    agg.columns = ["label" if col[0] == "label" else f"{col[0]}_{col[1]}" for col in agg.columns.to_flat_index()]
    save_dataframe(per_rep, reports_dir / "replicate_oracle_metrics.csv")
    save_dataframe(agg, reports_dir / "replicate_oracle_aggregate.csv")
    return per_rep, agg


def run_variant_with_retry(
    variant: ab_lab.AblationVariant,
    init_ckpt_path: str | Path,
    data_root: str | Path,
    train_cfg: ab_lab.TrainConfig,
    oracle_cfg: ab_lab.OracleEvalConfig,
    oracle_bundle: Dict[str, object],
    paths: Dict[str, Path],
    device: torch.device,
    min_batch_size: int = 128,
) -> Dict[str, object]:
    attempts: List[dict] = []
    best_ckpt = Path(paths["runs_dir"]) / variant.name / "checkpoints" / f"{variant.name}_best.pt"
    if best_ckpt.exists():
        try:
            existing = evaluate_checkpoint(
                ckpt_path=best_ckpt,
                label=variant.name,
                data_root=data_root,
                oracle_bundle=oracle_bundle,
                oracle_cfg=oracle_cfg,
                train_cfg=train_cfg,
                device=device,
                target_scale=float(variant.target_scale),
            )
            existing["skipped_existing"] = True
            return existing
        except BaseException:
            pass
    candidate_bs = int(train_cfg.batch_size)
    write_retry_budget = 1
    while candidate_bs >= int(min_batch_size):
        cfg_try = replace(train_cfg, batch_size=int(candidate_bs))
        run_dir = paths["runs_dir"] / variant.name
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
        _cleanup_cuda()
        try:
            result = ab_lab.run_variant_finetune(
                init_ckpt_path=init_ckpt_path,
                data_root=data_root,
                variant=variant,
                train_cfg=cfg_try,
                oracle_cfg=oracle_cfg,
                oracle_bundle=oracle_bundle,
                paths=paths,
                device=device,
            )
            result["runtime_train_cfg"] = asdict(cfg_try)
            result["oom_retry_attempts"] = attempts + [{"batch_size": int(candidate_bs), "ok": True}]
            reports_dir = paths["runs_dir"] / variant.name / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            save_json(result["oom_retry_attempts"], reports_dir / "oom_retry_attempts.json")
            return result
        except RuntimeError as exc:
            if _is_checkpoint_write_error(exc):
                attempts.append({"batch_size": int(candidate_bs), "ok": False, "error": str(exc), "kind": "checkpoint_write"})
                _cleanup_cuda()
                if write_retry_budget <= 0:
                    raise
                write_retry_budget -= 1
                redirect_report = maybe_redirect_followup_paths(paths)
                print(json.dumps({"variant": variant.name, "write_retry": True, "redirect": redirect_report}, indent=2))
                continue
            if not _is_cuda_oom(exc):
                raise
            attempts.append({"batch_size": int(candidate_bs), "ok": False, "error": str(exc)})
            _cleanup_cuda()
            next_bs = max(int(min_batch_size), (int(candidate_bs) // 2 // 32) * 32)
            if next_bs >= int(candidate_bs):
                next_bs = int(candidate_bs) - 32
            candidate_bs = int(next_bs)
    raise RuntimeError(
        f"All retry batch sizes failed for variant {variant.name}: "
        + json.dumps(attempts, ensure_ascii=False)
    )


def run_variant_group(
    variant_names: Sequence[str],
    variant_catalog: Dict[str, ab_lab.AblationVariant],
    init_ckpt_path: str | Path,
    data_root: str | Path,
    train_cfg: ab_lab.TrainConfig,
    oracle_cfg: ab_lab.OracleEvalConfig,
    oracle_bundle: Dict[str, object],
    paths: Dict[str, Path],
    device: torch.device,
) -> List[Dict[str, object]]:
    redirect_report = maybe_redirect_followup_paths(paths)
    print(json.dumps(redirect_report, indent=2))
    results: List[Dict[str, object]] = []
    for name in variant_names:
        variant = variant_catalog[name]
        print(f"\n===== RUN {variant.name} =====")
        print(variant.description)
        results.append(
            run_variant_with_retry(
                variant=variant,
                init_ckpt_path=init_ckpt_path,
                data_root=data_root,
                train_cfg=train_cfg,
                oracle_cfg=oracle_cfg,
                oracle_bundle=oracle_bundle,
                paths=paths,
                device=device,
            )
        )
    return results
