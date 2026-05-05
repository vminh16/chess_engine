from __future__ import annotations

import gc
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def _find_repo_root(start: Path) -> Path:
    start = start.resolve()
    markers = (
        ("model",),
        ("train_v2_TF1", "ft1_colab_helpers.py"),
        ("train_v4_report1", "report1_train_helpers.py"),
        ("runs",),
    )
    for candidate in [start.parent, *start.parents]:
        if all(candidate.joinpath(*parts).exists() for parts in markers):
            return candidate
    raise RuntimeError(f"Cannot resolve repository root from helper path: {start}")


def _import_module_from_file(module_name: str, module_path: Path):
    module_path = Path(module_path).resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"Missing module path for {module_name}: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


PROJECT_ROOT = _find_repo_root(Path(__file__).resolve())
MODEL_ROOT = PROJECT_ROOT / "model"
EXPERIMENT_DIRS = [
    PROJECT_ROOT / "experiments" / "teacher_root_cause_lab",
    PROJECT_ROOT / "experiments" / "root_cause_ablation_suite",
    PROJECT_ROOT / "experiments" / "objective_resolution_suite",
    PROJECT_ROOT / "experiments" / "failure_b_resolution_suite",
    PROJECT_ROOT / "train_v2_TF1",
    PROJECT_ROOT / "train_v4_report1",
]

for path in [PROJECT_ROOT, MODEL_ROOT, *EXPERIMENT_DIRS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import teacher_root_cause_helpers as base_lab  # noqa: E402
import objective_resolution_helpers as obj_lab  # noqa: E402

shared_ft1 = _import_module_from_file(
    "phase_b_offline_benchmark_ft1_helpers",
    PROJECT_ROOT / "train_v2_TF1" / "ft1_colab_helpers.py",
)

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "process"
DEFAULT_ORACLE_ROLE_BUNDLE_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "oc2_joint_oracle_full_model_pilot"
    / "outputs"
    / "cache"
    / "oracle_role_bundle"
)
DEFAULT_POOLED_CENTER_BUNDLE_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "failure_b_resolution_suite"
    / "outputs"
    / "cache"
    / "pooled_center_bundle"
)
DEFAULT_OUTPUTS_ROOT = PROJECT_ROOT / "evaluation" / "phase_b_offline_benchmark" / "outputs"


@dataclass
class OfflineBenchmarkConfig:
    benchmark_name: str
    candidate_checkpoint: str | Path
    candidate_label: str = "candidate"
    reference_checkpoint: Optional[str | Path] = None
    reference_label: str = "reference"
    data_root: str | Path = DEFAULT_DATA_ROOT
    oracle_role_bundle_dir: str | Path = DEFAULT_ORACLE_ROLE_BUNDLE_DIR
    pooled_center_bundle_dir: str | Path = DEFAULT_POOLED_CENTER_BUNDLE_DIR
    outputs_root: str | Path = DEFAULT_OUTPUTS_ROOT
    eval_batch_size: Optional[int] = None
    test_max_samples: Optional[int] = None
    test_num_shards: Optional[int] = None
    midband_rel_tol: float = 0.05
    slope_abs_tol: float = 0.02
    center_strong_threshold: Optional[float] = None
    run_bootstrap_compare: bool = True
    bootstrap_n: int = 2000
    bootstrap_seed: int = 123
    bootstrap_ci_alpha: float = 0.05
    abs_calibration_bins: int = 20


@dataclass
class ResolvedBenchmarkConfig:
    benchmark_name: str
    candidate_checkpoint: Path
    candidate_label: str
    reference_checkpoint: Optional[Path]
    reference_label: str
    data_root: Path
    oracle_role_bundle_dir: Path
    pooled_center_bundle_dir: Path
    outputs_root: Path
    eval_batch_size: int
    test_max_samples: int
    test_num_shards: int
    midband_rel_tol: float
    slope_abs_tol: float
    center_strong_threshold: Optional[float]
    run_bootstrap_compare: bool
    bootstrap_n: int
    bootstrap_seed: int
    bootstrap_ci_alpha: float
    abs_calibration_bins: int


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _json_ready(value: object) -> object:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    return value


def _save_json(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(dict(payload)), indent=2, ensure_ascii=False), encoding="utf-8")


def _normalize_repo_path(repo_root: Path, raw_path: object) -> Optional[Path]:
    if raw_path is None:
        return None
    text = str(raw_path).strip()
    if not text:
        return None
    direct = Path(text)
    if direct.exists():
        return direct.resolve()

    normalized = text.replace("\\", "/").rstrip("/")
    lowered = normalized.lower()
    colab_data_roots = {
        "/content/chess_engine_data/process": repo_root / "data" / "process",
        "/content/drive/mydrive/chess_engine": repo_root,
    }
    for raw_prefix, mapped in colab_data_roots.items():
        if lowered == raw_prefix:
            return mapped.resolve()
    marker = "/chess_engine/"
    idx = lowered.find(marker)
    if idx >= 0:
        suffix = normalized[idx + len(marker) :]
        return (repo_root / Path(suffix)).resolve()
    if lowered.endswith("/chess_engine"):
        return repo_root.resolve()
    return direct


def _count_split_shards(split_dir: Path) -> int:
    return len(sorted(split_dir.glob("X_*.npy")))


def _count_split_samples(split_dir: Path, num_shards: Optional[int] = None) -> int:
    y_paths = sorted(split_dir.glob("y_*.npy"))
    if num_shards is not None:
        y_paths = y_paths[: int(num_shards)]
    total = 0
    for path in y_paths:
        arr = np.load(path, mmap_mode="r")
        total += int(arr.shape[0])
    return int(total)


def _default_eval_batch_size(device: torch.device) -> int:
    if device.type != "cuda":
        return 512
    props = torch.cuda.get_device_properties(device)
    total_mem_gb = float(props.total_memory) / (1024.0**3)
    if total_mem_gb >= 14.0:
        return 4096
    if total_mem_gb >= 8.0:
        return 2048
    if total_mem_gb >= 6.0:
        return 1024
    return 512


CORE_METRIC_SPECS: Sequence[Mapping[str, str]] = (
    {"metric": "overall_mse", "direction": "lower", "tier": "core", "reliability": "high", "reason": "Full-test overall squared error."},
    {"metric": "overall_mae", "direction": "lower", "tier": "core", "reliability": "high", "reason": "Full-test overall absolute error."},
    {"metric": "overall_pearson", "direction": "higher", "tier": "core", "reliability": "high", "reason": "Full-test linear association across the whole target range."},
    {"metric": "test_mse_0.1eq", "direction": "lower", "tier": "core", "reliability": "high", "reason": "Near-center regression error on a large test slice."},
    {"metric": "test_mse_0.2eq", "direction": "lower", "tier": "core", "reliability": "high", "reason": "Near-center regression error on a large test slice."},
    {"metric": "test_mse_0.5eq", "direction": "lower", "tier": "core", "reliability": "high", "reason": "Stable mid-band regression error on a large test slice."},
    {"metric": "test_mse_0.7eq", "direction": "lower", "tier": "core", "reliability": "high", "reason": "Decisive-band regression error on a large test slice."},
    {"metric": "test_slope_0.2eq", "direction": "closer_to_1", "tier": "core", "reliability": "high", "reason": "Calibration slope on a moderate-variance test band."},
    {"metric": "test_slope_0.7eq", "direction": "closer_to_1", "tier": "core", "reliability": "high", "reason": "Calibration slope on the decisive test band."},
    {"metric": "center_false_decisive_0.1eq", "direction": "lower", "tier": "core", "reliability": "high", "reason": "False-decisive rate around true-center positions."},
    {"metric": "center_false_decisive_0.2eq", "direction": "lower", "tier": "core", "reliability": "high", "reason": "Severe false-decisive rate around true-center positions."},
    {"metric": "center_wrong_sign_0.1eq", "direction": "lower", "tier": "core", "reliability": "high", "reason": "Wrong-sign rate among near-center false positives."},
    {"metric": "center_wrong_sign_0.2eq", "direction": "lower", "tier": "core", "reliability": "high", "reason": "Wrong-sign rate among severe near-center false positives."},
    {"metric": "sign_match_0.05_0.2eq", "direction": "higher", "tier": "core", "reliability": "high", "reason": "Sign agreement on mildly non-center positions."},
    {"metric": "sign_match_0.2_0.5eq", "direction": "higher", "tier": "core", "reliability": "high", "reason": "Sign agreement on stable mid-band positions."},
    {"metric": "sign_match_0.5_0.7eq", "direction": "higher", "tier": "core", "reliability": "high", "reason": "Sign agreement on decisive positions."},
    {"metric": "abs_cal_gap_0.2_0.5eq", "direction": "lower", "tier": "core", "reliability": "high", "reason": "Absolute-value calibration gap on stable mid-band positions."},
    {"metric": "abs_cal_gap_0.5_0.7eq", "direction": "lower", "tier": "core", "reliability": "high", "reason": "Absolute-value calibration gap on decisive positions."},
)

SECONDARY_METRIC_SPECS: Sequence[Mapping[str, str]] = (
    {"metric": "oracle_midband_mae_sum_stable", "direction": "lower", "tier": "secondary", "reliability": "medium", "reason": "Oracle subset metric; useful for continuity with earlier suites but based on a much smaller sample."},
    {"metric": "oracle_stable_0.7_slope", "direction": "higher", "tier": "secondary", "reliability": "medium", "reason": "Oracle subset slope; useful as a secondary check only."},
)

DIAGNOSTIC_METRIC_SPECS: Sequence[Mapping[str, str]] = (
    {"metric": "test_slope_0.1eq", "direction": "closer_to_1", "tier": "diagnostic", "reliability": "low", "reason": "Near-zero slope is unstable because target variance is tiny."},
    {"metric": "center_spread_ratio", "direction": "closer_to_1", "tier": "diagnostic", "reliability": "medium", "reason": "Useful to spot over-amplification, but ratio values can look extreme near center."},
    {"metric": "max_midband_abs_cal_gap", "direction": "lower", "tier": "diagnostic", "reliability": "medium", "reason": "Worst-bucket metric is intentionally harsh and can move with local bucket noise."},
    {"metric": "oracle_center_score", "direction": "lower", "tier": "diagnostic", "reliability": "low", "reason": "Center bundle score is based on a very small curated set; do not use as a promotion gate."},
)


def build_default_paths(repo_root: Path, benchmark_name: str, outputs_root: Optional[Path] = None) -> Dict[str, Path]:
    suite_root = repo_root / "evaluation" / "phase_b_offline_benchmark"
    outputs_base = Path(outputs_root) if outputs_root is not None else DEFAULT_OUTPUTS_ROOT
    benchmark_dir = outputs_base / str(benchmark_name)
    reports_dir = benchmark_dir / "reports"
    plots_dir = benchmark_dir / "plots"
    return {
        "suite_root": suite_root,
        "outputs_base": outputs_base,
        "benchmark_dir": benchmark_dir,
        "reports_dir": reports_dir,
        "plots_dir": plots_dir,
    }


def _resolve_checkpoint_from_policy(run_dir: Path, checkpoint_policy: str, decision_payload: Mapping[str, object]) -> Path:
    policy = str(checkpoint_policy).strip().lower()
    checkpoints_dir = run_dir / "checkpoints"
    mapping = {
        "best_gate": decision_payload.get("best_gate_checkpoint"),
        "best_any": decision_payload.get("best_any_checkpoint"),
        "best_pareto_a": decision_payload.get("best_pareto_A_checkpoint"),
        "best_pareto_b": decision_payload.get("best_pareto_B_checkpoint"),
        "latest": decision_payload.get("latest_checkpoint"),
    }
    raw = mapping.get(policy)
    if raw:
        candidate = _normalize_repo_path(PROJECT_ROOT, raw)
        if candidate is not None and candidate.exists():
            return candidate

    fallback_names = {
        "best_gate": "ckpt_best_gate.pt",
        "best_any": "ckpt_best_any.pt",
        "best_pareto_a": "ckpt_best_pareto_A.pt",
        "best_pareto_b": "ckpt_best_pareto_B.pt",
        "latest": "ckpt_latest.pt",
    }
    if policy not in fallback_names:
        raise ValueError(
            "checkpoint_policy must be one of: best_gate, best_any, best_pareto_a, best_pareto_b, latest"
        )
    candidate = checkpoints_dir / fallback_names[policy]
    if not candidate.exists():
        raise FileNotFoundError(f"Checkpoint for policy={checkpoint_policy} not found: {candidate}")
    return candidate.resolve()


def resolve_config_from_run(
    run_dir: str | Path,
    *,
    checkpoint_policy: str = "best_gate",
    benchmark_name: Optional[str] = None,
    candidate_label: Optional[str] = None,
    reference_label: str = "L4_A1_plus_A2",
    data_root: Optional[str | Path] = None,
    reference_checkpoint: Optional[str | Path] = None,
    oracle_role_bundle_dir: Optional[str | Path] = None,
    pooled_center_bundle_dir: Optional[str | Path] = None,
    outputs_root: Optional[str | Path] = None,
    eval_batch_size: Optional[int] = None,
    test_max_samples: Optional[int] = None,
    test_num_shards: Optional[int] = None,
    midband_rel_tol: float = 0.05,
    slope_abs_tol: float = 0.02,
    center_strong_threshold: Optional[float] = None,
    run_bootstrap_compare: bool = True,
    bootstrap_n: int = 2000,
    bootstrap_seed: int = 123,
    bootstrap_ci_alpha: float = 0.05,
    abs_calibration_bins: int = 20,
) -> OfflineBenchmarkConfig:
    run_dir = Path(run_dir).resolve()
    reports_dir = run_dir / "reports"
    run_config_path = reports_dir / "run_config.json"
    decision_path = reports_dir / "decision_summary.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing run_config.json: {run_config_path}")
    if not decision_path.exists():
        raise FileNotFoundError(f"Missing decision_summary.json: {decision_path}")

    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    run_name = run_dir.name

    candidate_ckpt = _resolve_checkpoint_from_policy(run_dir, checkpoint_policy, decision)
    baseline_ckpt = _normalize_repo_path(PROJECT_ROOT, reference_checkpoint or run_config.get("l4_reference_ckpt"))
    role_dir = _normalize_repo_path(PROJECT_ROOT, oracle_role_bundle_dir or run_config.get("oracle_role_bundle_dir"))
    pooled_dir = _normalize_repo_path(PROJECT_ROOT, pooled_center_bundle_dir or run_config.get("pooled_center_bundle_dir"))
    resolved_data_root = (
        Path(data_root).resolve()
        if data_root is not None
        else _normalize_repo_path(PROJECT_ROOT, run_config.get("data_root")) or DEFAULT_DATA_ROOT
    )

    if benchmark_name is None or not str(benchmark_name).strip():
        benchmark_name = f"offline_{run_name}_{str(checkpoint_policy).strip().lower()}"
    if candidate_label is None or not str(candidate_label).strip():
        candidate_label = f"{run_name}:{str(checkpoint_policy).strip().lower()}"

    train_cfg = dict(run_config.get("train_cfg", {}))
    resolved_eval_batch = int(eval_batch_size) if eval_batch_size is not None else int(train_cfg.get("eval_batch_size", 4096))
    resolved_test_max = int(test_max_samples) if test_max_samples is not None else None
    resolved_test_shards = int(test_num_shards) if test_num_shards is not None else None

    return OfflineBenchmarkConfig(
        benchmark_name=str(benchmark_name),
        candidate_checkpoint=candidate_ckpt,
        candidate_label=str(candidate_label),
        reference_checkpoint=baseline_ckpt,
        reference_label=str(reference_label),
        data_root=resolved_data_root,
        oracle_role_bundle_dir=role_dir if role_dir is not None else DEFAULT_ORACLE_ROLE_BUNDLE_DIR,
        pooled_center_bundle_dir=pooled_dir if pooled_dir is not None else DEFAULT_POOLED_CENTER_BUNDLE_DIR,
        outputs_root=Path(outputs_root).resolve() if outputs_root is not None else DEFAULT_OUTPUTS_ROOT,
        eval_batch_size=resolved_eval_batch,
        test_max_samples=resolved_test_max,
        test_num_shards=resolved_test_shards,
        midband_rel_tol=float(midband_rel_tol),
        slope_abs_tol=float(slope_abs_tol),
        center_strong_threshold=center_strong_threshold,
        run_bootstrap_compare=bool(run_bootstrap_compare),
        bootstrap_n=int(bootstrap_n),
        bootstrap_seed=int(bootstrap_seed),
        bootstrap_ci_alpha=float(bootstrap_ci_alpha),
        abs_calibration_bins=int(abs_calibration_bins),
    )


def resolve_benchmark_config(config: OfflineBenchmarkConfig, device: torch.device) -> ResolvedBenchmarkConfig:
    data_root = Path(config.data_root).resolve()
    test_dir = data_root / "test"
    full_test_shards = _count_split_shards(test_dir)
    if full_test_shards <= 0:
        raise FileNotFoundError(f"No test shards found under: {test_dir}")

    requested_test_shards = full_test_shards if config.test_num_shards is None else int(config.test_num_shards)
    if requested_test_shards <= 0:
        raise ValueError("test_num_shards must be positive")
    test_num_shards = min(requested_test_shards, full_test_shards)

    available_samples = _count_split_samples(test_dir, num_shards=test_num_shards)
    requested_test_samples = available_samples if config.test_max_samples is None else int(config.test_max_samples)
    if requested_test_samples <= 0:
        raise ValueError("test_max_samples must be positive")
    test_max_samples = min(requested_test_samples, available_samples)

    eval_batch_size = int(config.eval_batch_size) if config.eval_batch_size is not None else _default_eval_batch_size(device)
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive")

    candidate_checkpoint = Path(config.candidate_checkpoint).resolve()
    reference_checkpoint = None if config.reference_checkpoint is None else Path(config.reference_checkpoint).resolve()
    oracle_role_bundle_dir = Path(config.oracle_role_bundle_dir).resolve()
    pooled_center_bundle_dir = Path(config.pooled_center_bundle_dir).resolve()
    outputs_root = Path(config.outputs_root).resolve()

    return ResolvedBenchmarkConfig(
        benchmark_name=str(config.benchmark_name),
        candidate_checkpoint=candidate_checkpoint,
        candidate_label=str(config.candidate_label),
        reference_checkpoint=reference_checkpoint,
        reference_label=str(config.reference_label),
        data_root=data_root,
        oracle_role_bundle_dir=oracle_role_bundle_dir,
        pooled_center_bundle_dir=pooled_center_bundle_dir,
        outputs_root=outputs_root,
        eval_batch_size=eval_batch_size,
        test_max_samples=test_max_samples,
        test_num_shards=test_num_shards,
        midband_rel_tol=float(config.midband_rel_tol),
        slope_abs_tol=float(config.slope_abs_tol),
        center_strong_threshold=(None if config.center_strong_threshold is None else float(config.center_strong_threshold)),
        run_bootstrap_compare=bool(config.run_bootstrap_compare),
        bootstrap_n=max(int(config.bootstrap_n), 0),
        bootstrap_seed=int(config.bootstrap_seed),
        bootstrap_ci_alpha=float(config.bootstrap_ci_alpha),
        abs_calibration_bins=max(int(config.abs_calibration_bins), 4),
    )


def validate_runtime_paths(config: ResolvedBenchmarkConfig) -> Dict[str, object]:
    test_dir = config.data_root / "test"
    paths = {
        "candidate_checkpoint": config.candidate_checkpoint,
        "data_root": config.data_root,
        "test_split": test_dir,
        "oracle_role_bundle_dir": config.oracle_role_bundle_dir,
        "pooled_center_bundle_dir": config.pooled_center_bundle_dir,
    }
    if config.reference_checkpoint is not None:
        paths["reference_checkpoint"] = config.reference_checkpoint
    missing = {name: str(path) for name, path in paths.items() if not Path(path).exists()}
    return {
        "ok": not missing,
        "paths": {name: str(path) for name, path in paths.items()},
        "missing": missing,
        "test_num_shards": int(config.test_num_shards),
        "test_max_samples": int(config.test_max_samples),
        "full_test_shards": int(_count_split_shards(test_dir)),
        "full_test_samples": int(_count_split_samples(test_dir)),
    }


def _load_bundles(config: ResolvedBenchmarkConfig) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, object]]:
    oracle_bundle = obj_lab.build_primary_oracle_bundle(config.data_root)
    pooled_center_bundle = shared_ft1.load_pooled_center_bundle(config.pooled_center_bundle_dir)
    role_bundle = shared_ft1.load_ft1_role_bundle(config.oracle_role_bundle_dir)
    return oracle_bundle, pooled_center_bundle, role_bundle


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    data_root: Path,
    oracle_bundle: Dict[str, object],
    pooled_center_bundle: Dict[str, object],
    role_bundle: Dict[str, object],
    device: torch.device,
    eval_batch_size: int,
    test_max_samples: int,
    test_num_shards: int,
) -> Dict[str, object]:
    model, _ = base_lab.load_model_from_checkpoint(checkpoint_path, device=device)
    try:
        result = shared_ft1.evaluate_ft1_model(
            model=model,
            data_root=data_root,
            device=device,
            eval_batch_size=eval_batch_size,
            split="test",
            max_samples=test_max_samples,
            num_shards=test_num_shards,
            oracle_bundle=oracle_bundle,
            pooled_center_bundle=pooled_center_bundle,
            role_bundle=role_bundle,
        )
        result = dict(result)
        result.update(dict(result.get("primary", {})))
        result["oracle_bundle"] = oracle_bundle
        return result
    finally:
        del model
        _cleanup_cuda()


def _compact_eval_result(
    eval_result: Mapping[str, object],
    *,
    split_diagnostics: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    split_eval = dict(eval_result["split_eval"])
    oracle_eval = dict(eval_result["oracle_eval"])
    role_eval = dict(eval_result["role_eval"])

    compact = {
        "primary": dict(eval_result["primary"]),
        "split_eval": {
            "metrics": _json_ready(split_eval.get("metrics", {})),
        },
        "oracle_eval": {
            "summary": _json_ready(oracle_eval.get("summary", {})),
            "standard_metrics": _json_ready(oracle_eval.get("standard_metrics", {})),
            "variant_standard_metrics": _json_ready(oracle_eval.get("variant_standard_metrics", {})),
        },
        "pooled_center_eval": _json_ready(eval_result.get("pooled_center_eval", {})),
        "role_eval": {
            "metrics": _json_ready(role_eval.get("metrics", {})),
        },
    }
    if split_diagnostics is not None:
        compact["split_diagnostics"] = {
            "derived_metrics": _json_ready(split_diagnostics.get("derived_metrics", {})),
            "sample_sizes": _json_ready(split_diagnostics.get("sample_sizes", {})),
        }
    compact.update(dict(eval_result["primary"]))
    return compact


def _extract_split_arrays(eval_result: Mapping[str, object]) -> Tuple[np.ndarray, np.ndarray]:
    split_eval = dict(eval_result["split_eval"])
    if "targets" not in split_eval or "preds" not in split_eval:
        raise KeyError("split_eval must contain raw targets and preds for plotting/diagnostics")
    targets = np.asarray(split_eval["targets"], dtype=np.float64).reshape(-1)
    preds = np.asarray(split_eval["preds"], dtype=np.float64).reshape(-1)
    if targets.shape != preds.shape:
        raise ValueError(f"split_eval targets/preds shape mismatch: {targets.shape} vs {preds.shape}")
    return targets, preds


def _extract_oracle_arrays(eval_result: Mapping[str, object]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    oracle_eval = dict(eval_result["oracle_eval"])
    oracle_targets = np.asarray(oracle_eval["oracle_targets"], dtype=np.float64).reshape(-1)
    preds = np.asarray(oracle_eval["preds"], dtype=np.float64).reshape(-1)
    rows = eval_result["oracle_bundle"]["rows"]
    stable_mask = rows["stability_group"].astype(str).eq("stable").to_numpy(dtype=bool)
    if oracle_targets.shape != preds.shape or oracle_targets.shape[0] != stable_mask.shape[0]:
        raise ValueError(
            "oracle_eval arrays and oracle bundle rows must have identical length "
            f"({oracle_targets.shape[0]}, {preds.shape[0]}, {stable_mask.shape[0]})"
        )
    return oracle_targets, preds, stable_mask


def _band_mask(abs_values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if lo <= 0.0:
        return abs_values <= hi
    return (abs_values > lo) & (abs_values <= hi)


def _safe_rate(mask: np.ndarray, values: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.mean(values[mask]))


def _safe_gap(mask: np.ndarray, lhs: np.ndarray, rhs: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(abs(float(np.mean(lhs[mask])) - float(np.mean(rhs[mask]))))


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _pooled_center_sample_size(eval_result: Mapping[str, object]) -> Optional[int]:
    pooled_center_eval = dict(eval_result.get("pooled_center_eval", {}))
    pred = pooled_center_eval.get("pred")
    if pred is None:
        return None
    return int(np.asarray(pred).reshape(-1).shape[0])


def build_split_diagnostics(eval_result: Mapping[str, object], *, abs_calibration_bins: int) -> Dict[str, object]:
    targets, preds = _extract_split_arrays(eval_result)
    abs_targets = np.abs(targets)
    abs_preds = np.abs(preds)

    center_mask = abs_targets <= 0.05
    band_010 = _band_mask(abs_targets, 0.0, 0.10)
    band_020 = _band_mask(abs_targets, 0.0, 0.20)
    band_050 = _band_mask(abs_targets, 0.0, 0.50)
    band_070 = _band_mask(abs_targets, 0.0, 0.70)
    band_005_02 = _band_mask(abs_targets, 0.05, 0.20)
    band_02_05 = _band_mask(abs_targets, 0.20, 0.50)
    band_05_07 = _band_mask(abs_targets, 0.50, 0.70)
    midband_mask = _band_mask(abs_targets, 0.05, 0.70)

    sign_match_005_02 = _safe_rate(band_005_02, (np.sign(preds) == np.sign(targets)).astype(np.float64))
    sign_match_02_05 = _safe_rate(band_02_05, (np.sign(preds) == np.sign(targets)).astype(np.float64))
    sign_match_05_07 = _safe_rate(band_05_07, (np.sign(preds) == np.sign(targets)).astype(np.float64))

    derived_metrics = {
        "sign_match_0.05_0.2eq": sign_match_005_02,
        "sign_match_0.2_0.5eq": sign_match_02_05,
        "sign_match_0.5_0.7eq": sign_match_05_07,
        "abs_cal_gap_0.2_0.5eq": _safe_gap(band_02_05, abs_preds, abs_targets),
        "abs_cal_gap_0.5_0.7eq": _safe_gap(band_05_07, abs_preds, abs_targets),
    }

    sample_sizes = {
        "overall_mse": int(targets.shape[0]),
        "overall_mae": int(targets.shape[0]),
        "overall_pearson": int(targets.shape[0]),
        "test_mse_0.1eq": int(np.count_nonzero(band_010)),
        "test_mse_0.2eq": int(np.count_nonzero(band_020)),
        "test_mse_0.5eq": int(np.count_nonzero(band_050)),
        "test_mse_0.7eq": int(np.count_nonzero(band_070)),
        "test_slope_0.1eq": int(np.count_nonzero(band_010)),
        "test_slope_0.2eq": int(np.count_nonzero(band_020)),
        "test_slope_0.7eq": int(np.count_nonzero(band_070)),
        "center_false_decisive_0.1eq": int(np.count_nonzero(center_mask)),
        "center_false_decisive_0.2eq": int(np.count_nonzero(center_mask)),
        "center_wrong_sign_0.1eq": int(np.count_nonzero(center_mask)),
        "center_wrong_sign_0.2eq": int(np.count_nonzero(center_mask)),
        "center_spread_ratio": int(np.count_nonzero(center_mask)),
        "max_midband_abs_cal_gap": int(np.count_nonzero(midband_mask)),
        "sign_match_0.05_0.2eq": int(np.count_nonzero(band_005_02)),
        "sign_match_0.2_0.5eq": int(np.count_nonzero(band_02_05)),
        "sign_match_0.5_0.7eq": int(np.count_nonzero(band_05_07)),
        "abs_cal_gap_0.2_0.5eq": int(np.count_nonzero(band_02_05)),
        "abs_cal_gap_0.5_0.7eq": int(np.count_nonzero(band_05_07)),
    }

    bin_edges = np.linspace(0.0, 1.0, int(abs_calibration_bins) + 1, dtype=np.float64)
    calibration_rows: List[Dict[str, object]] = []
    for idx in range(bin_edges.shape[0] - 1):
        lo = float(bin_edges[idx])
        hi = float(bin_edges[idx + 1])
        if idx == 0:
            mask = abs_targets <= hi
        else:
            mask = (abs_targets > lo) & (abs_targets <= hi)
        calibration_rows.append(
            {
                "bin_index": int(idx),
                "abs_target_lo": lo,
                "abs_target_hi": hi,
                "count": int(np.count_nonzero(mask)),
                "mean_abs_target": (float(np.mean(abs_targets[mask])) if np.any(mask) else float("nan")),
                "mean_abs_pred": (float(np.mean(abs_preds[mask])) if np.any(mask) else float("nan")),
                "abs_cal_gap": _safe_gap(mask, abs_preds, abs_targets),
                "mae": (float(np.mean(np.abs(preds[mask] - targets[mask]))) if np.any(mask) else float("nan")),
                "pearson": (_safe_pearson(targets[mask], preds[mask]) if np.any(mask) else float("nan")),
            }
        )
    absolute_calibration = pd.DataFrame(calibration_rows)

    band_rows = []
    for label, lo, hi, sign_key, cal_key in (
        ("0.05-0.20", 0.05, 0.20, "sign_match_0.05_0.2eq", None),
        ("0.20-0.50", 0.20, 0.50, "sign_match_0.2_0.5eq", "abs_cal_gap_0.2_0.5eq"),
        ("0.50-0.70", 0.50, 0.70, "sign_match_0.5_0.7eq", "abs_cal_gap_0.5_0.7eq"),
    ):
        mask = _band_mask(abs_targets, lo, hi)
        band_rows.append(
            {
                "band_label": label,
                "abs_target_lo": float(lo),
                "abs_target_hi": float(hi),
                "count": int(np.count_nonzero(mask)),
                "mae": (float(np.mean(np.abs(preds[mask] - targets[mask]))) if np.any(mask) else float("nan")),
                "sign_match": float(derived_metrics[sign_key]),
                "mean_abs_target": (float(np.mean(abs_targets[mask])) if np.any(mask) else float("nan")),
                "mean_abs_pred": (float(np.mean(abs_preds[mask])) if np.any(mask) else float("nan")),
                "abs_cal_gap": (float(derived_metrics[cal_key]) if cal_key is not None else float("nan")),
            }
        )
    band_diagnostics = pd.DataFrame(band_rows)

    return {
        "targets": targets,
        "preds": preds,
        "derived_metrics": derived_metrics,
        "sample_sizes": sample_sizes,
        "absolute_calibration": absolute_calibration,
        "band_diagnostics": band_diagnostics,
    }


def build_metric_map(eval_result: Mapping[str, object], split_diagnostics: Mapping[str, object]) -> Dict[str, float]:
    primary = dict(eval_result["primary"])
    metrics = {
        "overall_mse": float(eval_result["split_eval"]["metrics"]["overall"]["mse"]),
        "overall_mae": float(eval_result["split_eval"]["metrics"]["overall"]["mae"]),
        "overall_pearson": float(eval_result["split_eval"]["metrics"]["overall"]["pearson"]),
        "test_mse_0.1eq": float(primary["test_mse_0.1eq"]),
        "test_mse_0.2eq": float(primary["test_mse_0.2eq"]),
        "test_mse_0.5eq": float(primary["test_mse_0.5eq"]),
        "test_mse_0.7eq": float(primary["test_mse_0.7eq"]),
        "test_slope_0.1eq": float(primary["test_slope_0.1eq"]),
        "test_slope_0.2eq": float(primary["test_slope_0.2eq"]),
        "test_slope_0.7eq": float(primary["test_slope_0.7eq"]),
        "center_false_decisive_0.1eq": float(primary["test_center_false_0.1eq"]),
        "center_false_decisive_0.2eq": float(primary["test_center_false_0.2eq"]),
        "center_wrong_sign_0.1eq": float(primary["test_center_wrong_sign_0.1eq"]),
        "center_wrong_sign_0.2eq": float(primary["test_center_wrong_sign_0.2eq"]),
        "center_spread_ratio": float(primary["test_center_spread_ratio"]),
        "max_midband_abs_cal_gap": float(primary["test_max_midband_abs_cal_gap"]),
        "oracle_midband_mae_sum_stable": float(primary["oracle_midband_mae_sum_stable"]),
        "oracle_stable_0.7_slope": float(primary["oracle_stable_0.7_slope"]),
        "oracle_center_score": float(primary["center_score"]),
    }
    metrics.update({str(k): float(v) for k, v in dict(split_diagnostics["derived_metrics"]).items()})
    return metrics


def build_sample_sizes(
    eval_result: Mapping[str, object],
    split_diagnostics: Mapping[str, object],
) -> Dict[str, int]:
    sample_sizes = {str(k): int(v) for k, v in dict(split_diagnostics["sample_sizes"]).items()}
    oracle_targets, _oracle_preds, stable_mask = _extract_oracle_arrays(eval_result)
    abs_oracle = np.abs(oracle_targets)
    oracle_midband_mask = stable_mask & (abs_oracle > 0.05) & (abs_oracle <= 0.70)
    oracle_slope_mask = stable_mask & (abs_oracle <= 0.70)
    pooled_n = _pooled_center_sample_size(eval_result)
    sample_sizes["oracle_midband_mae_sum_stable"] = int(np.count_nonzero(oracle_midband_mask))
    sample_sizes["oracle_stable_0.7_slope"] = int(np.count_nonzero(oracle_slope_mask))
    if pooled_n is not None:
        sample_sizes["oracle_center_score"] = int(pooled_n)
    return sample_sizes


def _compare_metric_values(candidate_value: float, reference_value: float, direction: str) -> Tuple[Optional[bool], Dict[str, object]]:
    extras: Dict[str, object] = {}
    if not (math.isfinite(candidate_value) and math.isfinite(reference_value)):
        return None, extras
    if direction == "lower":
        return bool(candidate_value < reference_value), extras
    if direction == "higher":
        return bool(candidate_value > reference_value), extras
    if direction == "closer_to_1":
        candidate_distance = abs(candidate_value - 1.0)
        reference_distance = abs(reference_value - 1.0)
        extras["candidate_distance_to_one"] = candidate_distance
        extras["reference_distance_to_one"] = reference_distance
        extras["delta_distance_to_one"] = candidate_distance - reference_distance
        return bool(candidate_distance < reference_distance), extras
    raise ValueError(f"Unsupported metric direction: {direction}")


def _build_metric_frame(
    metric_specs: Sequence[Mapping[str, str]],
    *,
    candidate_metrics: Mapping[str, float],
    reference_metrics: Optional[Mapping[str, float]],
    sample_sizes: Mapping[str, int],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for spec in metric_specs:
        metric_name = str(spec["metric"])
        direction = str(spec["direction"])
        candidate_value = float(candidate_metrics[metric_name])
        row: Dict[str, object] = {
            "metric": metric_name,
            "direction": direction,
            "tier": str(spec["tier"]),
            "reliability": str(spec["reliability"]),
            "reason": str(spec["reason"]),
            "sample_size": sample_sizes.get(metric_name),
            "candidate_value": candidate_value,
        }
        if reference_metrics is not None and metric_name in reference_metrics:
            reference_value = float(reference_metrics[metric_name])
            better, extras = _compare_metric_values(candidate_value, reference_value, direction)
            row.update(
                {
                    "reference_value": reference_value,
                    "delta_candidate_minus_reference": (
                        candidate_value - reference_value if math.isfinite(candidate_value) and math.isfinite(reference_value) else None
                    ),
                    "candidate_better": better,
                }
            )
            row.update(extras)
        rows.append(row)
    return pd.DataFrame(rows)


def build_metric_tables(
    *,
    candidate_metrics: Mapping[str, float],
    reference_metrics: Optional[Mapping[str, float]],
    sample_sizes: Mapping[str, int],
) -> Dict[str, pd.DataFrame]:
    core_df = _build_metric_frame(
        CORE_METRIC_SPECS,
        candidate_metrics=candidate_metrics,
        reference_metrics=reference_metrics,
        sample_sizes=sample_sizes,
    )
    secondary_df = _build_metric_frame(
        SECONDARY_METRIC_SPECS,
        candidate_metrics=candidate_metrics,
        reference_metrics=reference_metrics,
        sample_sizes=sample_sizes,
    )
    diagnostic_df = _build_metric_frame(
        DIAGNOSTIC_METRIC_SPECS,
        candidate_metrics=candidate_metrics,
        reference_metrics=reference_metrics,
        sample_sizes=sample_sizes,
    )
    reliability_catalog = pd.DataFrame([*CORE_METRIC_SPECS, *SECONDARY_METRIC_SPECS, *DIAGNOSTIC_METRIC_SPECS])
    combined_df = pd.concat([core_df, secondary_df, diagnostic_df], ignore_index=True)
    return {
        "core": core_df,
        "secondary": secondary_df,
        "diagnostic": diagnostic_df,
        "catalog": reliability_catalog,
        "combined": combined_df,
    }


def build_decision_summary(
    candidate_result: Mapping[str, object],
    reference_result: Optional[Mapping[str, object]],
    config: ResolvedBenchmarkConfig,
    *,
    candidate_metrics: Mapping[str, float],
    reference_metrics: Optional[Mapping[str, float]],
    metric_tables: Mapping[str, pd.DataFrame],
    runtime_check: Mapping[str, object],
) -> Dict[str, object]:
    full_test_samples = int(runtime_check["full_test_samples"])
    full_test_shards = int(runtime_check["full_test_shards"])
    evaluated_test_samples = int(config.test_max_samples)
    evaluated_test_shards = int(config.test_num_shards)
    summary: Dict[str, object] = {
        "candidate_label": config.candidate_label,
        "reference_label": config.reference_label if reference_result is not None else None,
        "candidate_checkpoint": str(config.candidate_checkpoint),
        "reference_checkpoint": str(config.reference_checkpoint) if config.reference_checkpoint is not None else None,
        "benchmark_scope": {
            "evaluated_test_samples": evaluated_test_samples,
            "evaluated_test_shards": evaluated_test_shards,
            "full_test_samples": full_test_samples,
            "full_test_shards": full_test_shards,
            "is_full_test": bool(evaluated_test_samples == full_test_samples and evaluated_test_shards == full_test_shards),
        },
        "metric_policy": {
            "core_metrics": metric_tables["core"]["metric"].tolist(),
            "secondary_metrics": metric_tables["secondary"]["metric"].tolist(),
            "diagnostic_metrics": metric_tables["diagnostic"]["metric"].tolist(),
        },
        "oracle_midband_mae_sum_stable": float(candidate_metrics["oracle_midband_mae_sum_stable"]),
        "oracle_stable_0.7_slope": float(candidate_metrics["oracle_stable_0.7_slope"]),
        "center_score": float(candidate_metrics["oracle_center_score"]),
        "test_max_midband_abs_cal_gap": float(candidate_metrics["max_midband_abs_cal_gap"]),
        "test_center_false_0.1eq": float(candidate_metrics["center_false_decisive_0.1eq"]),
        "test_center_false_0.2eq": float(candidate_metrics["center_false_decisive_0.2eq"]),
        "test_center_wrong_sign_0.1eq": float(candidate_metrics["center_wrong_sign_0.1eq"]),
        "test_center_wrong_sign_0.2eq": float(candidate_metrics["center_wrong_sign_0.2eq"]),
        "test_center_spread_ratio": float(candidate_metrics["center_spread_ratio"]),
        "core_candidate_better_count": int(metric_tables["core"]["candidate_better"].fillna(False).astype(bool).sum())
        if "candidate_better" in metric_tables["core"].columns
        else None,
        "core_candidate_worse_metrics": (
            metric_tables["core"].loc[metric_tables["core"]["candidate_better"].eq(False), "metric"].tolist()
            if "candidate_better" in metric_tables["core"].columns
            else []
        ),
    }

    if reference_result is None or reference_metrics is None:
        summary["offline_gate_ready"] = None
        return summary

    midband_gate_max = float(reference_metrics["oracle_midband_mae_sum_stable"]) * (1.0 + float(config.midband_rel_tol))
    slope_gate_min = float(reference_metrics["oracle_stable_0.7_slope"]) - float(config.slope_abs_tol)
    center_non_regression_max = float(reference_metrics["oracle_center_score"])

    midband_pass = float(candidate_metrics["oracle_midband_mae_sum_stable"]) <= midband_gate_max
    slope_pass = float(candidate_metrics["oracle_stable_0.7_slope"]) >= slope_gate_min
    center_non_regression_pass = float(candidate_metrics["oracle_center_score"]) <= center_non_regression_max

    summary.update(
        {
            "reference_oracle_midband_mae_sum_stable": float(reference_metrics["oracle_midband_mae_sum_stable"]),
            "reference_oracle_stable_0.7_slope": float(reference_metrics["oracle_stable_0.7_slope"]),
            "reference_center_score": float(reference_metrics["oracle_center_score"]),
            "midband_gate_max": midband_gate_max,
            "slope_gate_min": slope_gate_min,
            "center_non_regression_max": center_non_regression_max,
            "oracle_midband_gate_pass": bool(midband_pass),
            "oracle_slope_gate_pass": bool(slope_pass),
            "center_non_regression_pass": bool(center_non_regression_pass),
            "offline_gate_ready": bool(midband_pass and slope_pass and center_non_regression_pass),
        }
    )
    if config.center_strong_threshold is not None:
        summary["center_strong_threshold"] = float(config.center_strong_threshold)
        summary["center_strong_pass"] = bool(float(candidate_metrics["oracle_center_score"]) <= float(config.center_strong_threshold))
    return summary


def _plot_core_metrics(core_df: pd.DataFrame, output_path: Path, *, candidate_label: str, reference_label: Optional[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    sections = [
        ("Overall", ["overall_mse", "overall_mae", "overall_pearson"]),
        ("Error by Band", ["test_mse_0.1eq", "test_mse_0.2eq", "test_mse_0.5eq", "test_mse_0.7eq"]),
        (
            "Center Safety",
            [
                "center_false_decisive_0.1eq",
                "center_false_decisive_0.2eq",
                "center_wrong_sign_0.1eq",
                "center_wrong_sign_0.2eq",
            ],
        ),
        (
            "Directional / Calibration",
            [
                "sign_match_0.05_0.2eq",
                "sign_match_0.2_0.5eq",
                "sign_match_0.5_0.7eq",
                "abs_cal_gap_0.2_0.5eq",
                "abs_cal_gap_0.5_0.7eq",
            ],
        ),
    ]
    for ax, (title, metric_names) in zip(axes.flat, sections):
        frame = core_df[core_df["metric"].isin(metric_names)].reset_index(drop=True)
        xpos = np.arange(len(frame))
        width = 0.38
        ax.bar(xpos - width / 2.0, frame["candidate_value"].to_numpy(dtype=float), width=width, label=candidate_label)
        if "reference_value" in frame.columns and frame["reference_value"].notna().any():
            ax.bar(xpos + width / 2.0, frame["reference_value"].to_numpy(dtype=float), width=width, label=reference_label or "reference")
        ax.set_title(title)
        ax.set_xticks(xpos)
        ax.set_xticklabels(frame["metric"].tolist(), rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_prediction_vs_target(
    candidate_targets: np.ndarray,
    candidate_preds: np.ndarray,
    *,
    candidate_label: str,
    reference_targets: Optional[np.ndarray],
    reference_preds: Optional[np.ndarray],
    reference_label: Optional[str],
    candidate_metrics: Mapping[str, float],
    reference_metrics: Optional[Mapping[str, float]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ncols = 2 if reference_targets is not None and reference_preds is not None and reference_metrics is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 7), squeeze=False)
    panels = [
        (candidate_targets, candidate_preds, candidate_label, candidate_metrics),
    ]
    if ncols == 2:
        panels.append((reference_targets, reference_preds, reference_label or "reference", reference_metrics))
    for ax, (targets, preds, label, metrics) in zip(axes.flat, panels):
        hb = ax.hexbin(targets, preds, gridsize=65, extent=(-1, 1, -1, 1), mincnt=1, cmap="viridis")
        ax.plot([-1, 1], [-1, 1], linestyle="--", color="white", linewidth=1.2)
        ax.axhline(0.0, color="white", linewidth=0.8, alpha=0.4)
        ax.axvline(0.0, color="white", linewidth=0.8, alpha=0.4)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("target y")
        ax.set_ylabel("prediction")
        ax.set_title(
            f"{label}\n"
            f"MSE={metrics['overall_mse']:.4f} | MAE={metrics['overall_mae']:.4f} | r={metrics['overall_pearson']:.4f}"
        )
        fig.colorbar(hb, ax=ax, shrink=0.85, label="count")
    fig.suptitle("Prediction vs target on full test split", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_absolute_calibration(
    candidate_df: pd.DataFrame,
    *,
    candidate_label: str,
    reference_df: Optional[pd.DataFrame],
    reference_label: Optional[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax_curve, ax_count = axes
    diagonal = np.linspace(0.0, 1.0, 200)
    ax_curve.plot(diagonal, diagonal, linestyle="--", color="black", linewidth=1.0, label="ideal |pred|=|target|")

    for frame, label in (
        (candidate_df, candidate_label),
        (reference_df, reference_label),
    ):
        if frame is None or label is None:
            continue
        valid = frame["count"].to_numpy(dtype=float) > 0
        ax_curve.plot(
            frame.loc[valid, "mean_abs_target"].to_numpy(dtype=float),
            frame.loc[valid, "mean_abs_pred"].to_numpy(dtype=float),
            marker="o",
            linewidth=2.0,
            label=label,
        )
        ax_count.plot(
            frame.loc[valid, "mean_abs_target"].to_numpy(dtype=float),
            frame.loc[valid, "count"].to_numpy(dtype=float),
            marker="o",
            linewidth=2.0,
            label=label,
        )

    ax_curve.set_title("Absolute-value calibration by target-magnitude bin")
    ax_curve.set_xlabel("mean |target|")
    ax_curve.set_ylabel("mean |prediction|")
    ax_curve.grid(alpha=0.25)
    ax_curve.legend(loc="best")

    ax_count.set_title("Test-sample density by magnitude bin")
    ax_count.set_xlabel("mean |target|")
    ax_count.set_ylabel("count")
    ax_count.grid(alpha=0.25)
    ax_count.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_center_behavior(
    candidate_targets: np.ndarray,
    candidate_preds: np.ndarray,
    *,
    candidate_label: str,
    reference_targets: Optional[np.ndarray],
    reference_preds: Optional[np.ndarray],
    reference_label: Optional[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    center_mask = np.abs(candidate_targets) <= 0.05
    bins = np.linspace(0.0, 1.0, 60)

    ax_hist, ax_cdf = axes
    ax_hist.hist(
        np.abs(candidate_preds[center_mask]),
        bins=bins,
        histtype="step",
        linewidth=2.0,
        density=True,
        label=candidate_label,
    )
    if reference_targets is not None and reference_preds is not None and reference_label is not None:
        ref_center_mask = np.abs(reference_targets) <= 0.05
        ax_hist.hist(
            np.abs(reference_preds[ref_center_mask]),
            bins=bins,
            histtype="step",
            linewidth=2.0,
            density=True,
            label=reference_label,
        )
        ref_sorted = np.sort(np.abs(reference_preds[ref_center_mask]))
        if ref_sorted.size > 0:
            ax_cdf.plot(ref_sorted, np.arange(1, ref_sorted.size + 1) / ref_sorted.size, linewidth=2.0, label=reference_label)

    for ax in axes:
        ax.axvline(0.10, linestyle="--", color="tab:red", linewidth=1.2, label="false decisive @0.10")
        ax.axvline(0.20, linestyle=":", color="tab:orange", linewidth=1.2, label="false decisive @0.20")
        ax.grid(alpha=0.25)
        ax.set_xlim(0.0, 1.0)

    cand_sorted = np.sort(np.abs(candidate_preds[center_mask]))
    if cand_sorted.size > 0:
        ax_cdf.plot(cand_sorted, np.arange(1, cand_sorted.size + 1) / cand_sorted.size, linewidth=2.0, label=candidate_label)

    ax_hist.set_title("Abs(pred) on true-center positions (|y|<=0.05)")
    ax_hist.set_xlabel("|prediction|")
    ax_hist.set_ylabel("density")
    ax_hist.legend(loc="best")

    ax_cdf.set_title("CDF of |prediction| on true-center positions")
    ax_cdf.set_xlabel("|prediction|")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_sign_match_by_band(
    candidate_df: pd.DataFrame,
    *,
    candidate_label: str,
    reference_df: Optional[pd.DataFrame],
    reference_label: Optional[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_sign, ax_gap = axes
    xpos = np.arange(candidate_df.shape[0])
    width = 0.35

    ax_sign.bar(xpos - width / 2.0, candidate_df["sign_match"].to_numpy(dtype=float), width=width, label=candidate_label)
    if reference_df is not None and reference_label is not None:
        ax_sign.bar(xpos + width / 2.0, reference_df["sign_match"].to_numpy(dtype=float), width=width, label=reference_label)
    ax_sign.set_title("Sign match by absolute target band")
    ax_sign.set_xticks(xpos)
    ax_sign.set_xticklabels(candidate_df["band_label"].tolist())
    ax_sign.set_ylim(0.0, 1.0)
    ax_sign.set_ylabel("rate")
    ax_sign.grid(axis="y", alpha=0.25)
    ax_sign.legend(loc="best")

    candidate_gap_df = candidate_df[candidate_df["abs_cal_gap"].notna()].reset_index(drop=True)
    if reference_df is not None:
        reference_gap_df = reference_df[reference_df["abs_cal_gap"].notna()].reset_index(drop=True)
    else:
        reference_gap_df = None
    xpos_gap = np.arange(candidate_gap_df.shape[0])
    ax_gap.bar(xpos_gap - width / 2.0, candidate_gap_df["abs_cal_gap"].to_numpy(dtype=float), width=width, label=candidate_label)
    if reference_gap_df is not None and reference_label is not None:
        ax_gap.bar(xpos_gap + width / 2.0, reference_gap_df["abs_cal_gap"].to_numpy(dtype=float), width=width, label=reference_label)
    ax_gap.set_title("Absolute calibration gap by band")
    ax_gap.set_xticks(xpos_gap)
    ax_gap.set_xticklabels(candidate_gap_df["band_label"].tolist())
    ax_gap.set_ylabel("|mean|pred| - mean|target||")
    ax_gap.grid(axis="y", alpha=0.25)
    ax_gap.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_offline_benchmark(config: OfflineBenchmarkConfig, *, device: Optional[torch.device] = None) -> Dict[str, object]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = resolve_benchmark_config(config, device=device)
    runtime_check = validate_runtime_paths(resolved)
    paths = build_default_paths(PROJECT_ROOT, resolved.benchmark_name, outputs_root=resolved.outputs_root)
    for directory in (paths["benchmark_dir"], paths["reports_dir"], paths["plots_dir"]):
        directory.mkdir(parents=True, exist_ok=True)

    _save_json(runtime_check, paths["reports_dir"] / "runtime_check.json")
    if not bool(runtime_check["ok"]):
        raise FileNotFoundError(f"Runtime check failed: {runtime_check['missing']}")

    benchmark_config_payload = {
        "resolved_config": asdict(resolved),
        "device": str(device),
        "device_name": (torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"),
    }
    _save_json(benchmark_config_payload, paths["reports_dir"] / "benchmark_config.json")

    oracle_bundle, pooled_center_bundle, role_bundle = _load_bundles(resolved)
    candidate_result = evaluate_checkpoint(
        checkpoint_path=resolved.candidate_checkpoint,
        data_root=resolved.data_root,
        oracle_bundle=oracle_bundle,
        pooled_center_bundle=pooled_center_bundle,
        role_bundle=role_bundle,
        device=device,
        eval_batch_size=resolved.eval_batch_size,
        test_max_samples=resolved.test_max_samples,
        test_num_shards=resolved.test_num_shards,
    )
    reference_result = None
    if resolved.reference_checkpoint is not None:
        reference_result = evaluate_checkpoint(
            checkpoint_path=resolved.reference_checkpoint,
            data_root=resolved.data_root,
            oracle_bundle=oracle_bundle,
            pooled_center_bundle=pooled_center_bundle,
            role_bundle=role_bundle,
            device=device,
            eval_batch_size=resolved.eval_batch_size,
            test_max_samples=resolved.test_max_samples,
            test_num_shards=resolved.test_num_shards,
        )

    candidate_split_diag = build_split_diagnostics(candidate_result, abs_calibration_bins=resolved.abs_calibration_bins)
    reference_split_diag = (
        None
        if reference_result is None
        else build_split_diagnostics(reference_result, abs_calibration_bins=resolved.abs_calibration_bins)
    )
    candidate_metrics = build_metric_map(candidate_result, candidate_split_diag)
    reference_metrics = (
        None if reference_result is None or reference_split_diag is None else build_metric_map(reference_result, reference_split_diag)
    )
    sample_sizes = build_sample_sizes(candidate_result, candidate_split_diag)
    metric_tables = build_metric_tables(
        candidate_metrics=candidate_metrics,
        reference_metrics=reference_metrics,
        sample_sizes=sample_sizes,
    )
    decision_summary = build_decision_summary(
        candidate_result,
        reference_result,
        resolved,
        candidate_metrics=candidate_metrics,
        reference_metrics=reference_metrics,
        metric_tables=metric_tables,
        runtime_check=runtime_check,
    )

    candidate_compact = _compact_eval_result(candidate_result, split_diagnostics=candidate_split_diag)
    reference_compact = (
        None if reference_result is None or reference_split_diag is None else _compact_eval_result(reference_result, split_diagnostics=reference_split_diag)
    )

    candidate_calibration_df = candidate_split_diag["absolute_calibration"].assign(label=resolved.candidate_label)
    reference_calibration_df = (
        None
        if reference_split_diag is None
        else reference_split_diag["absolute_calibration"].assign(label=resolved.reference_label)
    )
    calibration_df = (
        candidate_calibration_df
        if reference_calibration_df is None
        else pd.concat([candidate_calibration_df, reference_calibration_df], ignore_index=True)
    )
    candidate_band_df = candidate_split_diag["band_diagnostics"].assign(label=resolved.candidate_label)
    reference_band_df = (
        None
        if reference_split_diag is None
        else reference_split_diag["band_diagnostics"].assign(label=resolved.reference_label)
    )
    band_df = candidate_band_df if reference_band_df is None else pd.concat([candidate_band_df, reference_band_df], ignore_index=True)

    _save_json(candidate_compact, paths["reports_dir"] / "candidate_eval_summary.json")
    if reference_compact is not None:
        _save_json(reference_compact, paths["reports_dir"] / "reference_eval_summary.json")
    _save_json({"sample_sizes": sample_sizes}, paths["reports_dir"] / "sample_sizes.json")
    metric_tables["combined"].to_csv(paths["reports_dir"] / "metrics_table.csv", index=False)
    metric_tables["core"].to_csv(paths["reports_dir"] / "core_metrics_table.csv", index=False)
    metric_tables["secondary"].to_csv(paths["reports_dir"] / "secondary_metrics_table.csv", index=False)
    metric_tables["diagnostic"].to_csv(paths["reports_dir"] / "diagnostic_metrics_table.csv", index=False)
    metric_tables["catalog"].to_csv(paths["reports_dir"] / "metric_reliability_catalog.csv", index=False)
    calibration_df.to_csv(paths["reports_dir"] / "absolute_calibration_curve.csv", index=False)
    band_df.to_csv(paths["reports_dir"] / "band_diagnostics.csv", index=False)
    _save_json(decision_summary, paths["reports_dir"] / "decision_summary.json")

    _plot_core_metrics(
        metric_tables["core"],
        paths["plots_dir"] / "core_metrics.png",
        candidate_label=resolved.candidate_label,
        reference_label=resolved.reference_label if reference_result is not None else None,
    )
    _plot_prediction_vs_target(
        candidate_split_diag["targets"],
        candidate_split_diag["preds"],
        candidate_label=resolved.candidate_label,
        reference_targets=(None if reference_split_diag is None else reference_split_diag["targets"]),
        reference_preds=(None if reference_split_diag is None else reference_split_diag["preds"]),
        reference_label=(resolved.reference_label if reference_split_diag is not None else None),
        candidate_metrics=candidate_metrics,
        reference_metrics=reference_metrics,
        output_path=paths["plots_dir"] / "prediction_vs_target_hexbin.png",
    )
    _plot_absolute_calibration(
        candidate_calibration_df,
        candidate_label=resolved.candidate_label,
        reference_df=reference_calibration_df,
        reference_label=(resolved.reference_label if reference_calibration_df is not None else None),
        output_path=paths["plots_dir"] / "absolute_calibration.png",
    )
    _plot_center_behavior(
        candidate_split_diag["targets"],
        candidate_split_diag["preds"],
        candidate_label=resolved.candidate_label,
        reference_targets=(None if reference_split_diag is None else reference_split_diag["targets"]),
        reference_preds=(None if reference_split_diag is None else reference_split_diag["preds"]),
        reference_label=(resolved.reference_label if reference_split_diag is not None else None),
        output_path=paths["plots_dir"] / "center_behavior.png",
    )
    _plot_sign_match_by_band(
        candidate_band_df,
        candidate_label=resolved.candidate_label,
        reference_df=reference_band_df,
        reference_label=(resolved.reference_label if reference_band_df is not None else None),
        output_path=paths["plots_dir"] / "sign_match_by_band.png",
    )

    bootstrap_path = None
    if (
        resolved.run_bootstrap_compare
        and reference_result is not None
        and resolved.bootstrap_n > 0
    ):
        bootstrap_cfg = obj_lab.BootstrapConfig(
            n_bootstrap=int(resolved.bootstrap_n),
            seed=int(resolved.bootstrap_seed),
            ci_alpha=float(resolved.bootstrap_ci_alpha),
        )
        bootstrap_df = obj_lab.bootstrap_compare_to_baseline(
            baseline_result=reference_result,
            candidate_result=candidate_result,
            cfg=bootstrap_cfg,
        )
        bootstrap_path = paths["reports_dir"] / "oracle_bootstrap_compare.csv"
        bootstrap_df.to_csv(bootstrap_path, index=False)

    return {
        "paths": {k: str(v) for k, v in paths.items()},
        "runtime_check": runtime_check,
        "benchmark_config": benchmark_config_payload,
        "decision_summary": decision_summary,
        "metrics_table": metric_tables["combined"],
        "bootstrap_report_path": (str(bootstrap_path) if bootstrap_path is not None else None),
    }
