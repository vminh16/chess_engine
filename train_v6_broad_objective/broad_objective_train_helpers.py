from __future__ import annotations

import gc
import importlib.util
import json
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.amp import autocast


def _find_repo_root(start: Path) -> Path:
    start = start.resolve()
    markers = (
        ("model",),
        ("experiments", "objective_resolution_suite"),
        ("train_v2_TF1", "ft1_colab_helpers.py"),
    )
    for candidate in [start.parent, *start.parents]:
        if all((candidate.joinpath(*parts)).exists() for parts in markers):
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
]

for path in [PROJECT_ROOT, MODEL_ROOT, *EXPERIMENT_DIRS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from architecture_v2.model import DGRNChessNetV2  # noqa: E402
import teacher_root_cause_helpers as base_lab  # noqa: E402
import root_cause_ablation_helpers as ab_lab  # noqa: E402
import objective_resolution_helpers as obj_lab  # noqa: E402

shared_ft1 = _import_module_from_file(
    "train_v2_tf1_ft1_colab_helpers",
    PROJECT_ROOT / "train_v2_TF1" / "ft1_colab_helpers.py",
)


@dataclass
class Report1TrainConfig:
    run_name: str = "dgrn_5m_phasec1_broad_objective_sglobal_16b_256d_random_run1"
    phase_name: str = "C1_BROAD_OBJECTIVE"
    epochs: int = 15
    sampling_mode: str = "random"
    main_batch_size: int = 768
    clean_center_batch_size: int = 0
    ambiguous_center_batch_size: int = 0
    grad_accum_steps: int = 1
    eval_batch_size: int = 4096
    learning_rate: float = 1.0e-4
    min_lr: float = 1.0e-5
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 1.0
    seed: int = 123
    train_num_shards: Optional[int] = None
    val_num_shards: int = 2
    test_num_shards: int = 4
    val_max_samples: int = 100_000
    test_max_samples: int = 200_000
    log_every_steps: int = 200
    grad_monitor_every_steps: int = 1000
    use_amp: bool = True
    amp_dtype: str = "float16"
    amp_loss_scale: float = 128.0
    preload_shard_dtype: str = "auto"
    channels_last: bool = True
    cudnn_benchmark: bool = True
    pin_memory_batches: bool = False
    prefetch_shards: bool = True
    prefetch_workers: int = 1
    benchmark_steps: int = 12
    benchmark_warmup_steps: int = 3
    benchmark_num_shards: int = 1
    max_profile_mem_ratio: float = 0.82
    periodic_save_minutes: int = 30
    save_epoch_checkpoints: bool = False
    resume_if_exists: bool = True
    main_center_tau_y600: float = 0.10
    main_center_min_weight: float = 0.35
    main_center_weight_power: float = 1.0
    lambda_clean_center: float = 0.20
    lambda_ambiguous_center: float = 0.10
    aux_margin_y600: float = 0.08
    aux_margin_weight: float = 0.40
    aux_huber_delta: float = 0.05
    aux_ramp_epochs: int = 4
    use_backbone_pcgrad: bool = True
    pcgrad_eps: float = 1.0e-12
    broad_y_huber_delta: float = 0.10
    broad_z_huber_weight: float = 0.20
    broad_z_huber_delta: float = 1.00
    broad_center_tau_y600: float = 0.05
    broad_center_pred_margin_y600: float = 0.10
    broad_center_margin_weight: float = 2.00
    broad_abs_calibration_weight: float = 0.50
    broad_abs_calibration_min_count: int = 8
    broad_abs_calibration_edges_y600: Tuple[float, ...] = (0.00, 0.05, 0.10, 0.20, 0.50, 0.70)


@dataclass
class Report1GateConfig:
    midband_mae_rel_tol: float = 0.05
    stable_slope_abs_tol: float = 0.02
    broad_overall_mse_rel_tol: float = 0.02
    broad_mse_0p1_rel_tol: float = 0.02
    broad_center_false_0p1_abs_tol: float = 0.01
    broad_abs_cal_rel_tol: float = 0.05


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_global_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_rng_state() -> Dict[str, object]:
    state: Dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: object) -> None:
    if not isinstance(state, dict):
        return
    try:
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch_cpu" in state:
            torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and ("torch_cuda" in state):
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except Exception as exc:
        print(f"[phase_c1][resume] Failed to restore RNG state ({exc}); continuing with seeded RNG")


def _resolve_amp_dtype(amp_dtype: object) -> torch.dtype:
    value = str(amp_dtype).strip().lower()
    if value in {"float16", "fp16", "half"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    return torch.float16


def _resolve_preload_numpy_dtype(preload_dtype: object, use_amp: bool, amp_dtype: object) -> Optional[np.dtype]:
    value = str(preload_dtype).strip().lower()
    if value in {"", "none", "false", "off", "uint8"}:
        return None
    if value == "auto":
        return np.float16 if (bool(use_amp) and _resolve_amp_dtype(amp_dtype) == torch.float16) else np.float32
    if value in {"float16", "fp16", "half"}:
        return np.float16
    if value in {"float32", "fp32"}:
        return np.float32
    raise ValueError(f"Unsupported preload_shard_dtype: {preload_dtype}")


def _unscale_grad_list(
    grads: Sequence[Optional[torch.Tensor]],
    loss_scale: float,
) -> List[Optional[torch.Tensor]]:
    loss_scale = float(loss_scale)
    if abs(loss_scale - 1.0) <= 1e-12:
        return [None if grad is None else grad.detach() for grad in grads]
    inv_scale = 1.0 / loss_scale
    return [None if grad is None else (grad.detach() * inv_scale) for grad in grads]


def _all_grads_finite(params: Sequence[nn.Parameter]) -> bool:
    for param in params:
        grad = param.grad
        if grad is None:
            continue
        if not bool(torch.isfinite(grad).all()):
            return False
    return True


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return [_json_ready(item) for item in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return {str(k): _json_ready(v) for k, v in value.to_dict().items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _torch_load_trusted_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> object:
    """Load an internally generated training checkpoint.

    Phase C1 checkpoints store Python and NumPy RNG states for exact resume.
    PyTorch 2.6+ defaults ``torch.load`` to ``weights_only=True``, which rejects
    those non-tensor objects. These files are produced by this repo's own
    training loop, so using ``weights_only=False`` is the correct resume path.
    """
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def _report1_train_cfg_from_mapping(raw_cfg: object, fallback: Report1TrainConfig) -> Report1TrainConfig:
    allowed = {field.name for field in fields(Report1TrainConfig)}
    values = asdict(fallback)
    if isinstance(raw_cfg, dict):
        values.update({str(key): value for key, value in raw_cfg.items() if str(key) in allowed})
    return Report1TrainConfig(**values)


def _apply_resume_safe_train_cfg_overrides(saved_cfg: Report1TrainConfig, requested_cfg: Report1TrainConfig) -> Report1TrainConfig:
    values = asdict(saved_cfg)
    for key in (
        "epochs",
        "resume_if_exists",
        "log_every_steps",
        "grad_monitor_every_steps",
        "benchmark_steps",
        "benchmark_warmup_steps",
        "periodic_save_minutes",
        "save_epoch_checkpoints",
    ):
        values[key] = getattr(requested_cfg, key)
    return Report1TrainConfig(**values)


def _load_existing_run_config(paths: Dict[str, Path], latest_ckpt: Path) -> Optional[Dict[str, object]]:
    run_config_path = paths["reports_dir"] / "run_config.json"
    if run_config_path.exists():
        return json.loads(run_config_path.read_text(encoding="utf-8"))
    if latest_ckpt.exists():
        payload = _torch_load_trusted_checkpoint(latest_ckpt, map_location="cpu")
        if isinstance(payload, dict):
            config = payload.get("config")
            if isinstance(config, dict):
                return dict(config)
    return None


def _assert_resume_model_cfg_compatible(saved_config: Optional[Dict[str, object]], requested_model_cfg: Dict[str, object]) -> None:
    if not isinstance(saved_config, dict):
        return
    saved_model_cfg = saved_config.get("model_cfg")
    if not isinstance(saved_model_cfg, dict):
        return
    saved = {str(key): value for key, value in saved_model_cfg.items()}
    requested = {str(key): value for key, value in requested_model_cfg.items()}
    if saved != requested:
        diffs = []
        for key in sorted(set(saved) | set(requested)):
            if saved.get(key) != requested.get(key):
                diffs.append(f"{key}: checkpoint={saved.get(key)!r}, requested={requested.get(key)!r}")
        raise RuntimeError(
            "Resume model_cfg mismatch for existing report1 run. "
            "Use the original experiment settings or a new run_name. "
            f"Diffs: {'; '.join(diffs[:12])}"
        )


def validate_report1_train_config(train_cfg: Report1TrainConfig) -> None:
    positive_int_fields = {
        "epochs": train_cfg.epochs,
        "main_batch_size": train_cfg.main_batch_size,
        "grad_accum_steps": train_cfg.grad_accum_steps,
        "eval_batch_size": train_cfg.eval_batch_size,
        "log_every_steps": train_cfg.log_every_steps,
        "grad_monitor_every_steps": train_cfg.grad_monitor_every_steps,
        "benchmark_steps": train_cfg.benchmark_steps,
        "benchmark_num_shards": train_cfg.benchmark_num_shards,
        "prefetch_workers": train_cfg.prefetch_workers,
        "periodic_save_minutes": train_cfg.periodic_save_minutes,
    }
    for name, value in positive_int_fields.items():
        if int(value) <= 0:
            raise ValueError(f"train_cfg.{name} must be > 0")
    non_negative_int_fields = {
        "clean_center_batch_size": train_cfg.clean_center_batch_size,
        "ambiguous_center_batch_size": train_cfg.ambiguous_center_batch_size,
        "broad_abs_calibration_min_count": train_cfg.broad_abs_calibration_min_count,
    }
    for name, value in non_negative_int_fields.items():
        if int(value) < 0:
            raise ValueError(f"train_cfg.{name} must be >= 0")
    if int(train_cfg.benchmark_warmup_steps) < 0:
        raise ValueError("train_cfg.benchmark_warmup_steps must be >= 0")
    if float(train_cfg.learning_rate) <= 0.0:
        raise ValueError("train_cfg.learning_rate must be > 0")
    if float(train_cfg.min_lr) < 0.0:
        raise ValueError("train_cfg.min_lr must be >= 0")
    if float(train_cfg.min_lr) > float(train_cfg.learning_rate):
        raise ValueError("train_cfg.min_lr must be <= train_cfg.learning_rate")
    if float(train_cfg.amp_loss_scale) <= 0.0:
        raise ValueError("train_cfg.amp_loss_scale must be > 0")
    if not (0.0 < float(train_cfg.max_profile_mem_ratio) <= 1.0):
        raise ValueError("train_cfg.max_profile_mem_ratio must be in (0, 1]")
    sampling_mode = str(train_cfg.sampling_mode).strip().lower()
    if sampling_mode not in {"random", "band_balanced", "sign_stratified"}:
        raise ValueError("train_cfg.sampling_mode must be one of: random, band_balanced, sign_stratified")
    if float(train_cfg.broad_y_huber_delta) <= 0.0:
        raise ValueError("train_cfg.broad_y_huber_delta must be > 0")
    if float(train_cfg.broad_z_huber_weight) < 0.0:
        raise ValueError("train_cfg.broad_z_huber_weight must be >= 0")
    if float(train_cfg.broad_z_huber_delta) <= 0.0:
        raise ValueError("train_cfg.broad_z_huber_delta must be > 0")
    if float(train_cfg.broad_center_tau_y600) <= 0.0:
        raise ValueError("train_cfg.broad_center_tau_y600 must be > 0")
    if float(train_cfg.broad_center_pred_margin_y600) <= 0.0:
        raise ValueError("train_cfg.broad_center_pred_margin_y600 must be > 0")
    if float(train_cfg.broad_center_margin_weight) < 0.0:
        raise ValueError("train_cfg.broad_center_margin_weight must be >= 0")
    if float(train_cfg.broad_abs_calibration_weight) < 0.0:
        raise ValueError("train_cfg.broad_abs_calibration_weight must be >= 0")
    edges = tuple(float(v) for v in train_cfg.broad_abs_calibration_edges_y600)
    if len(edges) < 2 or any(edges[idx] >= edges[idx + 1] for idx in range(len(edges) - 1)):
        raise ValueError("train_cfg.broad_abs_calibration_edges_y600 must be strictly increasing")


PROFILE_TUNE_KEYS = {
    "main_batch_size",
    "clean_center_batch_size",
    "ambiguous_center_batch_size",
    "grad_accum_steps",
    "eval_batch_size",
    "preload_shard_dtype",
    "prefetch_workers",
    "benchmark_num_shards",
}


def default_device_profile(gpu_name: Optional[str], total_mem_gb: float) -> Dict[str, object]:
    gpu_name = (gpu_name or "").upper()
    if "T4" in gpu_name and total_mem_gb >= 14.0:
        return {
            "main_batch_size": 1536,
            "clean_center_batch_size": 192,
            "ambiguous_center_batch_size": 384,
            "grad_accum_steps": 1,
            "eval_batch_size": 6144,
            "preload_shard_dtype": "none",
            "prefetch_workers": 1,
            "benchmark_num_shards": 2,
        }
    if total_mem_gb >= 14.0:
        return {
            "main_batch_size": 1280,
            "clean_center_batch_size": 160,
            "ambiguous_center_batch_size": 320,
            "grad_accum_steps": 1,
            "eval_batch_size": 5120,
            "preload_shard_dtype": "none",
            "prefetch_workers": 1,
            "benchmark_num_shards": 2,
        }
    if total_mem_gb >= 8.0:
        return {
            "main_batch_size": 384,
            "clean_center_batch_size": 48,
            "ambiguous_center_batch_size": 96,
            "grad_accum_steps": 1,
            "eval_batch_size": 2048,
            "preload_shard_dtype": "auto",
            "prefetch_workers": 1,
            "benchmark_num_shards": 1,
        }
    return {
        "main_batch_size": 256,
        "clean_center_batch_size": 32,
        "ambiguous_center_batch_size": 64,
        "grad_accum_steps": 2,
        "eval_batch_size": 1280,
        "preload_shard_dtype": "auto",
        "prefetch_workers": 1,
        "benchmark_num_shards": 1,
    }


def candidate_device_profiles(gpu_name: Optional[str], total_mem_gb: float) -> List[Dict[str, object]]:
    gpu_name = (gpu_name or "").upper()
    candidates: List[Dict[str, object]] = []
    if "T4" in gpu_name and total_mem_gb >= 14.0:
        candidates.extend(
            [
                {"main_batch_size": 2304, "clean_center_batch_size": 288, "ambiguous_center_batch_size": 576, "grad_accum_steps": 1, "eval_batch_size": 8192, "preload_shard_dtype": "none", "prefetch_workers": 1, "benchmark_num_shards": 2},
                {"main_batch_size": 2048, "clean_center_batch_size": 256, "ambiguous_center_batch_size": 512, "grad_accum_steps": 1, "eval_batch_size": 8192, "preload_shard_dtype": "none", "prefetch_workers": 1, "benchmark_num_shards": 2},
                {"main_batch_size": 1792, "clean_center_batch_size": 224, "ambiguous_center_batch_size": 448, "grad_accum_steps": 1, "eval_batch_size": 7168, "preload_shard_dtype": "none", "prefetch_workers": 1, "benchmark_num_shards": 2},
                {"main_batch_size": 1536, "clean_center_batch_size": 192, "ambiguous_center_batch_size": 384, "grad_accum_steps": 1, "eval_batch_size": 6144, "preload_shard_dtype": "none", "prefetch_workers": 1, "benchmark_num_shards": 2},
                {"main_batch_size": 1280, "clean_center_batch_size": 160, "ambiguous_center_batch_size": 320, "grad_accum_steps": 1, "eval_batch_size": 5120, "preload_shard_dtype": "none", "prefetch_workers": 1, "benchmark_num_shards": 2},
                {"main_batch_size": 1536, "clean_center_batch_size": 192, "ambiguous_center_batch_size": 384, "grad_accum_steps": 1, "eval_batch_size": 6144, "preload_shard_dtype": "auto", "prefetch_workers": 1, "benchmark_num_shards": 2},
                {"main_batch_size": 1024, "clean_center_batch_size": 128, "ambiguous_center_batch_size": 256, "grad_accum_steps": 1, "eval_batch_size": 4096, "preload_shard_dtype": "auto", "prefetch_workers": 1, "benchmark_num_shards": 2},
            ]
        )
    elif total_mem_gb >= 8.0:
        candidates.extend(
            [
                {"main_batch_size": 512, "clean_center_batch_size": 64, "ambiguous_center_batch_size": 128, "grad_accum_steps": 1, "eval_batch_size": 3072, "preload_shard_dtype": "auto", "prefetch_workers": 1, "benchmark_num_shards": 1},
                {"main_batch_size": 448, "clean_center_batch_size": 56, "ambiguous_center_batch_size": 112, "grad_accum_steps": 1, "eval_batch_size": 2560, "preload_shard_dtype": "auto", "prefetch_workers": 1, "benchmark_num_shards": 1},
                {"main_batch_size": 384, "clean_center_batch_size": 48, "ambiguous_center_batch_size": 96, "grad_accum_steps": 1, "eval_batch_size": 2048, "preload_shard_dtype": "auto", "prefetch_workers": 1, "benchmark_num_shards": 1},
                {"main_batch_size": 320, "clean_center_batch_size": 40, "ambiguous_center_batch_size": 80, "grad_accum_steps": 1, "eval_batch_size": 2048, "preload_shard_dtype": "auto", "prefetch_workers": 1, "benchmark_num_shards": 1},
            ]
        )
    else:
        candidates.extend(
            [
                {"main_batch_size": 320, "clean_center_batch_size": 40, "ambiguous_center_batch_size": 80, "grad_accum_steps": 2, "eval_batch_size": 1280, "preload_shard_dtype": "auto", "prefetch_workers": 1, "benchmark_num_shards": 1},
                {"main_batch_size": 288, "clean_center_batch_size": 36, "ambiguous_center_batch_size": 72, "grad_accum_steps": 2, "eval_batch_size": 1280, "preload_shard_dtype": "auto", "prefetch_workers": 1, "benchmark_num_shards": 1},
                {"main_batch_size": 256, "clean_center_batch_size": 32, "ambiguous_center_batch_size": 64, "grad_accum_steps": 2, "eval_batch_size": 1280, "preload_shard_dtype": "auto", "prefetch_workers": 1, "benchmark_num_shards": 1},
                {"main_batch_size": 224, "clean_center_batch_size": 28, "ambiguous_center_batch_size": 56, "grad_accum_steps": 2, "eval_batch_size": 1024, "preload_shard_dtype": "auto", "prefetch_workers": 1, "benchmark_num_shards": 1},
                {"main_batch_size": 192, "clean_center_batch_size": 24, "ambiguous_center_batch_size": 48, "grad_accum_steps": 2, "eval_batch_size": 1024, "preload_shard_dtype": "auto", "prefetch_workers": 1, "benchmark_num_shards": 1},
                {"main_batch_size": 160, "clean_center_batch_size": 20, "ambiguous_center_batch_size": 40, "grad_accum_steps": 2, "eval_batch_size": 768, "preload_shard_dtype": "auto", "prefetch_workers": 1, "benchmark_num_shards": 1},
                {"main_batch_size": 128, "clean_center_batch_size": 16, "ambiguous_center_batch_size": 32, "grad_accum_steps": 2, "eval_batch_size": 768, "preload_shard_dtype": "auto", "prefetch_workers": 1, "benchmark_num_shards": 1},
            ]
        )
    candidates.append(default_device_profile(gpu_name, total_mem_gb))
    dedup: List[Dict[str, object]] = []
    seen: set[Tuple[int, int, int, int, int, str, int, int]] = set()
    for item in candidates:
        item = dict(item)
        item["clean_center_batch_size"] = 0
        item["ambiguous_center_batch_size"] = 0
        key = (
            int(item["main_batch_size"]),
            int(item["clean_center_batch_size"]),
            int(item["ambiguous_center_batch_size"]),
            int(item["grad_accum_steps"]),
            int(item["eval_batch_size"]),
            str(item.get("preload_shard_dtype", "auto")).strip().lower(),
            int(item.get("prefetch_workers", 1)),
            int(item.get("benchmark_num_shards", 1)),
        )
        if key in seen:
            continue
        seen.add(key)
        dedup.append(item)
    return dedup


def build_default_paths(repo_root: Path, runs_root: Path, run_name: str) -> Dict[str, Path]:
    run_dir = Path(runs_root) / str(run_name)
    checkpoints_dir = run_dir / "checkpoints"
    reports_dir = run_dir / "reports"
    plots_dir = run_dir / "plots"
    for path in (run_dir, checkpoints_dir, reports_dir, plots_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "repo_root": Path(repo_root),
        "run_dir": run_dir,
        "checkpoints_dir": checkpoints_dir,
        "reports_dir": reports_dir,
        "plots_dir": plots_dir,
        "l4_reference_ckpt": repo_root
        / "experiments"
        / "objective_resolution_suite"
        / "outputs"
        / "runs"
        / "L4_A1_plus_A2"
        / "checkpoints"
        / "L4_A1_plus_A2_best.pt",
        "oracle_role_bundle_dir": repo_root
        / "experiments"
        / "oc2_joint_oracle_full_model_pilot"
        / "outputs"
        / "cache"
        / "oracle_role_bundle",
        "pooled_center_bundle_dir": repo_root
        / "experiments"
        / "failure_b_resolution_suite"
        / "outputs"
        / "cache"
        / "pooled_center_bundle",
    }


def validate_report1_runtime_paths(data_root: Path, paths: Dict[str, Path]) -> Dict[str, object]:
    required = {
        "data_root": data_root,
        "split_train": data_root / "train",
        "split_val": data_root / "val",
        "split_test": data_root / "test",
        "l4_reference_ckpt": paths["l4_reference_ckpt"],
        "oracle_role_bundle_dir": paths["oracle_role_bundle_dir"],
        "pooled_center_bundle_dir": paths["pooled_center_bundle_dir"],
    }
    missing = {k: str(v) for k, v in required.items() if not Path(v).exists()}
    return {"ok": not missing, "paths": {k: str(v) for k, v in required.items()}, "missing": missing}


def _sample_aux_indices(role_bundle: Dict[str, object], cfg: Report1TrainConfig, rng: np.random.Generator) -> np.ndarray:
    clean_pool = np.asarray(role_bundle["indices_by_role"]["clean_center"], dtype=np.int64)
    ambiguous_pool = np.asarray(role_bundle["indices_by_role"]["center_ambiguous"], dtype=np.int64)
    clean_size = max(int(cfg.clean_center_batch_size), 0)
    ambiguous_size = max(int(cfg.ambiguous_center_batch_size), 0)
    if clean_size == 0 and ambiguous_size == 0:
        return np.empty((0,), dtype=np.int64)
    clean = rng.choice(
        clean_pool,
        size=clean_size,
        replace=clean_pool.size < clean_size,
    ).astype(np.int64)
    ambiguous = rng.choice(
        ambiguous_pool,
        size=ambiguous_size,
        replace=ambiguous_pool.size < ambiguous_size,
    ).astype(np.int64)
    out = np.concatenate([clean, ambiguous], axis=0).astype(np.int64)
    rng.shuffle(out)
    return out


def _to_device_batch(
    array: np.ndarray,
    device: torch.device,
    channels_last: bool,
    pin_memory: bool,
    use_amp: bool,
    amp_dtype: object,
) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    fp16_inputs = bool(use_amp) and device.type == "cuda" and (_resolve_amp_dtype(amp_dtype) == torch.float16)
    target_dtype = torch.float16 if fp16_inputs else torch.float32
    if pin_memory and device.type == "cuda":
        tensor = tensor.pin_memory()
    needs_cast = False
    if tensor.dtype == torch.uint8:
        needs_cast = True
    elif tensor.is_floating_point():
        needs_cast = tensor.dtype != target_dtype
    else:
        needs_cast = True
    if needs_cast:
        tensor = tensor.to(device=device, dtype=target_dtype, non_blocking=(device.type == "cuda"))
    else:
        tensor = tensor.to(device=device, non_blocking=(device.type == "cuda"))
    if channels_last and tensor.ndim == 4:
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    return tensor


def _scale_value_from_y600(value: float, target_scale: float) -> float:
    return float(ab_lab.canonical_y600_to_scale_value(float(value), float(target_scale)))


def _zero_like_objective(logits: torch.Tensor) -> torch.Tensor:
    return logits.view(-1).sum() * 0.0


def _safe_masked_mean(values: torch.Tensor, mask: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=values.dtype)
    count = mask_f.sum()
    mean = (values * mask_f).sum() / torch.clamp(count, min=1.0)
    return torch.where(count > 0.0, mean, fallback)


def _batch_abs_calibration_penalty(
    pred: torch.Tensor,
    y: torch.Tensor,
    variant: object,
    cfg: Report1TrainConfig,
) -> torch.Tensor:
    zero = pred.sum() * 0.0
    min_count = max(int(cfg.broad_abs_calibration_min_count), 1)
    edges_y600 = tuple(float(v) for v in cfg.broad_abs_calibration_edges_y600)
    edges = [
        _scale_value_from_y600(edge, float(variant.target_scale))
        for edge in edges_y600
    ]
    abs_y = torch.abs(y)
    abs_pred = torch.abs(pred)
    penalties: List[torch.Tensor] = []
    active_terms: List[torch.Tensor] = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (abs_y >= float(left)) & (abs_y < float(right))
        mask_f = mask.to(dtype=pred.dtype)
        count = mask_f.sum()
        active = (count >= float(min_count)).to(dtype=pred.dtype)
        mean_pred = (abs_pred * mask_f).sum() / torch.clamp(count, min=1.0)
        mean_y = (abs_y * mask_f).sum() / torch.clamp(count, min=1.0)
        gap = mean_pred - mean_y
        penalties.append((gap * gap) * active)
        active_terms.append(active)
    if not penalties:
        return zero
    active_count = torch.stack(active_terms).sum()
    return torch.stack(penalties).sum() / torch.clamp(active_count, min=1.0)


def compute_broad_main_terms(
    logits: torch.Tensor,
    y_source: torch.Tensor,
    variant: object,
    cfg: Report1TrainConfig,
) -> Dict[str, torch.Tensor]:
    y = ab_lab.remap_target_torch(y_source.view(-1), to_scale=float(variant.target_scale))
    logits = logits.view(-1)
    pred = torch.tanh(logits)
    zero = pred.sum() * 0.0

    y_residual = pred - y
    y_huber_per = ab_lab.huber_per_sample(y_residual, float(cfg.broad_y_huber_delta))
    y_huber = y_huber_per.mean()

    y_logits = ab_lab.target_to_logits(y, eps=float(variant.target_clamp_eps))
    z_residual = logits - y_logits
    z_huber = ab_lab.huber_per_sample(z_residual, float(cfg.broad_z_huber_delta)).mean()

    center_tau = _scale_value_from_y600(float(cfg.broad_center_tau_y600), float(variant.target_scale))
    center_margin = _scale_value_from_y600(float(cfg.broad_center_pred_margin_y600), float(variant.target_scale))
    center_mask = torch.abs(y) <= float(center_tau)
    center_margin_per = torch.relu(torch.abs(pred) - float(center_margin)) ** 2
    center_margin_penalty = _safe_masked_mean(center_margin_per, center_mask, zero)

    abs_calibration_penalty = _batch_abs_calibration_penalty(pred, y, variant, cfg)
    objective = (
        y_huber
        + float(cfg.broad_z_huber_weight) * z_huber
        + float(cfg.broad_center_margin_weight) * center_margin_penalty
        + float(cfg.broad_abs_calibration_weight) * abs_calibration_penalty
    )
    return {
        "objective": objective,
        "pred": pred,
        "mean_main_weight": torch.ones((), device=logits.device, dtype=logits.dtype),
        "downweighted_frac": torch.zeros((), device=logits.device, dtype=logits.dtype),
        "mean_y_term": y_huber.detach(),
        "mean_z_term": z_huber.detach(),
        "center_margin_penalty": center_margin_penalty,
        "abs_calibration_penalty": abs_calibration_penalty,
    }


def compute_disabled_aux_terms(
    logits: torch.Tensor,
    oracle_y: torch.Tensor,
    role_code: torch.Tensor,
    cfg: Report1TrainConfig,
) -> Dict[str, torch.Tensor]:
    zero = _zero_like_objective(logits)
    pred = torch.tanh(logits.view(-1))
    return {
        "objective": zero,
        "pred": pred,
        "clean_loss": zero.detach(),
        "ambiguous_loss": zero.detach(),
        "margin_penalty": zero.detach(),
        "clean_frac": zero.detach(),
        "ambiguous_frac": zero.detach(),
    }


def build_sign_stratified_order(
    y: np.ndarray,
    batch_size: int,
    band_edges_y600: Sequence[float],
    rng: np.random.Generator,
    target_scale: float,
) -> np.ndarray:
    """Balanced sampler that preserves FT1 absolute bands and adds sign separation outside the center band.

    Bucket layout:
    - band 0: |y| in [0.0, 0.05) kept unsplit because sign is the noisiest near zero
    - bands >= 1: split into negative / positive buckets
    """
    y = np.asarray(y, dtype=np.float64)
    abs_y = np.abs(y)
    band_edges = np.asarray(
        [ab_lab.canonical_y600_to_scale_value(v, target_scale) for v in band_edges_y600],
        dtype=np.float64,
    )
    bins = np.clip(np.digitize(abs_y, band_edges[1:-1], right=False), 0, len(band_edges) - 2)
    bucket_indices: List[np.ndarray] = []

    center_idx = np.flatnonzero(bins == 0).astype(np.int64)
    if center_idx.size > 0:
        bucket_indices.append(rng.permutation(center_idx).astype(np.int64))

    for band_idx in range(1, len(band_edges) - 1):
        band_mask = bins == band_idx
        neg_idx = np.flatnonzero(band_mask & (y < 0.0)).astype(np.int64)
        pos_idx = np.flatnonzero(band_mask & (y > 0.0)).astype(np.int64)
        if neg_idx.size > 0:
            bucket_indices.append(rng.permutation(neg_idx).astype(np.int64))
        if pos_idx.size > 0:
            bucket_indices.append(rng.permutation(pos_idx).astype(np.int64))

    if not bucket_indices:
        return np.arange(y.shape[0], dtype=np.int64)

    pointers = [0 for _ in bucket_indices]
    quotas = [batch_size // len(bucket_indices) for _ in bucket_indices]
    for idx in range(batch_size % len(bucket_indices)):
        quotas[idx] += 1

    batches: List[np.ndarray] = []
    while True:
        active = [idx for idx, arr in enumerate(bucket_indices) if pointers[idx] < arr.size]
        if not active:
            break
        total_remaining = int(sum(bucket_indices[idx].size - pointers[idx] for idx in active))
        target_slots = min(int(batch_size), total_remaining)
        alloc = [0 for _ in bucket_indices]
        leftover = target_slots
        for idx in active:
            take = min(quotas[idx], int(bucket_indices[idx].size - pointers[idx]), leftover)
            alloc[idx] = take
            leftover -= take
        while leftover > 0:
            progressed = False
            for idx in active:
                cap = int(bucket_indices[idx].size - pointers[idx] - alloc[idx])
                if cap > 0 and leftover > 0:
                    alloc[idx] += 1
                    leftover -= 1
                    progressed = True
            if not progressed:
                break
        parts: List[np.ndarray] = []
        for idx in active:
            take = alloc[idx]
            if take <= 0:
                continue
            arr = bucket_indices[idx]
            parts.append(arr[pointers[idx] : pointers[idx] + take])
            pointers[idx] += take
        if not parts:
            break
        batch = np.concatenate(parts, axis=0)
        rng.shuffle(batch)
        batches.append(batch)

    if not batches:
        return np.arange(y.shape[0], dtype=np.int64)

    order = np.concatenate(batches, axis=0).astype(np.int64)
    if order.size != y.shape[0]:
        remaining_mask = np.ones(y.shape[0], dtype=bool)
        remaining_mask[order] = False
        remaining = np.flatnonzero(remaining_mask).astype(np.int64)
        if remaining.size > 0:
            rng.shuffle(remaining)
            order = np.concatenate([order, remaining], axis=0)
    return order


def build_main_order(
    y: np.ndarray,
    batch_size: int,
    variant: object,
    rng: np.random.Generator,
    sampling_mode: str,
) -> np.ndarray:
    mode = str(sampling_mode).strip().lower()
    if mode == "random":
        return rng.permutation(int(y.shape[0])).astype(np.int64)
    if mode == "band_balanced":
        return ab_lab.build_band_balanced_order(
            abs_y=np.abs(y.astype(np.float64, copy=False)),
            batch_size=int(batch_size),
            band_edges_y600=variant.balance_band_edges_y600,
            rng=rng,
            target_scale=float(variant.target_scale),
        )
    if mode == "sign_stratified":
        return build_sign_stratified_order(
            y=y,
            batch_size=int(batch_size),
            band_edges_y600=variant.balance_band_edges_y600,
            rng=rng,
            target_scale=float(variant.target_scale),
        )
    raise ValueError(f"Unsupported sampling_mode: {sampling_mode}")


class ShardPrefetcher:
    def __init__(
        self,
        train_shards: Sequence[Tuple[int, Path, Path]],
        variant: object,
        train_cfg: Report1TrainConfig,
        epoch: int,
        preload_x_dtype: Optional[np.dtype],
        seed: int,
        enabled: bool,
    ) -> None:
        self.train_shards = list(train_shards)
        self.variant = variant
        self.train_cfg = train_cfg
        self.epoch = int(epoch)
        self.preload_x_dtype = preload_x_dtype
        self.seed = int(seed)
        self.enabled = bool(enabled)
        self.executor: Optional[ThreadPoolExecutor] = None
        self.future = None
        self.index = 0
        if self.enabled:
            self.executor = ThreadPoolExecutor(max_workers=max(int(train_cfg.prefetch_workers), 1))

    def _load(self, shard_index: int) -> Dict[str, object]:
        shard_id, x_path, y_path = self.train_shards[shard_index]
        X = np.load(x_path, mmap_mode="r")
        if self.preload_x_dtype is not None:
            X = np.asarray(X, dtype=self.preload_x_dtype)
        y = np.load(y_path, mmap_mode="r").astype(np.float32, copy=False)
        order = build_main_order(
            y=y,
            batch_size=int(self.train_cfg.main_batch_size),
            variant=self.variant,
            rng=np.random.default_rng(self.seed + self.epoch * 10_000 + int(shard_id)),
            sampling_mode=self.train_cfg.sampling_mode,
        )
        return {"shard_id": int(shard_id), "X": X, "y": y, "order": order}

    def __iter__(self) -> "ShardPrefetcher":
        self.index = 0
        if self.executor is not None and self.train_shards:
            self.future = self.executor.submit(self._load, 0)
        return self

    def __next__(self) -> Dict[str, object]:
        if self.index >= len(self.train_shards):
            if self.executor is not None:
                self.executor.shutdown(wait=False, cancel_futures=False)
                self.executor = None
            raise StopIteration
        if self.executor is None:
            pack = self._load(self.index)
        else:
            if self.future is None:
                self.future = self.executor.submit(self._load, self.index)
            pack = self.future.result()
            next_index = self.index + 1
            self.future = self.executor.submit(self._load, next_index) if next_index < len(self.train_shards) else None
        self.index += 1
        return pack


def _optimizer_steps_per_epoch(train_shards: Sequence[Tuple[int, Path, Path]], batch_size: int, grad_accum_steps: int) -> int:
    total_micro_steps = sum(int(math.ceil(int(np.load(y_path, mmap_mode="r").shape[0]) / max(int(batch_size), 1))) for _, _, y_path in train_shards)
    return int(math.ceil(total_micro_steps / max(int(grad_accum_steps), 1)))


def _epoch_main_samples(train_shards: Sequence[Tuple[int, Path, Path]]) -> int:
    return int(sum(int(np.load(y_path, mmap_mode="r").shape[0]) for _, _, y_path in train_shards))


def benchmark_report1_profile(
    model_cfg: Dict[str, object],
    data_root: str | Path,
    role_bundle: Dict[str, object],
    train_cfg: Report1TrainConfig,
    device: torch.device,
    num_shards: int,
) -> Dict[str, float]:
    if device.type != "cuda":
        return {"ok": False, "reason": "cuda_required"}

    variant = shared_ft1.build_l4_variant()
    train_shards = ab_lab.resolve_split_shards(data_root, "train", num_shards=int(num_shards))
    if not train_shards:
        raise RuntimeError("No train shards available for report1 benchmark.")

    preload_x_dtype = _resolve_preload_numpy_dtype(train_cfg.preload_shard_dtype, train_cfg.use_amp, train_cfg.amp_dtype)
    shard_id, x_path, y_path = train_shards[0]
    X = np.load(x_path, mmap_mode="r")
    if preload_x_dtype is not None:
        X = np.asarray(X, dtype=preload_x_dtype)
    y = np.load(y_path, mmap_mode="r").astype(np.float32, copy=False)
    order = build_main_order(
        y=y,
        batch_size=int(train_cfg.main_batch_size),
        variant=variant,
        rng=np.random.default_rng(int(train_cfg.seed) + int(shard_id)),
        sampling_mode=train_cfg.sampling_mode,
    )
    aux_rng = np.random.default_rng(int(train_cfg.seed) + 20260417)

    model = DGRNChessNetV2(**model_cfg).to(device)
    if bool(train_cfg.channels_last):
        model = model.to(memory_format=torch.channels_last)
    model.train()
    optimizer = ab_lab.build_optimizer(model, lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    backbone_params = [param for name, param in model.named_parameters() if param.requires_grad and (name.startswith("stem.") or name.startswith("blocks."))]
    head_params = [param for name, param in model.named_parameters() if param.requires_grad and not (name.startswith("stem.") or name.startswith("blocks."))]
    all_params = [param for param in model.parameters() if param.requires_grad]
    amp_enabled = bool(train_cfg.use_amp) and device.type == "cuda"
    amp_dtype = _resolve_amp_dtype(train_cfg.amp_dtype)
    amp_grad_scale = max(float(train_cfg.amp_loss_scale), 1.0) if (amp_enabled and amp_dtype == torch.float16) else 1.0

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    warmup = max(int(train_cfg.benchmark_warmup_steps), 0)
    target_steps = max(int(train_cfg.benchmark_steps), 1)
    total_steps = warmup + target_steps
    positions = 0
    step_times: List[float] = []
    micro_step = 0
    aux_scale = min(1.0, 1.0 / max(int(train_cfg.aux_ramp_epochs), 1))

    try:
        for step in range(total_steps):
            start_idx = (step * int(train_cfg.main_batch_size)) % int(order.shape[0])
            if start_idx + int(train_cfg.main_batch_size) <= int(order.shape[0]):
                idx = order[start_idx : start_idx + int(train_cfg.main_batch_size)]
            else:
                idx = np.concatenate(
                    [
                        order[start_idx:],
                        order[: (start_idx + int(train_cfg.main_batch_size)) - int(order.shape[0])],
                    ],
                    axis=0,
                ).astype(np.int64)
            aux_idx = _sample_aux_indices(role_bundle, train_cfg, aux_rng)

            xb_main = _to_device_batch(
                X[idx],
                device,
                bool(train_cfg.channels_last),
                bool(train_cfg.pin_memory_batches),
                bool(train_cfg.use_amp),
                train_cfg.amp_dtype,
            )
            yb_main = torch.from_numpy(np.asarray(y[idx], dtype=np.float32)).to(device=device, non_blocking=True).view(-1)
            xb_aux = _to_device_batch(
                role_bundle["X_train"][aux_idx],
                device,
                bool(train_cfg.channels_last),
                bool(train_cfg.pin_memory_batches),
                bool(train_cfg.use_amp),
                train_cfg.amp_dtype,
            )
            yb_aux = torch.from_numpy(np.asarray(role_bundle["oracle_y"][aux_idx], dtype=np.float32)).to(device=device, non_blocking=True).view(-1)
            role_aux = torch.from_numpy(np.asarray(role_bundle["role_code"][aux_idx], dtype=np.int64)).to(device=device, non_blocking=True).view(-1)
            xb_mix = torch.cat([xb_main, xb_aux], dim=0)

            t0 = time.perf_counter()
            accum_scale = 1.0 / max(int(train_cfg.grad_accum_steps), 1)
            with autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype if amp_enabled else None):
                logits_mix = model.forward_logits(xb_mix).view(-1)
                main_logits = logits_mix[: yb_main.numel()]
                aux_logits = logits_mix[yb_main.numel() :]
                main_terms = compute_broad_main_terms(main_logits, yb_main, variant, train_cfg)
                aux_terms = compute_disabled_aux_terms(aux_logits, yb_aux, role_aux, train_cfg)
                total_objective = main_terms["objective"] + aux_scale * aux_terms["objective"]
                main_for_grad = main_terms["objective"] * accum_scale
                aux_for_grad = aux_scale * aux_terms["objective"] * accum_scale
                total_for_grad = total_objective * accum_scale

            main_backbone_grads = torch.autograd.grad(main_for_grad * float(amp_grad_scale), backbone_params, retain_graph=True, allow_unused=True)
            aux_backbone_grads = torch.autograd.grad(aux_for_grad * float(amp_grad_scale), backbone_params, retain_graph=True, allow_unused=True)
            total_head_grads = torch.autograd.grad(total_for_grad * float(amp_grad_scale), head_params, retain_graph=False, allow_unused=True)
            main_backbone_grads = _unscale_grad_list(main_backbone_grads, amp_grad_scale)
            aux_backbone_grads = _unscale_grad_list(aux_backbone_grads, amp_grad_scale)
            total_head_grads = _unscale_grad_list(total_head_grads, amp_grad_scale)
            shared_backbone_grads, _ = shared_ft1.project_backbone_conflicts(
                main_backbone_grads,
                aux_backbone_grads,
                eps=float(train_cfg.pcgrad_eps),
            )
            if not bool(train_cfg.use_backbone_pcgrad):
                shared_backbone_grads = [
                    shared_ft1._optional_add(
                        mg.detach().clone() if mg is not None else None,
                        ag.detach().clone() if ag is not None else None,
                    )
                    for mg, ag in zip(main_backbone_grads, aux_backbone_grads)
                ]

            for param, grad in zip(backbone_params, shared_backbone_grads):
                if grad is None:
                    continue
                if param.grad is None:
                    param.grad = grad.detach().clone()
                else:
                    param.grad.add_(grad.detach())
            for param, grad in zip(head_params, total_head_grads):
                if grad is None:
                    continue
                if param.grad is None:
                    param.grad = grad.detach().clone()
                else:
                    param.grad.add_(grad.detach())

            micro_step += 1
            should_step = (micro_step % max(int(train_cfg.grad_accum_steps), 1)) == 0
            if should_step:
                if train_cfg.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(all_params, float(train_cfg.grad_clip_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - t0

            if step >= warmup:
                step_times.append(elapsed)
                positions += int(yb_main.numel())
        if micro_step % max(int(train_cfg.grad_accum_steps), 1) != 0:
            optimizer.zero_grad(set_to_none=True)
        peak_mem_bytes = int(torch.cuda.max_memory_allocated(device))
        return {
            "ok": True,
            "main_positions_per_sec": float(positions / max(sum(step_times), 1e-9)),
            "mean_step_sec": float(np.mean(step_times)) if step_times else float("nan"),
            "peak_mem_bytes": peak_mem_bytes,
            "peak_mem_gb": float(peak_mem_bytes / 1024**3),
        }
    finally:
        del model
        _cleanup_cuda()


def autotune_report1_profile(
    model_cfg: Dict[str, object],
    data_root: str | Path,
    role_bundle: Dict[str, object],
    device: torch.device,
    gpu_name: Optional[str],
    total_mem_gb: float,
    base_cfg: Report1TrainConfig,
) -> Dict[str, object]:
    base_profile = default_device_profile(gpu_name, total_mem_gb)
    if device.type != "cuda":
        selected = dict(base_profile)
        return {
            "device": str(device),
            "gpu_name": gpu_name,
            "total_mem_gb": total_mem_gb,
            "candidates": [],
            "selected_profile": selected,
            "reason": "cpu_or_non_cuda",
        }

    reports: List[Dict[str, object]] = []
    best: Optional[Dict[str, object]] = None
    best_speed = float("-inf")
    total_train_samples = _epoch_main_samples(ab_lab.resolve_split_shards(data_root, "train", base_cfg.train_num_shards))
    for candidate in candidate_device_profiles(gpu_name, total_mem_gb):
        cfg = Report1TrainConfig(**{**asdict(base_cfg), **candidate})
        try:
            bench = benchmark_report1_profile(
                model_cfg=model_cfg,
                data_root=data_root,
                role_bundle=role_bundle,
                train_cfg=cfg,
                device=device,
                num_shards=max(int(cfg.benchmark_num_shards), 1),
            )
        except RuntimeError as exc:
            text = str(exc).lower()
            bench = {"ok": False, "error": str(exc), "oom": ("out of memory" in text)}
        except torch.cuda.OutOfMemoryError as exc:
            bench = {"ok": False, "error": str(exc), "oom": True}
        peak_ratio = float(bench.get("peak_mem_bytes", 0)) / max(float(total_mem_gb) * 1024**3, 1.0) if bench.get("ok") else None
        main_sps = float(bench.get("main_positions_per_sec", float("nan"))) if bench.get("ok") else None
        epoch_hours_estimate = (
            float(total_train_samples / max(main_sps, 1e-9) / 3600.0)
            if (main_sps is not None and math.isfinite(main_sps))
            else None
        )
        row = {
            **candidate,
            **bench,
            "peak_mem_ratio": peak_ratio,
            "epoch_hours_estimate": epoch_hours_estimate,
        }
        reports.append(_json_ready(row))
        if not bench.get("ok"):
            continue
        if peak_ratio is not None and peak_ratio > float(base_cfg.max_profile_mem_ratio):
            continue
        if main_sps is not None and main_sps > best_speed:
            best_speed = main_sps
            best = row
    selected_profile = (
        {key: best[key] for key in PROFILE_TUNE_KEYS if key in best}
        if best is not None
        else {key: base_profile[key] for key in PROFILE_TUNE_KEYS if key in base_profile}
    )
    return {
        "device": str(device),
        "gpu_name": gpu_name,
        "total_mem_gb": total_mem_gb,
        "candidates": reports,
        "selected_profile": selected_profile,
        "used_fallback": best is None,
    }


def evaluate_reference_checkpoint(
    checkpoint_path: Path,
    data_root: Path,
    pooled_center_bundle: Dict[str, object],
    role_bundle: Dict[str, object],
    device: torch.device,
    eval_batch_size: int,
    val_max_samples: int,
    val_num_shards: int,
) -> Dict[str, object]:
    oracle_bundle = obj_lab.build_primary_oracle_bundle(data_root)
    model, _ = base_lab.load_model_from_checkpoint(checkpoint_path, device=device)
    try:
        return shared_ft1.evaluate_ft1_model(
            model=model,
            data_root=data_root,
            device=device,
            eval_batch_size=eval_batch_size,
            split="val",
            max_samples=val_max_samples,
            num_shards=val_num_shards,
            oracle_bundle=oracle_bundle,
            pooled_center_bundle=pooled_center_bundle,
            role_bundle=role_bundle,
        )
    finally:
        del model
        _cleanup_cuda()


def _resume_signature(model_cfg: Dict[str, object], train_cfg: Report1TrainConfig, gate_cfg: Report1GateConfig) -> Dict[str, object]:
    train_cfg_dict = asdict(train_cfg)
    ignored = {
        "epochs",
        "resume_if_exists",
        "log_every_steps",
        "grad_monitor_every_steps",
        "benchmark_steps",
        "benchmark_warmup_steps",
        "periodic_save_minutes",
        "save_epoch_checkpoints",
    }
    filtered_train_cfg = {k: train_cfg_dict[k] for k in sorted(train_cfg_dict) if k not in ignored}
    return _json_ready({
        "model_cfg": dict(model_cfg),
        "train_cfg": filtered_train_cfg,
        "gate_cfg": asdict(gate_cfg),
    })


def _diff_resume_signature(existing: Dict[str, object], current: Dict[str, object]) -> List[str]:
    diffs: List[str] = []
    for section in ("model_cfg", "train_cfg", "gate_cfg"):
        existing_section = dict(existing.get(section, {}))
        current_section = dict(current.get(section, {}))
        keys = sorted(set(existing_section) | set(current_section))
        for key in keys:
            if existing_section.get(key) != current_section.get(key):
                diffs.append(
                    f"{section}.{key}: existing={existing_section.get(key)!r} current={current_section.get(key)!r}"
                )
    return diffs


def _load_resume_state(
    latest_ckpt: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    expected_scheduler_t_max: Optional[int] = None,
) -> Dict[str, object]:
    if not latest_ckpt.exists():
        return {
            "start_epoch": 0,
            "global_step": 0,
            "best_any_center_score": float("inf"),
            "best_gate_center_score": float("inf"),
            "history_rows": [],
            "resume_shard_index": 0,
            "resume_next_start": 0,
            "resume_is_epoch_end": True,
            "resume_rng_state": None,
            "resume_aux_rng_state": None,
            "resume_signature": {},
        }
    resume = _torch_load_trusted_checkpoint(latest_ckpt, map_location="cpu")
    if not isinstance(resume, dict):
        raise TypeError(f"Resume checkpoint must be a dict payload: {latest_ckpt}")
    resume_model_state = resume.get("model_state", resume.get("model"))
    if resume_model_state is None:
        raise KeyError("Resume checkpoint missing model state.")
    model.load_state_dict(resume_model_state, strict=True)
    resume_optimizer_state = resume.get("optimizer_state")
    if resume_optimizer_state is None:
        raise KeyError("Resume checkpoint missing optimizer_state.")
    optimizer.load_state_dict(resume_optimizer_state)
    resume_scheduler_state = resume.get("scheduler_state")
    if resume_scheduler_state is not None:
        scheduler.load_state_dict(resume_scheduler_state)
        if expected_scheduler_t_max is not None:
            saved_t_max = getattr(scheduler, "T_max", None)
            if saved_t_max != int(expected_scheduler_t_max):
                print(
                    "[phase_c1][resume] Updating scheduler T_max from checkpoint "
                    f"{saved_t_max} -> {int(expected_scheduler_t_max)} to match requested total steps."
                )
                scheduler.T_max = int(expected_scheduler_t_max)
    resume_epoch = int(resume.get("epoch", -1))
    resume_is_epoch_end = bool(resume.get("is_epoch_end", True))
    start_epoch = (resume_epoch + 1) if resume_is_epoch_end else max(0, resume_epoch)
    return {
        "start_epoch": start_epoch,
        "global_step": int(resume.get("global_step", 0)),
        "best_any_center_score": float(resume.get("best_any_center_score", float("inf"))),
        "best_gate_center_score": float(resume.get("best_gate_center_score", float("inf"))),
        "history_rows": list(resume.get("history", [])),
        "resume_shard_index": int(resume.get("resume_shard_index", 0)),
        "resume_next_start": int(resume.get("resume_next_start", 0)),
        "resume_is_epoch_end": resume_is_epoch_end,
        "resume_rng_state": resume.get("rng_state"),
        "resume_aux_rng_state": resume.get("aux_rng_state"),
        "resume_signature": dict(resume.get("config", {}).get("resume_signature", {})),
    }


def _epoch_checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    model_cfg: Dict[str, object],
    train_cfg: Report1TrainConfig,
    gate_cfg: Report1GateConfig,
    history_rows: List[dict],
    epoch: int,
    global_step: int,
    is_epoch_end: bool,
    best_any_center_score: float,
    best_gate_center_score: float,
    resume_shard_index: int = 0,
    resume_next_start: int = 0,
    rng_state: Optional[Dict[str, object]] = None,
    aux_rng_state: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "model_cfg": dict(model_cfg),
        "train_cfg": asdict(train_cfg),
        "gate_cfg": asdict(gate_cfg),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "history": list(history_rows),
        "is_epoch_end": bool(is_epoch_end),
        "best_any_center_score": float(best_any_center_score),
        "best_gate_center_score": float(best_gate_center_score),
        "resume_shard_index": int(resume_shard_index),
        "resume_next_start": int(resume_next_start),
        "rng_state": rng_state,
        "aux_rng_state": aux_rng_state,
        "config": {
            "model_cfg": dict(model_cfg),
            "train_cfg": asdict(train_cfg),
            "gate_cfg": asdict(gate_cfg),
            "resume_signature": _resume_signature(model_cfg=model_cfg, train_cfg=train_cfg, gate_cfg=gate_cfg),
        },
    }


def _pareto_a_key(metrics: Dict[str, object]) -> Tuple[float, float]:
    return (
        float(metrics.get("oracle_midband_mae_sum_stable", float("inf"))),
        -float(metrics.get("oracle_stable_0.7_slope", float("-inf"))),
    )


def _finite_float(value: object, default: float = float("inf")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _overall_metric(eval_result: Dict[str, object], metric: str, default: float = float("inf")) -> float:
    try:
        return _finite_float(eval_result["split_eval"]["metrics"]["overall"][metric], default=default)
    except (KeyError, TypeError):
        return float(default)


def _ratio(candidate: float, reference: float) -> float:
    if not math.isfinite(candidate):
        return float("inf")
    if not math.isfinite(reference) or abs(reference) <= 1.0e-12:
        return float("inf")
    return float(candidate / reference)


def _broad_selection_report(
    epoch_eval: Dict[str, object],
    l4_reference: Dict[str, object],
    gate_cfg: Report1GateConfig,
) -> Dict[str, object]:
    primary = dict(epoch_eval.get("primary", {}))
    ref_primary = dict(l4_reference.get("primary", {}))
    candidate = {
        "overall_mse": _overall_metric(epoch_eval, "mse"),
        "overall_mae": _overall_metric(epoch_eval, "mae"),
        "overall_pearson": _overall_metric(epoch_eval, "pearson", default=float("-inf")),
        "test_mse_0.1eq": _finite_float(primary.get("test_mse_0.1eq")),
        "test_mse_0.2eq": _finite_float(primary.get("test_mse_0.2eq")),
        "test_mse_0.5eq": _finite_float(primary.get("test_mse_0.5eq")),
        "test_mse_0.7eq": _finite_float(primary.get("test_mse_0.7eq")),
        "test_center_false_0.1eq": _finite_float(primary.get("test_center_false_0.1eq")),
        "test_center_false_0.2eq": _finite_float(primary.get("test_center_false_0.2eq")),
        "test_max_midband_abs_cal_gap": _finite_float(primary.get("test_max_midband_abs_cal_gap")),
    }
    reference = {
        "overall_mse": _overall_metric(l4_reference, "mse"),
        "overall_mae": _overall_metric(l4_reference, "mae"),
        "overall_pearson": _overall_metric(l4_reference, "pearson", default=float("-inf")),
        "test_mse_0.1eq": _finite_float(ref_primary.get("test_mse_0.1eq")),
        "test_mse_0.2eq": _finite_float(ref_primary.get("test_mse_0.2eq")),
        "test_mse_0.5eq": _finite_float(ref_primary.get("test_mse_0.5eq")),
        "test_mse_0.7eq": _finite_float(ref_primary.get("test_mse_0.7eq")),
        "test_center_false_0.1eq": _finite_float(ref_primary.get("test_center_false_0.1eq")),
        "test_center_false_0.2eq": _finite_float(ref_primary.get("test_center_false_0.2eq")),
        "test_max_midband_abs_cal_gap": _finite_float(ref_primary.get("test_max_midband_abs_cal_gap")),
    }
    components = {
        "overall_mse_ratio": _ratio(candidate["overall_mse"], reference["overall_mse"]),
        "mse_0p1_ratio": _ratio(candidate["test_mse_0.1eq"], reference["test_mse_0.1eq"]),
        "mse_0p2_ratio": _ratio(candidate["test_mse_0.2eq"], reference["test_mse_0.2eq"]),
        "mse_0p5_ratio": _ratio(candidate["test_mse_0.5eq"], reference["test_mse_0.5eq"]),
        "mse_0p7_ratio": _ratio(candidate["test_mse_0.7eq"], reference["test_mse_0.7eq"]),
        "center_false_0p1_ratio": _ratio(candidate["test_center_false_0.1eq"], reference["test_center_false_0.1eq"]),
        "center_false_0p2_ratio": _ratio(candidate["test_center_false_0.2eq"], reference["test_center_false_0.2eq"]),
        "abs_cal_ratio": _ratio(candidate["test_max_midband_abs_cal_gap"], reference["test_max_midband_abs_cal_gap"]),
    }
    broad_score = (
        1.00 * components["overall_mse_ratio"]
        + 1.25 * components["mse_0p1_ratio"]
        + 0.75 * components["mse_0p2_ratio"]
        + 0.75 * components["mse_0p5_ratio"]
        + 0.50 * components["mse_0p7_ratio"]
        + 1.25 * components["center_false_0p1_ratio"]
        + 0.75 * components["center_false_0p2_ratio"]
        + 0.75 * components["abs_cal_ratio"]
    ) / 7.00
    gate_pass = (
        candidate["overall_mse"] <= reference["overall_mse"] * (1.0 + float(gate_cfg.broad_overall_mse_rel_tol))
        and candidate["test_mse_0.1eq"] <= reference["test_mse_0.1eq"] * (1.0 + float(gate_cfg.broad_mse_0p1_rel_tol))
        and candidate["test_center_false_0.1eq"] <= reference["test_center_false_0.1eq"] + float(gate_cfg.broad_center_false_0p1_abs_tol)
        and candidate["test_max_midband_abs_cal_gap"] <= reference["test_max_midband_abs_cal_gap"] * (1.0 + float(gate_cfg.broad_abs_cal_rel_tol))
    )
    return {
        "broad_score": float(broad_score),
        "broad_gate_pass": bool(gate_pass),
        "broad_candidate": candidate,
        "broad_reference": reference,
        "broad_components": components,
    }


def run_report1_training(
    repo_root: str | Path,
    runs_root: str | Path,
    data_root: str | Path,
    model_cfg: Dict[str, object],
    train_cfg: Report1TrainConfig,
    gate_cfg: Report1GateConfig,
    autotune_profile: bool = True,
) -> Dict[str, object]:
    repo_root = Path(repo_root)
    runs_root = Path(runs_root)
    data_root = Path(data_root)
    validate_report1_train_config(train_cfg)
    paths = build_default_paths(repo_root=repo_root, runs_root=runs_root, run_name=train_cfg.run_name)
    latest_ckpt = paths["checkpoints_dir"] / "ckpt_latest.pt"
    existing_run_config = (
        _load_existing_run_config(paths, latest_ckpt)
        if bool(train_cfg.resume_if_exists) and latest_ckpt.exists()
        else None
    )
    _assert_resume_model_cfg_compatible(existing_run_config, model_cfg)
    runtime_check = validate_report1_runtime_paths(data_root=data_root, paths=paths)
    base_lab.save_json(runtime_check, paths["reports_dir"] / "runtime_check.json")
    if not bool(runtime_check["ok"]):
        raise RuntimeError("Phase C1 runtime validation failed: " + json.dumps(runtime_check["missing"], ensure_ascii=False))

    device = base_lab.choose_device(prefer_cuda=True)
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    total_mem_gb = float(torch.cuda.get_device_properties(device).total_memory / 1024**3) if device.type == "cuda" else 0.0
    if bool(train_cfg.cudnn_benchmark) and device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    tuned_cfg = train_cfg
    role_bundle = shared_ft1.load_ft1_role_bundle(paths["oracle_role_bundle_dir"])
    autotune_report = None
    if existing_run_config is not None and isinstance(existing_run_config.get("train_cfg"), dict):
        tuned_cfg = _apply_resume_safe_train_cfg_overrides(
            _report1_train_cfg_from_mapping(existing_run_config.get("train_cfg"), train_cfg),
            train_cfg,
        )
        validate_report1_train_config(tuned_cfg)
        autotune_report = existing_run_config.get("autotune_report")
        print(
            "[phase_c1][resume] Reusing saved train_cfg/profile from existing run_config; "
            "skipping autotune to keep resume_signature stable."
        )
    elif autotune_profile:
        autotune_report = autotune_report1_profile(
            model_cfg=model_cfg,
            data_root=data_root,
            role_bundle=role_bundle,
            device=device,
            gpu_name=gpu_name,
            total_mem_gb=total_mem_gb,
            base_cfg=train_cfg,
        )
        selected = dict(autotune_report["selected_profile"])
        tuned_cfg = Report1TrainConfig(**{**asdict(train_cfg), **selected})
        validate_report1_train_config(tuned_cfg)
        base_lab.save_json(_json_ready(autotune_report), paths["reports_dir"] / "train_batch_autotune.json")

    preload_x_dtype = _resolve_preload_numpy_dtype(tuned_cfg.preload_shard_dtype, tuned_cfg.use_amp, tuned_cfg.amp_dtype)
    role_bundle["X_train"] = (
        np.asarray(role_bundle["X"], dtype=preload_x_dtype)
        if preload_x_dtype is not None
        else np.asarray(role_bundle["X"], dtype=np.uint8)
    )

    variant = shared_ft1.build_l4_variant()
    oracle_bundle = obj_lab.build_primary_oracle_bundle(data_root)
    pooled_center_bundle = shared_ft1.load_pooled_center_bundle(paths["pooled_center_bundle_dir"])
    l4_reference = evaluate_reference_checkpoint(
        checkpoint_path=paths["l4_reference_ckpt"],
        data_root=data_root,
        pooled_center_bundle=pooled_center_bundle,
        role_bundle=role_bundle,
        device=device,
        eval_batch_size=tuned_cfg.eval_batch_size,
        val_max_samples=tuned_cfg.val_max_samples,
        val_num_shards=tuned_cfg.val_num_shards,
    )
    base_lab.save_json(
        _json_ready(
            {
                "checkpoint": str(paths["l4_reference_ckpt"]),
                "primary": l4_reference["primary"],
                "pooled_center_eval": l4_reference["pooled_center_eval"],
                "role_metrics": l4_reference["role_eval"]["metrics"],
            }
        ),
        paths["reports_dir"] / "l4_reference.json",
    )

    train_shards = ab_lab.resolve_split_shards(data_root, "train", tuned_cfg.train_num_shards)
    if not train_shards:
        raise RuntimeError(f"No report1 train shards found under {data_root / 'train'}")
    optimizer_steps_per_epoch = _optimizer_steps_per_epoch(train_shards, tuned_cfg.main_batch_size, tuned_cfg.grad_accum_steps)
    total_optimizer_steps = optimizer_steps_per_epoch * int(tuned_cfg.epochs)
    total_main_samples = _epoch_main_samples(train_shards)

    run_config = {
        "model_cfg": dict(model_cfg),
        "train_cfg": asdict(tuned_cfg),
        "gate_cfg": asdict(gate_cfg),
        "runtime": {
            "device": str(device),
            "gpu_name": gpu_name,
            "total_mem_gb": total_mem_gb,
        },
        "data_root": str(data_root),
        "autotune_report": autotune_report,
        "l4_reference_ckpt": str(paths["l4_reference_ckpt"]),
        "oracle_role_bundle_dir": str(paths["oracle_role_bundle_dir"]),
        "pooled_center_bundle_dir": str(paths["pooled_center_bundle_dir"]),
        "total_main_samples": int(total_main_samples),
        "optimizer_steps_per_epoch": int(optimizer_steps_per_epoch),
        "sampler": {
            "mode": str(tuned_cfg.sampling_mode),
            "band_edges_y600": list(variant.balance_band_edges_y600),
        },
        "objective": {
            "name": "broad_objective_v1",
            "oracle_aux_gradient_enabled": bool(
                int(tuned_cfg.clean_center_batch_size) > 0
                or int(tuned_cfg.ambiguous_center_batch_size) > 0
            ),
            "broad_y_huber_delta": float(tuned_cfg.broad_y_huber_delta),
            "broad_z_huber_weight": float(tuned_cfg.broad_z_huber_weight),
            "broad_z_huber_delta": float(tuned_cfg.broad_z_huber_delta),
            "broad_center_tau_y600": float(tuned_cfg.broad_center_tau_y600),
            "broad_center_pred_margin_y600": float(tuned_cfg.broad_center_pred_margin_y600),
            "broad_center_margin_weight": float(tuned_cfg.broad_center_margin_weight),
            "broad_abs_calibration_weight": float(tuned_cfg.broad_abs_calibration_weight),
            "broad_abs_calibration_edges_y600": list(tuned_cfg.broad_abs_calibration_edges_y600),
        },
        "selection_policy": "broad_validation_score_v1",
        "resume_signature": _resume_signature(model_cfg=model_cfg, train_cfg=tuned_cfg, gate_cfg=gate_cfg),
    }
    base_lab.save_json(_json_ready(run_config), paths["reports_dir"] / "run_config.json")

    set_global_seed(tuned_cfg.seed)
    model = DGRNChessNetV2(**model_cfg).to(device)
    if bool(tuned_cfg.channels_last):
        model = model.to(memory_format=torch.channels_last)
    optimizer = ab_lab.build_optimizer(model, lr=tuned_cfg.learning_rate, weight_decay=tuned_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(total_optimizer_steps), 1),
        eta_min=float(tuned_cfg.min_lr),
    )
    backbone_params = [param for name, param in model.named_parameters() if param.requires_grad and (name.startswith("stem.") or name.startswith("blocks."))]
    head_params = [param for name, param in model.named_parameters() if param.requires_grad and not (name.startswith("stem.") or name.startswith("blocks."))]
    all_params = [param for param in model.parameters() if param.requires_grad]

    best_any_ckpt = paths["checkpoints_dir"] / "ckpt_best_any.pt"
    best_gate_ckpt = paths["checkpoints_dir"] / "ckpt_best_gate.pt"
    best_pareto_a_ckpt = paths["checkpoints_dir"] / "ckpt_best_pareto_A.pt"
    best_pareto_b_ckpt = paths["checkpoints_dir"] / "ckpt_best_pareto_B.pt"
    epoch_ckpts_dir = (paths["checkpoints_dir"] / "epochs") if bool(tuned_cfg.save_epoch_checkpoints) else None
    if epoch_ckpts_dir is not None:
        epoch_ckpts_dir.mkdir(parents=True, exist_ok=True)

    resume = (
        _load_resume_state(latest_ckpt, model, optimizer, scheduler, expected_scheduler_t_max=total_optimizer_steps)
        if bool(tuned_cfg.resume_if_exists)
        else {
            "start_epoch": 0,
            "global_step": 0,
            "best_any_center_score": float("inf"),
            "best_gate_center_score": float("inf"),
            "history_rows": [],
            "resume_shard_index": 0,
            "resume_next_start": 0,
            "resume_is_epoch_end": True,
            "resume_rng_state": None,
            "resume_aux_rng_state": None,
            "resume_signature": {},
        }
    )
    if latest_ckpt.exists() and not bool(tuned_cfg.resume_if_exists):
        raise RuntimeError(
            f"Found existing checkpoint at {latest_ckpt} while resume_if_exists=False. "
            "Use a new run_name or enable resume explicitly."
        )
    if latest_ckpt.exists():
        diffs = _diff_resume_signature(dict(resume.get("resume_signature", {})), run_config["resume_signature"])
        if diffs:
            diff_text = "; ".join(diffs[:12])
            raise RuntimeError(
                "Resume config mismatch for existing report1 run. "
                f"Use a new run_name or restore the old config. Diffs: {diff_text}"
            )

    history_rows: List[dict] = list(resume["history_rows"])
    step_history_path = paths["reports_dir"] / "step_history.csv"
    if latest_ckpt.exists() and step_history_path.exists():
        try:
            step_rows: List[dict] = pd.read_csv(step_history_path).to_dict("records")
        except Exception as exc:
            print(f"[phase_c1][resume] Failed to load existing step_history.csv ({exc}); starting step log from this session")
            step_rows = []
    else:
        step_rows = []
    best_any_center_score = float(resume["best_any_center_score"])
    best_gate_center_score = float(resume["best_gate_center_score"])
    best_pareto_a_key = (float("inf"), float("inf"))
    best_pareto_b_center_score = float("inf")
    for row in history_rows:
        best_pareto_a_key = min(best_pareto_a_key, _pareto_a_key(row))
        if "center_score" in row:
            best_pareto_b_center_score = min(best_pareto_b_center_score, float(row["center_score"]))
    global_step = int(resume["global_step"])
    amp_enabled = bool(tuned_cfg.use_amp) and (device.type == "cuda")
    amp_dtype = _resolve_amp_dtype(tuned_cfg.amp_dtype)
    amp_grad_scale = max(float(tuned_cfg.amp_loss_scale), 1.0) if (amp_enabled and amp_dtype == torch.float16) else 1.0
    next_periodic_save_time = time.time() + max(int(tuned_cfg.periodic_save_minutes), 1) * 60
    aux_rng = np.random.default_rng(int(tuned_cfg.seed) + 20260417)
    if isinstance(resume["resume_aux_rng_state"], dict):
        try:
            aux_rng.bit_generator.state = resume["resume_aux_rng_state"]
        except Exception as exc:
            print(f"[phase_c1][resume] Failed to restore aux RNG state ({exc}); continuing with seeded RNG")

    def _maybe_save_periodic_latest_checkpoint(epoch: int, resume_shard_index: int, resume_next_start: int) -> None:
        nonlocal next_periodic_save_time
        now = time.time()
        if now < next_periodic_save_time:
            return
        payload = _epoch_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            model_cfg=model_cfg,
            train_cfg=tuned_cfg,
            gate_cfg=gate_cfg,
            history_rows=history_rows,
            epoch=epoch,
            global_step=global_step,
            is_epoch_end=False,
            best_any_center_score=best_any_center_score,
            best_gate_center_score=best_gate_center_score,
            resume_shard_index=resume_shard_index,
            resume_next_start=resume_next_start,
            rng_state=_capture_rng_state(),
            aux_rng_state=aux_rng.bit_generator.state,
        )
        ab_lab.atomic_torch_save(payload, latest_ckpt)
        next_periodic_save_time = now + max(int(tuned_cfg.periodic_save_minutes), 1) * 60

    for epoch in range(int(resume["start_epoch"]), int(tuned_cfg.epochs)):
        is_partial_resume_epoch = (epoch == int(resume["start_epoch"]) and not bool(resume["resume_is_epoch_end"]))
        if is_partial_resume_epoch and isinstance(resume["resume_rng_state"], dict):
            _restore_rng_state(resume["resume_rng_state"])
        else:
            set_global_seed(int(tuned_cfg.seed) + int(epoch))

        model.train()
        optimizer.zero_grad(set_to_none=True)
        aux_scale = float(min(1.0, (int(epoch) + 1) / max(int(tuned_cfg.aux_ramp_epochs), 1)))
        epoch_running = {
            "total_objective": 0.0,
            "main_objective": 0.0,
            "aux_objective": 0.0,
            "clean_loss": 0.0,
            "ambiguous_loss": 0.0,
            "margin_penalty": 0.0,
            "mean_main_weight": 0.0,
            "downweighted_frac": 0.0,
            "y_huber": 0.0,
            "z_huber": 0.0,
            "center_margin_penalty": 0.0,
            "abs_calibration_penalty": 0.0,
            "n": 0,
        }
        epoch_main_samples_seen = 0
        epoch_optimizer_steps_done = 0
        grad_samples: List[dict] = []
        t0 = time.time()
        accum_micro_step = 0
        epoch_resume_shard_index = int(resume["resume_shard_index"]) if is_partial_resume_epoch else 0
        epoch_resume_next_start = int(resume["resume_next_start"]) if is_partial_resume_epoch else 0

        prefetcher = ShardPrefetcher(
            train_shards=train_shards,
            variant=variant,
            train_cfg=tuned_cfg,
            epoch=epoch,
            preload_x_dtype=preload_x_dtype,
            seed=int(tuned_cfg.seed),
            enabled=bool(tuned_cfg.prefetch_shards),
        )

        for shard_index, pack in enumerate(prefetcher):
            if shard_index < epoch_resume_shard_index:
                continue
            X_shard = pack["X"]
            y_shard = pack["y"]
            order = pack["order"]
            shard_start_offset = epoch_resume_next_start if shard_index == epoch_resume_shard_index else 0
            shard_start_offset = min(max(0, int(shard_start_offset)), int(order.shape[0]))
            for start in range(shard_start_offset, int(order.shape[0]), int(tuned_cfg.main_batch_size)):
                idx = order[start : start + int(tuned_cfg.main_batch_size)]
                aux_idx = _sample_aux_indices(role_bundle, tuned_cfg, aux_rng)

                xb_main = _to_device_batch(
                    X_shard[idx],
                    device,
                    bool(tuned_cfg.channels_last),
                    bool(tuned_cfg.pin_memory_batches),
                    bool(tuned_cfg.use_amp),
                    tuned_cfg.amp_dtype,
                )
                yb_main = torch.from_numpy(np.asarray(y_shard[idx], dtype=np.float32)).to(device=device, non_blocking=True).view(-1)
                xb_aux = _to_device_batch(
                    role_bundle["X_train"][aux_idx],
                    device,
                    bool(tuned_cfg.channels_last),
                    bool(tuned_cfg.pin_memory_batches),
                    bool(tuned_cfg.use_amp),
                    tuned_cfg.amp_dtype,
                )
                yb_aux = torch.from_numpy(np.asarray(role_bundle["oracle_y"][aux_idx], dtype=np.float32)).to(device=device, non_blocking=True).view(-1)
                role_aux = torch.from_numpy(np.asarray(role_bundle["role_code"][aux_idx], dtype=np.int64)).to(device=device, non_blocking=True).view(-1)
                xb_mix = torch.cat([xb_main, xb_aux], dim=0)

                accum_scale = 1.0 / max(int(tuned_cfg.grad_accum_steps), 1)
                with autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype if amp_enabled else None):
                    logits_mix = model.forward_logits(xb_mix).view(-1)
                    main_logits = logits_mix[: yb_main.numel()]
                    aux_logits = logits_mix[yb_main.numel() :]
                    main_terms = compute_broad_main_terms(main_logits, yb_main, variant, tuned_cfg)
                    aux_terms = compute_disabled_aux_terms(aux_logits, yb_aux, role_aux, tuned_cfg)
                    aux_objective_scaled = aux_scale * aux_terms["objective"]
                    total_objective = main_terms["objective"] + aux_objective_scaled
                    main_for_grad = main_terms["objective"] * accum_scale
                    aux_for_grad = aux_objective_scaled * accum_scale
                    total_for_grad = total_objective * accum_scale

                if not bool(torch.isfinite(total_objective)):
                    optimizer.zero_grad(set_to_none=True)
                    accum_micro_step = 0
                    print(
                        f"[phase_c1][epoch={epoch}] non-finite objective at shard={pack['shard_id']}, "
                        "dropping current accumulation window"
                    )
                    continue

                main_backbone_grads = torch.autograd.grad(main_for_grad * float(amp_grad_scale), backbone_params, retain_graph=True, allow_unused=True)
                aux_backbone_grads = torch.autograd.grad(aux_for_grad * float(amp_grad_scale), backbone_params, retain_graph=True, allow_unused=True)
                should_step = ((accum_micro_step + 1) % max(int(tuned_cfg.grad_accum_steps), 1)) == 0
                next_global_step = int(global_step) + (1 if should_step else 0)
                should_monitor = should_step and (next_global_step % max(int(tuned_cfg.grad_monitor_every_steps), 1) == 0)
                if should_monitor:
                    main_all_grads = torch.autograd.grad(main_for_grad * float(amp_grad_scale), all_params, retain_graph=True, allow_unused=True)
                    aux_all_grads = torch.autograd.grad(aux_for_grad * float(amp_grad_scale), all_params, retain_graph=True, allow_unused=True)
                total_head_grads = torch.autograd.grad(total_for_grad * float(amp_grad_scale), head_params, retain_graph=False, allow_unused=True)

                main_backbone_grads = _unscale_grad_list(main_backbone_grads, amp_grad_scale)
                aux_backbone_grads = _unscale_grad_list(aux_backbone_grads, amp_grad_scale)
                total_head_grads = _unscale_grad_list(total_head_grads, amp_grad_scale)
                shared_backbone_grads, pcgrad_report = shared_ft1.project_backbone_conflicts(
                    main_backbone_grads,
                    aux_backbone_grads,
                    eps=float(tuned_cfg.pcgrad_eps),
                )
                if not bool(tuned_cfg.use_backbone_pcgrad):
                    shared_backbone_grads = [
                        shared_ft1._optional_add(
                            mg.detach().clone() if mg is not None else None,
                            ag.detach().clone() if ag is not None else None,
                        )
                        for mg, ag in zip(main_backbone_grads, aux_backbone_grads)
                    ]
                    pcgrad_report["grad_cosine_backbone_post"] = pcgrad_report["grad_cosine_backbone"]
                    pcgrad_report["grad_norm_shared_backbone"] = float(torch.norm(shared_ft1._flatten_grads(shared_backbone_grads))) if shared_ft1._flatten_grads(shared_backbone_grads).numel() > 0 else float("nan")
                    pcgrad_report["grad_conflict_backbone"] = 0.0
                    pcgrad_report["grad_projection_scale"] = 0.0

                for param, grad in zip(backbone_params, shared_backbone_grads):
                    if grad is None:
                        continue
                    if param.grad is None:
                        param.grad = grad.detach().clone()
                    else:
                        param.grad.add_(grad.detach())
                for param, grad in zip(head_params, total_head_grads):
                    if grad is None:
                        continue
                    if param.grad is None:
                        param.grad = grad.detach().clone()
                    else:
                        param.grad.add_(grad.detach())

                accum_micro_step += 1
                if not _all_grads_finite(all_params):
                    optimizer.zero_grad(set_to_none=True)
                    accum_micro_step = 0
                    print(
                        f"[phase_c1][epoch={epoch}] non-finite accumulated gradients at shard={pack['shard_id']}; "
                        "dropping current accumulation window"
                    )
                    continue

                if should_monitor:
                    main_all_grads = _unscale_grad_list(main_all_grads, amp_grad_scale)
                    aux_all_grads = _unscale_grad_list(aux_all_grads, amp_grad_scale)
                    grad_report = shared_ft1.collect_gradient_monitor(
                        main_grads_backbone=main_backbone_grads,
                        aux_grads_backbone=aux_backbone_grads,
                        shared_grads_backbone=shared_backbone_grads,
                        main_grads_all=main_all_grads,
                        aux_grads_all=aux_all_grads,
                    )
                    grad_report.update(pcgrad_report)
                    grad_samples.append(grad_report)
                    step_rows.append(
                        {
                            "global_step": int(next_global_step),
                            "train_total_objective": float(total_objective.item()),
                            "train_main_objective": float(main_terms["objective"].item()),
                            "train_aux_objective": float(aux_objective_scaled.item()),
                            "aux_scale": float(aux_scale),
                            "grad_cosine_backbone": float(grad_report.get("grad_cosine_backbone", float("nan"))),
                            "grad_cosine_backbone_pre": float(grad_report.get("grad_cosine_backbone", float("nan"))),
                            "grad_cosine_backbone_post": float(grad_report.get("grad_cosine_backbone_post", float("nan"))),
                            "grad_norm_main_backbone": float(grad_report.get("grad_norm_main_backbone", float("nan"))),
                            "grad_norm_aux_backbone": float(grad_report.get("grad_norm_aux_backbone", float("nan"))),
                            "grad_norm_shared_backbone": float(grad_report.get("grad_norm_shared_backbone", float("nan"))),
                            "grad_conflict_backbone": float(grad_report.get("grad_conflict_backbone", float("nan"))),
                            "grad_projection_scale": float(grad_report.get("grad_projection_scale", float("nan"))),
                            "grad_cosine_all": float(grad_report.get("grad_cosine_all", float("nan"))),
                            "grad_norm_main_all": float(grad_report.get("grad_norm_main_all", float("nan"))),
                            "grad_norm_aux_all": float(grad_report.get("grad_norm_aux_all", float("nan"))),
                        }
                    )

                bs = int(yb_main.numel())
                epoch_running["total_objective"] += float(total_objective.item()) * bs
                epoch_running["main_objective"] += float(main_terms["objective"].item()) * bs
                epoch_running["aux_objective"] += float(aux_objective_scaled.item()) * bs
                epoch_running["clean_loss"] += float(aux_terms["clean_loss"].item()) * bs
                epoch_running["ambiguous_loss"] += float(aux_terms["ambiguous_loss"].item()) * bs
                epoch_running["margin_penalty"] += float(aux_terms["margin_penalty"].item()) * bs
                epoch_running["mean_main_weight"] += float(main_terms["mean_main_weight"].item()) * bs
                epoch_running["downweighted_frac"] += float(main_terms["downweighted_frac"].item()) * bs
                epoch_running["y_huber"] += float(main_terms["mean_y_term"].item()) * bs
                epoch_running["z_huber"] += float(main_terms["mean_z_term"].item()) * bs
                epoch_running["center_margin_penalty"] += float(main_terms["center_margin_penalty"].item()) * bs
                epoch_running["abs_calibration_penalty"] += float(main_terms["abs_calibration_penalty"].item()) * bs
                epoch_running["n"] += bs
                epoch_main_samples_seen += bs

                if should_step:
                    if not _all_grads_finite(all_params):
                        optimizer.zero_grad(set_to_none=True)
                        accum_micro_step = 0
                        print(
                            f"[phase_c1][epoch={epoch}] non-finite gradients before step={next_global_step}; "
                            "dropping current accumulation window"
                        )
                        continue
                    if tuned_cfg.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(all_params, float(tuned_cfg.grad_clip_norm))
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step = next_global_step
                    epoch_optimizer_steps_done += 1
                    accum_micro_step = 0
                    _maybe_save_periodic_latest_checkpoint(
                        epoch,
                        resume_shard_index=shard_index,
                        resume_next_start=int(start) + int(tuned_cfg.main_batch_size),
                    )
                    if global_step % max(int(tuned_cfg.log_every_steps), 1) == 0:
                        print(
                            f"[phase_c1][epoch={epoch}] step={global_step} "
                            f"main_obj={epoch_running['main_objective'] / max(epoch_running['n'], 1):.6f} "
                            f"aux_obj={epoch_running['aux_objective'] / max(epoch_running['n'], 1):.6f}"
                        )

        if accum_micro_step != 0:
            if not _all_grads_finite(all_params):
                optimizer.zero_grad(set_to_none=True)
                print(
                    f"[phase_c1][epoch={epoch}] non-finite gradients at tail step={global_step + 1}; "
                    "dropping tail accumulation window"
                )
            else:
                if tuned_cfg.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(all_params, float(tuned_cfg.grad_clip_norm))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                epoch_optimizer_steps_done += 1
                accum_micro_step = 0
                _maybe_save_periodic_latest_checkpoint(
                    epoch,
                    resume_shard_index=len(train_shards),
                    resume_next_start=0,
                )

        epoch_eval = shared_ft1.evaluate_ft1_model(
            model=model,
            data_root=data_root,
            device=device,
            eval_batch_size=tuned_cfg.eval_batch_size,
            split="val",
            max_samples=tuned_cfg.val_max_samples,
            num_shards=tuned_cfg.val_num_shards,
            oracle_bundle=oracle_bundle,
            pooled_center_bundle=pooled_center_bundle,
            role_bundle=role_bundle,
        )
        primary = dict(epoch_eval["primary"])
        epoch_time_sec = float(time.time() - t0)
        broad_report = _broad_selection_report(epoch_eval, l4_reference, gate_cfg)
        gate_pass = bool(broad_report["broad_gate_pass"])
        broad_score = float(broad_report["broad_score"])
        center_score = float(primary["center_score"])
        selected: List[str] = []
        if broad_score < best_any_center_score:
            best_any_center_score = broad_score
            selected.append("best_any")
        if gate_pass and broad_score < best_gate_center_score:
            best_gate_center_score = broad_score
            selected.append("best_gate")
        pareto_a_key = _pareto_a_key(primary)
        if pareto_a_key < best_pareto_a_key:
            best_pareto_a_key = pareto_a_key
            selected.append("best_pareto_A")
        if center_score < best_pareto_b_center_score:
            best_pareto_b_center_score = center_score
            selected.append("best_pareto_B")

        row = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "lr": float(scheduler.get_last_lr()[0]),
            "epoch_time_sec": epoch_time_sec,
            "optimizer_steps_per_sec": float(epoch_optimizer_steps_done / max(epoch_time_sec, 1e-6)),
            "main_samples_per_sec": float(epoch_main_samples_seen / max(epoch_time_sec, 1e-6)),
            "train_total_objective": float(epoch_running["total_objective"] / max(epoch_running["n"], 1)),
            "train_main_objective": float(epoch_running["main_objective"] / max(epoch_running["n"], 1)),
            "train_aux_objective": float(epoch_running["aux_objective"] / max(epoch_running["n"], 1)),
            "train_clean_loss": float(epoch_running["clean_loss"] / max(epoch_running["n"], 1)),
            "train_ambiguous_loss": float(epoch_running["ambiguous_loss"] / max(epoch_running["n"], 1)),
            "train_margin_penalty": float(epoch_running["margin_penalty"] / max(epoch_running["n"], 1)),
            "train_mean_main_weight": float(epoch_running["mean_main_weight"] / max(epoch_running["n"], 1)),
            "train_downweighted_frac": float(epoch_running["downweighted_frac"] / max(epoch_running["n"], 1)),
            "train_y_huber": float(epoch_running["y_huber"] / max(epoch_running["n"], 1)),
            "train_z_huber": float(epoch_running["z_huber"] / max(epoch_running["n"], 1)),
            "train_center_margin_penalty": float(epoch_running["center_margin_penalty"] / max(epoch_running["n"], 1)),
            "train_abs_calibration_penalty": float(epoch_running["abs_calibration_penalty"] / max(epoch_running["n"], 1)),
            "aux_scale": float(aux_scale),
            "sampling_mode": str(tuned_cfg.sampling_mode),
            "midband_gate_pass": bool(gate_pass),
            "broad_gate_pass": bool(gate_pass),
            "broad_score": float(broad_score),
            "selected_checkpoint": "|".join(selected) if selected else "none",
            **primary,
            **{f"broad_{key}": value for key, value in dict(broad_report["broad_components"]).items()},
            "overall_mse": float(dict(broad_report["broad_candidate"])["overall_mse"]),
            "overall_mae": float(dict(broad_report["broad_candidate"])["overall_mae"]),
            "overall_pearson": float(dict(broad_report["broad_candidate"])["overall_pearson"]),
        }
        if grad_samples:
            grad_df = pd.DataFrame(grad_samples)
            row["grad_conflict_backbone"] = float(grad_df["grad_conflict_backbone"].mean())
            row["grad_cosine_backbone"] = float(grad_df["grad_cosine_backbone"].mean())
            row["grad_cosine_backbone_post"] = float(grad_df["grad_cosine_backbone_post"].mean())
            row["grad_cosine_all"] = float(grad_df["grad_cosine_all"].mean())
        history_rows.append(row)

        payload = _epoch_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            model_cfg=model_cfg,
            train_cfg=tuned_cfg,
            gate_cfg=gate_cfg,
            history_rows=history_rows,
            epoch=epoch,
            global_step=global_step,
            is_epoch_end=True,
            best_any_center_score=best_any_center_score,
            best_gate_center_score=best_gate_center_score,
            rng_state=_capture_rng_state(),
            aux_rng_state=aux_rng.bit_generator.state,
        )
        ab_lab.atomic_torch_save(payload, latest_ckpt)
        if epoch_ckpts_dir is not None:
            ab_lab.atomic_torch_save(payload, epoch_ckpts_dir / f"ckpt_epoch_{int(epoch):03d}.pt")
        if "best_any" in selected:
            ab_lab.atomic_torch_save(payload, best_any_ckpt)
        if "best_gate" in selected:
            ab_lab.atomic_torch_save(payload, best_gate_ckpt)
        if "best_pareto_A" in selected:
            ab_lab.atomic_torch_save(payload, best_pareto_a_ckpt)
        if "best_pareto_B" in selected:
            ab_lab.atomic_torch_save(payload, best_pareto_b_ckpt)

        selected_if_stop_now = (
            "best_gate"
            if best_gate_ckpt.exists()
            else ("best_any" if best_any_ckpt.exists() else "latest")
        )
        decision_payload = {
            "selection_policy": "broad_validation_score_v1",
            "best_any_broad_score": float(best_any_center_score),
            "best_gate_broad_score": float(best_gate_center_score) if math.isfinite(best_gate_center_score) else None,
            "best_any_center_score": float(best_any_center_score),
            "best_gate_center_score": float(best_gate_center_score) if math.isfinite(best_gate_center_score) else None,
            "best_pareto_A_midband": float(best_pareto_a_key[0]),
            "best_pareto_A_slope": float(-best_pareto_a_key[1]),
            "best_pareto_B_center_score": float(best_pareto_b_center_score),
            "has_best_gate": bool(math.isfinite(best_gate_center_score)),
            "latest_checkpoint": str(latest_ckpt),
            "best_any_checkpoint": str(best_any_ckpt) if best_any_ckpt.exists() else None,
            "best_gate_checkpoint": str(best_gate_ckpt) if best_gate_ckpt.exists() else None,
            "best_pareto_A_checkpoint": str(best_pareto_a_ckpt) if best_pareto_a_ckpt.exists() else None,
            "best_pareto_B_checkpoint": str(best_pareto_b_ckpt) if best_pareto_b_ckpt.exists() else None,
            "last_epoch": int(epoch),
            "last_center_score": center_score,
            "last_broad_score": broad_score,
            "last_gate_pass": bool(gate_pass),
            "last_broad_gate_pass": bool(gate_pass),
            "selected_checkpoint_if_stopped_now": selected_if_stop_now,
            "last_epoch_tags": "|".join(selected) if selected else "none",
            "last_broad_components": dict(broad_report["broad_components"]),
            "last_broad_candidate": dict(broad_report["broad_candidate"]),
            "broad_reference": dict(broad_report["broad_reference"]),
        }
        base_lab.save_json(_json_ready(decision_payload), paths["reports_dir"] / "decision_summary.json")
        shared_ft1.save_history_outputs(history_rows, step_rows, paths["reports_dir"])
        print(
            f"[phase_c1][epoch={epoch}] phase={tuned_cfg.phase_name} "
            f"time={epoch_time_sec / 3600.0:.2f}h "
            f"main_sps={row['main_samples_per_sec']:.1f} broad={broad_score:.4f} gate={gate_pass}"
        )

    selected_ckpt = best_gate_ckpt if best_gate_ckpt.exists() else (best_any_ckpt if best_any_ckpt.exists() else latest_ckpt)
    model_eval, _ = base_lab.load_model_from_checkpoint(selected_ckpt, device=device)
    try:
        final_eval = shared_ft1.evaluate_ft1_model(
            model=model_eval,
            data_root=data_root,
            device=device,
            eval_batch_size=tuned_cfg.eval_batch_size,
            split="test",
            max_samples=tuned_cfg.test_max_samples,
            num_shards=tuned_cfg.test_num_shards,
            oracle_bundle=oracle_bundle,
            pooled_center_bundle=pooled_center_bundle,
            role_bundle=role_bundle,
        )
        final_eval = dict(final_eval)
        final_eval.update(dict(final_eval.get("primary", {})))
    finally:
        del model_eval
        _cleanup_cuda()

    base_lab.save_json(_json_ready(final_eval), paths["reports_dir"] / "selected_checkpoint_eval.json")
    base_lab.save_json(_json_ready(run_config), paths["reports_dir"] / "run_config.json")
    return {
        "paths": paths,
        "runtime_check": runtime_check,
        "l4_reference": l4_reference,
        "selected_checkpoint": str(selected_ckpt),
        "final_eval": final_eval,
        "autotune_report": autotune_report,
        "run_config": run_config,
    }
