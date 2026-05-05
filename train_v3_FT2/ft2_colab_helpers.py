from __future__ import annotations

import gc
import importlib.util
import json
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

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
    PROJECT_ROOT / "experiments" / "oc2_joint_oracle_full_model_pilot",
    PROJECT_ROOT / "train_v2_TF1",
]

for path in [PROJECT_ROOT, MODEL_ROOT, *EXPERIMENT_DIRS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from architecture_v2.model import DGRNChessNetV2  # noqa: E402
import teacher_root_cause_helpers as base_lab  # noqa: E402
import root_cause_ablation_helpers as ab_lab  # noqa: E402
import objective_resolution_helpers as obj_lab  # noqa: E402

ft1_lab = _import_module_from_file(
    "train_v2_tf1_ft1_colab_helpers",
    PROJECT_ROOT / "train_v2_TF1" / "ft1_colab_helpers.py",
)


ROLE_CLEAN_CENTER = 0
ROLE_AMBIGUOUS_CENTER = 1


@dataclass
class FT2TrainConfig:
    run_name: str = "dgrn_5m_ft2_t4_run1"
    epochs: int = 50
    main_batch_size: int = 448
    clean_center_batch_size: int = 56
    ambiguous_center_batch_size: int = 112
    grad_accum_steps: int = 1
    eval_batch_size: int = 2560
    learning_rate: float = 1.0e-4
    min_lr: float = 1.0e-5
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 1.0
    seed: int = 123
    train_num_shards: Optional[int] = None
    val_num_shards: int = 2
    test_num_shards: int = 4
    val_max_samples: int = 200_000
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
    main_center_tau_y600: float = 0.10
    main_center_min_weight: float = 0.35
    main_center_weight_power: float = 1.0
    lambda_main_init: float = 1.0
    lambda_clean_init: float = 0.20
    lambda_ambiguous_init: float = 0.10
    gradnorm_alpha: float = 1.0
    gradnorm_weight_sum: float = 1.30
    gradnorm_warmup_epochs: int = 4
    gradnorm_adapt_rate: float = 0.05
    aux_margin_y600: float = 0.08
    aux_margin_weight: float = 0.40
    aux_huber_delta: float = 0.05
    use_backbone_pcgrad: bool = True
    pcgrad_eps: float = 1.0e-12
    benchmark_steps: int = 8
    benchmark_warmup_steps: int = 2
    benchmark_num_shards: int = 1
    max_profile_mem_ratio: float = 0.82
    periodic_save_minutes: int = 30
    save_epoch_checkpoints: bool = False
    resume_if_exists: bool = True
    role_val_frac: float = 0.20
    role_split_seed: int = 123
    role_refresh_cache: bool = False


@dataclass
class FT2GateConfig:
    midband_mae_rel_tol: float = 0.05
    stable_slope_abs_tol: float = 0.02


@dataclass
class FT2RoleSplitConfig:
    val_frac: float = 0.20
    seed: int = 123


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
        print(f"[ft2][resume] Failed to restore RNG state ({exc}); continuing with seeded RNG")


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


def validate_ft2_train_config(train_cfg: FT2TrainConfig) -> None:
    positive_int_fields = {
        "epochs": train_cfg.epochs,
        "main_batch_size": train_cfg.main_batch_size,
        "clean_center_batch_size": train_cfg.clean_center_batch_size,
        "ambiguous_center_batch_size": train_cfg.ambiguous_center_batch_size,
        "grad_accum_steps": train_cfg.grad_accum_steps,
        "eval_batch_size": train_cfg.eval_batch_size,
        "benchmark_steps": train_cfg.benchmark_steps,
        "periodic_save_minutes": train_cfg.periodic_save_minutes,
    }
    for name, value in positive_int_fields.items():
        if int(value) <= 0:
            raise ValueError(f"train_cfg.{name} must be > 0")
    if int(train_cfg.benchmark_warmup_steps) < 0:
        raise ValueError("train_cfg.benchmark_warmup_steps must be >= 0")
    if float(train_cfg.learning_rate) <= 0.0:
        raise ValueError("train_cfg.learning_rate must be > 0")
    if float(train_cfg.min_lr) < 0.0:
        raise ValueError("train_cfg.min_lr must be >= 0")
    if float(train_cfg.min_lr) > float(train_cfg.learning_rate):
        raise ValueError("train_cfg.min_lr must be <= train_cfg.learning_rate")
    if float(train_cfg.gradnorm_weight_sum) <= 0.0:
        raise ValueError("train_cfg.gradnorm_weight_sum must be > 0")
    if float(train_cfg.amp_loss_scale) <= 0.0:
        raise ValueError("train_cfg.amp_loss_scale must be > 0")
    if not (0.0 < float(train_cfg.role_val_frac) < 1.0):
        raise ValueError("train_cfg.role_val_frac must be in (0, 1)")
    if not (0.0 < float(train_cfg.max_profile_mem_ratio) <= 1.0):
        raise ValueError("train_cfg.max_profile_mem_ratio must be in (0, 1]")


def _pareto_a_key(metrics: Dict[str, object]) -> Tuple[float, float]:
    return (
        float(metrics.get("oracle_midband_mae_sum_stable", float("inf"))),
        -float(metrics.get("oracle_stable_0.7_slope", float("-inf"))),
    )


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
    return value


def default_colab_profile(gpu_name: Optional[str], total_mem_gb: float) -> Dict[str, int]:
    gpu_name = (gpu_name or "").upper()
    if "T4" in gpu_name and total_mem_gb >= 14.0:
        return {
            "main_batch_size": 448,
            "clean_center_batch_size": 56,
            "ambiguous_center_batch_size": 112,
            "grad_accum_steps": 1,
            "eval_batch_size": 2560,
        }
    if total_mem_gb >= 14.0:
        return {
            "main_batch_size": 384,
            "clean_center_batch_size": 48,
            "ambiguous_center_batch_size": 96,
            "grad_accum_steps": 1,
            "eval_batch_size": 2048,
        }
    if total_mem_gb >= 8.0:
        return {
            "main_batch_size": 256,
            "clean_center_batch_size": 32,
            "ambiguous_center_batch_size": 64,
            "grad_accum_steps": 2,
            "eval_batch_size": 1536,
        }
    return {
        "main_batch_size": 128,
        "clean_center_batch_size": 16,
        "ambiguous_center_batch_size": 32,
        "grad_accum_steps": 2,
        "eval_batch_size": 768,
    }


def candidate_colab_profiles(gpu_name: Optional[str], total_mem_gb: float) -> List[Dict[str, int]]:
    gpu_name = (gpu_name or "").upper()
    out: List[Dict[str, int]] = []
    if "T4" in gpu_name and total_mem_gb >= 14.0:
        out.extend(
            [
                {"main_batch_size": 512, "clean_center_batch_size": 64, "ambiguous_center_batch_size": 128, "grad_accum_steps": 1, "eval_batch_size": 3072},
                {"main_batch_size": 448, "clean_center_batch_size": 56, "ambiguous_center_batch_size": 112, "grad_accum_steps": 1, "eval_batch_size": 2560},
                {"main_batch_size": 384, "clean_center_batch_size": 48, "ambiguous_center_batch_size": 96, "grad_accum_steps": 1, "eval_batch_size": 2048},
                {"main_batch_size": 320, "clean_center_batch_size": 40, "ambiguous_center_batch_size": 80, "grad_accum_steps": 1, "eval_batch_size": 2048},
            ]
        )
    out.append(default_colab_profile(gpu_name, total_mem_gb))
    dedup: List[Dict[str, int]] = []
    seen: set[Tuple[int, int, int, int, int]] = set()
    for item in out:
        key = (
            int(item["main_batch_size"]),
            int(item["clean_center_batch_size"]),
            int(item["ambiguous_center_batch_size"]),
            int(item["grad_accum_steps"]),
            int(item["eval_batch_size"]),
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
    cache_dir = run_dir / "cache"
    for path in (run_dir, checkpoints_dir, reports_dir, plots_dir, cache_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "repo_root": Path(repo_root),
        "run_dir": run_dir,
        "checkpoints_dir": checkpoints_dir,
        "reports_dir": reports_dir,
        "plots_dir": plots_dir,
        "cache_dir": cache_dir,
        "l4_reference_ckpt": repo_root / "experiments" / "objective_resolution_suite" / "outputs" / "runs" / "L4_A1_plus_A2" / "checkpoints" / "L4_A1_plus_A2_best.pt",
        "oracle_role_bundle_dir": repo_root / "experiments" / "oc2_joint_oracle_full_model_pilot" / "outputs" / "cache" / "oracle_role_bundle",
        "pooled_center_bundle_dir": repo_root / "experiments" / "failure_b_resolution_suite" / "outputs" / "cache" / "pooled_center_bundle",
    }


def validate_ft2_runtime_paths(data_root: Path, paths: Dict[str, Path]) -> Dict[str, object]:
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


def _role_split_cache_paths(split_dir: Path) -> Dict[str, Path]:
    split_dir.mkdir(parents=True, exist_ok=True)
    return {
        "manifest": split_dir / "ft2_role_split_manifest.json",
        "train_idx": split_dir / "train_indices.npy",
        "val_idx": split_dir / "val_indices.npy",
    }


def _bundle_select(bundle: Dict[str, object], indices: np.ndarray, preload_x_dtype: Optional[np.dtype]) -> Dict[str, object]:
    indices = np.asarray(indices, dtype=np.int64)
    X = np.asarray(bundle["X"][indices], dtype=np.uint8)
    X_train = np.asarray(X, dtype=preload_x_dtype) if preload_x_dtype is not None else X
    oracle_y = np.asarray(bundle["oracle_y"][indices], dtype=np.float32)
    role_code = np.asarray(bundle["role_code"][indices], dtype=np.int64)
    rows = bundle["rows"].iloc[indices].reset_index(drop=True)
    return {
        "manifest": dict(bundle["manifest"]),
        "rows": rows,
        "X": X,
        "X_train": X_train,
        "oracle_y": oracle_y,
        "role_code": role_code,
        "indices_by_role": {
            "clean_center": np.flatnonzero(role_code == ROLE_CLEAN_CENTER).astype(np.int64),
            "center_ambiguous": np.flatnonzero(role_code == ROLE_AMBIGUOUS_CENTER).astype(np.int64),
        },
    }


def _bundle_identity_columns(rows: pd.DataFrame) -> List[str]:
    if "fen" in rows.columns:
        return ["fen"]
    if {"shard_id", "local_index"}.issubset(rows.columns):
        return ["shard_id", "local_index"]
    return []


def _split_bundle_indices_by_identity(
    base_bundle: Dict[str, object],
    split_cfg: FT2RoleSplitConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    rows = base_bundle["rows"].reset_index(drop=True)
    identity_columns = _bundle_identity_columns(rows)
    total_rows = int(rows.shape[0])
    roles = ("clean_center", "center_ambiguous")
    target_val = {
        role: max(1, int(round(int(base_bundle["indices_by_role"][role].size) * float(split_cfg.val_frac))))
        for role in roles
    }

    grouped: List[Dict[str, object]] = []
    work_rows = rows.copy()
    work_rows["bundle_index"] = np.arange(total_rows, dtype=np.int64)
    if identity_columns:
        grouped_iter = work_rows.groupby(identity_columns, sort=False, dropna=False)
    else:
        grouped_iter = ((idx, work_rows.iloc[[idx]]) for idx in range(total_rows))
    for _, group in grouped_iter:
        idx = np.asarray(group["bundle_index"].to_numpy(dtype=np.int64), dtype=np.int64)
        role_code = np.asarray(base_bundle["role_code"][idx], dtype=np.int64)
        grouped.append(
            {
                "indices": idx,
                "role_counts": {
                    "clean_center": int(np.sum(role_code == ROLE_CLEAN_CENTER)),
                    "center_ambiguous": int(np.sum(role_code == ROLE_AMBIGUOUS_CENTER)),
                },
            }
        )

    order = np.arange(len(grouped), dtype=np.int64)
    np.random.default_rng(int(split_cfg.seed)).shuffle(order)
    val_groups: List[Dict[str, object]] = []
    train_groups: List[Dict[str, object]] = []
    val_counts = {role: 0 for role in roles}
    train_counts = {
        "clean_center": int(base_bundle["indices_by_role"]["clean_center"].size),
        "center_ambiguous": int(base_bundle["indices_by_role"]["center_ambiguous"].size),
    }
    for group_idx in order:
        group = grouped[int(group_idx)]
        needs_val = any(
            val_counts[role] < target_val[role] and int(group["role_counts"][role]) > 0
            for role in roles
        )
        if needs_val:
            val_groups.append(group)
            for role in roles:
                delta = int(group["role_counts"][role])
                val_counts[role] += delta
                train_counts[role] -= delta
        else:
            train_groups.append(group)

    def _move_group(
        source: List[Dict[str, object]],
        target: List[Dict[str, object]],
        role: str,
        direction: str,
    ) -> bool:
        for idx in range(len(source) - 1, -1, -1):
            group = source[idx]
            if int(group["role_counts"][role]) <= 0:
                continue
            source.pop(idx)
            target.append(group)
            for role_name in roles:
                delta = int(group["role_counts"][role_name])
                if direction == "to_train":
                    train_counts[role_name] += delta
                    val_counts[role_name] -= delta
                else:
                    train_counts[role_name] -= delta
                    val_counts[role_name] += delta
            return True
        return False

    for role in roles:
        if train_counts[role] <= 0:
            if not _move_group(val_groups, train_groups, role, direction="to_train"):
                raise RuntimeError(f"Unable to keep role={role} in aux_train during identity split.")
        if val_counts[role] <= 0:
            if not _move_group(train_groups, val_groups, role, direction="to_val"):
                raise RuntimeError(f"Unable to keep role={role} in aux_val during identity split.")

    train_idx = np.sort(
        np.concatenate([np.asarray(group["indices"], dtype=np.int64) for group in train_groups], axis=0)
    )
    val_idx = np.sort(
        np.concatenate([np.asarray(group["indices"], dtype=np.int64) for group in val_groups], axis=0)
    )
    split_report = {
        "identity_columns": identity_columns,
        "target_val_counts": {k: int(v) for k, v in target_val.items()},
        "actual_val_counts": {k: int(v) for k, v in val_counts.items()},
        "actual_train_counts": {k: int(v) for k, v in train_counts.items()},
        "num_identity_groups": int(len(grouped)),
        "num_train_groups": int(len(train_groups)),
        "num_val_groups": int(len(val_groups)),
    }
    return train_idx, val_idx, split_report


def prepare_ft2_role_bundle(
    bundle_dir: str | Path,
    split_dir: str | Path,
    split_cfg: FT2RoleSplitConfig,
    preload_x_dtype: Optional[np.dtype],
    refresh_split: bool = False,
) -> Dict[str, object]:
    base_bundle = ft1_lab.load_ft1_role_bundle(bundle_dir)
    split_dir = Path(split_dir)
    cache_paths = _role_split_cache_paths(split_dir)
    cache_signature = {
        "base_manifest": dict(base_bundle["manifest"]),
        "val_frac": float(split_cfg.val_frac),
        "seed": int(split_cfg.seed),
        "identity_columns": _bundle_identity_columns(base_bundle["rows"]),
    }
    if (
        not refresh_split
        and cache_paths["manifest"].exists()
        and cache_paths["train_idx"].exists()
        and cache_paths["val_idx"].exists()
    ):
        manifest = json.loads(cache_paths["manifest"].read_text(encoding="utf-8"))
        if all(manifest.get(key) == value for key, value in cache_signature.items()):
            train_idx = np.load(cache_paths["train_idx"])
            val_idx = np.load(cache_paths["val_idx"])
            return {
                "base": base_bundle,
                "train": _bundle_select(base_bundle, train_idx, preload_x_dtype),
                "val": _bundle_select(base_bundle, val_idx, preload_x_dtype),
                "all": _bundle_select(base_bundle, np.arange(base_bundle["X"].shape[0], dtype=np.int64), preload_x_dtype),
                "split_manifest": manifest,
            }

    if not (0.0 < float(split_cfg.val_frac) < 1.0):
        raise ValueError("split_cfg.val_frac must be in (0, 1)")
    train_idx, val_idx, split_report = _split_bundle_indices_by_identity(base_bundle, split_cfg)
    expected_manifest = {
        **cache_signature,
        **split_report,
    }
    np.save(cache_paths["train_idx"], train_idx.astype(np.int64))
    np.save(cache_paths["val_idx"], val_idx.astype(np.int64))
    cache_paths["manifest"].write_text(json.dumps(expected_manifest, indent=2), encoding="utf-8")
    return {
        "base": base_bundle,
        "train": _bundle_select(base_bundle, train_idx, preload_x_dtype),
        "val": _bundle_select(base_bundle, val_idx, preload_x_dtype),
        "all": _bundle_select(base_bundle, np.arange(base_bundle["X"].shape[0], dtype=np.int64), preload_x_dtype),
        "split_manifest": expected_manifest,
    }


def _huber_mean(residual: torch.Tensor, delta: float) -> torch.Tensor:
    delta = float(delta)
    abs_residual = residual.abs()
    value = torch.where(
        abs_residual <= delta,
        0.5 * residual * residual,
        delta * (abs_residual - 0.5 * delta),
    )
    return torch.mean(value)


def compute_ft2_aux_components(
    logits: torch.Tensor,
    oracle_y: torch.Tensor,
    role_code: torch.Tensor,
    cfg: FT2TrainConfig,
) -> Dict[str, torch.Tensor]:
    pred = torch.tanh(logits.view(-1))
    oracle_y = oracle_y.view(-1)
    role_code = role_code.view(-1).long()
    clean_mask = role_code == ROLE_CLEAN_CENTER
    ambiguous_mask = role_code == ROLE_AMBIGUOUS_CENTER

    clean_fit = _huber_mean(pred[clean_mask] - oracle_y[clean_mask], cfg.aux_huber_delta) if torch.any(clean_mask) else pred.new_tensor(0.0)
    ambiguous = _huber_mean(pred[ambiguous_mask] - oracle_y[ambiguous_mask], cfg.aux_huber_delta) if torch.any(ambiguous_mask) else pred.new_tensor(0.0)
    margin = (
        torch.mean(torch.relu(torch.abs(pred[clean_mask]) - float(cfg.aux_margin_y600)) ** 2)
        if torch.any(clean_mask)
        else pred.new_tensor(0.0)
    )
    clean_total = clean_fit + float(cfg.aux_margin_weight) * margin
    return {
        "pred": pred,
        "clean_fit": clean_fit,
        "margin": margin,
        "clean_total": clean_total,
        "ambiguous": ambiguous,
        "clean_frac": torch.mean(clean_mask.float()),
        "ambiguous_frac": torch.mean(ambiguous_mask.float()),
    }


def _flatten_grads(grads: Sequence[Optional[torch.Tensor]]) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    for grad in grads:
        if grad is None:
            continue
        parts.append(grad.detach().float().reshape(-1))
    if not parts:
        return torch.empty(0)
    return torch.cat(parts, dim=0)


def _clone_grad_list(grads: Sequence[Optional[torch.Tensor]]) -> List[Optional[torch.Tensor]]:
    out: List[Optional[torch.Tensor]] = []
    for grad in grads:
        out.append(None if grad is None else grad.detach().clone())
    return out


def _optional_add(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if a is None:
        return None if b is None else b.detach().clone()
    if b is None:
        return a.detach().clone()
    return a.detach().clone() + b.detach()


def _cosine_from_vectors(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    denom = float(torch.norm(a) * torch.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(torch.dot(a, b) / denom)


def _grad_norm(grads: Sequence[Optional[torch.Tensor]]) -> float:
    flat = _flatten_grads(grads)
    if flat.numel() == 0:
        return 0.0
    return float(torch.norm(flat))


def _task_pairwise_conflict(task_vectors: Dict[str, torch.Tensor]) -> float:
    values: List[float] = []
    keys = list(task_vectors.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            cos = _cosine_from_vectors(task_vectors[keys[i]], task_vectors[keys[j]])
            if math.isnan(cos):
                continue
            if cos < 0.0:
                values.append(abs(cos))
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def project_multi_task_backbone_conflicts(
    task_grads: Dict[str, Sequence[Optional[torch.Tensor]]],
    eps: float = 1.0e-12,
) -> Tuple[List[Optional[torch.Tensor]], Dict[str, float]]:
    names = list(task_grads.keys())
    projected = {name: _clone_grad_list(task_grads[name]) for name in names}
    flat_before = {name: _flatten_grads(task_grads[name]) for name in names}
    for name in names:
        order = [other for other in names if other != name]
        for other in order:
            g = _flatten_grads(projected[name])
            h = flat_before[other]
            if g.numel() == 0 or h.numel() == 0:
                continue
            dot = float(torch.dot(g, h))
            if dot >= 0.0:
                continue
            denom = float(torch.dot(h, h)) + float(eps)
            coeff = dot / denom
            for idx, grad in enumerate(projected[name]):
                other_grad = task_grads[other][idx]
                if grad is None or other_grad is None:
                    continue
                projected[name][idx] = grad - coeff * other_grad
    merged: List[Optional[torch.Tensor]] = []
    num_params = len(next(iter(projected.values())))
    for idx in range(num_params):
        value: Optional[torch.Tensor] = None
        for name in names:
            value = _optional_add(value, projected[name][idx])
        if value is not None:
            value = value / float(len(names))
        merged.append(value)
    flat_after = {name: _flatten_grads(projected[name]) for name in names}
    report = {
        "grad_conflict_backbone": _task_pairwise_conflict(flat_before),
        "grad_conflict_backbone_post": _task_pairwise_conflict(flat_after),
        "grad_norm_shared_backbone": _grad_norm(merged),
        "grad_cosine_main_clean": _cosine_from_vectors(flat_before.get("main", torch.empty(0)), flat_before.get("clean_total", torch.empty(0))),
        "grad_cosine_main_amb": _cosine_from_vectors(flat_before.get("main", torch.empty(0)), flat_before.get("ambiguous", torch.empty(0))),
        "grad_cosine_clean_amb": _cosine_from_vectors(flat_before.get("clean_total", torch.empty(0)), flat_before.get("ambiguous", torch.empty(0))),
    }
    return merged, report


class GradNormController:
    def __init__(
        self,
        alpha: float,
        weight_sum: float,
        warmup_epochs: int,
        adapt_rate: float,
        init_weights: Dict[str, float],
        eps: float = 1.0e-8,
    ) -> None:
        self.alpha = float(alpha)
        self.weight_sum = float(weight_sum)
        self.warmup_epochs = int(warmup_epochs)
        self.adapt_rate = float(adapt_rate)
        self.eps = float(eps)
        self.task_names = list(init_weights.keys())
        self.weights = {name: float(init_weights[name]) for name in self.task_names}
        self.initial_losses: Dict[str, float] = {}

    def state_dict(self) -> Dict[str, object]:
        return {
            "alpha": self.alpha,
            "weight_sum": self.weight_sum,
            "warmup_epochs": self.warmup_epochs,
            "adapt_rate": self.adapt_rate,
            "eps": self.eps,
            "task_names": list(self.task_names),
            "weights": dict(self.weights),
            "initial_losses": dict(self.initial_losses),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if not isinstance(state, dict):
            return
        self.weights.update({k: float(v) for k, v in dict(state.get("weights", {})).items() if k in self.weights})
        self.initial_losses.update({k: float(v) for k, v in dict(state.get("initial_losses", {})).items() if k in self.weights})

    def current_weights(self, epoch: int) -> Dict[str, float]:
        return dict(self.weights)

    def update(self, epoch: int, losses: Dict[str, float], grad_norms: Dict[str, float]) -> Dict[str, float]:
        for name in self.task_names:
            value = max(float(losses.get(name, 0.0)), self.eps)
            if name not in self.initial_losses:
                self.initial_losses[name] = value
        if int(epoch) < self.warmup_epochs:
            return dict(self.weights)
        ratios = {
            name: max(float(losses.get(name, 0.0)), self.eps) / max(float(self.initial_losses[name]), self.eps)
            for name in self.task_names
        }
        mean_ratio = sum(ratios.values()) / max(len(ratios), 1)
        mean_grad = sum(max(float(grad_norms.get(name, 0.0)), self.eps) for name in self.task_names) / max(len(self.task_names), 1)
        for name in self.task_names:
            target = mean_grad * math.pow(max(ratios[name] / max(mean_ratio, self.eps), self.eps), self.alpha)
            current = max(float(grad_norms.get(name, 0.0)), self.eps)
            ratio = target / current
            self.weights[name] *= math.pow(ratio, self.adapt_rate)
        total = sum(max(v, self.eps) for v in self.weights.values())
        scale = self.weight_sum / max(total, self.eps)
        for name in self.task_names:
            self.weights[name] = max(self.eps, float(self.weights[name]) * scale)
        return dict(self.weights)


@contextmanager
def freeze_batchnorm_running_stats(model: nn.Module) -> Iterator[None]:
    modules: List[Tuple[nn.Module, bool]] = []
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            modules.append((module, module.training))
            module.eval()
    try:
        yield
    finally:
        for module, was_training in modules:
            module.train(was_training)


def _to_device_batch(
    array: np.ndarray,
    device: torch.device,
    channels_last: bool,
    pin_memory: bool,
) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if channels_last and tensor.ndim == 4:
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    if pin_memory and device.type == "cuda":
        tensor = tensor.pin_memory()
    tensor = tensor.to(device=device, non_blocking=(device.type == "cuda"))
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
    return tensor


class ShardPrefetcher:
    def __init__(
        self,
        train_shards: Sequence[Tuple[int, Path, Path]],
        variant: ab_lab.AblationVariant,
        train_cfg: FT2TrainConfig,
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
        abs_y = np.abs(y.astype(np.float64, copy=False))
        order = ab_lab.build_band_balanced_order(
            abs_y=abs_y,
            batch_size=int(self.train_cfg.main_batch_size),
            band_edges_y600=self.variant.balance_band_edges_y600,
            rng=np.random.default_rng(self.seed + self.epoch * 10_000 + int(shard_id)),
            target_scale=float(self.variant.target_scale),
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


def _sample_aux_indices(role_bundle: Dict[str, object], cfg: FT2TrainConfig, rng: np.random.Generator) -> np.ndarray:
    clean_pool = np.asarray(role_bundle["indices_by_role"]["clean_center"], dtype=np.int64)
    ambiguous_pool = np.asarray(role_bundle["indices_by_role"]["center_ambiguous"], dtype=np.int64)
    clean = rng.choice(clean_pool, size=int(cfg.clean_center_batch_size), replace=clean_pool.size < int(cfg.clean_center_batch_size)).astype(np.int64)
    ambiguous = rng.choice(
        ambiguous_pool,
        size=int(cfg.ambiguous_center_batch_size),
        replace=ambiguous_pool.size < int(cfg.ambiguous_center_batch_size),
    ).astype(np.int64)
    out = np.concatenate([clean, ambiguous], axis=0).astype(np.int64)
    rng.shuffle(out)
    return out


def benchmark_ft2_profile(
    model_cfg: Dict[str, object],
    data_root: str | Path,
    role_bundle_train: Dict[str, object],
    train_cfg: FT2TrainConfig,
    device: torch.device,
    num_shards: int,
) -> Dict[str, float]:
    if device.type != "cuda":
        return {"ok": False, "reason": "cuda_required"}
    variant = ft1_lab.build_l4_variant()
    train_shards = ab_lab.resolve_split_shards(data_root, "train", num_shards=int(num_shards))
    if not train_shards:
        raise RuntimeError("No train shards available for FT2 benchmark.")
    preload_x_dtype = _resolve_preload_numpy_dtype(train_cfg.preload_shard_dtype, train_cfg.use_amp, train_cfg.amp_dtype)
    shard_id, x_path, y_path = train_shards[0]
    X = np.load(x_path, mmap_mode="r")
    if preload_x_dtype is not None:
        X = np.asarray(X, dtype=preload_x_dtype)
    y = np.load(y_path, mmap_mode="r").astype(np.float32, copy=False)
    abs_y = np.abs(y.astype(np.float64, copy=False))
    order = ab_lab.build_band_balanced_order(
        abs_y=abs_y,
        batch_size=int(train_cfg.main_batch_size),
        band_edges_y600=variant.balance_band_edges_y600,
        rng=np.random.default_rng(int(train_cfg.seed) + int(shard_id)),
        target_scale=float(variant.target_scale),
    )
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
    amp_grad_scale = 1.0
    if amp_enabled and amp_dtype == torch.float16:
        amp_grad_scale = max(float(train_cfg.amp_loss_scale), 1.0)
    controller = GradNormController(
        alpha=train_cfg.gradnorm_alpha,
        weight_sum=train_cfg.gradnorm_weight_sum,
        warmup_epochs=train_cfg.gradnorm_warmup_epochs,
        adapt_rate=train_cfg.gradnorm_adapt_rate,
        init_weights={
            "main": train_cfg.lambda_main_init,
            "clean_total": train_cfg.lambda_clean_init,
            "ambiguous": train_cfg.lambda_ambiguous_init,
        },
    )
    rng = np.random.default_rng(int(train_cfg.seed))
    torch.cuda.reset_peak_memory_stats(device)
    warmup = int(train_cfg.benchmark_warmup_steps)
    steps = int(train_cfg.benchmark_steps)
    total_positions = 0
    start_time = None
    for loop_idx in range(warmup + steps):
        if loop_idx == warmup:
            torch.cuda.synchronize(device)
            start_time = time.perf_counter()
            total_positions = 0
        measure_idx = max(loop_idx - warmup, 0)
        begin = (measure_idx % max(steps, 1)) * int(train_cfg.main_batch_size)
        idx = order[begin : begin + int(train_cfg.main_batch_size)]
        if idx.size < int(train_cfg.main_batch_size):
            idx = order[: int(train_cfg.main_batch_size)]
        aux_idx = _sample_aux_indices(role_bundle_train, train_cfg, rng)
        xb_main = _to_device_batch(X[idx], device, bool(train_cfg.channels_last), bool(train_cfg.pin_memory_batches))
        yb_main = torch.from_numpy(np.asarray(y[idx], dtype=np.float32)).to(device=device, non_blocking=True).view(-1)
        xb_aux = _to_device_batch(role_bundle_train["X_train"][aux_idx], device, bool(train_cfg.channels_last), bool(train_cfg.pin_memory_batches))
        yb_aux = torch.from_numpy(np.asarray(role_bundle_train["oracle_y"][aux_idx], dtype=np.float32)).to(device=device, non_blocking=True).view(-1)
        role_aux = torch.from_numpy(np.asarray(role_bundle_train["role_code"][aux_idx], dtype=np.int64)).to(device=device, non_blocking=True).view(-1)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype if amp_enabled else None):
            main_logits = model.forward_logits(xb_main).view(-1)
            main_terms = ft1_lab.compute_l4_main_terms(main_logits, yb_main, variant, train_cfg)
        with freeze_batchnorm_running_stats(model):
            with autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype if amp_enabled else None):
                aux_logits = model.forward_logits(xb_aux).view(-1)
                aux_terms = compute_ft2_aux_components(aux_logits, yb_aux, role_aux, train_cfg)
        task_weights = controller.current_weights(epoch=0)
        scaled_main = float(task_weights["main"]) * main_terms["objective"]
        scaled_clean = float(task_weights["clean_total"]) * aux_terms["clean_total"]
        scaled_amb = float(task_weights["ambiguous"]) * aux_terms["ambiguous"]
        total_head = scaled_main + scaled_clean + scaled_amb
        if not bool(torch.isfinite(total_head)):
            raise RuntimeError("Non-finite FT2 benchmark objective")
        main_grads = torch.autograd.grad(scaled_main * float(amp_grad_scale), backbone_params, retain_graph=True, allow_unused=True)
        clean_grads = torch.autograd.grad(scaled_clean * float(amp_grad_scale), backbone_params, retain_graph=True, allow_unused=True)
        amb_grads = torch.autograd.grad(scaled_amb * float(amp_grad_scale), backbone_params, retain_graph=True, allow_unused=True)
        main_grads = _unscale_grad_list(main_grads, amp_grad_scale)
        clean_grads = _unscale_grad_list(clean_grads, amp_grad_scale)
        amb_grads = _unscale_grad_list(amb_grads, amp_grad_scale)
        merged_backbone, _ = project_multi_task_backbone_conflicts(
            {"main": main_grads, "clean_total": clean_grads, "ambiguous": amb_grads},
            eps=float(train_cfg.pcgrad_eps),
        )
        head_grads = torch.autograd.grad(total_head * float(amp_grad_scale), head_params, retain_graph=False, allow_unused=True)
        head_grads = _unscale_grad_list(head_grads, amp_grad_scale)
        for param, grad in zip(backbone_params, merged_backbone):
            if grad is not None:
                param.grad = grad.detach().clone()
        for param, grad in zip(head_params, head_grads):
            if grad is not None:
                param.grad = grad.detach().clone()
        if not _all_grads_finite(all_params):
            raise RuntimeError("Non-finite FT2 benchmark gradients")
        if train_cfg.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.grad_clip_norm))
        optimizer.step()
        torch.cuda.synchronize(device)
        if loop_idx >= warmup:
            total_positions += int(yb_main.numel())
    torch.cuda.synchronize(device)
    elapsed = max(time.perf_counter() - float(start_time or time.perf_counter()), 1e-6)
    steps_per_sec = float(steps / elapsed)
    positions_per_sec = float(total_positions / elapsed)
    total_samples = float(_epoch_main_samples(train_shards))
    epoch_hours_estimate = total_samples / max(positions_per_sec, 1e-6) / 3600.0
    total_mem_gb = float(torch.cuda.get_device_properties(device).total_memory / 1024**3)
    peak_mem_gb = float(torch.cuda.max_memory_allocated(device) / 1024**3)
    report = {
        "steps_per_sec": steps_per_sec,
        "main_positions_per_sec": positions_per_sec,
        "epoch_hours_estimate": float(epoch_hours_estimate),
        "peak_mem_gb": peak_mem_gb,
        "mem_ratio": peak_mem_gb / max(total_mem_gb, 1e-6),
        "total_mem_gb": total_mem_gb,
    }
    del model
    _cleanup_cuda()
    return report


def autotune_ft2_profile(
    model_cfg: Dict[str, object],
    data_root: str | Path,
    role_bundle_train: Dict[str, object],
    device: torch.device,
    gpu_name: Optional[str],
    total_mem_gb: float,
    base_cfg: FT2TrainConfig,
) -> Dict[str, object]:
    if device.type != "cuda":
        profile = default_colab_profile(gpu_name, total_mem_gb)
        return {"selected_profile": profile, "attempts": [], "device": str(device)}
    attempts: List[dict] = []
    selected: Optional[Dict[str, int]] = None
    selected_report: Optional[Dict[str, float]] = None
    best_speed = -1.0
    for candidate in candidate_colab_profiles(gpu_name, total_mem_gb):
        trial_cfg = FT2TrainConfig(**{**asdict(base_cfg), **candidate})
        try:
            report = benchmark_ft2_profile(
                model_cfg=model_cfg,
                data_root=data_root,
                role_bundle_train=role_bundle_train,
                train_cfg=trial_cfg,
                device=device,
                num_shards=int(base_cfg.benchmark_num_shards),
            )
            ok = report["mem_ratio"] <= float(base_cfg.max_profile_mem_ratio)
            attempts.append({"profile": candidate, "ok": ok, **report})
            if ok and report["main_positions_per_sec"] > best_speed:
                best_speed = float(report["main_positions_per_sec"])
                selected = dict(candidate)
                selected_report = dict(report)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            attempts.append({"profile": candidate, "ok": False, "error": str(exc)})
            _cleanup_cuda()
    if selected is None:
        selected = default_colab_profile(gpu_name, total_mem_gb)
    return {
        "selected_profile": selected,
        "selected_report": selected_report,
        "attempts": attempts,
        "device": str(device),
        "gpu_name": gpu_name,
        "total_mem_gb": total_mem_gb,
    }


def _save_history_outputs(history_rows: List[dict], step_rows: List[dict], reports_dir: Path) -> None:
    pd.DataFrame(history_rows).to_csv(reports_dir / "history.csv", index=False)
    pd.DataFrame(step_rows).to_csv(reports_dir / "step_history.csv", index=False)
    base_lab.save_json({"history": history_rows}, reports_dir / "history.json")


def _epoch_checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    controller: GradNormController,
    model_cfg: Dict[str, object],
    train_cfg: FT2TrainConfig,
    gate_cfg: FT2GateConfig,
    history_rows: List[dict],
    epoch: int,
    global_step: int,
    is_epoch_end: bool,
    best_any_center_score: float,
    best_gate_center_score: float,
    resume_shard_index: int = 0,
    resume_next_start: int = 0,
    rng_state: Optional[Dict[str, object]] = None,
    aux_rng_state: Optional[dict] = None,
) -> Dict[str, object]:
    return {
        "config": {"model_cfg": dict(model_cfg), "train_cfg": asdict(train_cfg), "gate_cfg": asdict(gate_cfg)},
        "epoch": int(epoch),
        "global_step": int(global_step),
        "is_epoch_end": bool(is_epoch_end),
        "resume_shard_index": int(resume_shard_index),
        "resume_next_start": int(resume_next_start),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "gradnorm_state": controller.state_dict(),
        "history": list(history_rows),
        "best_any_center_score": float(best_any_center_score),
        "best_gate_center_score": float(best_gate_center_score),
        "rng_state": rng_state or _capture_rng_state(),
        "aux_rng_state": aux_rng_state,
    }


def _load_resume_state(
    latest_ckpt: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    controller: GradNormController,
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
        }
    payload = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    controller.load_state_dict(payload.get("gradnorm_state", {}))
    epoch = int(payload.get("epoch", 0))
    is_epoch_end = bool(payload.get("is_epoch_end", True))
    return {
        "start_epoch": epoch + 1 if is_epoch_end else epoch,
        "global_step": int(payload.get("global_step", 0)),
        "best_any_center_score": float(payload.get("best_any_center_score", float("inf"))),
        "best_gate_center_score": float(payload.get("best_gate_center_score", float("inf"))),
        "history_rows": list(payload.get("history", [])),
        "resume_shard_index": int(payload.get("resume_shard_index", 0)),
        "resume_next_start": int(payload.get("resume_next_start", 0)),
        "resume_is_epoch_end": is_epoch_end,
        "resume_rng_state": payload.get("rng_state"),
        "resume_aux_rng_state": payload.get("aux_rng_state"),
    }


def evaluate_reference_checkpoint(
    checkpoint_path: str | Path,
    data_root: str | Path,
    pooled_center_bundle: Dict[str, object],
    role_bundle_eval: Dict[str, object],
    device: torch.device,
    eval_batch_size: int,
    val_max_samples: int,
    val_num_shards: int,
) -> Dict[str, object]:
    oracle_bundle = obj_lab.build_primary_oracle_bundle(data_root)
    model, _ = base_lab.load_model_from_checkpoint(checkpoint_path, device=device)
    try:
        return ft1_lab.evaluate_ft1_model(
            model=model,
            data_root=data_root,
            device=device,
            eval_batch_size=eval_batch_size,
            split="val",
            max_samples=val_max_samples,
            num_shards=val_num_shards,
            oracle_bundle=oracle_bundle,
            pooled_center_bundle=pooled_center_bundle,
            role_bundle=role_bundle_eval,
        )
    finally:
        del model
        _cleanup_cuda()


def run_ft2_training(
    repo_root: str | Path,
    runs_root: str | Path,
    data_root: str | Path,
    model_cfg: Dict[str, object],
    train_cfg: FT2TrainConfig,
    gate_cfg: FT2GateConfig,
    autotune_profile: bool = True,
) -> Dict[str, object]:
    repo_root = Path(repo_root)
    runs_root = Path(runs_root)
    data_root = Path(data_root)
    validate_ft2_train_config(train_cfg)
    paths = build_default_paths(repo_root=repo_root, runs_root=runs_root, run_name=train_cfg.run_name)
    runtime_check = validate_ft2_runtime_paths(data_root=data_root, paths=paths)
    base_lab.save_json(runtime_check, paths["reports_dir"] / "runtime_check.json")
    if not bool(runtime_check["ok"]):
        raise RuntimeError("FT2 runtime validation failed: " + json.dumps(runtime_check["missing"], ensure_ascii=False))

    device = base_lab.choose_device(prefer_cuda=True)
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    total_mem_gb = float(torch.cuda.get_device_properties(device).total_memory / 1024**3) if device.type == "cuda" else 0.0
    if bool(train_cfg.cudnn_benchmark) and device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    preload_x_dtype = _resolve_preload_numpy_dtype(train_cfg.preload_shard_dtype, train_cfg.use_amp, train_cfg.amp_dtype)

    role_splits = prepare_ft2_role_bundle(
        bundle_dir=paths["oracle_role_bundle_dir"],
        split_dir=paths["cache_dir"] / "role_split",
        split_cfg=FT2RoleSplitConfig(val_frac=train_cfg.role_val_frac, seed=train_cfg.role_split_seed),
        preload_x_dtype=preload_x_dtype,
        refresh_split=bool(train_cfg.role_refresh_cache),
    )
    base_lab.save_json(role_splits["split_manifest"], paths["reports_dir"] / "role_split_manifest.json")

    tuned_cfg = train_cfg
    autotune_report = None
    if autotune_profile:
        autotune_report = autotune_ft2_profile(
            model_cfg=model_cfg,
            data_root=data_root,
            role_bundle_train=role_splits["train"],
            device=device,
            gpu_name=gpu_name,
            total_mem_gb=total_mem_gb,
            base_cfg=train_cfg,
        )
        selected = dict(autotune_report["selected_profile"])
        tuned_cfg = FT2TrainConfig(**{**asdict(train_cfg), **selected})
        validate_ft2_train_config(tuned_cfg)
        base_lab.save_json(autotune_report, paths["reports_dir"] / "train_batch_autotune.json")

    oracle_bundle = obj_lab.build_primary_oracle_bundle(data_root)
    pooled_center_bundle = ft1_lab.load_pooled_center_bundle(paths["pooled_center_bundle_dir"])
    l4_reference = evaluate_reference_checkpoint(
        checkpoint_path=paths["l4_reference_ckpt"],
        data_root=data_root,
        pooled_center_bundle=pooled_center_bundle,
        role_bundle_eval=role_splits["val"],
        device=device,
        eval_batch_size=tuned_cfg.eval_batch_size,
        val_max_samples=tuned_cfg.val_max_samples,
        val_num_shards=tuned_cfg.val_num_shards,
    )
    base_lab.save_json(
        _json_ready({
            "checkpoint": str(paths["l4_reference_ckpt"]),
            "primary": l4_reference["primary"],
            "pooled_center_eval": l4_reference["pooled_center_eval"],
            "role_metrics": l4_reference["role_eval"]["metrics"],
        }),
        paths["reports_dir"] / "l4_reference.json",
    )

    train_shards = ab_lab.resolve_split_shards(data_root, "train", tuned_cfg.train_num_shards)
    if not train_shards:
        raise RuntimeError(f"No FT2 train shards found under {data_root / 'train'}")
    optimizer_steps_per_epoch = _optimizer_steps_per_epoch(train_shards, tuned_cfg.main_batch_size, tuned_cfg.grad_accum_steps)
    total_optimizer_steps = optimizer_steps_per_epoch * int(tuned_cfg.epochs)
    total_main_samples = _epoch_main_samples(train_shards)

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
    controller = GradNormController(
        alpha=tuned_cfg.gradnorm_alpha,
        weight_sum=tuned_cfg.gradnorm_weight_sum,
        warmup_epochs=tuned_cfg.gradnorm_warmup_epochs,
        adapt_rate=tuned_cfg.gradnorm_adapt_rate,
        init_weights={
            "main": tuned_cfg.lambda_main_init,
            "clean_total": tuned_cfg.lambda_clean_init,
            "ambiguous": tuned_cfg.lambda_ambiguous_init,
        },
    )
    backbone_params = [param for name, param in model.named_parameters() if param.requires_grad and (name.startswith("stem.") or name.startswith("blocks."))]
    head_params = [param for name, param in model.named_parameters() if param.requires_grad and not (name.startswith("stem.") or name.startswith("blocks."))]
    all_params = [param for param in model.parameters() if param.requires_grad]

    latest_ckpt = paths["checkpoints_dir"] / "ckpt_latest.pt"
    best_any_ckpt = paths["checkpoints_dir"] / "ckpt_best_any.pt"
    best_gate_ckpt = paths["checkpoints_dir"] / "ckpt_best_gate.pt"
    best_pareto_a_ckpt = paths["checkpoints_dir"] / "ckpt_best_pareto_A.pt"
    best_pareto_b_ckpt = paths["checkpoints_dir"] / "ckpt_best_pareto_B.pt"
    epoch_ckpts_dir = (paths["checkpoints_dir"] / "epochs") if bool(tuned_cfg.save_epoch_checkpoints) else None
    if epoch_ckpts_dir is not None:
        epoch_ckpts_dir.mkdir(parents=True, exist_ok=True)
    resume = _load_resume_state(latest_ckpt, model, optimizer, scheduler, controller) if bool(tuned_cfg.resume_if_exists) else {
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
    }
    history_rows: List[dict] = list(resume["history_rows"])
    step_rows: List[dict] = []
    best_any_center_score = float(resume["best_any_center_score"])
    best_gate_center_score = float(resume["best_gate_center_score"])
    global_step = int(resume["global_step"])
    best_pareto_a_key = (float("inf"), float("inf"))
    best_pareto_b_center_score = float("inf")
    for row in history_rows:
        best_pareto_a_key = min(best_pareto_a_key, _pareto_a_key(row))
        if "center_score" in row:
            best_pareto_b_center_score = min(best_pareto_b_center_score, float(row["center_score"]))
    variant = ft1_lab.build_l4_variant()
    amp_enabled = bool(tuned_cfg.use_amp) and (device.type == "cuda")
    amp_dtype = _resolve_amp_dtype(tuned_cfg.amp_dtype)
    amp_grad_scale = 1.0
    if amp_enabled and amp_dtype == torch.float16:
        amp_grad_scale = max(float(tuned_cfg.amp_loss_scale), 1.0)
    next_periodic_save_time = time.time() + max(int(tuned_cfg.periodic_save_minutes), 1) * 60
    aux_rng = np.random.default_rng(int(tuned_cfg.seed) + 20260410)
    if isinstance(resume["resume_aux_rng_state"], dict):
        try:
            aux_rng.bit_generator.state = resume["resume_aux_rng_state"]
        except Exception as exc:
            print(f"[ft2][resume] Failed to restore aux RNG state ({exc}); continuing with seeded RNG")

    def _maybe_save_periodic_latest_checkpoint(epoch: int, resume_shard_index: int, resume_next_start: int) -> None:
        nonlocal next_periodic_save_time
        now = time.time()
        if now < next_periodic_save_time:
            return
        payload = _epoch_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            controller=controller,
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
        epoch_running = {
            "main_objective": 0.0,
            "clean_fit": 0.0,
            "ambiguous": 0.0,
            "margin": 0.0,
            "mean_main_weight": 0.0,
            "downweighted_frac": 0.0,
            "n": 0,
        }
        grad_samples: List[dict] = []
        t0 = time.time()
        accum_micro_step = 0
        pending_controller_losses: Optional[Dict[str, float]] = None
        pending_controller_grad_norms: Optional[Dict[str, float]] = None
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
                aux_idx = _sample_aux_indices(role_splits["train"], tuned_cfg, aux_rng)
                xb_main = _to_device_batch(X_shard[idx], device, bool(tuned_cfg.channels_last), bool(tuned_cfg.pin_memory_batches))
                yb_main = torch.from_numpy(np.asarray(y_shard[idx], dtype=np.float32)).to(device=device, non_blocking=True).view(-1)
                xb_aux = _to_device_batch(role_splits["train"]["X_train"][aux_idx], device, bool(tuned_cfg.channels_last), bool(tuned_cfg.pin_memory_batches))
                yb_aux = torch.from_numpy(np.asarray(role_splits["train"]["oracle_y"][aux_idx], dtype=np.float32)).to(device=device, non_blocking=True).view(-1)
                role_aux = torch.from_numpy(np.asarray(role_splits["train"]["role_code"][aux_idx], dtype=np.int64)).to(device=device, non_blocking=True).view(-1)

                accum_scale = 1.0 / max(int(tuned_cfg.grad_accum_steps), 1)
                with autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype if amp_enabled else None):
                    main_logits = model.forward_logits(xb_main).view(-1)
                    main_terms = ft1_lab.compute_l4_main_terms(main_logits, yb_main, variant, tuned_cfg)
                with freeze_batchnorm_running_stats(model):
                    with autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype if amp_enabled else None):
                        aux_logits = model.forward_logits(xb_aux).view(-1)
                        aux_terms = compute_ft2_aux_components(aux_logits, yb_aux, role_aux, tuned_cfg)

                weights = controller.current_weights(epoch)
                weighted_main = float(weights["main"]) * main_terms["objective"]
                weighted_clean = float(weights["clean_total"]) * aux_terms["clean_total"]
                weighted_amb = float(weights["ambiguous"]) * aux_terms["ambiguous"]

                main_for_grad = weighted_main * accum_scale
                clean_for_grad = weighted_clean * accum_scale
                amb_for_grad = weighted_amb * accum_scale
                total_for_grad = main_for_grad + clean_for_grad + amb_for_grad
                total_objective = weighted_main + weighted_clean + weighted_amb
                if not bool(torch.isfinite(total_objective)):
                    optimizer.zero_grad(set_to_none=True)
                    accum_micro_step = 0
                    pending_controller_losses = None
                    pending_controller_grad_norms = None
                    print(
                        f"[ft2][epoch={epoch}] non-finite objective at shard={pack['shard_id']}, "
                        f"accum_micro_step={accum_micro_step}; dropping current accumulation window"
                    )
                    continue

                main_backbone_grads = torch.autograd.grad(main_for_grad * float(amp_grad_scale), backbone_params, retain_graph=True, allow_unused=True)
                clean_backbone_grads = torch.autograd.grad(clean_for_grad * float(amp_grad_scale), backbone_params, retain_graph=True, allow_unused=True)
                amb_backbone_grads = torch.autograd.grad(amb_for_grad * float(amp_grad_scale), backbone_params, retain_graph=True, allow_unused=True)
                main_backbone_grads = _unscale_grad_list(main_backbone_grads, amp_grad_scale)
                clean_backbone_grads = _unscale_grad_list(clean_backbone_grads, amp_grad_scale)
                amb_backbone_grads = _unscale_grad_list(amb_backbone_grads, amp_grad_scale)
                merged_backbone, pcgrad_report = project_multi_task_backbone_conflicts(
                    {"main": main_backbone_grads, "clean_total": clean_backbone_grads, "ambiguous": amb_backbone_grads},
                    eps=float(tuned_cfg.pcgrad_eps),
                )
                if not bool(tuned_cfg.use_backbone_pcgrad):
                    merged_backbone = []
                    for idx_param in range(len(backbone_params)):
                        value = None
                        for grads in (main_backbone_grads, clean_backbone_grads, amb_backbone_grads):
                            value = _optional_add(value, grads[idx_param])
                        merged_backbone.append(value)
                    pcgrad_report["grad_conflict_backbone_post"] = pcgrad_report["grad_conflict_backbone"]
                head_grads = torch.autograd.grad(total_for_grad * float(amp_grad_scale), head_params, retain_graph=False, allow_unused=True)
                head_grads = _unscale_grad_list(head_grads, amp_grad_scale)

                should_step = ((accum_micro_step + 1) % max(int(tuned_cfg.grad_accum_steps), 1)) == 0
                next_global_step = int(global_step) + (1 if should_step else 0)
                should_monitor = (
                    should_step
                    and (next_global_step % max(int(tuned_cfg.grad_monitor_every_steps), 1) == 0)
                )
                pending_controller_grad_norms = {
                    "main": _grad_norm(main_backbone_grads),
                    "clean_total": _grad_norm(clean_backbone_grads),
                    "ambiguous": _grad_norm(amb_backbone_grads),
                }
                pending_controller_losses = {
                    "main": float(main_terms["objective"].item()),
                    "clean_total": float(aux_terms["clean_total"].item()),
                    "ambiguous": float(aux_terms["ambiguous"].item()),
                }

                for param, grad in zip(backbone_params, merged_backbone):
                    if grad is None:
                        continue
                    if param.grad is None:
                        param.grad = grad.detach().clone()
                    else:
                        param.grad.add_(grad.detach())
                for param, grad in zip(head_params, head_grads):
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
                    pending_controller_losses = None
                    pending_controller_grad_norms = None
                    print(
                        f"[ft2][epoch={epoch}] non-finite accumulated gradients at accum_micro_step={accum_micro_step}; "
                        "dropping current accumulation window"
                    )
                    continue

                if should_monitor:
                    grad_samples.append(
                        {
                            "global_step": int(next_global_step),
                            **pcgrad_report,
                            "weight_main": float(weights["main"]),
                            "weight_clean_total": float(weights["clean_total"]),
                            "weight_ambiguous": float(weights["ambiguous"]),
                        }
                    )
                    step_rows.append(
                        {
                            "global_step": int(next_global_step),
                            "train_main_objective": float(main_terms["objective"].item()),
                            "train_clean_fit": float(aux_terms["clean_fit"].item()),
                            "train_clean_total": float(aux_terms["clean_total"].item()),
                            "train_ambiguous": float(aux_terms["ambiguous"].item()),
                            "train_margin": float(aux_terms["margin"].item()),
                            **pcgrad_report,
                            "weight_main": float(weights["main"]),
                            "weight_clean_total": float(weights["clean_total"]),
                            "weight_ambiguous": float(weights["ambiguous"]),
                        }
                    )

                bs = int(yb_main.numel())
                epoch_running["main_objective"] += float(main_terms["objective"].item()) * bs
                epoch_running["clean_fit"] += float(aux_terms["clean_fit"].item()) * bs
                epoch_running["ambiguous"] += float(aux_terms["ambiguous"].item()) * bs
                epoch_running["margin"] += float(aux_terms["margin"].item()) * bs
                epoch_running["mean_main_weight"] += float(main_terms["mean_main_weight"].item()) * bs
                epoch_running["downweighted_frac"] += float(main_terms["downweighted_frac"].item()) * bs
                epoch_running["n"] += bs

                if should_step:
                    if not _all_grads_finite(all_params):
                        optimizer.zero_grad(set_to_none=True)
                        accum_micro_step = 0
                        pending_controller_losses = None
                        pending_controller_grad_norms = None
                        print(
                            f"[ft2][epoch={epoch}] non-finite gradients before step={next_global_step}; "
                            "dropping current accumulation window"
                        )
                        continue
                    controller.update(
                        epoch=epoch,
                        losses=pending_controller_losses or {},
                        grad_norms=pending_controller_grad_norms or {},
                    )
                    if tuned_cfg.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(tuned_cfg.grad_clip_norm))
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step = next_global_step
                    accum_micro_step = 0
                    pending_controller_losses = None
                    pending_controller_grad_norms = None
                    _maybe_save_periodic_latest_checkpoint(
                        epoch,
                        resume_shard_index=shard_index,
                        resume_next_start=int(start) + int(tuned_cfg.main_batch_size),
                    )
                    if global_step % max(int(tuned_cfg.log_every_steps), 1) == 0:
                        print(
                            f"[ft2][epoch={epoch}] step={global_step} "
                            f"main_obj={epoch_running['main_objective'] / max(epoch_running['n'], 1):.6f} "
                            f"clean={epoch_running['clean_fit'] / max(epoch_running['n'], 1):.6f} "
                            f"amb={epoch_running['ambiguous'] / max(epoch_running['n'], 1):.6f}"
                        )

        if accum_micro_step != 0:
            if not _all_grads_finite(all_params):
                optimizer.zero_grad(set_to_none=True)
                accum_micro_step = 0
                pending_controller_losses = None
                pending_controller_grad_norms = None
                print(
                    f"[ft2][epoch={epoch}] non-finite gradients at tail step={global_step + 1}; "
                    "dropping tail accumulation window"
                )
            else:
                controller.update(
                    epoch=epoch,
                    losses=pending_controller_losses or {},
                    grad_norms=pending_controller_grad_norms or {},
                )
                if tuned_cfg.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(tuned_cfg.grad_clip_norm))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                accum_micro_step = 0
                pending_controller_losses = None
                pending_controller_grad_norms = None
                _maybe_save_periodic_latest_checkpoint(
                    epoch,
                    resume_shard_index=len(train_shards),
                    resume_next_start=0,
                )
                if global_step % max(int(tuned_cfg.log_every_steps), 1) == 0:
                    print(
                        f"[ft2][epoch={epoch}] step={global_step} "
                        f"main_obj={epoch_running['main_objective'] / max(epoch_running['n'], 1):.6f} "
                        f"clean={epoch_running['clean_fit'] / max(epoch_running['n'], 1):.6f} "
                        f"amb={epoch_running['ambiguous'] / max(epoch_running['n'], 1):.6f}"
                    )

        epoch_eval = ft1_lab.evaluate_ft1_model(
            model=model,
            data_root=data_root,
            device=device,
            eval_batch_size=tuned_cfg.eval_batch_size,
            split="val",
            max_samples=tuned_cfg.val_max_samples,
            num_shards=tuned_cfg.val_num_shards,
            oracle_bundle=oracle_bundle,
            pooled_center_bundle=pooled_center_bundle,
            role_bundle=role_splits["val"],
        )
        primary = dict(epoch_eval["primary"])
        epoch_time_sec = float(time.time() - t0)
        gate_pass = (
            float(primary["oracle_midband_mae_sum_stable"]) <= float(l4_reference["primary"]["oracle_midband_mae_sum_stable"]) * (1.0 + float(gate_cfg.midband_mae_rel_tol))
            and float(primary["oracle_stable_0.7_slope"]) >= float(l4_reference["primary"]["oracle_stable_0.7_slope"]) - float(gate_cfg.stable_slope_abs_tol)
        )
        center_score = float(primary["center_score"])
        selected: List[str] = []
        if center_score < best_any_center_score:
            best_any_center_score = center_score
            selected.append("best_any")
        if gate_pass and center_score < best_gate_center_score:
            best_gate_center_score = center_score
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
            "optimizer_steps_per_sec": float(optimizer_steps_per_epoch / max(epoch_time_sec, 1e-6)),
            "main_samples_per_sec": float(total_main_samples / max(epoch_time_sec, 1e-6)),
            "train_main_objective": float(epoch_running["main_objective"] / max(epoch_running["n"], 1)),
            "train_clean_fit": float(epoch_running["clean_fit"] / max(epoch_running["n"], 1)),
            "train_ambiguous": float(epoch_running["ambiguous"] / max(epoch_running["n"], 1)),
            "train_margin": float(epoch_running["margin"] / max(epoch_running["n"], 1)),
            "train_mean_main_weight": float(epoch_running["mean_main_weight"] / max(epoch_running["n"], 1)),
            "train_downweighted_frac": float(epoch_running["downweighted_frac"] / max(epoch_running["n"], 1)),
            "midband_gate_pass": bool(gate_pass),
            "weight_main": float(controller.weights["main"]),
            "weight_clean_total": float(controller.weights["clean_total"]),
            "weight_ambiguous": float(controller.weights["ambiguous"]),
            "selected_checkpoint": "|".join(selected) if selected else "none",
            **primary,
        }
        if grad_samples:
            grad_df = pd.DataFrame(grad_samples)
            row["grad_conflict_backbone"] = float(grad_df["grad_conflict_backbone"].mean())
            row["grad_conflict_backbone_post"] = float(grad_df["grad_conflict_backbone_post"].mean())
            row["grad_cosine_main_clean"] = float(grad_df["grad_cosine_main_clean"].mean())
            row["grad_cosine_main_amb"] = float(grad_df["grad_cosine_main_amb"].mean())
            row["grad_cosine_clean_amb"] = float(grad_df["grad_cosine_clean_amb"].mean())
        history_rows.append(row)

        payload = _epoch_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            controller=controller,
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
        decision_payload = {
            "best_any_center_score": float(best_any_center_score),
            "best_gate_center_score": float(best_gate_center_score),
            "best_pareto_A_midband": float(best_pareto_a_key[0]),
            "best_pareto_A_slope": float(-best_pareto_a_key[1]),
            "best_pareto_B_center_score": float(best_pareto_b_center_score),
            "has_best_gate": math.isfinite(best_gate_center_score),
            "latest_checkpoint": str(latest_ckpt),
            "best_any_checkpoint": str(best_any_ckpt) if best_any_ckpt.exists() else None,
            "best_gate_checkpoint": str(best_gate_ckpt) if best_gate_ckpt.exists() else None,
            "best_pareto_A_checkpoint": str(best_pareto_a_ckpt) if best_pareto_a_ckpt.exists() else None,
            "best_pareto_B_checkpoint": str(best_pareto_b_ckpt) if best_pareto_b_ckpt.exists() else None,
            "last_epoch": int(epoch),
            "last_center_score": center_score,
            "last_gate_pass": bool(gate_pass),
            "selected_checkpoint": "|".join(selected) if selected else "none",
        }
        base_lab.save_json(decision_payload, paths["reports_dir"] / "decision_summary.json")
        _save_history_outputs(history_rows, step_rows, paths["reports_dir"])
        print(
            f"[ft2][epoch={epoch}] time={epoch_time_sec / 3600.0:.2f}h "
            f"main_sps={row['main_samples_per_sec']:.1f} center={primary['center_score']:.4f} gate={gate_pass}"
        )

    selected_ckpt = best_gate_ckpt if best_gate_ckpt.exists() else (best_any_ckpt if best_any_ckpt.exists() else latest_ckpt)
    model_eval, _ = base_lab.load_model_from_checkpoint(selected_ckpt, device=device)
    try:
        final_eval = ft1_lab.evaluate_ft1_model(
            model=model_eval,
            data_root=data_root,
            device=device,
            eval_batch_size=tuned_cfg.eval_batch_size,
            split="test",
            max_samples=tuned_cfg.test_max_samples,
            num_shards=tuned_cfg.test_num_shards,
            oracle_bundle=oracle_bundle,
            pooled_center_bundle=pooled_center_bundle,
            role_bundle=role_splits["val"],
        )
        final_eval = dict(final_eval)
        final_eval.update(dict(final_eval.get("primary", {})))
    finally:
        del model_eval
        _cleanup_cuda()
    base_lab.save_json(_json_ready(final_eval), paths["reports_dir"] / "selected_checkpoint_eval.json")
    base_lab.save_json(
        _json_ready({
            "device": str(device),
            "gpu_name": gpu_name,
            "total_mem_gb": total_mem_gb,
            "optimizer_steps_per_epoch": int(optimizer_steps_per_epoch),
            "main_samples_per_epoch": int(total_main_samples),
            "model_cfg": dict(model_cfg),
            "train_cfg": asdict(tuned_cfg),
            "gate_cfg": asdict(gate_cfg),
            "autotune_report": autotune_report,
            "role_split_manifest": role_splits["split_manifest"],
        }),
        paths["reports_dir"] / "run_config.json",
    )
    return {
        "paths": paths,
        "runtime_check": runtime_check,
        "l4_reference": l4_reference,
        "selected_checkpoint": str(selected_ckpt),
        "final_eval": final_eval,
        "autotune_report": autotune_report,
        "role_split_manifest": role_splits["split_manifest"],
    }
