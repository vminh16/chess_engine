from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.amp import autocast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "model"
EXPERIMENT_DIRS = [
    PROJECT_ROOT / "experiments" / "teacher_root_cause_lab",
    PROJECT_ROOT / "experiments" / "root_cause_ablation_suite",
    PROJECT_ROOT / "experiments" / "objective_resolution_suite",
    PROJECT_ROOT / "experiments" / "failure_b_resolution_suite",
]

for path in [PROJECT_ROOT, MODEL_ROOT, *EXPERIMENT_DIRS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from architecture_v2.model import DGRNChessNetV2  # noqa: E402
import teacher_root_cause_helpers as base_lab  # noqa: E402
import root_cause_ablation_helpers as ab_lab  # noqa: E402
import objective_resolution_helpers as obj_lab  # noqa: E402
import failure_b_resolution_helpers as fb_lab  # noqa: E402


SOURCE_ROLE_CENTER_ANCHOR = 0
SOURCE_ROLE_CENTER_HARD = 1
SOURCE_ROLE_CENTER_AMBIGUOUS = 2
ROLE_CLEAN_CENTER = 0
ROLE_AMBIGUOUS_CENTER = 1
ROLE_NAME_BY_CODE = {
    ROLE_CLEAN_CENTER: "clean_center",
    ROLE_AMBIGUOUS_CENTER: "center_ambiguous",
}


@dataclass
class FT1TrainConfig:
    run_name: str = "dgrn_5m_ft1_colab_t4_run1"
    epochs: int = 50
    main_batch_size: int = 256
    clean_center_batch_size: int = 32
    ambiguous_center_batch_size: int = 64
    grad_accum_steps: int = 2
    use_amp: bool = True
    amp_dtype: str = "float16"
    amp_loss_scale: float = 128.0
    preload_shard_dtype: str = "auto"
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
    eval_batch_size: int = 1024
    log_every_steps: int = 200
    main_center_tau_y600: float = 0.10
    main_center_min_weight: float = 0.35
    main_center_weight_power: float = 1.0
    lambda_clean_center: float = 0.20
    lambda_ambiguous_center: float = 0.10
    aux_margin_y600: float = 0.08
    aux_margin_weight: float = 0.40
    aux_huber_delta: float = 0.05
    aux_ramp_epochs: int = 4
    grad_monitor_every_steps: int = 1000
    use_backbone_pcgrad: bool = True
    pcgrad_eps: float = 1.0e-12
    resume_if_exists: bool = False


@dataclass
class FT1GateConfig:
    midband_mae_rel_tol: float = 0.05
    stable_slope_abs_tol: float = 0.02


def _cleanup_cuda() -> None:
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
        print(f"[resume] Failed to restore RNG state ({exc}); continuing with seeded RNG")


def default_colab_profile(gpu_name: Optional[str], total_mem_gb: float, cpu_count: Optional[int]) -> Dict[str, int]:
    gpu_name = (gpu_name or "").upper()
    cpu_count = int(cpu_count or 2)
    if "T4" in gpu_name and total_mem_gb >= 14.0:
        return {
            "main_batch_size": 320,
            "clean_center_batch_size": 40,
            "ambiguous_center_batch_size": 80,
            "grad_accum_steps": 2,
            "eval_batch_size": 2048,
        }
    if total_mem_gb >= 14.0:
        return {
            "main_batch_size": 320,
            "clean_center_batch_size": 32,
            "ambiguous_center_batch_size": 64,
            "grad_accum_steps": 3,
            "eval_batch_size": 1024,
        }
    if total_mem_gb >= 8.0:
        return {
            "main_batch_size": 192,
            "clean_center_batch_size": 24,
            "ambiguous_center_batch_size": 48,
            "grad_accum_steps": 4,
            "eval_batch_size": 768,
        }
    return {
        "main_batch_size": 96,
        "clean_center_batch_size": 16,
        "ambiguous_center_batch_size": 32,
        "grad_accum_steps": 4,
        "eval_batch_size": 384,
    }


def _candidate_colab_profiles(gpu_name: Optional[str], total_mem_gb: float) -> List[Dict[str, int]]:
    gpu_name = (gpu_name or "").upper()
    candidates: List[Dict[str, int]] = []
    if "T4" in gpu_name and total_mem_gb >= 14.0:
        candidates.extend(
            [
                {"main_batch_size": 512, "clean_center_batch_size": 64, "ambiguous_center_batch_size": 128, "grad_accum_steps": 1, "eval_batch_size": 3072},
                {"main_batch_size": 448, "clean_center_batch_size": 56, "ambiguous_center_batch_size": 112, "grad_accum_steps": 1, "eval_batch_size": 2560},
                {"main_batch_size": 384, "clean_center_batch_size": 48, "ambiguous_center_batch_size": 96, "grad_accum_steps": 1, "eval_batch_size": 2048},
            ]
        )
    candidates.append(default_colab_profile(gpu_name=gpu_name, total_mem_gb=total_mem_gb, cpu_count=2))
    dedup: List[Dict[str, int]] = []
    seen: set[Tuple[int, int, int, int, int]] = set()
    for item in candidates:
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


def _resolve_amp_dtype(amp_dtype: object) -> torch.dtype:
    value = str(amp_dtype).strip().lower()
    if value in {"float16", "fp16", "half"}:
        return torch.float16
    if value in {"bfloat16", "bf16"}:
        return torch.bfloat16
    return torch.float16


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


def _ft1_resume_signature(model_cfg: Dict[str, object], train_cfg: FT1TrainConfig, gate_cfg: FT1GateConfig) -> Dict[str, object]:
    train_cfg_dict = asdict(train_cfg)
    ignored = {"epochs", "resume_if_exists", "log_every_steps", "grad_monitor_every_steps"}
    filtered_train_cfg = {k: train_cfg_dict[k] for k in sorted(train_cfg_dict) if k not in ignored}
    return {
        "model_cfg": dict(model_cfg),
        "train_cfg": filtered_train_cfg,
        "gate_cfg": asdict(gate_cfg),
    }


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


def build_l4_variant() -> ab_lab.AblationVariant:
    catalog = obj_lab.build_variant_catalog()
    variant = catalog.get("L4_A1_plus_A2")
    if variant is None:
        raise KeyError("Cannot resolve L4_A1_plus_A2 from objective resolution catalog.")
    return variant


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


def validate_ft1_runtime_paths(data_root: str | Path, paths: Dict[str, Path]) -> Dict[str, object]:
    data_root = Path(data_root)
    required = {
        "data_root": data_root,
        "l4_reference_ckpt": paths["l4_reference_ckpt"],
        "oracle_role_bundle_dir": paths["oracle_role_bundle_dir"],
        "pooled_center_bundle_dir": paths["pooled_center_bundle_dir"],
    }
    split_dirs = {f"split_{split}": data_root / split for split in ("train", "val", "test")}
    missing = {name: str(path) for name, path in {**required, **split_dirs}.items() if not Path(path).exists()}
    status = {
        "ok": len(missing) == 0,
        "data_root": str(data_root),
        "paths": {name: str(path) for name, path in {**required, **split_dirs}.items()},
        "missing": missing,
    }
    if missing:
        missing_text = ", ".join(f"{name}={path}" for name, path in missing.items())
        raise FileNotFoundError(f"FT1 runtime path check failed: {missing_text}")
    return status


def _resolve_split_shards(data_root: str | Path, split: str, num_shards: Optional[int]) -> List[Tuple[int, Path, Path]]:
    pairs = base_lab.resolve_split_pairs(data_root, split)
    pairs = base_lab.select_pairs_evenly(pairs, num_shards)
    rows: List[Tuple[int, Path, Path]] = []
    for x_path, y_path in pairs:
        shard_id = int(Path(x_path).stem.split("_")[1])
        rows.append((shard_id, Path(x_path), Path(y_path)))
    return rows


def load_ft1_role_bundle(bundle_dir: str | Path) -> Dict[str, object]:
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    npz_path = bundle_dir / "oracle_role_bundle.npz"
    rows_path = bundle_dir / "oracle_role_rows.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing oracle role manifest: {manifest_path}")
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing oracle role bundle npz: {npz_path}")
    if not rows_path.exists():
        raise FileNotFoundError(f"Missing oracle role rows csv: {rows_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    npz = np.load(npz_path, allow_pickle=False)
    rows = pd.read_csv(rows_path)

    source_role_code = np.asarray(npz["role_code"], dtype=np.int64)
    final_role_code = np.where(
        source_role_code == SOURCE_ROLE_CENTER_AMBIGUOUS,
        ROLE_AMBIGUOUS_CENTER,
        ROLE_CLEAN_CENTER,
    ).astype(np.int64)
    if rows.shape[0] != final_role_code.shape[0]:
        raise ValueError(
            f"Role row mismatch: rows={rows.shape[0]} vs role_code={final_role_code.shape[0]}"
        )

    indices_by_role = {
        "clean_center": np.flatnonzero(final_role_code == ROLE_CLEAN_CENTER).astype(np.int64),
        "center_ambiguous": np.flatnonzero(final_role_code == ROLE_AMBIGUOUS_CENTER).astype(np.int64),
    }
    if indices_by_role["clean_center"].size == 0 or indices_by_role["center_ambiguous"].size == 0:
        raise RuntimeError(
            "FT1 oracle bundle must contain both clean_center and center_ambiguous samples."
        )

    return {
        "manifest": manifest,
        "rows": rows,
        "X": np.asarray(npz["X"], dtype=np.uint8),
        "X_train": np.asarray(npz["X"], dtype=np.uint8),
        "oracle_y": np.asarray(npz["oracle_y"], dtype=np.float32),
        "source_role_code": source_role_code,
        "role_code": final_role_code,
        "indices_by_role": indices_by_role,
    }


def load_pooled_center_bundle(bundle_dir: str | Path) -> Dict[str, object]:
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    npz_path = bundle_dir / "pooled_center_bundle.npz"
    rows_path = bundle_dir / "pooled_center_rows.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing pooled center manifest: {manifest_path}")
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing pooled center npz: {npz_path}")
    if not rows_path.exists():
        raise FileNotFoundError(f"Missing pooled center rows csv: {rows_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    npz = np.load(npz_path, allow_pickle=False)
    return {
        "manifest": manifest,
        "rows": pd.read_csv(rows_path),
        "X": np.asarray(npz["X"], dtype=np.uint8),
        "oracle_y600": np.asarray(npz["oracle_y600"], dtype=np.float64),
        "train_y": np.asarray(npz["train_y"], dtype=np.float64),
    }


def _main_center_weights(y_source: torch.Tensor, cfg: FT1TrainConfig) -> torch.Tensor:
    tau = max(float(cfg.main_center_tau_y600), 1e-6)
    ratio = torch.clamp(torch.abs(y_source).float() / tau, 0.0, 1.0)
    smooth = torch.pow(ratio, float(cfg.main_center_weight_power))
    return float(cfg.main_center_min_weight) + (1.0 - float(cfg.main_center_min_weight)) * smooth


def compute_l4_main_terms(
    logits: torch.Tensor,
    y_source: torch.Tensor,
    variant: ab_lab.AblationVariant,
    cfg: FT1TrainConfig,
) -> Dict[str, torch.Tensor]:
    y = ab_lab.remap_target_torch(y_source.view(-1), to_scale=variant.target_scale)
    pred = torch.tanh(logits.view(-1))
    mse_per = (pred - y) ** 2
    y_logits = ab_lab.target_to_logits(y, eps=variant.target_clamp_eps)
    residual = logits.view(-1) - y_logits
    y_clamped = torch.clamp(y, -1.0 + variant.target_clamp_eps, 1.0 - variant.target_clamp_eps)
    z_weight = torch.pow(
        torch.clamp(1.0 - y_clamped * y_clamped, min=variant.target_clamp_eps),
        variant.z_loss_beta,
    )
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
        "pred": pred,
        "mean_main_weight": torch.mean(sample_weight),
        "downweighted_frac": torch.mean((sample_weight < 0.999).float()),
        "mean_y_term": torch.mean(y_term_per),
        "mean_z_term": torch.mean(z_huber_per),
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


def compute_ft1_aux_terms(
    logits: torch.Tensor,
    oracle_y: torch.Tensor,
    role_code: torch.Tensor,
    cfg: FT1TrainConfig,
) -> Dict[str, torch.Tensor]:
    pred = torch.tanh(logits.view(-1))
    oracle_y = oracle_y.view(-1)
    role_code = role_code.view(-1).long()

    clean_mask = role_code == ROLE_CLEAN_CENTER
    ambiguous_mask = role_code == ROLE_AMBIGUOUS_CENTER

    clean_loss = _huber_mean(pred[clean_mask] - oracle_y[clean_mask], cfg.aux_huber_delta) if torch.any(clean_mask) else pred.new_tensor(0.0)
    ambiguous_loss = _huber_mean(pred[ambiguous_mask] - oracle_y[ambiguous_mask], cfg.aux_huber_delta) if torch.any(ambiguous_mask) else pred.new_tensor(0.0)
    margin_penalty = (
        torch.mean(torch.relu(torch.abs(pred[clean_mask]) - float(cfg.aux_margin_y600)) ** 2)
        if torch.any(clean_mask)
        else pred.new_tensor(0.0)
    )

    objective = (
        float(cfg.lambda_clean_center) * clean_loss
        + float(cfg.lambda_ambiguous_center) * ambiguous_loss
        + float(cfg.aux_margin_weight) * margin_penalty
    )
    return {
        "objective": objective,
        "pred": pred,
        "clean_loss": clean_loss,
        "ambiguous_loss": ambiguous_loss,
        "margin_penalty": margin_penalty,
        "clean_frac": torch.mean(clean_mask.float()),
        "ambiguous_frac": torch.mean(ambiguous_mask.float()),
    }


def autotune_ft1_profile(
    model_cfg: Dict[str, object],
    device: torch.device,
    gpu_name: Optional[str],
    total_mem_gb: float,
    base_profile: Optional[Dict[str, int]] = None,
    use_amp: bool = True,
    amp_dtype: str = "float16",
    amp_loss_scale: float = 128.0,
) -> Dict[str, int]:
    if device.type != "cuda":
        return dict(base_profile or default_colab_profile(gpu_name, total_mem_gb, cpu_count=2))

    l4_variant = build_l4_variant()
    candidates = _candidate_colab_profiles(gpu_name=gpu_name, total_mem_gb=total_mem_gb)
    if base_profile is not None:
        candidates.append(dict(base_profile))
    resolved_amp_dtype = _resolve_amp_dtype(amp_dtype)
    amp_ctx_kwargs: Dict[str, object] = {"device_type": "cuda", "enabled": bool(use_amp)}
    grad_scale = 1.0
    if bool(use_amp):
        amp_ctx_kwargs["dtype"] = resolved_amp_dtype
        if resolved_amp_dtype == torch.float16:
            grad_scale = max(float(amp_loss_scale), 1.0)

    for candidate in candidates:
        model = None
        try:
            torch.cuda.empty_cache()
            model = DGRNChessNetV2(**model_cfg).to(device)
            model.train()
            backbone_params = [
                param
                for name, param in model.named_parameters()
                if param.requires_grad and (name.startswith("stem.") or name.startswith("blocks."))
            ]
            head_params = [
                param
                for name, param in model.named_parameters()
                if param.requires_grad and not (name.startswith("stem.") or name.startswith("blocks."))
            ]
            main_bs = int(candidate["main_batch_size"])
            clean_bs = int(candidate["clean_center_batch_size"])
            ambiguous_bs = int(candidate["ambiguous_center_batch_size"])
            total_aux = clean_bs + ambiguous_bs
            xb_main = torch.randint(0, 2, (main_bs, 18, 8, 8), device=device, dtype=torch.uint8).float()
            yb_main = torch.empty(main_bs, device=device, dtype=torch.float32).uniform_(-1.0, 1.0)
            xb_aux = torch.randint(0, 2, (total_aux, 18, 8, 8), device=device, dtype=torch.uint8).float()
            yb_aux = torch.empty(total_aux, device=device, dtype=torch.float32).uniform_(-0.2, 0.2)
            role_aux = torch.cat(
                [
                    torch.full((clean_bs,), ROLE_CLEAN_CENTER, device=device, dtype=torch.long),
                    torch.full((ambiguous_bs,), ROLE_AMBIGUOUS_CENTER, device=device, dtype=torch.long),
                ],
                dim=0,
            )
            xb_mix = torch.cat([xb_main, xb_aux], dim=0)
            with autocast(**amp_ctx_kwargs):
                logits_mix = model.forward_logits(xb_mix).view(-1)
                main_logits = logits_mix[:main_bs]
                aux_logits = logits_mix[main_bs:]
                probe_cfg = FT1TrainConfig(
                    main_batch_size=main_bs,
                    clean_center_batch_size=clean_bs,
                    ambiguous_center_batch_size=ambiguous_bs,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    amp_loss_scale=amp_loss_scale,
                )
                main_terms = compute_l4_main_terms(main_logits, yb_main, l4_variant, probe_cfg)
                aux_terms = compute_ft1_aux_terms(aux_logits, yb_aux, role_aux, probe_cfg)
                main_for_grad = main_terms["objective"] * grad_scale
                aux_for_grad = aux_terms["objective"] * grad_scale
                total_for_grad = (main_terms["objective"] + aux_terms["objective"]) * grad_scale
            main_backbone_grads = torch.autograd.grad(main_for_grad, backbone_params, retain_graph=True, allow_unused=True)
            aux_backbone_grads = torch.autograd.grad(aux_for_grad, backbone_params, retain_graph=True, allow_unused=True)
            _ = project_backbone_conflicts(
                _unscale_grad_list(main_backbone_grads, grad_scale),
                _unscale_grad_list(aux_backbone_grads, grad_scale),
                eps=1.0e-12,
            )
            _ = torch.autograd.grad(total_for_grad, head_params, retain_graph=False, allow_unused=True)
            torch.cuda.synchronize(device)
            return dict(candidate)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
        finally:
            if model is not None:
                del model
            torch.cuda.empty_cache()
    return dict(base_profile or default_colab_profile(gpu_name, total_mem_gb, cpu_count=2))


def _flatten_grads(grads: Sequence[Optional[torch.Tensor]]) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    for grad in grads:
        if grad is None:
            continue
        parts.append(grad.detach().float().reshape(-1))
    if not parts:
        return torch.empty(0)
    return torch.cat(parts, dim=0)


def _cosine_from_vectors(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    denom = float(torch.norm(a) * torch.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(torch.dot(a, b) / denom)


def _clone_grad_list(grads: Sequence[Optional[torch.Tensor]]) -> List[Optional[torch.Tensor]]:
    out: List[Optional[torch.Tensor]] = []
    for grad in grads:
        if grad is None:
            out.append(None)
        else:
            out.append(grad.detach().clone())
    return out


def _optional_add(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if a is None:
        return None if b is None else b
    if b is None:
        return a
    return a + b


def project_backbone_conflicts(
    main_grads: Sequence[Optional[torch.Tensor]],
    aux_grads: Sequence[Optional[torch.Tensor]],
    eps: float,
) -> Tuple[List[Optional[torch.Tensor]], Dict[str, float]]:
    main_vec = _flatten_grads(main_grads)
    aux_vec = _flatten_grads(aux_grads)
    cosine_pre = _cosine_from_vectors(main_vec, aux_vec)
    norm_main = float(torch.norm(main_vec)) if main_vec.numel() > 0 else float("nan")
    norm_aux = float(torch.norm(aux_vec)) if aux_vec.numel() > 0 else float("nan")

    projected_aux = _clone_grad_list(aux_grads)
    projection_applied = False
    projection_scale = 0.0
    if main_vec.numel() > 0 and aux_vec.numel() > 0:
        dot = float(torch.dot(main_vec, aux_vec))
        denom = float(torch.dot(main_vec, main_vec))
        if dot < 0.0 and denom > float(eps):
            projection_applied = True
            projection_scale = dot / (denom + float(eps))
            projected_aux = []
            for main_grad, aux_grad in zip(main_grads, aux_grads):
                if aux_grad is None:
                    projected_aux.append(None)
                    continue
                if main_grad is None:
                    projected_aux.append(aux_grad.detach().clone())
                    continue
                projected_aux.append((aux_grad - projection_scale * main_grad).detach().clone())

    proj_vec = _flatten_grads(projected_aux)
    cosine_post = _cosine_from_vectors(main_vec, proj_vec)
    shared_grads = [_optional_add(mg.detach().clone() if mg is not None else None, pg) for mg, pg in zip(main_grads, projected_aux)]
    shared_vec = _flatten_grads(shared_grads)
    norm_shared = float(torch.norm(shared_vec)) if shared_vec.numel() > 0 else float("nan")
    return shared_grads, {
        "grad_cosine_backbone": cosine_pre,
        "grad_cosine_backbone_post": cosine_post,
        "grad_norm_main_backbone": norm_main,
        "grad_norm_aux_backbone": norm_aux,
        "grad_norm_shared_backbone": norm_shared,
        "grad_conflict_backbone": float(1.0 if projection_applied else 0.0),
        "grad_projection_scale": float(projection_scale),
    }


def collect_gradient_monitor(
    main_grads_backbone: Sequence[Optional[torch.Tensor]],
    aux_grads_backbone: Sequence[Optional[torch.Tensor]],
    shared_grads_backbone: Sequence[Optional[torch.Tensor]],
    main_grads_all: Sequence[Optional[torch.Tensor]],
    aux_grads_all: Sequence[Optional[torch.Tensor]],
) -> Dict[str, float]:
    main_backbone = _flatten_grads(main_grads_backbone)
    aux_backbone = _flatten_grads(aux_grads_backbone)
    shared_backbone = _flatten_grads(shared_grads_backbone)
    main_all = _flatten_grads(main_grads_all)
    aux_all = _flatten_grads(aux_grads_all)
    return {
        "grad_cosine_backbone": _cosine_from_vectors(main_backbone, aux_backbone),
        "grad_cosine_backbone_post": _cosine_from_vectors(main_backbone, shared_backbone),
        "grad_norm_main_backbone": float(torch.norm(main_backbone)) if main_backbone.numel() > 0 else float("nan"),
        "grad_norm_aux_backbone": float(torch.norm(aux_backbone)) if aux_backbone.numel() > 0 else float("nan"),
        "grad_norm_shared_backbone": float(torch.norm(shared_backbone)) if shared_backbone.numel() > 0 else float("nan"),
        "grad_conflict_backbone": float(1.0 if (_cosine_from_vectors(main_backbone, aux_backbone) < 0.0) else 0.0),
        "grad_cosine_all": _cosine_from_vectors(main_all, aux_all),
        "grad_norm_main_all": float(torch.norm(main_all)) if main_all.numel() > 0 else float("nan"),
        "grad_norm_aux_all": float(torch.norm(aux_all)) if aux_all.numel() > 0 else float("nan"),
    }


def collect_bn_sanity(model: nn.Module) -> Dict[str, float]:
    running_means: List[np.ndarray] = []
    running_vars: List[np.ndarray] = []
    nonfinite = 0
    layer_count = 0
    for module in model.modules():
        if not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            continue
        layer_count += 1
        rm = module.running_mean.detach().cpu().float().numpy()
        rv = module.running_var.detach().cpu().float().numpy()
        running_means.append(rm)
        running_vars.append(rv)
        if (not np.all(np.isfinite(rm))) or (not np.all(np.isfinite(rv))):
            nonfinite += 1
    if not running_means:
        return {
            "bn_layer_count": 0,
            "bn_mean_abs_running_mean": float("nan"),
            "bn_max_abs_running_mean": float("nan"),
            "bn_mean_running_var": float("nan"),
            "bn_min_running_var": float("nan"),
            "bn_max_running_var": float("nan"),
            "bn_nonfinite_layers": 0,
        }
    abs_mean = np.concatenate([np.abs(x).reshape(-1) for x in running_means], axis=0)
    var = np.concatenate([x.reshape(-1) for x in running_vars], axis=0)
    return {
        "bn_layer_count": int(layer_count),
        "bn_mean_abs_running_mean": float(abs_mean.mean()),
        "bn_max_abs_running_mean": float(abs_mean.max()),
        "bn_mean_running_var": float(var.mean()),
        "bn_min_running_var": float(var.min()),
        "bn_max_running_var": float(var.max()),
        "bn_nonfinite_layers": int(nonfinite),
    }


@torch.no_grad()
def predict_tanh(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    preds: List[np.ndarray] = []
    model.eval()
    for start in range(0, int(X.shape[0]), int(batch_size)):
        xb = torch.from_numpy(np.array(X[start : start + batch_size], dtype=np.float32, copy=True)).to(device=device)
        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model.forward_logits(xb).view(-1)
            pred = torch.tanh(logits)
        preds.append(pred.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(preds, axis=0)


def role_group_metrics(pred: np.ndarray, oracle: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    return {
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


def evaluate_role_bundle(
    model: nn.Module,
    role_bundle: Dict[str, object],
    device: torch.device,
    batch_size: int,
) -> Dict[str, object]:
    pred = predict_tanh(model, role_bundle["X"], device=device, batch_size=batch_size)
    oracle = np.asarray(role_bundle["oracle_y"], dtype=np.float64)
    role_code = np.asarray(role_bundle["role_code"], dtype=np.int64)

    metrics = {"all": role_group_metrics(pred, oracle)}
    for code, name in ROLE_NAME_BY_CODE.items():
        mask = role_code == int(code)
        metrics[name] = role_group_metrics(pred[mask], oracle[mask])
    return {
        "pred": pred,
        "oracle": oracle,
        "role_code": role_code,
        "metrics": metrics,
    }


def center_only_score(pooled_center_eval: Dict[str, float]) -> float:
    return float(
        float(pooled_center_eval["mae_vs_oracle"])
        + 0.30 * float(pooled_center_eval["false_decisive_0.1"])
        + 0.20 * float(pooled_center_eval["false_decisive_0.2"])
        + 0.10 * max(0.0, float(pooled_center_eval["amp_ratio"]) - 2.5)
    )


def passes_midband_gate(primary: Dict[str, object], l4_primary: Dict[str, object], gate_cfg: FT1GateConfig) -> bool:
    midband_limit = float(l4_primary["oracle_midband_mae_sum_stable"]) * (1.0 + float(gate_cfg.midband_mae_rel_tol))
    slope_limit = float(l4_primary["oracle_stable_0.7_slope"]) - float(gate_cfg.stable_slope_abs_tol)
    return bool(
        float(primary["oracle_midband_mae_sum_stable"]) <= midband_limit
        and float(primary["oracle_stable_0.7_slope"]) >= slope_limit
    )


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    history_rows: List[dict],
    epoch: int,
    global_step: int,
    best_any_center_score: float,
    best_gate_center_score: float,
    run_config: Dict[str, object],
    decision_summary: Dict[str, object],
    is_epoch_end: bool = True,
    resume_shard_index: Optional[int] = None,
    resume_next_start: Optional[int] = None,
    aux_rng_state: Optional[Dict[str, object]] = None,
    rng_state: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    payload = {
        "epoch": int(epoch),
        "is_epoch_end": bool(is_epoch_end),
        "global_step": int(global_step),
        "config": run_config,
        "history": history_rows,
        "model": model.state_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_any_center_score": float(best_any_center_score),
        "best_gate_center_score": float(best_gate_center_score),
        "decision_summary": decision_summary,
    }
    if resume_shard_index is not None:
        payload["resume_shard_index"] = int(resume_shard_index)
    if resume_next_start is not None:
        payload["resume_next_start"] = int(resume_next_start)
    if aux_rng_state is not None:
        payload["aux_rng_state"] = aux_rng_state
    if rng_state is not None:
        payload["rng_state"] = rng_state
    return payload


def _load_checkpoint_state(path: Path) -> Dict[str, object]:
    return torch.load(path, map_location="cpu", weights_only=False)


def evaluate_ft1_model(
    model: nn.Module,
    data_root: str | Path,
    device: torch.device,
    eval_batch_size: int,
    split: str,
    max_samples: int,
    num_shards: int,
    oracle_bundle: Dict[str, object],
    pooled_center_bundle: Dict[str, object],
    role_bundle: Dict[str, object],
) -> Dict[str, object]:
    oracle_cfg = ab_lab.OracleEvalConfig()
    eval_batch_size = int(eval_batch_size)
    split_eval = ab_lab.evaluate_model_on_split_scale_aware(
        model=model,
        data_root=data_root,
        split=str(split),
        device=device,
        max_samples=max_samples,
        num_shards=num_shards,
        batch_size=eval_batch_size,
        target_scale=600.0,
        oracle_cfg=oracle_cfg,
    )
    oracle_eval = ab_lab.evaluate_model_on_oracle_subset(
        model=model,
        oracle_bundle=oracle_bundle,
        device=device,
        target_scale=600.0,
        oracle_cfg=oracle_cfg,
    )
    primary = obj_lab.extract_primary_metrics(label="ft1_current", split_eval=split_eval, oracle_eval=oracle_eval)
    pooled_center_eval = fb_lab.evaluate_model_on_center_bundle(
        model=model,
        bundle=pooled_center_bundle,
        device=device,
        batch_size=eval_batch_size,
    )
    role_eval = evaluate_role_bundle(model, role_bundle, device=device, batch_size=max(eval_batch_size, 512))
    primary.update(
        {
            "pooled_center_mae": float(pooled_center_eval["mae_vs_oracle"]),
            "pooled_center_amp_ratio": float(pooled_center_eval["amp_ratio"]),
            "pooled_center_false_0.1eq": float(pooled_center_eval["false_decisive_0.1"]),
            "pooled_center_false_0.2eq": float(pooled_center_eval["false_decisive_0.2"]),
            "center_score": float(center_only_score(pooled_center_eval)),
            "clean_center_mae": float(role_eval["metrics"]["clean_center"]["mae_vs_oracle"]),
            "clean_center_amp_ratio": float(role_eval["metrics"]["clean_center"]["amp_ratio"]),
            "clean_center_false_0.1eq": float(role_eval["metrics"]["clean_center"]["false_decisive_0.1"]),
            "clean_center_false_0.2eq": float(role_eval["metrics"]["clean_center"]["false_decisive_0.2"]),
            "ambiguous_center_mae": float(role_eval["metrics"]["center_ambiguous"]["mae_vs_oracle"]),
            "ambiguous_center_amp_ratio": float(role_eval["metrics"]["center_ambiguous"]["amp_ratio"]),
            "ambiguous_center_false_0.1eq": float(role_eval["metrics"]["center_ambiguous"]["false_decisive_0.1"]),
            "ambiguous_center_false_0.2eq": float(role_eval["metrics"]["center_ambiguous"]["false_decisive_0.2"]),
        }
    )
    return {
        "split_eval": split_eval,
        "oracle_eval": oracle_eval,
        "pooled_center_eval": pooled_center_eval,
        "role_eval": role_eval,
        "primary": primary,
    }


def evaluate_saved_checkpoint(
    checkpoint_path: str | Path,
    data_root: str | Path,
    pooled_center_bundle_dir: str | Path,
    oracle_role_bundle_dir: str | Path,
    device: torch.device,
    eval_batch_size: int = 1024,
    test_max_samples: int = 200_000,
    test_num_shards: int = 4,
) -> Dict[str, object]:
    oracle_bundle = obj_lab.build_primary_oracle_bundle(data_root)
    pooled_center_bundle = load_pooled_center_bundle(pooled_center_bundle_dir)
    role_bundle = load_ft1_role_bundle(oracle_role_bundle_dir)
    model, _ = base_lab.load_model_from_checkpoint(checkpoint_path, device=device)
    try:
        eval_result = evaluate_ft1_model(
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
        # Backward compatibility for notebook cells that read primary metrics at top-level.
        eval_result = dict(eval_result)
        eval_result.update(dict(eval_result.get("primary", {})))
        return eval_result
    finally:
        del model
        _cleanup_cuda()


def evaluate_l4_reference(
    checkpoint_path: str | Path,
    data_root: str | Path,
    pooled_center_bundle_dir: str | Path,
    oracle_role_bundle_dir: str | Path,
    device: torch.device,
    eval_batch_size: int = 1024,
    val_max_samples: int = 100_000,
    val_num_shards: int = 2,
) -> Dict[str, object]:
    oracle_bundle = obj_lab.build_primary_oracle_bundle(data_root)
    pooled_center_bundle = load_pooled_center_bundle(pooled_center_bundle_dir)
    role_bundle = load_ft1_role_bundle(oracle_role_bundle_dir)
    model, _ = base_lab.load_model_from_checkpoint(checkpoint_path, device=device)
    try:
        return evaluate_ft1_model(
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


def _aux_weight_scale(epoch: int, cfg: FT1TrainConfig) -> float:
    ramp_epochs = max(int(cfg.aux_ramp_epochs), 1)
    return float(min(1.0, (int(epoch) + 1) / ramp_epochs))


def _sample_aux_indices(role_bundle: Dict[str, object], cfg: FT1TrainConfig, rng: np.random.Generator) -> np.ndarray:
    clean_pool = np.asarray(role_bundle["indices_by_role"]["clean_center"], dtype=np.int64)
    ambiguous_pool = np.asarray(role_bundle["indices_by_role"]["center_ambiguous"], dtype=np.int64)
    clean = rng.choice(
        clean_pool,
        size=int(cfg.clean_center_batch_size),
        replace=clean_pool.size < int(cfg.clean_center_batch_size),
    ).astype(np.int64)
    ambiguous = rng.choice(
        ambiguous_pool,
        size=int(cfg.ambiguous_center_batch_size),
        replace=ambiguous_pool.size < int(cfg.ambiguous_center_batch_size),
    ).astype(np.int64)
    out = np.concatenate([clean, ambiguous], axis=0).astype(np.int64)
    rng.shuffle(out)
    return out


def _load_shard_arrays(x_path: Path, y_path: Path, preload_x_dtype: Optional[np.dtype] = None) -> Tuple[np.ndarray, np.ndarray]:
    X = np.load(x_path, mmap_mode="r")
    if preload_x_dtype is not None:
        X = np.asarray(X, dtype=preload_x_dtype)
    y = np.load(y_path, mmap_mode="r").astype(np.float32, copy=False)
    return X, y


def _format_history_row(row: Dict[str, object]) -> Dict[str, object]:
    out = {}
    for key, value in row.items():
        if isinstance(value, np.generic):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def save_history_outputs(history_rows: List[dict], step_rows: List[dict], reports_dir: Path) -> None:
    history_df = pd.DataFrame(history_rows)
    step_columns = [
        "global_step",
        "train_total_objective",
        "train_main_objective",
        "train_aux_objective",
        "aux_scale",
        "grad_cosine_backbone",
        "grad_cosine_backbone_pre",
        "grad_cosine_backbone_post",
        "grad_norm_main_backbone",
        "grad_norm_aux_backbone",
        "grad_norm_shared_backbone",
        "grad_conflict_backbone",
        "grad_projection_scale",
        "grad_cosine_all",
        "grad_norm_main_all",
        "grad_norm_aux_all",
    ]
    step_df = pd.DataFrame(step_rows, columns=step_columns)
    base_lab.save_dataframe(history_df, reports_dir / "history.csv")
    base_lab.save_dataframe(step_df, reports_dir / "step_history.csv")
    base_lab.save_json(
        {
            "history": [_format_history_row(row) for row in history_rows],
            "step_history_rows": int(step_df.shape[0]),
        },
        reports_dir / "history.json",
    )


def run_ft1_full_retrain(
    runs_root: str | Path,
    data_root: str | Path,
    model_cfg: Dict[str, object],
    train_cfg: FT1TrainConfig,
    gate_cfg: FT1GateConfig,
    device: torch.device,
    repo_root: str | Path | None = None,
) -> Dict[str, object]:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]
    if int(train_cfg.epochs) <= 0:
        raise ValueError("train_cfg.epochs must be > 0")
    if int(train_cfg.main_batch_size) <= 0:
        raise ValueError("train_cfg.main_batch_size must be > 0")
    if int(train_cfg.eval_batch_size) <= 0:
        raise ValueError("train_cfg.eval_batch_size must be > 0")
    if int(train_cfg.grad_accum_steps) <= 0:
        raise ValueError("train_cfg.grad_accum_steps must be > 0")
    if int(train_cfg.clean_center_batch_size) < 0 or int(train_cfg.ambiguous_center_batch_size) < 0:
        raise ValueError("train_cfg clean/ambiguous center batch sizes must be >= 0")
    if int(train_cfg.grad_monitor_every_steps) <= 0:
        raise ValueError("train_cfg.grad_monitor_every_steps must be > 0")
    if float(getattr(train_cfg, "amp_loss_scale", 128.0)) <= 0.0:
        raise ValueError("train_cfg.amp_loss_scale must be > 0")
    repo_root = Path(repo_root)
    runs_root = Path(runs_root)
    data_root = Path(data_root)
    paths = build_default_paths(repo_root=repo_root, runs_root=runs_root, run_name=train_cfg.run_name)
    runtime_check = validate_ft1_runtime_paths(data_root=data_root, paths=paths)
    base_lab.save_json(runtime_check, paths["reports_dir"] / "runtime_check.json")
    current_signature = _ft1_resume_signature(model_cfg=model_cfg, train_cfg=train_cfg, gate_cfg=gate_cfg)

    l4_variant = build_l4_variant()
    oracle_bundle = obj_lab.build_primary_oracle_bundle(data_root)
    pooled_center_bundle = load_pooled_center_bundle(paths["pooled_center_bundle_dir"])
    role_bundle = load_ft1_role_bundle(paths["oracle_role_bundle_dir"])
    preload_x_dtype = _resolve_preload_numpy_dtype(
        preload_dtype=getattr(train_cfg, "preload_shard_dtype", "auto"),
        use_amp=bool(getattr(train_cfg, "use_amp", True)),
        amp_dtype=getattr(train_cfg, "amp_dtype", "float16"),
    )
    if preload_x_dtype is not None:
        role_bundle["X_train"] = np.asarray(role_bundle["X"], dtype=preload_x_dtype)

    l4_reference = evaluate_l4_reference(
        checkpoint_path=paths["l4_reference_ckpt"],
        data_root=data_root,
        pooled_center_bundle_dir=paths["pooled_center_bundle_dir"],
        oracle_role_bundle_dir=paths["oracle_role_bundle_dir"],
        device=device,
        eval_batch_size=train_cfg.eval_batch_size,
        val_max_samples=train_cfg.val_max_samples,
        val_num_shards=train_cfg.val_num_shards,
    )
    base_lab.save_json(
        {
            "checkpoint": str(paths["l4_reference_ckpt"]),
            "primary": l4_reference["primary"],
            "pooled_center_eval": l4_reference["pooled_center_eval"],
            "role_metrics": l4_reference["role_eval"]["metrics"],
        },
        paths["reports_dir"] / "l4_reference.json",
    )

    train_shards = _resolve_split_shards(data_root, "train", train_cfg.train_num_shards)
    if not train_shards:
        raise RuntimeError(f"No training shards found at {data_root / 'train'}")
    total_samples = int(sum(int(np.load(y_path, mmap_mode="r").shape[0]) for _, _, y_path in train_shards))
    total_micro_steps = sum(
        int(math.ceil(int(np.load(y_path, mmap_mode="r").shape[0]) / train_cfg.main_batch_size))
        for _, _, y_path in train_shards
    )
    total_optimizer_steps = int(math.ceil(total_micro_steps / max(int(train_cfg.grad_accum_steps), 1)) * int(train_cfg.epochs))

    model = DGRNChessNetV2(**model_cfg).to(device)
    if not hasattr(model, "forward_logits"):
        raise RuntimeError("FT1 training requires model.forward_logits().")
    optimizer = ab_lab.build_optimizer(model, lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_optimizer_steps, 1),
        eta_min=train_cfg.min_lr,
    )

    latest_ckpt = paths["checkpoints_dir"] / "ckpt_latest.pt"
    best_gate_ckpt = paths["checkpoints_dir"] / "ckpt_best_gate.pt"
    best_any_ckpt = paths["checkpoints_dir"] / "ckpt_best_any.pt"
    decision_summary_path = paths["reports_dir"] / "decision_summary.json"

    history_rows: List[dict] = []
    step_rows: List[dict] = []
    decision_summary_out: Dict[str, object] = {}
    global_step = 0
    start_epoch = 0
    resume_is_epoch_end = True
    resume_shard_index = 0
    resume_next_start = 0
    resume_aux_rng_state: Optional[Dict[str, object]] = None
    resume_rng_state: Optional[Dict[str, object]] = None
    best_any_center_score = float("inf")
    best_gate_center_score = float("inf")
    if decision_summary_path.exists():
        try:
            decision_summary_out = json.loads(decision_summary_path.read_text(encoding="utf-8"))
        except Exception:
            decision_summary_out = {}
    if latest_ckpt.exists():
        if not bool(train_cfg.resume_if_exists):
            raise RuntimeError(
                f"Found existing checkpoint at {latest_ckpt} while resume_if_exists=False. "
                f"Use a new run_name or enable resume explicitly."
            )
        existing_signature: Dict[str, object] = {}
        existing_run_config_path = paths["reports_dir"] / "run_config.json"
        if existing_run_config_path.exists():
            existing_run_config = json.loads(existing_run_config_path.read_text(encoding="utf-8"))
        resume = _load_checkpoint_state(latest_ckpt)
        if not existing_signature:
            existing_signature = dict(resume.get("config", {}).get("resume_signature", {}))
        if existing_signature:
            diffs = _diff_resume_signature(existing_signature, current_signature)
            if diffs:
                diff_text = "; ".join(diffs[:12])
                raise RuntimeError(
                    "Resume config mismatch for existing FT1 run. "
                    f"Use a new run_name or restore the old config. Diffs: {diff_text}"
                )
        resume_model_state = resume.get("model_state", resume.get("model"))
        if resume_model_state is None:
            raise KeyError("Resume checkpoint missing model state: expected 'model_state' or 'model'")
        model.load_state_dict(resume_model_state, strict=True)
        resume_optimizer_state = resume.get("optimizer_state")
        if resume_optimizer_state is None:
            raise KeyError("Resume checkpoint missing optimizer_state")
        optimizer.load_state_dict(resume_optimizer_state)
        resume_scheduler_state = resume.get("scheduler_state")
        if resume_scheduler_state is not None:
            scheduler.load_state_dict(resume_scheduler_state)
        history_rows = list(resume.get("history", []))
        step_history_path = paths["reports_dir"] / "step_history.csv"
        if step_history_path.exists():
            step_rows = pd.read_csv(step_history_path).to_dict(orient="records")
        global_step = int(resume.get("global_step", 0))
        resume_epoch = int(resume.get("epoch", -1))
        resume_is_epoch_end = bool(resume.get("is_epoch_end", True))
        resume_aux_rng_state = resume.get("aux_rng_state")
        resume_rng_state = resume.get("rng_state")
        start_epoch = (resume_epoch + 1) if resume_is_epoch_end else max(0, resume_epoch)
        if not resume_is_epoch_end:
            resume_shard_index = min(max(0, int(resume.get("resume_shard_index", 0))), len(train_shards))
            resume_next_start = max(0, int(resume.get("resume_next_start", 0)))
        best_any_center_score = float(resume.get("best_any_center_score", best_any_center_score))
        best_gate_center_score = float(resume.get("best_gate_center_score", best_gate_center_score))
        print(
            f"[resume] Loaded {latest_ckpt} at epoch={start_epoch} "
            f"global_step={global_step} is_epoch_end={resume_is_epoch_end} "
            f"resume_shard_index={resume_shard_index} resume_next_start={resume_next_start}"
        )

    run_config = {
        "model_cfg": dict(model_cfg),
        "train_cfg": asdict(train_cfg),
        "gate_cfg": asdict(gate_cfg),
        "resume_signature": current_signature,
        "l4_variant": asdict(l4_variant),
        "data_root": str(data_root),
        "l4_reference_ckpt": str(paths["l4_reference_ckpt"]),
        "oracle_role_bundle_dir": str(paths["oracle_role_bundle_dir"]),
        "pooled_center_bundle_dir": str(paths["pooled_center_bundle_dir"]),
        "total_samples": int(total_samples),
        "train_shards": [str(x_path) for _, x_path, _ in train_shards],
    }
    base_lab.save_json(run_config, paths["reports_dir"] / "run_config.json")

    periodic_latest_interval_sec = 20.0 * 60.0
    next_periodic_latest_save_time = time.time() + periodic_latest_interval_sec

    def _maybe_save_periodic_latest_checkpoint(
        current_epoch: int,
        resume_shard_index: Optional[int] = None,
        resume_next_start: Optional[int] = None,
        aux_rng_state: Optional[Dict[str, object]] = None,
        rng_state: Optional[Dict[str, object]] = None,
    ) -> None:
        nonlocal next_periodic_latest_save_time
        now = time.time()
        if now < next_periodic_latest_save_time:
            return
        periodic_payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            history_rows=history_rows,
            epoch=current_epoch,
            global_step=global_step,
            best_any_center_score=best_any_center_score,
            best_gate_center_score=best_gate_center_score,
            run_config=run_config,
            decision_summary=dict(decision_summary_out),
            is_epoch_end=False,
            resume_shard_index=resume_shard_index,
            resume_next_start=resume_next_start,
            aux_rng_state=aux_rng_state,
            rng_state=rng_state,
        )
        ab_lab.atomic_torch_save(periodic_payload, latest_ckpt)
        while next_periodic_latest_save_time <= now:
            next_periodic_latest_save_time += periodic_latest_interval_sec

    aux_rng = np.random.default_rng(train_cfg.seed + 2026)
    if isinstance(resume_aux_rng_state, dict):
        try:
            aux_rng.bit_generator.state = resume_aux_rng_state
        except Exception as exc:
            print(f"[resume] Failed to restore aux RNG state ({exc}); continuing with seeded RNG")
    backbone_params = [
        param
        for name, param in model.named_parameters()
        if param.requires_grad and (name.startswith("stem.") or name.startswith("blocks."))
    ]
    head_params = [
        param
        for name, param in model.named_parameters()
        if param.requires_grad and not (name.startswith("stem.") or name.startswith("blocks."))
    ]
    all_params = [param for param in model.parameters() if param.requires_grad]
    use_train_amp = bool(getattr(train_cfg, "use_amp", True)) and (device.type == "cuda")
    amp_ctx_kwargs: Dict[str, object] = {
        "device_type": device.type,
        "enabled": use_train_amp,
    }
    amp_grad_scale = 1.0
    if use_train_amp:
        resolved_amp_dtype = _resolve_amp_dtype(getattr(train_cfg, "amp_dtype", "float16"))
        amp_ctx_kwargs["dtype"] = resolved_amp_dtype
        if resolved_amp_dtype == torch.float16:
            amp_grad_scale = max(float(getattr(train_cfg, "amp_loss_scale", 128.0)), 1.0)

    for epoch in range(start_epoch, int(train_cfg.epochs)):
        is_partial_resume_epoch = (epoch == start_epoch and not resume_is_epoch_end)
        if is_partial_resume_epoch and isinstance(resume_rng_state, dict):
            _restore_rng_state(resume_rng_state)
        else:
            set_global_seed(train_cfg.seed + epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        aux_scale = _aux_weight_scale(epoch, train_cfg)
        epoch_running = {
            "main_objective": 0.0,
            "aux_objective": 0.0,
            "clean_loss": 0.0,
            "ambiguous_loss": 0.0,
            "margin_penalty": 0.0,
            "mean_main_weight": 0.0,
            "downweighted_frac": 0.0,
            "n": 0,
        }
        grad_samples: List[Dict[str, float]] = []
        t0 = time.time()
        epoch_resume_shard_index = resume_shard_index if (epoch == start_epoch and not resume_is_epoch_end) else 0
        epoch_resume_next_start = resume_next_start if (epoch == start_epoch and not resume_is_epoch_end) else 0

        micro_step = 0
        for shard_index, (shard_id, x_path, y_path) in enumerate(train_shards):
            if shard_index < epoch_resume_shard_index:
                continue
            X_shard, y_shard = _load_shard_arrays(x_path, y_path, preload_x_dtype=preload_x_dtype)
            abs_y = np.abs(y_shard.astype(np.float64, copy=False))
            order = ab_lab.build_band_balanced_order(
                abs_y=abs_y,
                batch_size=int(train_cfg.main_batch_size),
                band_edges_y600=l4_variant.balance_band_edges_y600,
                rng=np.random.default_rng(train_cfg.seed + epoch * 10_000 + shard_id),
                target_scale=float(l4_variant.target_scale),
            )

            shard_start_offset = epoch_resume_next_start if shard_index == epoch_resume_shard_index else 0
            shard_start_offset = min(max(0, int(shard_start_offset)), int(order.shape[0]))
            for start in range(shard_start_offset, int(order.shape[0]), int(train_cfg.main_batch_size)):
                idx = order[start : start + int(train_cfg.main_batch_size)]
                xb_main = torch.from_numpy(np.array(X_shard[idx], copy=False)).to(device=device, non_blocking=True)
                yb_main = torch.from_numpy(np.array(y_shard[idx], dtype=np.float32, copy=True)).to(device=device, non_blocking=True).view(-1)

                aux_idx = _sample_aux_indices(role_bundle, train_cfg, aux_rng)
                xb_aux = torch.from_numpy(np.array(role_bundle["X_train"][aux_idx], copy=False)).to(device=device, non_blocking=True)
                yb_aux = torch.from_numpy(np.array(role_bundle["oracle_y"][aux_idx], dtype=np.float32, copy=True)).to(device=device, non_blocking=True).view(-1)
                role_aux = torch.from_numpy(np.array(role_bundle["role_code"][aux_idx], dtype=np.int64, copy=True)).to(device=device, non_blocking=True).view(-1)

                xb_mix = torch.cat([xb_main, xb_aux], dim=0)
                accum_scale = 1.0 / max(int(train_cfg.grad_accum_steps), 1)
                with autocast(**amp_ctx_kwargs):
                    logits_mix = model.forward_logits(xb_mix).view(-1)
                    main_logits = logits_mix[: yb_main.numel()]
                    aux_logits = logits_mix[yb_main.numel() :]
                    main_terms = compute_l4_main_terms(main_logits, yb_main, l4_variant, train_cfg)
                    aux_terms = compute_ft1_aux_terms(aux_logits, yb_aux, role_aux, train_cfg)
                    aux_objective_scaled = float(aux_scale) * aux_terms["objective"]
                    total_objective = main_terms["objective"] + aux_objective_scaled
                    main_for_grad = main_terms["objective"] * accum_scale
                    aux_for_grad = aux_objective_scaled * accum_scale
                    total_for_grad = total_objective * accum_scale

                if not bool(torch.isfinite(total_objective)):
                    print(
                        f"[ft1][epoch={epoch}] non-finite objective at shard={shard_id}, "
                        f"micro_step={micro_step}; skipping micro-batch"
                    )
                    continue

                main_for_grad_scaled = main_for_grad * float(amp_grad_scale)
                aux_for_grad_scaled = aux_for_grad * float(amp_grad_scale)
                total_for_grad_scaled = total_for_grad * float(amp_grad_scale)

                main_backbone_grads = torch.autograd.grad(main_for_grad_scaled, backbone_params, retain_graph=True, allow_unused=True)
                aux_backbone_grads = torch.autograd.grad(aux_for_grad_scaled, backbone_params, retain_graph=True, allow_unused=True)
                should_step = ((micro_step + 1) % max(int(train_cfg.grad_accum_steps), 1)) == 0
                next_global_step = int(global_step) + (1 if should_step else 0)
                should_monitor = (
                    should_step
                    and torch.isfinite(total_objective)
                    and (next_global_step % int(train_cfg.grad_monitor_every_steps) == 0)
                )
                if should_monitor:
                    main_all_grads = torch.autograd.grad(main_for_grad_scaled, all_params, retain_graph=True, allow_unused=True)
                    aux_all_grads = torch.autograd.grad(aux_for_grad_scaled, all_params, retain_graph=True, allow_unused=True)
                total_head_grads = list(torch.autograd.grad(total_for_grad_scaled, head_params, retain_graph=False, allow_unused=True))

                main_backbone_grads = _unscale_grad_list(main_backbone_grads, amp_grad_scale)
                aux_backbone_grads = _unscale_grad_list(aux_backbone_grads, amp_grad_scale)
                total_head_grads = _unscale_grad_list(total_head_grads, amp_grad_scale)
                if should_monitor:
                    main_all_grads = _unscale_grad_list(main_all_grads, amp_grad_scale)
                    aux_all_grads = _unscale_grad_list(aux_all_grads, amp_grad_scale)

                shared_backbone_grads, pcgrad_report = project_backbone_conflicts(
                    main_backbone_grads,
                    aux_backbone_grads,
                    eps=float(train_cfg.pcgrad_eps),
                )
                if not bool(train_cfg.use_backbone_pcgrad):
                    shared_backbone_grads = [
                        _optional_add(mg.detach().clone() if mg is not None else None, ag.detach().clone() if ag is not None else None)
                        for mg, ag in zip(main_backbone_grads, aux_backbone_grads)
                    ]
                    pcgrad_report["grad_cosine_backbone_post"] = pcgrad_report["grad_cosine_backbone"]
                    pcgrad_report["grad_norm_shared_backbone"] = float(torch.norm(_flatten_grads(shared_backbone_grads))) if _flatten_grads(shared_backbone_grads).numel() > 0 else float("nan")
                    pcgrad_report["grad_conflict_backbone"] = 0.0
                    pcgrad_report["grad_projection_scale"] = 0.0

                if should_monitor:
                    grad_report = collect_gradient_monitor(
                        main_grads_backbone=main_backbone_grads,
                        aux_grads_backbone=aux_backbone_grads,
                        shared_grads_backbone=shared_backbone_grads,
                        main_grads_all=main_all_grads,
                        aux_grads_all=aux_all_grads,
                    )
                    grad_report.update(pcgrad_report)
                    grad_report["global_step"] = int(next_global_step)
                    grad_samples.append(grad_report)
                    step_rows.append(
                        {
                            "global_step": int(next_global_step),
                            "train_total_objective": float(total_objective.item()),
                            "train_main_objective": float(main_terms["objective"].item()),
                            "train_aux_objective": float(aux_terms["objective"].item()),
                            "aux_scale": float(aux_scale),
                            "grad_cosine_backbone_pre": float(grad_report.get("grad_cosine_backbone", float("nan"))),
                            **grad_report,
                        }
                    )

                for param, grad in zip(backbone_params, shared_backbone_grads):
                    if grad is None:
                        continue
                    grad_detached = grad.detach()
                    if param.grad is None:
                        param.grad = grad_detached.clone()
                    else:
                        param.grad.add_(grad_detached)

                for param, grad in zip(head_params, total_head_grads):
                    if grad is None:
                        continue
                    grad_detached = grad.detach()
                    if param.grad is None:
                        param.grad = grad_detached.clone()
                    else:
                        param.grad.add_(grad_detached)
                micro_step += 1

                bs = int(yb_main.numel())
                epoch_running["main_objective"] += float(main_terms["objective"].item()) * bs
                epoch_running["aux_objective"] += float(aux_terms["objective"].item()) * bs
                epoch_running["clean_loss"] += float(aux_terms["clean_loss"].item()) * bs
                epoch_running["ambiguous_loss"] += float(aux_terms["ambiguous_loss"].item()) * bs
                epoch_running["margin_penalty"] += float(aux_terms["margin_penalty"].item()) * bs
                epoch_running["mean_main_weight"] += float(main_terms["mean_main_weight"].item()) * bs
                epoch_running["downweighted_frac"] += float(main_terms["downweighted_frac"].item()) * bs
                epoch_running["n"] += bs

                if should_step:
                    if not _all_grads_finite(all_params):
                        optimizer.zero_grad(set_to_none=True)
                        print(
                            f"[ft1][epoch={epoch}] non-finite gradients before step={next_global_step}; "
                            "skipping optimizer step"
                        )
                        continue
                    if train_cfg.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.grad_clip_norm))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1
                    _maybe_save_periodic_latest_checkpoint(
                        epoch,
                        resume_shard_index=shard_index,
                        resume_next_start=int(start) + int(train_cfg.main_batch_size),
                        aux_rng_state=aux_rng.bit_generator.state,
                        rng_state=_capture_rng_state(),
                    )

                    if global_step % int(train_cfg.log_every_steps) == 0:
                        print(
                            f"[ft1][epoch={epoch}] step={global_step}/{total_optimizer_steps} "
                            f"main_obj={epoch_running['main_objective'] / max(epoch_running['n'], 1):.6f} "
                            f"aux_obj={epoch_running['aux_objective'] / max(epoch_running['n'], 1):.6f} "
                            f"aux_scale={aux_scale:.3f}"
                        )

            print(f"[ft1][epoch={epoch}] finished shard {shard_index + 1}/{len(train_shards)} (shard_id={shard_id})")

        if (micro_step % max(int(train_cfg.grad_accum_steps), 1)) != 0:
            if not _all_grads_finite(all_params):
                optimizer.zero_grad(set_to_none=True)
                print(
                    f"[ft1][epoch={epoch}] non-finite gradients at tail step={global_step + 1}; "
                    "skipping optimizer step"
                )
            else:
                if train_cfg.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.grad_clip_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                _maybe_save_periodic_latest_checkpoint(
                    epoch,
                    resume_shard_index=len(train_shards),
                    resume_next_start=0,
                    aux_rng_state=aux_rng.bit_generator.state,
                    rng_state=_capture_rng_state(),
                )

        model.eval()
        epoch_eval = evaluate_ft1_model(
            model=model,
            data_root=data_root,
            device=device,
            eval_batch_size=train_cfg.eval_batch_size,
            split="val",
            max_samples=train_cfg.val_max_samples,
            num_shards=train_cfg.val_num_shards,
            oracle_bundle=oracle_bundle,
            pooled_center_bundle=pooled_center_bundle,
            role_bundle=role_bundle,
        )
        primary = dict(epoch_eval["primary"])
        pooled_center_eval = dict(epoch_eval["pooled_center_eval"])
        role_metrics = epoch_eval["role_eval"]["metrics"]
        bn_sanity = collect_bn_sanity(model)
        if grad_samples:
            grad_frame = pd.DataFrame(grad_samples)
            grad_mean = {
                key: float(grad_frame[key].mean())
                for key in (
                    "grad_cosine_backbone",
                    "grad_cosine_backbone_post",
                    "grad_norm_main_backbone",
                    "grad_norm_aux_backbone",
                    "grad_norm_shared_backbone",
                    "grad_conflict_backbone",
                    "grad_projection_scale",
                    "grad_cosine_all",
                    "grad_norm_main_all",
                    "grad_norm_aux_all",
                )
                if key in grad_frame.columns
            }
        else:
            grad_mean = {
                "grad_cosine_backbone": float("nan"),
                "grad_cosine_backbone_post": float("nan"),
                "grad_norm_main_backbone": float("nan"),
                "grad_norm_aux_backbone": float("nan"),
                "grad_norm_shared_backbone": float("nan"),
                "grad_conflict_backbone": float("nan"),
                "grad_projection_scale": float("nan"),
                "grad_cosine_all": float("nan"),
                "grad_norm_main_all": float("nan"),
                "grad_norm_aux_all": float("nan"),
            }

        gate_pass = passes_midband_gate(primary, l4_reference["primary"], gate_cfg)
        center_score = float(primary["center_score"])
        decision_summary = {
            "epoch": int(epoch),
            "center_score": center_score,
            "midband_gate_pass": bool(gate_pass),
            "best_any_center_score": float(best_any_center_score),
            "best_gate_center_score": float(best_gate_center_score),
            "selected_checkpoint": "none",
        }

        selected: List[str] = []
        if center_score < best_any_center_score:
            best_any_center_score = center_score
            selected.append("best_any")
        if gate_pass and (center_score < best_gate_center_score):
            best_gate_center_score = center_score
            selected.append("best_gate")

        row = {
            "epoch": int(epoch),
            "train_main_objective": epoch_running["main_objective"] / max(epoch_running["n"], 1),
            "train_aux_objective": epoch_running["aux_objective"] / max(epoch_running["n"], 1),
            "train_clean_loss": epoch_running["clean_loss"] / max(epoch_running["n"], 1),
            "train_ambiguous_loss": epoch_running["ambiguous_loss"] / max(epoch_running["n"], 1),
            "train_margin_penalty": epoch_running["margin_penalty"] / max(epoch_running["n"], 1),
            "train_mean_main_weight": epoch_running["mean_main_weight"] / max(epoch_running["n"], 1),
            "train_downweighted_frac": epoch_running["downweighted_frac"] / max(epoch_running["n"], 1),
            "aux_scale": float(aux_scale),
            "global_step": int(global_step),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_time_sec": float(time.time() - t0),
            "midband_gate_pass": bool(gate_pass),
            **grad_mean,
            **bn_sanity,
            **primary,
            "pooled_center_wrong_sign_0.1eq": float(pooled_center_eval["wrong_sign_0.1"]),
            "pooled_center_wrong_sign_0.2eq": float(pooled_center_eval["wrong_sign_0.2"]),
            "pooled_center_spread_ratio": float(pooled_center_eval["spread_ratio"]),
            "clean_center_wrong_sign_0.1eq": float(role_metrics["clean_center"]["wrong_sign_0.1"]),
            "clean_center_wrong_sign_0.2eq": float(role_metrics["clean_center"]["wrong_sign_0.2"]),
            "clean_center_spread_ratio": float(role_metrics["clean_center"]["spread_ratio"]),
            "ambiguous_center_wrong_sign_0.1eq": float(role_metrics["center_ambiguous"]["wrong_sign_0.1"]),
            "ambiguous_center_wrong_sign_0.2eq": float(role_metrics["center_ambiguous"]["wrong_sign_0.2"]),
            "ambiguous_center_spread_ratio": float(role_metrics["center_ambiguous"]["spread_ratio"]),
            "selected_checkpoint": "|".join(selected) if selected else "none",
        }
        history_rows.append(row)
        save_history_outputs(history_rows, step_rows, paths["reports_dir"])

        decision_summary["selected_checkpoint"] = "|".join(selected) if selected else "none"
        decision_summary["best_any_center_score"] = float(best_any_center_score)
        decision_summary["best_gate_center_score"] = float(best_gate_center_score)

        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            history_rows=history_rows,
            epoch=epoch,
            global_step=global_step,
            best_any_center_score=best_any_center_score,
            best_gate_center_score=best_gate_center_score,
            run_config=run_config,
            decision_summary=decision_summary,
            aux_rng_state=aux_rng.bit_generator.state,
            rng_state=_capture_rng_state(),
        )
        ab_lab.atomic_torch_save(payload, latest_ckpt)
        if "best_any" in selected:
            ab_lab.atomic_torch_save(payload, best_any_ckpt)
        if "best_gate" in selected:
            ab_lab.atomic_torch_save(payload, best_gate_ckpt)

        decision_payload = {
            "best_any_center_score": float(best_any_center_score),
            "best_gate_center_score": float(best_gate_center_score),
            "has_best_gate": bool(best_gate_ckpt.exists()),
            "latest_checkpoint": str(latest_ckpt),
            "best_any_checkpoint": (str(best_any_ckpt) if best_any_ckpt.exists() else None),
            "best_gate_checkpoint": (str(best_gate_ckpt) if best_gate_ckpt.exists() else None),
            "l4_reference_checkpoint": str(paths["l4_reference_ckpt"]),
            "last_epoch": int(epoch),
            "last_center_score": center_score,
            "last_gate_pass": bool(gate_pass),
        }
        base_lab.save_json(decision_payload, decision_summary_path)
        decision_summary_out = dict(decision_payload)
        print(json.dumps(_format_history_row(row), ensure_ascii=False, indent=2))

    selected_checkpoint = best_gate_ckpt if best_gate_ckpt.exists() else (best_any_ckpt if best_any_ckpt.exists() else latest_ckpt)
    final_eval = evaluate_saved_checkpoint(
        checkpoint_path=selected_checkpoint,
        data_root=data_root,
        pooled_center_bundle_dir=paths["pooled_center_bundle_dir"],
        oracle_role_bundle_dir=paths["oracle_role_bundle_dir"],
        device=device,
        eval_batch_size=train_cfg.eval_batch_size,
        test_max_samples=train_cfg.test_max_samples,
        test_num_shards=train_cfg.test_num_shards,
    )
    base_lab.save_json(
        {
            "selected_checkpoint": str(selected_checkpoint),
            "primary": final_eval["primary"],
            "pooled_center_eval": final_eval["pooled_center_eval"],
            "role_metrics": final_eval["role_eval"]["metrics"],
        },
        paths["reports_dir"] / "selected_checkpoint_eval.json",
    )
    return {
        "paths": paths,
        "run_dir": paths["run_dir"],
        "decision_summary": decision_summary_out,
        "history": pd.DataFrame(history_rows),
        "selected_checkpoint": selected_checkpoint,
        "l4_reference": l4_reference,
        "final_eval": final_eval,
    }


def resolve_ft1_run_dir(runs_root: str | Path, run_name: str) -> Path:
    return Path(runs_root) / str(run_name)


def load_history_frame(run_dir: str | Path) -> pd.DataFrame:
    run_dir = Path(run_dir)
    history_path = run_dir / "reports" / "history.csv"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing FT1 history.csv: {history_path}")
    return pd.read_csv(history_path)


def load_step_history_frame(run_dir: str | Path) -> pd.DataFrame:
    run_dir = Path(run_dir)
    step_path = run_dir / "reports" / "step_history.csv"
    if not step_path.exists():
        raise FileNotFoundError(f"Missing FT1 step_history.csv: {step_path}")
    return pd.read_csv(step_path)


def load_decision_summary(run_dir: str | Path) -> Dict[str, object]:
    run_dir = Path(run_dir)
    path = run_dir / "reports" / "decision_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing FT1 decision summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_l4_reference(run_dir: str | Path) -> Dict[str, object]:
    run_dir = Path(run_dir)
    path = run_dir / "reports" / "l4_reference.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing FT1 l4_reference.json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_selected_checkpoint(run_dir: str | Path, prefer_gate: bool = True) -> Path:
    run_dir = Path(run_dir)
    checkpoints_dir = run_dir / "checkpoints"
    gate_ckpt = checkpoints_dir / "ckpt_best_gate.pt"
    any_ckpt = checkpoints_dir / "ckpt_best_any.pt"
    latest_ckpt = checkpoints_dir / "ckpt_latest.pt"
    if prefer_gate and gate_ckpt.exists():
        return gate_ckpt
    if any_ckpt.exists():
        return any_ckpt
    if latest_ckpt.exists():
        return latest_ckpt
    raise FileNotFoundError(f"No FT1 checkpoint found under {checkpoints_dir}")
