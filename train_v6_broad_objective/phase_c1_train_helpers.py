from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


def _find_repo_root(start: Path) -> Path:
    start = start.resolve()
    markers = (
        ("model", "architecture_v2", "model.py"),
        ("train_v6_broad_objective", "broad_objective_train_helpers.py"),
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from architecture_v2.model import DGRNChessNetV2  # noqa: E402

broad_train = _import_module_from_file(
    "train_v6_broad_objective_broad_objective_train_helpers",
    PROJECT_ROOT / "train_v6_broad_objective" / "broad_objective_train_helpers.py",
)


@dataclass(frozen=True)
class PhaseC1Experiment:
    experiment_id: str
    title: str
    summary: str
    model_cfg: Dict[str, object]
    train_cfg: object
    gate_cfg: object
    notes: Tuple[str, ...] = ()


def _make_model_cfg(
    *,
    num_blocks: int = 16,
    hidden_dim: int = 256,
    head_type: str = "simplified_global",
    drop_path_rate: float = 0.05,
    head_dropout: float = 0.10,
) -> Dict[str, object]:
    return {
        "num_blocks": int(num_blocks),
        "hidden_dim": int(hidden_dim),
        "input_channels": 18,
        "drop_path_rate": float(drop_path_rate),
        "head_hidden_dim": int(hidden_dim // 2),
        "head_type": str(head_type).strip().lower(),
        "head_dropout": float(head_dropout),
        "output_mode": "tanh",
    }


def _parameter_count(model_cfg: Dict[str, object]) -> int:
    model = DGRNChessNetV2(**model_cfg)
    return int(sum(param.numel() for param in model.parameters()))


def _suffix_run_name(run_name: str, run_suffix: str) -> str:
    suffix = str(run_suffix or "").strip()
    if not suffix:
        return run_name
    suffix = suffix if suffix.startswith("_") else f"_{suffix}"
    return f"{run_name}{suffix}"


def build_phase_c1_experiment(
    *,
    run_suffix: str = "",
    epochs_override: Optional[int] = None,
) -> PhaseC1Experiment:
    model_cfg = _make_model_cfg()
    train_cfg = broad_train.Report1TrainConfig()
    train_cfg.run_name = _suffix_run_name(
        "dgrn_5m_phasec1_broad_objective_sglobal_16b_256d_random_run1",
        run_suffix,
    )
    train_cfg.phase_name = "C1_BROAD_OBJECTIVE"
    train_cfg.epochs = 12 if epochs_override is None else int(epochs_override)
    train_cfg.sampling_mode = "random"
    train_cfg.clean_center_batch_size = 0
    train_cfg.ambiguous_center_batch_size = 0
    train_cfg.lambda_clean_center = 0.0
    train_cfg.lambda_ambiguous_center = 0.0
    train_cfg.aux_margin_weight = 0.0
    train_cfg.aux_ramp_epochs = 1
    train_cfg.use_backbone_pcgrad = False
    train_cfg.val_num_shards = 4
    train_cfg.val_max_samples = 200_000
    train_cfg.test_num_shards = 4
    train_cfg.test_max_samples = 200_000
    train_cfg.periodic_save_minutes = 20
    train_cfg.save_epoch_checkpoints = False
    train_cfg.resume_if_exists = True
    gate_cfg = broad_train.Report1GateConfig(
        broad_overall_mse_rel_tol=0.02,
        broad_mse_0p1_rel_tol=0.02,
        broad_center_false_0p1_abs_tol=0.01,
        broad_abs_cal_rel_tol=0.05,
    )
    return PhaseC1Experiment(
        experiment_id="C1",
        title="Phase C1 Broad Objective Refresh",
        summary=(
            "Train from scratch with natural/random sampling, no narrow-oracle gradient steering, "
            "and broad validation checkpoint selection."
        ),
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        gate_cfg=gate_cfg,
        notes=(
            f"param_count={_parameter_count(model_cfg)}",
            "best_gate means broad-validation gate in this phase, not the Phase B oracle gate.",
            "The 78-row oracle role bundle is loaded only for compatibility with existing evaluation helpers.",
        ),
    )


def build_phase_c1_from_env() -> PhaseC1Experiment:
    run_suffix = os.environ.get("CHESS_RUN_SUFFIX", "")
    raw_epochs = os.environ.get("CHESS_EPOCHS_OVERRIDE", "").strip()
    epochs_override = int(raw_epochs) if raw_epochs else None
    return build_phase_c1_experiment(run_suffix=run_suffix, epochs_override=epochs_override)


def run_phase_c1_training(
    *,
    repo_root: str | Path,
    runs_root: str | Path,
    data_root: str | Path,
    run_suffix: str = "",
    epochs_override: Optional[int] = None,
    autotune_profile: bool = True,
) -> Dict[str, object]:
    experiment = build_phase_c1_experiment(run_suffix=run_suffix, epochs_override=epochs_override)
    result = broad_train.run_report1_training(
        repo_root=repo_root,
        runs_root=runs_root,
        data_root=data_root,
        model_cfg=experiment.model_cfg,
        train_cfg=experiment.train_cfg,
        gate_cfg=experiment.gate_cfg,
        autotune_profile=autotune_profile,
    )
    result["phase_c1_experiment"] = {
        "experiment_id": experiment.experiment_id,
        "title": experiment.title,
        "summary": experiment.summary,
        "model_cfg": dict(experiment.model_cfg),
        "train_cfg": asdict(experiment.train_cfg),
        "gate_cfg": asdict(experiment.gate_cfg),
        "notes": list(experiment.notes),
    }
    return result
