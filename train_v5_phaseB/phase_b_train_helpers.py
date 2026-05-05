from __future__ import annotations

import contextlib
import copy
import importlib.util
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


def _find_repo_root(start: Path) -> Path:
    start = start.resolve()
    markers = (
        ("model", "architecture_v2", "model.py"),
        ("train_v4_report1", "report1_train_helpers.py"),
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

report1 = _import_module_from_file(
    "train_v4_report1_report1_train_helpers",
    PROJECT_ROOT / "train_v4_report1" / "report1_train_helpers.py",
)


@dataclass(frozen=True)
class PhaseBExperiment:
    experiment_id: str
    title: str
    summary: str
    model_cfg: Dict[str, object]
    train_cfg: object
    source_experiment: Optional[str] = None
    source_sampling_mode: Optional[str] = None
    notes: Tuple[str, ...] = ()


def _normalize_experiment_id(value: object) -> str:
    experiment_id = str(value).strip().upper()
    if experiment_id not in {"B1", "B2", "B3", "B4"}:
        raise ValueError("experiment_id must be one of: B1, B2, B3, B4")
    return experiment_id


def _normalize_sampling_mode(value: object) -> str:
    mode = str(value).strip().lower()
    if mode not in {"band_balanced", "sign_stratified"}:
        raise ValueError("sampling_mode must be one of: band_balanced, sign_stratified")
    return mode


def _normalize_source_experiment(value: object, *, allow_b3: bool = True) -> str:
    source = str(value).strip().upper()
    allowed = {"B1", "B2", "B3"} if allow_b3 else {"B1", "B2"}
    if source not in allowed:
        raise ValueError(f"source_experiment must be one of: {', '.join(sorted(allowed))}")
    return source


def _expected_sampling_from_source(source_experiment: str) -> Optional[str]:
    source = str(source_experiment).strip().upper()
    if source == "B1":
        return "band_balanced"
    if source == "B2":
        return "sign_stratified"
    if source == "B3":
        return None
    raise ValueError(f"Unsupported source_experiment: {source_experiment}")


def _head_short_name(head_type: str) -> str:
    mapping = {
        "residual_gain": "resgain",
        "simplified_global": "sglobal",
        "regime_separated": "rsep",
    }
    return mapping.get(str(head_type).strip().lower(), str(head_type).strip().lower())


def _make_model_cfg(
    *,
    num_blocks: int,
    hidden_dim: int,
    head_type: str,
    drop_path_rate: float = 0.05,
    head_hidden_dim: Optional[int] = None,
    head_dropout: float = 0.10,
) -> Dict[str, object]:
    return {
        "num_blocks": int(num_blocks),
        "hidden_dim": int(hidden_dim),
        "input_channels": 18,
        "drop_path_rate": float(drop_path_rate),
        "head_hidden_dim": int(hidden_dim // 2 if head_hidden_dim is None else head_hidden_dim),
        "head_type": str(head_type).strip().lower(),
        "head_dropout": float(head_dropout),
        "output_mode": "tanh",
    }


def _make_train_cfg(
    *,
    run_name: str,
    phase_name: str,
    epochs: int,
    sampling_mode: str,
    generic_main_batch_size: int,
    generic_clean_center_batch_size: int,
    generic_ambiguous_center_batch_size: int,
    generic_grad_accum_steps: int,
    generic_eval_batch_size: int,
    generic_preload_shard_dtype: str,
) -> object:
    cfg = report1.Report1TrainConfig()
    cfg.run_name = str(run_name)
    cfg.phase_name = str(phase_name)
    cfg.epochs = int(epochs)
    cfg.sampling_mode = _normalize_sampling_mode(sampling_mode)
    cfg.main_batch_size = int(generic_main_batch_size)
    cfg.clean_center_batch_size = int(generic_clean_center_batch_size)
    cfg.ambiguous_center_batch_size = int(generic_ambiguous_center_batch_size)
    cfg.grad_accum_steps = int(generic_grad_accum_steps)
    cfg.eval_batch_size = int(generic_eval_batch_size)
    cfg.preload_shard_dtype = str(generic_preload_shard_dtype)
    cfg.channels_last = True
    cfg.cudnn_benchmark = True
    cfg.pin_memory_batches = False
    cfg.prefetch_shards = True
    cfg.prefetch_workers = 1
    cfg.benchmark_steps = 12
    cfg.benchmark_warmup_steps = 3
    cfg.benchmark_num_shards = 1
    cfg.max_profile_mem_ratio = 0.82
    cfg.periodic_save_minutes = 30
    cfg.save_epoch_checkpoints = False
    cfg.resume_if_exists = True
    cfg.learning_rate = 1.0e-4
    cfg.min_lr = 1.0e-5
    cfg.weight_decay = 1.0e-4
    cfg.grad_clip_norm = 1.0
    cfg.use_amp = True
    cfg.amp_dtype = "float16"
    cfg.amp_loss_scale = 128.0
    cfg.val_num_shards = 2
    cfg.test_num_shards = 4
    cfg.val_max_samples = 100_000
    cfg.test_max_samples = 200_000
    cfg.log_every_steps = 200
    cfg.grad_monitor_every_steps = 1000
    cfg.main_center_tau_y600 = 0.10
    cfg.main_center_min_weight = 0.35
    cfg.main_center_weight_power = 1.0
    cfg.lambda_clean_center = 0.20
    cfg.lambda_ambiguous_center = 0.10
    cfg.aux_margin_y600 = 0.08
    cfg.aux_margin_weight = 0.40
    cfg.aux_huber_delta = 0.05
    cfg.aux_ramp_epochs = 4
    cfg.use_backbone_pcgrad = True
    cfg.pcgrad_eps = 1.0e-12
    return cfg


def _default_run_name(experiment_id: str, model_cfg: Dict[str, object], sampling_mode: str, source_experiment: Optional[str]) -> str:
    blocks = int(model_cfg["num_blocks"])
    hidden = int(model_cfg["hidden_dim"])
    head_short = _head_short_name(str(model_cfg["head_type"]))
    sample_short = "sign" if sampling_mode == "sign_stratified" else "band"
    run_name = f"dgrn_5m_phaseb_{experiment_id.lower()}_{head_short}_{blocks}b_{hidden}d_{sample_short}_run1"
    if source_experiment:
        run_name = f"{run_name}_{str(source_experiment).strip().lower()}"
    return run_name


def _parameter_count(model_cfg: Dict[str, object]) -> int:
    model = DGRNChessNetV2(**model_cfg)
    return int(sum(param.numel() for param in model.parameters()))


def _phase_b_generic_defaults(model_cfg: Dict[str, object]) -> Dict[str, object]:
    blocks = int(model_cfg["num_blocks"])
    hidden = int(model_cfg["hidden_dim"])
    if blocks >= 20 and hidden >= 256:
        return {
            "main_batch_size": 384,
            "clean_center_batch_size": 48,
            "ambiguous_center_batch_size": 96,
            "grad_accum_steps": 1,
            "eval_batch_size": 2048,
            "preload_shard_dtype": "auto",
        }
    if blocks >= 16 and hidden >= 256:
        return {
            "main_batch_size": 512,
            "clean_center_batch_size": 64,
            "ambiguous_center_batch_size": 128,
            "grad_accum_steps": 1,
            "eval_batch_size": 3072,
            "preload_shard_dtype": "auto",
        }
    return {
        "main_batch_size": 384,
        "clean_center_batch_size": 48,
        "ambiguous_center_batch_size": 96,
        "grad_accum_steps": 1,
        "eval_batch_size": 2048,
        "preload_shard_dtype": "auto",
    }


def build_phase_b_experiment(
    experiment_id: str,
    *,
    source_experiment: Optional[str] = None,
    source_sampling_mode: Optional[str] = None,
    run_suffix: str = "",
    epochs_override: Optional[int] = None,
) -> PhaseBExperiment:
    experiment_id = _normalize_experiment_id(experiment_id)
    source_sampling = None if source_sampling_mode is None or str(source_sampling_mode).strip() == "" else _normalize_sampling_mode(source_sampling_mode)

    if experiment_id == "B1":
        if source_experiment is not None or source_sampling is not None:
            raise ValueError("B1 does not accept source_experiment/source_sampling_mode; clear stale Phase B source env vars.")
        model_cfg = _make_model_cfg(num_blocks=16, hidden_dim=256, head_type="simplified_global")
        defaults = _phase_b_generic_defaults(model_cfg)
        sampling_mode = "band_balanced"
        run_name = _default_run_name("B1", model_cfg, sampling_mode, None)
        train_cfg = _make_train_cfg(
            run_name=run_name,
            phase_name="B1",
            epochs=12 if epochs_override is None else int(epochs_override),
            sampling_mode=sampling_mode,
            generic_main_batch_size=int(defaults["main_batch_size"]),
            generic_clean_center_batch_size=int(defaults["clean_center_batch_size"]),
            generic_ambiguous_center_batch_size=int(defaults["ambiguous_center_batch_size"]),
            generic_grad_accum_steps=int(defaults["grad_accum_steps"]),
            generic_eval_batch_size=int(defaults["eval_batch_size"]),
            generic_preload_shard_dtype=str(defaults["preload_shard_dtype"]),
        )
        notes = (
            "Head-first baseline cho Pha B.",
            "Giữ sampling sạch để đo tác động thuần của SimplifiedGlobalHead.",
        )
    elif experiment_id == "B2":
        if source_experiment is not None or source_sampling is not None:
            raise ValueError("B2 does not accept source_experiment/source_sampling_mode; clear stale Phase B source env vars.")
        model_cfg = _make_model_cfg(num_blocks=16, hidden_dim=256, head_type="simplified_global")
        defaults = _phase_b_generic_defaults(model_cfg)
        sampling_mode = "sign_stratified"
        run_name = _default_run_name("B2", model_cfg, sampling_mode, None)
        train_cfg = _make_train_cfg(
            run_name=run_name,
            phase_name="B2",
            epochs=12 if epochs_override is None else int(epochs_override),
            sampling_mode=sampling_mode,
            generic_main_batch_size=int(defaults["main_batch_size"]),
            generic_clean_center_batch_size=int(defaults["clean_center_batch_size"]),
            generic_ambiguous_center_batch_size=int(defaults["ambiguous_center_batch_size"]),
            generic_grad_accum_steps=int(defaults["grad_accum_steps"]),
            generic_eval_batch_size=int(defaults["eval_batch_size"]),
            generic_preload_shard_dtype=str(defaults["preload_shard_dtype"]),
        )
        notes = (
            "Mang ingredient tốt nhất của R2 sang head sạch hơn.",
            "Dùng để kiểm tra liệu lợi ích B có còn giữ được khi bỏ current head.",
        )
    elif experiment_id == "B3":
        if source_experiment is not None:
            source_experiment = _normalize_source_experiment(source_experiment, allow_b3=False)
            expected_sampling = _expected_sampling_from_source(source_experiment)
            if source_sampling is not None and expected_sampling is not None and source_sampling != expected_sampling:
                raise ValueError(
                    "B3 source_sampling_mode conflicts with source_experiment. "
                    f"{source_experiment} implies sampling_mode={expected_sampling}."
                )
        if source_sampling is None:
            if source_experiment is None:
                raise ValueError("B3 requires source_sampling_mode or source_experiment from {B1, B2}.")
            source_sampling = _expected_sampling_from_source(source_experiment)
        model_cfg = _make_model_cfg(num_blocks=16, hidden_dim=256, head_type="regime_separated")
        defaults = _phase_b_generic_defaults(model_cfg)
        sampling_mode = source_sampling
        run_name = _default_run_name("B3", model_cfg, sampling_mode, source_experiment or source_sampling.replace("_", ""))
        train_cfg = _make_train_cfg(
            run_name=run_name,
            phase_name="B3",
            epochs=12 if epochs_override is None else int(epochs_override),
            sampling_mode=sampling_mode,
            generic_main_batch_size=int(defaults["main_batch_size"]),
            generic_clean_center_batch_size=int(defaults["clean_center_batch_size"]),
            generic_ambiguous_center_batch_size=int(defaults["ambiguous_center_batch_size"]),
            generic_grad_accum_steps=int(defaults["grad_accum_steps"]),
            generic_eval_batch_size=int(defaults["eval_batch_size"]),
            generic_preload_shard_dtype=str(defaults["preload_shard_dtype"]),
        )
        notes = (
            "Tách sign/magnitude để đo đúng giả thuyết semantic conflict ở head.",
            f"Sampling hiện tại: {sampling_mode}.",
        )
    else:
        if source_experiment is None:
            raise ValueError("B4 requires source_experiment from {B1, B2, B3}.")
        source_experiment = _normalize_source_experiment(source_experiment, allow_b3=True)
        expected_sampling = _expected_sampling_from_source(source_experiment)
        if expected_sampling is not None and source_sampling is not None and source_sampling != expected_sampling:
            raise ValueError(
                "B4 source_sampling_mode conflicts with source_experiment. "
                f"{source_experiment} implies sampling_mode={expected_sampling}."
            )
        if source_experiment == "B1":
            head_type = "simplified_global"
            sampling_mode = "band_balanced"
        elif source_experiment == "B2":
            head_type = "simplified_global"
            sampling_mode = "sign_stratified"
        else:
            head_type = "regime_separated"
            if source_sampling is None:
                raise ValueError("B4 with source_experiment=B3 requires source_sampling_mode.")
            sampling_mode = source_sampling
        model_cfg = _make_model_cfg(num_blocks=20, hidden_dim=256, head_type=head_type)
        defaults = _phase_b_generic_defaults(model_cfg)
        run_name = _default_run_name("B4", model_cfg, sampling_mode, source_experiment)
        train_cfg = _make_train_cfg(
            run_name=run_name,
            phase_name="B4",
            epochs=10 if epochs_override is None else int(epochs_override),
            sampling_mode=sampling_mode,
            generic_main_batch_size=int(defaults["main_batch_size"]),
            generic_clean_center_batch_size=int(defaults["clean_center_batch_size"]),
            generic_ambiguous_center_batch_size=int(defaults["ambiguous_center_batch_size"]),
            generic_grad_accum_steps=int(defaults["grad_accum_steps"]),
            generic_eval_batch_size=int(defaults["eval_batch_size"]),
            generic_preload_shard_dtype=str(defaults["preload_shard_dtype"]),
        )
        notes = (
            f"Run xác nhận trên torso lớn hơn, kế thừa recipe thắng từ {source_experiment}.",
            "B4 không phải run khám phá; chỉ dùng để xác nhận scale-up.",
        )

    if run_suffix:
        suffix = str(run_suffix).strip().replace(" ", "_")
        if suffix:
            train_cfg.run_name = f"{train_cfg.run_name}_{suffix}"

    parameter_count = _parameter_count(model_cfg)
    title_map = {
        "B1": "Phase B1 - 16/256 + SimplifiedGlobalHead + baseline objective",
        "B2": "Phase B2 - 16/256 + SimplifiedGlobalHead + sign-stratified",
        "B3": "Phase B3 - 16/256 + RegimeSeparatedHead + best sampling",
        "B4": "Phase B4 - 20/256 + winning head recipe",
    }
    summary_map = {
        "B1": "Head-first baseline sạch nhất của Pha B.",
        "B2": "Kiểm tra liệu lợi ích B từ sign-stratified có còn giữ được khi head sạch hơn.",
        "B3": "Kiểm tra semantic decoupling của sign/magnitude trong value readout.",
        "B4": "Xác nhận recipe thắng trên torso 20/256 lớn hơn.",
    }
    notes = tuple(notes + (f"parameter_count={parameter_count:,}",))
    return PhaseBExperiment(
        experiment_id=experiment_id,
        title=title_map[experiment_id],
        summary=summary_map[experiment_id],
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        source_experiment=source_experiment,
        source_sampling_mode=source_sampling,
        notes=notes,
    )


def _phase_b_t4_profiles(model_cfg: Dict[str, object]) -> List[Dict[str, object]]:
    blocks = int(model_cfg["num_blocks"])
    hidden = int(model_cfg["hidden_dim"])
    if blocks >= 20 and hidden >= 256:
        return [
            {"main_batch_size": 768, "clean_center_batch_size": 96, "ambiguous_center_batch_size": 192, "grad_accum_steps": 1, "eval_batch_size": 4096, "preload_shard_dtype": "none", "prefetch_workers": 1, "benchmark_num_shards": 2},
            {"main_batch_size": 640, "clean_center_batch_size": 80, "ambiguous_center_batch_size": 160, "grad_accum_steps": 1, "eval_batch_size": 4096, "preload_shard_dtype": "none", "prefetch_workers": 1, "benchmark_num_shards": 2},
            {"main_batch_size": 512, "clean_center_batch_size": 64, "ambiguous_center_batch_size": 128, "grad_accum_steps": 1, "eval_batch_size": 3072, "preload_shard_dtype": "none", "prefetch_workers": 1, "benchmark_num_shards": 2},
            {"main_batch_size": 448, "clean_center_batch_size": 56, "ambiguous_center_batch_size": 112, "grad_accum_steps": 1, "eval_batch_size": 3072, "preload_shard_dtype": "none", "prefetch_workers": 1, "benchmark_num_shards": 2},
            {"main_batch_size": 384, "clean_center_batch_size": 48, "ambiguous_center_batch_size": 96, "grad_accum_steps": 1, "eval_batch_size": 2048, "preload_shard_dtype": "none", "prefetch_workers": 1, "benchmark_num_shards": 2},
        ]
    if blocks >= 16 and hidden >= 256:
        return [
            {"main_batch_size": 1024, "clean_center_batch_size": 128, "ambiguous_center_batch_size": 256, "grad_accum_steps": 1, "eval_batch_size": 4096, "preload_shard_dtype": "none", "prefetch_workers": 1, "benchmark_num_shards": 2},
            {"main_batch_size": 896, "clean_center_batch_size": 112, "ambiguous_center_batch_size": 224, "grad_accum_steps": 1, "eval_batch_size": 4096, "preload_shard_dtype": "none", "prefetch_workers": 1, "benchmark_num_shards": 2},
            {"main_batch_size": 768, "clean_center_batch_size": 96, "ambiguous_center_batch_size": 192, "grad_accum_steps": 1, "eval_batch_size": 4096, "preload_shard_dtype": "none", "prefetch_workers": 1, "benchmark_num_shards": 2},
            {"main_batch_size": 640, "clean_center_batch_size": 80, "ambiguous_center_batch_size": 160, "grad_accum_steps": 1, "eval_batch_size": 3072, "preload_shard_dtype": "none", "prefetch_workers": 1, "benchmark_num_shards": 2},
            {"main_batch_size": 512, "clean_center_batch_size": 64, "ambiguous_center_batch_size": 128, "grad_accum_steps": 1, "eval_batch_size": 3072, "preload_shard_dtype": "none", "prefetch_workers": 1, "benchmark_num_shards": 2},
        ]
    return report1.candidate_device_profiles("T4", 14.5)


def _phase_b_small_gpu_profiles(model_cfg: Dict[str, object]) -> List[Dict[str, object]]:
    blocks = int(model_cfg["num_blocks"])
    hidden = int(model_cfg["hidden_dim"])
    if blocks >= 20 and hidden >= 256:
        return [
            {"main_batch_size": 192, "clean_center_batch_size": 24, "ambiguous_center_batch_size": 48, "grad_accum_steps": 2, "eval_batch_size": 768, "preload_shard_dtype": "auto", "prefetch_workers": 1, "benchmark_num_shards": 1},
            {"main_batch_size": 160, "clean_center_batch_size": 20, "ambiguous_center_batch_size": 40, "grad_accum_steps": 2, "eval_batch_size": 768, "preload_shard_dtype": "auto", "prefetch_workers": 1, "benchmark_num_shards": 1},
            {"main_batch_size": 128, "clean_center_batch_size": 16, "ambiguous_center_batch_size": 32, "grad_accum_steps": 2, "eval_batch_size": 768, "preload_shard_dtype": "auto", "prefetch_workers": 1, "benchmark_num_shards": 1},
            {"main_batch_size": 96, "clean_center_batch_size": 16, "ambiguous_center_batch_size": 32, "grad_accum_steps": 2, "eval_batch_size": 512, "preload_shard_dtype": "auto", "prefetch_workers": 1, "benchmark_num_shards": 1},
        ]
    if blocks >= 16 and hidden >= 256:
        return [
            {"main_batch_size": 256, "clean_center_batch_size": 32, "ambiguous_center_batch_size": 64, "grad_accum_steps": 2, "eval_batch_size": 1024, "preload_shard_dtype": "auto", "prefetch_workers": 1, "benchmark_num_shards": 1},
            {"main_batch_size": 224, "clean_center_batch_size": 28, "ambiguous_center_batch_size": 56, "grad_accum_steps": 2, "eval_batch_size": 1024, "preload_shard_dtype": "auto", "prefetch_workers": 1, "benchmark_num_shards": 1},
            {"main_batch_size": 192, "clean_center_batch_size": 24, "ambiguous_center_batch_size": 48, "grad_accum_steps": 2, "eval_batch_size": 1024, "preload_shard_dtype": "auto", "prefetch_workers": 1, "benchmark_num_shards": 1},
            {"main_batch_size": 160, "clean_center_batch_size": 20, "ambiguous_center_batch_size": 40, "grad_accum_steps": 2, "eval_batch_size": 768, "preload_shard_dtype": "auto", "prefetch_workers": 1, "benchmark_num_shards": 1},
            {"main_batch_size": 128, "clean_center_batch_size": 16, "ambiguous_center_batch_size": 32, "grad_accum_steps": 2, "eval_batch_size": 768, "preload_shard_dtype": "auto", "prefetch_workers": 1, "benchmark_num_shards": 1},
        ]
    return report1.candidate_device_profiles("", 4.0)


def phase_b_candidate_device_profiles(model_cfg: Dict[str, object], gpu_name: Optional[str], total_mem_gb: float) -> List[Dict[str, object]]:
    gpu_upper = (gpu_name or "").upper()
    if "T4" in gpu_upper and float(total_mem_gb) >= 14.0:
        return _phase_b_t4_profiles(model_cfg)
    if float(total_mem_gb) <= 4.5:
        return _phase_b_small_gpu_profiles(model_cfg)
    return _phase_b_small_gpu_profiles(model_cfg)


def phase_b_default_device_profile(model_cfg: Dict[str, object], gpu_name: Optional[str], total_mem_gb: float) -> Dict[str, object]:
    return copy.deepcopy(phase_b_candidate_device_profiles(model_cfg, gpu_name, total_mem_gb)[0])


@contextlib.contextmanager
def _patch_report1_profiles(model_cfg: Dict[str, object]):
    original_default = report1.default_device_profile
    original_candidates = report1.candidate_device_profiles
    report1.default_device_profile = lambda gpu_name, total_mem_gb: phase_b_default_device_profile(model_cfg, gpu_name, total_mem_gb)
    report1.candidate_device_profiles = lambda gpu_name, total_mem_gb: phase_b_candidate_device_profiles(model_cfg, gpu_name, total_mem_gb)
    try:
        yield
    finally:
        report1.default_device_profile = original_default
        report1.candidate_device_profiles = original_candidates


def describe_phase_b_experiment(experiment: PhaseBExperiment) -> Dict[str, object]:
    return {
        "experiment_id": experiment.experiment_id,
        "title": experiment.title,
        "summary": experiment.summary,
        "model_cfg": dict(experiment.model_cfg),
        "train_cfg": asdict(experiment.train_cfg),
        "source_experiment": experiment.source_experiment,
        "source_sampling_mode": experiment.source_sampling_mode,
        "notes": list(experiment.notes),
        "parameter_count": _parameter_count(experiment.model_cfg),
    }


def run_phase_b_training(
    *,
    repo_root: str | Path,
    runs_root: str | Path,
    data_root: str | Path,
    experiment_id: str,
    source_experiment: Optional[str] = None,
    source_sampling_mode: Optional[str] = None,
    autotune_profile: bool = True,
    run_suffix: str = "",
    epochs_override: Optional[int] = None,
) -> Dict[str, object]:
    experiment = build_phase_b_experiment(
        experiment_id,
        source_experiment=source_experiment,
        source_sampling_mode=source_sampling_mode,
        run_suffix=run_suffix,
        epochs_override=epochs_override,
    )
    with _patch_report1_profiles(experiment.model_cfg):
        artifacts = report1.run_report1_training(
            repo_root=repo_root,
            runs_root=runs_root,
            data_root=data_root,
            model_cfg=experiment.model_cfg,
            train_cfg=experiment.train_cfg,
            gate_cfg=report1.Report1GateConfig(),
            autotune_profile=bool(autotune_profile),
        )
    artifacts["experiment"] = describe_phase_b_experiment(experiment)
    return artifacts
