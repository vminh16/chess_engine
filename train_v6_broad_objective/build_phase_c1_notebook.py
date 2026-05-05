from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "train_phase_c1_broad_objective.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


NOTEBOOK = {
    "cells": [
        md(
            """# Phase C1 Broad Objective Train

Train from scratch: 16/256 + SimplifiedGlobalHead + random sampling + broad objective.

Before a long run, use:

```python
%env CHESS_EPOCHS_OVERRIDE=1
%env CHESS_RUN_SUFFIX=smoke
```
"""
        ),
        code(
            """import os
from pathlib import Path

try:
    from google.colab import drive  # type: ignore
    IN_COLAB = True
except Exception:
    drive = None
    IN_COLAB = False

if IN_COLAB:
    drive.mount('/content/drive', force_remount=False)
    print('Mounted Google Drive.')
else:
    print('Running outside Colab.')
"""
        ),
        code(
            """import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, '').strip().lower()
    if raw == '':
        return bool(default)
    return raw in {'1', 'true', 'yes', 'y', 'on'}


def candidate_repo_roots() -> list[Path]:
    candidates: list[Path] = []
    env_root = os.environ.get('CHESS_REPO_ROOT', '').strip()
    if env_root:
        candidates.append(Path(env_root))
    if IN_COLAB:
        candidates.extend([
            Path('/content/drive/MyDrive/chess_engine'),
            Path('/content/chess_engine'),
        ])
    candidates.append(Path.cwd())
    candidates.append(Path.cwd().parent)
    return candidates


def resolve_repo_root() -> Path:
    for candidate in candidate_repo_roots():
        helper_path = candidate / 'train_v6_broad_objective' / 'phase_c1_train_helpers.py'
        if helper_path.exists():
            return candidate.resolve()
    raise RuntimeError('Cannot resolve repository root. Set CHESS_REPO_ROOT explicitly.')


REPO_ROOT = resolve_repo_root()
RUNS_ROOT = Path(os.environ.get('CHESS_RUNS_ROOT', '').strip()) if os.environ.get('CHESS_RUNS_ROOT', '').strip() else (REPO_ROOT / 'runs')
HELPER_PATH = REPO_ROOT / 'train_v6_broad_objective' / 'phase_c1_train_helpers.py'

DATA_ROOT_OVERRIDE = os.environ.get('CHESS_DATA_ROOT', '').strip()
DATA_ROOT = Path(DATA_ROOT_OVERRIDE) if DATA_ROOT_OVERRIDE else (REPO_ROOT / 'data' / 'process')
STAGE_DATA_LOCAL = IN_COLAB and env_bool('CHESS_STAGE_DATA_LOCAL', True)
FORCE_RESTAGE = IN_COLAB and env_bool('CHESS_FORCE_RESTAGE', False)

if STAGE_DATA_LOCAL:
    DATA_ROOT_ACTIVE = Path('/content/chess_engine_data/process')
    if FORCE_RESTAGE and DATA_ROOT_ACTIVE.exists():
        shutil.rmtree(DATA_ROOT_ACTIVE)
    if not DATA_ROOT_ACTIVE.exists():
        DATA_ROOT_ACTIVE.parent.mkdir(parents=True, exist_ok=True)
        print(f'Staging data from {DATA_ROOT} -> {DATA_ROOT_ACTIVE}')
        shutil.copytree(DATA_ROOT, DATA_ROOT_ACTIVE)
    else:
        print(f'Reusing staged data at {DATA_ROOT_ACTIVE}')
else:
    DATA_ROOT_ACTIVE = DATA_ROOT

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE == 'cuda' else None
GPU_CAPABILITY = torch.cuda.get_device_capability(0) if DEVICE == 'cuda' else None
TF32_SUPPORTED = bool(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8)
if TF32_SUPPORTED:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

random.seed(123)

print('REPO_ROOT =', REPO_ROOT)
print('RUNS_ROOT =', RUNS_ROOT)
print('DATA_ROOT_ACTIVE =', DATA_ROOT_ACTIVE)
print('DEVICE =', DEVICE)
print('GPU_NAME =', GPU_NAME)
print('GPU_CAPABILITY =', GPU_CAPABILITY)
print('TF32_SUPPORTED =', TF32_SUPPORTED)
"""
        ),
        code(
            """def import_phase_c1_helper(helper_path: Path):
    spec = importlib.util.spec_from_file_location('train_v6_phase_c1_train_helpers', helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot import helper from {helper_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


lab = import_phase_c1_helper(HELPER_PATH)
experiment = lab.build_phase_c1_from_env()

RUN_DIR = RUNS_ROOT / experiment.train_cfg.run_name
print(json.dumps({
    'experiment_id': experiment.experiment_id,
    'title': experiment.title,
    'summary': experiment.summary,
    'run_dir': str(RUN_DIR),
    'model_cfg': experiment.model_cfg,
    'train_cfg': experiment.train_cfg.__dict__,
    'gate_cfg': experiment.gate_cfg.__dict__,
    'notes': experiment.notes,
}, indent=2, ensure_ascii=False, default=str))
"""
        ),
        code(
            """existing_config_path = RUN_DIR / 'reports' / 'run_config.json'
if existing_config_path.exists():
    existing = json.loads(existing_config_path.read_text(encoding='utf-8'))
    print('Existing run_config found. Resume will reuse saved stable profile/config where required.')
    print(json.dumps({
        'run_name': existing.get('train_cfg', {}).get('run_name'),
        'phase_name': existing.get('train_cfg', {}).get('phase_name'),
        'sampling_mode': existing.get('train_cfg', {}).get('sampling_mode'),
        'clean_center_batch_size': existing.get('train_cfg', {}).get('clean_center_batch_size'),
        'ambiguous_center_batch_size': existing.get('train_cfg', {}).get('ambiguous_center_batch_size'),
        'selection_policy': existing.get('selection_policy'),
    }, indent=2, ensure_ascii=False))
else:
    print('No existing run_config. New C1 run will be created from scratch.')
"""
        ),
        code(
            """run_artifacts = lab.run_phase_c1_training(
    repo_root=REPO_ROOT,
    runs_root=RUNS_ROOT,
    data_root=DATA_ROOT_ACTIVE,
    run_suffix=os.environ.get('CHESS_RUN_SUFFIX', ''),
    epochs_override=int(os.environ['CHESS_EPOCHS_OVERRIDE']) if os.environ.get('CHESS_EPOCHS_OVERRIDE', '').strip() else None,
    autotune_profile=True,
)
print('Selected checkpoint:', run_artifacts['selected_checkpoint'])
print(json.dumps(run_artifacts['final_eval'].get('primary', {}), indent=2, ensure_ascii=False))
"""
        ),
        code(
            """history_path = RUN_DIR / 'reports' / 'history.csv'
decision_path = RUN_DIR / 'reports' / 'decision_summary.json'

if history_path.exists():
    history = pd.read_csv(history_path)
    display_cols = [
        'epoch',
        'broad_score',
        'broad_gate_pass',
        'overall_mse',
        'test_mse_0.1eq',
        'test_center_false_0.1eq',
        'test_max_midband_abs_cal_gap',
        'train_total_objective',
        'train_center_margin_penalty',
        'train_abs_calibration_penalty',
        'selected_checkpoint',
    ]
    display_cols = [col for col in display_cols if col in history.columns]
    display(history[display_cols].tail(20))

if decision_path.exists():
    decision = json.loads(decision_path.read_text(encoding='utf-8'))
    print(json.dumps(decision, indent=2, ensure_ascii=False))
"""
        ),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    OUT.write_text(json.dumps(NOTEBOOK, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
