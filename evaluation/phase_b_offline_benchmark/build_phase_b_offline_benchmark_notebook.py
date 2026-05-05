from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = PROJECT_ROOT / "evaluation" / "phase_b_offline_benchmark" / "phase_b_offline_benchmark.ipynb"


def _lines(text: str) -> List[str]:
    return [line + "\n" for line in textwrap.dedent(text).strip("\n").splitlines()]


def _markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _lines(text),
    }


def _code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _lines(text),
    }


def build_notebook() -> dict:
    cells = [
        _markdown_cell(
            """
            # Experiment: Phase B Offline Benchmark

            Mục tiêu:
            - Benchmark checkpoint champion của `Pha B` trên **toàn bộ test set** với cùng logic metric đã dùng trong pipeline `FT1/Report1`.
            - Chạy được cả local và Colab.
            - Nếu là Colab thì chỉ stage `data/process/test` về disk local của runtime để giảm nghẽn I/O.

            Outputs:
            - `evaluation/phase_b_offline_benchmark/outputs/<benchmark_name>/reports`
            - `evaluation/phase_b_offline_benchmark/outputs/<benchmark_name>/plots`

            Chính sách metric:
            - `core`: metric full-test đủ mạnh để quyết định offline quality
            - `secondary`: metric oracle subset để continuity với các suite cũ
            - `diagnostic`: metric hữu ích để đọc failure mode nhưng không nên dùng làm gate chính
            """
        ),
        _code_cell(
            """
            import os
            from pathlib import Path

            try:
                from google.colab import drive  # type: ignore
                IN_COLAB = True
            except Exception:
                drive = None
                IN_COLAB = False

            if IN_COLAB:
                drive.mount('/content/drive', force_remount=False)
            """
        ),
        _code_cell(
            """
            import json
            import os
            import shutil
            import sys
            from pathlib import Path

            import matplotlib.pyplot as plt
            import pandas as pd
            import torch
            from IPython.display import Image, display


            def env_bool(name: str, default: bool) -> bool:
                raw = os.environ.get(name, '').strip().lower()
                if not raw:
                    return bool(default)
                return raw in {'1', 'true', 'yes', 'y'}


            def env_int(name: str, default: int) -> int:
                raw = os.environ.get(name, '').strip()
                return int(raw) if raw else int(default)


            def env_optional_int(name: str):
                raw = os.environ.get(name, '').strip()
                return int(raw) if raw else None


            def env_optional_float(name: str):
                raw = os.environ.get(name, '').strip()
                return float(raw) if raw else None


            def find_repo_root(start: Path) -> Path:
                markers = (
                    ('model',),
                    ('train_v2_TF1', 'ft1_colab_helpers.py'),
                    ('train_v5_phaseB', 'phase_b_train_helpers.py'),
                    ('evaluation', 'phase_b_offline_benchmark', 'offline_benchmark_helpers.py'),
                )
                for candidate in [start.resolve(), *start.resolve().parents]:
                    if all(candidate.joinpath(*parts).exists() for parts in markers):
                        return candidate
                raise RuntimeError(f'Cannot resolve repo root from {start}')


            def resolve_colab_repo_root() -> Path:
                env_root = os.environ.get('CHESS_REPO_ROOT', '').strip()
                if env_root:
                    candidate = Path(env_root)
                    if candidate.exists():
                        return candidate.resolve()
                candidates = [Path('/content/drive/MyDrive/chess_engine')]
                shortcut_root = Path('/content/drive/.shortcut-targets-by-id')
                if shortcut_root.exists():
                    candidates.extend(shortcut_root.glob('*/chess_engine'))
                for candidate in candidates:
                    if candidate.exists():
                        return candidate.resolve()
                raise FileNotFoundError('Cannot locate repo root in Colab. Set CHESS_REPO_ROOT explicitly.')


            RUN_NAME = os.environ.get('CHESS_BENCHMARK_RUN_NAME', 'dgrn_5m_phaseb_b2_sglobal_16b_256d_sign_run1').strip()
            CHECKPOINT_POLICY = os.environ.get('CHESS_BENCHMARK_CHECKPOINT_POLICY', 'best_gate').strip().lower()
            BENCHMARK_NAME = os.environ.get('CHESS_BENCHMARK_NAME', '').strip()
            CANDIDATE_CHECKPOINT_OVERRIDE = os.environ.get('CHESS_BENCHMARK_CHECKPOINT', '').strip()
            REFERENCE_CHECKPOINT_OVERRIDE = os.environ.get('CHESS_REFERENCE_CHECKPOINT', '').strip()
            STAGE_TEST_LOCAL = env_bool('CHESS_STAGE_TEST_LOCAL', IN_COLAB)
            EVAL_BATCH_SIZE_OVERRIDE = env_optional_int('CHESS_BENCHMARK_EVAL_BATCH_SIZE')
            TEST_MAX_SAMPLES_OVERRIDE = env_optional_int('CHESS_BENCHMARK_MAX_SAMPLES')
            TEST_NUM_SHARDS_OVERRIDE = env_optional_int('CHESS_BENCHMARK_NUM_SHARDS')
            RUN_BOOTSTRAP = env_bool('CHESS_BENCHMARK_BOOTSTRAP', True)
            BOOTSTRAP_N = env_int('CHESS_BENCHMARK_BOOTSTRAP_N', 2000)
            CENTER_STRONG_THRESHOLD = env_optional_float('CHESS_CENTER_STRONG_THRESHOLD')
            ABS_CALIBRATION_BINS = env_int('CHESS_BENCHMARK_ABS_CAL_BINS', 20)

            if IN_COLAB:
                REPO_ROOT = resolve_colab_repo_root()
            else:
                REPO_ROOT = find_repo_root(Path.cwd())

            RUNS_ROOT = REPO_ROOT / 'runs'
            RUN_DIR = RUNS_ROOT / RUN_NAME
            OUTPUTS_ROOT = REPO_ROOT / 'evaluation' / 'phase_b_offline_benchmark' / 'outputs'

            DATA_ROOT_OVERRIDE = os.environ.get('CHESS_DATA_ROOT', '').strip()
            DATA_ROOT_SOURCE = Path(DATA_ROOT_OVERRIDE).resolve() if DATA_ROOT_OVERRIDE else (REPO_ROOT / 'data' / 'process')

            if IN_COLAB and STAGE_TEST_LOCAL:
                stage_root = Path('/content/chess_engine_data/process')
                src_test = DATA_ROOT_SOURCE / 'test'
                dst_test = stage_root / 'test'
                if not src_test.exists():
                    raise FileNotFoundError(f'Missing source test split: {src_test}')
                if dst_test.exists():
                    shutil.rmtree(dst_test)
                dst_test.mkdir(parents=True, exist_ok=True)
                for item in sorted(src_test.iterdir()):
                    if item.is_file():
                        shutil.copy2(item, dst_test / item.name)
                DATA_ROOT_ACTIVE = stage_root
            else:
                DATA_ROOT_ACTIVE = DATA_ROOT_SOURCE

            if not BENCHMARK_NAME:
                suffix = CANDIDATE_CHECKPOINT_OVERRIDE and 'explicit' or CHECKPOINT_POLICY
                BENCHMARK_NAME = f'offline_{RUN_NAME}_{suffix}'

            print('IN_COLAB =', IN_COLAB)
            print('REPO_ROOT =', REPO_ROOT)
            print('RUN_DIR =', RUN_DIR)
            print('DATA_ROOT_ACTIVE =', DATA_ROOT_ACTIVE)
            print('CHECKPOINT_POLICY =', CHECKPOINT_POLICY)
            print('BENCHMARK_NAME =', BENCHMARK_NAME)
            print('TEST_MAX_SAMPLES_OVERRIDE =', TEST_MAX_SAMPLES_OVERRIDE)
            print('TEST_NUM_SHARDS_OVERRIDE =', TEST_NUM_SHARDS_OVERRIDE)
            print('CUDA_AVAILABLE =', torch.cuda.is_available())
            if torch.cuda.is_available():
                print('GPU =', torch.cuda.get_device_name(0))
            """
        ),
        _code_cell(
            """
            import importlib.util


            def import_benchmark_helper(repo_root: Path):
                helper_path = repo_root / 'evaluation' / 'phase_b_offline_benchmark' / 'offline_benchmark_helpers.py'
                spec = importlib.util.spec_from_file_location('phase_b_offline_benchmark_helpers_runtime', helper_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f'Cannot import helper from {helper_path}')
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                return module


            lab = import_benchmark_helper(REPO_ROOT)
            """
        ),
        _code_cell(
            """
            if CANDIDATE_CHECKPOINT_OVERRIDE:
                cfg = lab.OfflineBenchmarkConfig(
                    benchmark_name=BENCHMARK_NAME,
                    candidate_checkpoint=CANDIDATE_CHECKPOINT_OVERRIDE,
                    reference_checkpoint=REFERENCE_CHECKPOINT_OVERRIDE or None,
                    data_root=DATA_ROOT_ACTIVE,
                    outputs_root=OUTPUTS_ROOT,
                    eval_batch_size=EVAL_BATCH_SIZE_OVERRIDE,
                    test_max_samples=TEST_MAX_SAMPLES_OVERRIDE,
                    test_num_shards=TEST_NUM_SHARDS_OVERRIDE,
                    center_strong_threshold=CENTER_STRONG_THRESHOLD,
                    run_bootstrap_compare=RUN_BOOTSTRAP,
                    bootstrap_n=BOOTSTRAP_N,
                    abs_calibration_bins=ABS_CALIBRATION_BINS,
                )
            else:
                cfg = lab.resolve_config_from_run(
                    run_dir=RUN_DIR,
                    checkpoint_policy=CHECKPOINT_POLICY,
                    benchmark_name=BENCHMARK_NAME,
                    data_root=DATA_ROOT_ACTIVE,
                    reference_checkpoint=REFERENCE_CHECKPOINT_OVERRIDE or None,
                    outputs_root=OUTPUTS_ROOT,
                    eval_batch_size=EVAL_BATCH_SIZE_OVERRIDE,
                    test_max_samples=TEST_MAX_SAMPLES_OVERRIDE,
                    test_num_shards=TEST_NUM_SHARDS_OVERRIDE,
                    center_strong_threshold=CENTER_STRONG_THRESHOLD,
                    run_bootstrap_compare=RUN_BOOTSTRAP,
                    bootstrap_n=BOOTSTRAP_N,
                    abs_calibration_bins=ABS_CALIBRATION_BINS,
                )

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            bench_artifacts = lab.run_offline_benchmark(cfg, device=device)
            bench_paths = {k: Path(v) for k, v in bench_artifacts['paths'].items()}
            print(json.dumps(bench_artifacts['decision_summary'], indent=2, ensure_ascii=False))
            print('Reports dir:', bench_paths['reports_dir'])
            print('Plots dir:', bench_paths['plots_dir'])
            """
        ),
        _code_cell(
            """
            reports_dir = bench_paths['reports_dir']
            plots_dir = bench_paths['plots_dir']

            core_df = pd.read_csv(reports_dir / 'core_metrics_table.csv')
            secondary_df = pd.read_csv(reports_dir / 'secondary_metrics_table.csv')
            diagnostic_df = pd.read_csv(reports_dir / 'diagnostic_metrics_table.csv')
            reliability_df = pd.read_csv(reports_dir / 'metric_reliability_catalog.csv')
            calibration_df = pd.read_csv(reports_dir / 'absolute_calibration_curve.csv')
            band_df = pd.read_csv(reports_dir / 'band_diagnostics.csv')
            decision = json.loads((reports_dir / 'decision_summary.json').read_text(encoding='utf-8'))
            runtime_check = json.loads((reports_dir / 'runtime_check.json').read_text(encoding='utf-8'))
            benchmark_config = json.loads((reports_dir / 'benchmark_config.json').read_text(encoding='utf-8'))
            sample_sizes = json.loads((reports_dir / 'sample_sizes.json').read_text(encoding='utf-8'))
            candidate_eval = json.loads((reports_dir / 'candidate_eval_summary.json').read_text(encoding='utf-8'))
            reference_eval_path = reports_dir / 'reference_eval_summary.json'
            reference_eval = json.loads(reference_eval_path.read_text(encoding='utf-8')) if reference_eval_path.exists() else None
            bootstrap_path = reports_dir / 'oracle_bootstrap_compare.csv'
            bootstrap_df = pd.read_csv(bootstrap_path) if bootstrap_path.exists() else None

            print('Decision summary:')
            print(json.dumps(decision, indent=2, ensure_ascii=False))
            print('Runtime check:')
            print(json.dumps(runtime_check, indent=2, ensure_ascii=False))
            print('Resolved benchmark config:')
            print(json.dumps(benchmark_config, indent=2, ensure_ascii=False))
            print('Sample sizes:')
            print(json.dumps(sample_sizes, indent=2, ensure_ascii=False))

            display(core_df)
            display(secondary_df)
            display(diagnostic_df)
            display(reliability_df)
            display(calibration_df.head(12))
            display(band_df)
            if bootstrap_df is not None:
                display(bootstrap_df)

            for plot_name in [
                'core_metrics.png',
                'prediction_vs_target_hexbin.png',
                'absolute_calibration.png',
                'center_behavior.png',
                'sign_match_by_band.png',
            ]:
                plot_path = plots_dir / plot_name
                if plot_path.exists():
                    display(Image(filename=str(plot_path)))
            """
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
