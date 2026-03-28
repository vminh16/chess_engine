from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from uuid import uuid4


NOTEBOOK_PATH = Path(__file__).resolve().parent / "oracle_root_cause_diagnostic.ipynb"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid4().hex[:8],
        "metadata": {},
        "source": dedent(source).strip("\n").splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip("\n").splitlines(keepends=True),
    }


cells = [
    md(
        """
        # Oracle Root-Cause Diagnostic

        Notebook này không cố sửa model ngay.

        Nó dùng `Stockfish fixed-node oracle` trên một subset chẩn đoán để trả lời trực tiếp:

        - train label hiện tại lệch khỏi oracle mạnh đến mức nào
        - teacher có đang sai thật, hay đang “sai so với label nhưng gần oracle hơn”
        - volatility có liên quan tới error của teacher hay không
        - trên stable subset, lỗi calibration còn lại lớn đến đâu
        - scale `tanh(cp / c)` nào khớp teacher hơn trên oracle cố định
        """
    ),
    code(
        r"""
        from dataclasses import asdict
        from pathlib import Path
        import json
        import sys

        import pandas as pd
        import torch
        from IPython.display import display

        PROJECT_ROOT = Path(r"C:\Users\USER\Desktop\chess_engine")
        EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "oracle_root_cause_diagnostic"
        RUN_DIR = Path(r"C:\Users\USER\Downloads\dgrn_5m_v3_stage2_polish_run1")
        DATA_ROOT = PROJECT_ROOT / "data" / "process"

        if str(EXPERIMENT_DIR) not in sys.path:
            sys.path.insert(0, str(EXPERIMENT_DIR))

        import oracle_diagnostic_helpers as lab

        torch.set_float32_matmul_precision("high")
        lab.set_global_seed(123)
        paths = lab.build_default_paths(run_dir=RUN_DIR, data_root=DATA_ROOT, experiment_dir=EXPERIMENT_DIR)
        lab.export_paths_json(paths, paths["output_dir"] / "paths.json")

        if "envs\\chess_engine" not in sys.executable.lower():
            raise RuntimeError(
                f"Notebook is running under the wrong interpreter: {sys.executable}. "
                "Select the 'chess_engine' Jupyter kernel."
            )

        DEVICE = lab.choose_device(prefer_cuda=True)
        if DEVICE.type != "cuda":
            raise RuntimeError("CUDA is required for this notebook.")

        CHECKPOINT = paths["run_dir"] / "ckpt_best.pt"
        assert CHECKPOINT.exists(), f"Missing checkpoint: {CHECKPOINT}"

        CFG = lab.OracleDiagnosticConfig(
            split="test",
            sample_abs_y_edges=(0.0, 0.05, 0.20, 0.50, 0.70, 1.00),
            sample_per_band=48,
            err_quantiles=(1.0 / 3.0, 2.0 / 3.0),
            oracle_scales=(400.0, 600.0, 800.0, 1200.0),
            stockfish_path=r"D:\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe",
            stockfish_threads=1,
            stockfish_hash_mb=32,
            stockfish_node_budgets=(8_000, 32_000, 128_000),
            stockfish_command_pause_ms=50,
            stockfish_timeout_sec=20.0,
            prediction_batch_size=2048,
            sample_seed=123,
            benchmark_train_batch_size=640,
            decode_validation_samples=64,
            subset_num_shards=None,
        )
        cfg_validation = lab.validate_diagnostic_config(CFG)
        lab.save_json(asdict(CFG), paths["output_dir"] / "runtime_config.json")
        lab.save_json(cfg_validation, paths["reports_dir"] / "config_validation.json")

        print("python:", sys.executable)
        print("device:", DEVICE)
        print("gpu_name:", torch.cuda.get_device_name(0))
        display(pd.DataFrame({"path_key": list(paths.keys()), "path_value": [str(v) for v in paths.values()]}))
        display(pd.DataFrame([cfg_validation]))
        """
    ),
    md(
        """
        ## Runtime Self-Check

        Cell này kiểm tra:
        - GPU forward benchmark
        - Stockfish determinism
        - decode -> FEN compatibility
        """
    ),
    code(
        r"""
        benchmark = lab.benchmark_single_train_step(
            init_ckpt_path=CHECKPOINT,
            data_root=paths["data_root"],
            device=DEVICE,
            cfg=CFG,
        )
        lab.save_json(benchmark, paths["reports_dir"] / "runtime_benchmark.json")

        sf_cfg = lab.build_stockfish_cfg(CFG)
        stockfish_benchmark = lab.sw_lab.benchmark_stockfish_proxy(sf_cfg)
        stockfish_validation = lab.sw_lab.validate_stockfish_proxy(sf_cfg)
        stockfish_decode = lab.sw_lab.validate_stockfish_compatible_decoding(
            data_root=paths["data_root"],
            split=CFG.split,
            sample_count=CFG.decode_validation_samples,
            num_shards=1,
        )
        stockfish_dataset_validation = lab.sw_lab.validate_stockfish_proxy_on_dataset_sample(
            data_root=paths["data_root"],
            split=CFG.split,
            cfg=sf_cfg,
            sample_index=0,
            num_shards=1,
        )

        lab.save_json(stockfish_benchmark, paths["reports_dir"] / "stockfish_proxy_benchmark.json")
        lab.save_json(stockfish_validation, paths["reports_dir"] / "stockfish_proxy_validation.json")
        lab.save_json(stockfish_decode, paths["reports_dir"] / "stockfish_decode_validation.json")
        lab.save_json(stockfish_dataset_validation, paths["reports_dir"] / "stockfish_proxy_dataset_validation.json")

        split_summary = pd.DataFrame(
            [
                lab.sw_lab.summarize_split_layout(paths["data_root"], "train"),
                lab.sw_lab.summarize_split_layout(paths["data_root"], "val"),
                lab.sw_lab.summarize_split_layout(paths["data_root"], "test"),
            ]
        )
        lab.save_dataframe(split_summary, paths["reports_dir"] / "split_summary.csv")
        assert stockfish_validation["bestmove_match"], stockfish_validation
        assert stockfish_validation["target_match"], stockfish_validation
        if stockfish_decode["checked_with_python_chess"]:
            assert stockfish_decode["invalid_fens"] == 0, stockfish_decode
        assert stockfish_dataset_validation["bestmove_match"], stockfish_dataset_validation
        assert stockfish_dataset_validation["target_match"], stockfish_dataset_validation
        display(pd.DataFrame([benchmark]))
        display(pd.DataFrame(stockfish_benchmark["per_query"]))
        display(pd.DataFrame([stockfish_validation]))
        display(pd.DataFrame([stockfish_dataset_validation]))
        display(pd.DataFrame([stockfish_decode]))
        display(split_summary)
        """
    ),
    md(
        """
        ## Build Diagnostic Subset

        Trình tự:
        1. chạy teacher hiện tại trên toàn bộ split chẩn đoán
        2. stratify theo `|y|` và `teacher_abs_err`
        3. lấy subset cân bằng để hỏi Stockfish oracle
        """
    ),
    code(
        r"""
        pred_cache = lab.precompute_split_prediction_cache(
            init_ckpt_path=CHECKPOINT,
            data_root=paths["data_root"],
            split=CFG.split,
            output_dir=paths["output_dir"],
            device=DEVICE,
            batch_size=CFG.prediction_batch_size,
            num_shards=CFG.subset_num_shards,
        )

        subset = lab.build_stratified_subset(
            data_root=paths["data_root"],
            pred_cache_dir=paths["split_pred_cache_dir"],
            split=CFG.split,
            cfg=CFG,
            num_shards=CFG.subset_num_shards,
        )
        lab.save_dataframe(subset["count_table"], paths["reports_dir"] / "subset_candidate_count_table.csv")
        lab.save_dataframe(subset["quota_table"], paths["reports_dir"] / "subset_quota_table.csv")
        lab.save_dataframe(subset["sampled_summary"], paths["reports_dir"] / "subset_sampled_summary.csv")
        display(subset["sampled_summary"])
        display(subset["quota_table"].head(15))
        """
    ),
    md(
        """
        ## Run Oracle And Analyze

        Cell này là phần quan trọng nhất.

        Quy ước:
        - `oracle reference` = kết quả ở node budget lớn nhất
        - các node budget nhỏ hơn chỉ dùng để đo stability / disagreement curve

        Nó tạo ra:
        - `oracle_subset_rows.csv`
        - `oracle_band_summary.csv`
        - `oracle_budget_alignment.csv`
        - `oracle_stability_summary.csv`
        - `oracle_scale_sweep.csv`
        - `oracle_root_cause_summary.json`
        """
    ),
    code(
        r"""
        oracle_rows = lab.run_stockfish_oracle_on_subset(
            subset=subset["samples"],
            cfg=CFG,
            output_dir=paths["output_dir"],
        )
        analysis = lab.run_diagnostic_analysis(
            df=oracle_rows,
            cfg=CFG,
            output_dir=paths["output_dir"],
        )

        lab.save_dataframe(oracle_rows.head(120), paths["reports_dir"] / "oracle_subset_rows_preview.csv")
        display(pd.DataFrame([analysis["summary"]]))
        display(analysis["band_summary"])
        display(analysis["budget_summary"])
        display(analysis["stability_summary"])
        display(analysis["scale_sweep"])
        display(analysis["stable_bucket"].head(12))
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "chess_engine",
            "language": "python",
            "name": "chess_engine",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"wrote {NOTEBOOK_PATH}")
