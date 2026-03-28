from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from uuid import uuid4


NOTEBOOK_PATH = Path(__file__).resolve().parent / "stability_weighted_near_zero_finetune.ipynb"


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
        # Stability-Weighted Near-Zero Fine-Tune

        Notebook này kiểm tra giả thuyết:

        - near-zero label volatility là một root cause thật sự
        - dùng trọng số ổn định trên vùng `|y| <= 0.2` có thể giảm failure mode nguy hiểm
        - chưa cần full production retrain; chỉ cần fine-tune ngắn từ `ckpt_best.pt`

        Tất cả artifact được lưu trong:
        `C:\\Users\\USER\\Desktop\\chess_engine\\experiments\\stability_weighted_near_zero_finetune\\outputs`
        """
    ),
    code(
        r"""
        from dataclasses import asdict
        from pathlib import Path
        import json
        import sys

        import numpy as np
        import pandas as pd
        import torch
        from IPython.display import display

        PROJECT_ROOT = Path(r"C:\Users\USER\Desktop\chess_engine")
        EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "stability_weighted_near_zero_finetune"
        RUN_DIR = Path(r"C:\Users\USER\Downloads\dgrn_5m_v3_stage2_polish_run1")
        DATA_ROOT = PROJECT_ROOT / "data" / "process"

        if str(EXPERIMENT_DIR) not in sys.path:
            sys.path.insert(0, str(EXPERIMENT_DIR))

        import stability_weighted_helpers as lab

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

        CHECKPOINTS = {
            "best": paths["run_dir"] / "ckpt_best.pt",
            "latest": paths["run_dir"] / "ckpt_latest.pt",
        }
        for name, ckpt_path in CHECKPOINTS.items():
            assert ckpt_path.exists(), f"Missing checkpoint: {name} -> {ckpt_path}"

        WEIGHT_CFG = lab.StabilityWeightConfig(
            near_zero_thr=0.20,
            calibration_abs_y_sample_edges=(0.0, 0.05, 0.10, 0.15, 0.20),
            sample_per_abs_y_band=24,
            stockfish_path=r"D:\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe",
            stockfish_threads=1,
            stockfish_hash_mb=32,
            stockfish_node_budgets=(2_000, 8_000, 32_000),
            stockfish_command_pause_ms=50,
            stockfish_timeout_sec=15.0,
            calibration_seed=123,
            prediction_batch_size=2560,
            weight_abs_y_edges=(0.0, 0.025, 0.05, 0.10, 0.15, 0.20),
            teacher_abs_err_quantiles=(0.2, 0.4, 0.6, 0.8),
            smoothing_prior=3.0,
            weight_strength=0.45,
            weight_min=0.55,
        )

        TRAIN_CFG = lab.FineTuneConfig(
            lambda_y=0.99,
            z_loss_beta=1.0,
            z_huber_delta=0.5,
            target_clamp_eps=1e-3,
            learning_rate=3e-6,
            min_lr=1e-6,
            weight_decay=2e-4,
            grad_clip_norm=1.0,
            batch_size=640,
            epochs=1,
            log_every_steps=200,
            seed=123,
            eval_val_samples=100_000,
            eval_test_samples=200_000,
            eval_val_num_shards=2,
            eval_test_num_shards=4,
            train_num_shards=None,
        )

        NOTEBOOK_CONFIG = {
            "weight_cfg": asdict(WEIGHT_CFG),
            "train_cfg": asdict(TRAIN_CFG),
            "prediction_cache_split": "train",
        }
        assert Path(WEIGHT_CFG.stockfish_path).exists(), f"Missing Stockfish binary: {WEIGHT_CFG.stockfish_path}"
        weight_cfg_validation = lab.validate_stability_weight_config(WEIGHT_CFG)
        train_cfg_validation = lab.validate_finetune_config(TRAIN_CFG)
        lab.save_json(NOTEBOOK_CONFIG, paths["output_dir"] / "runtime_config.json")
        lab.save_json(weight_cfg_validation, paths["reports_dir"] / "weight_cfg_validation.json")
        lab.save_json(train_cfg_validation, paths["reports_dir"] / "train_cfg_validation.json")
        print("python:", sys.executable)
        print("device:", DEVICE)
        print("gpu_name:", torch.cuda.get_device_name(0))
        display(pd.DataFrame({"path_key": list(paths.keys()), "path_value": [str(v) for v in paths.values()]}))
        display(pd.DataFrame([weight_cfg_validation]))
        display(pd.DataFrame([train_cfg_validation]))
        """
    ),
    md(
        """
        ## Runtime Self-Check

        Cell này xác minh:
        - batch size có thực tế trên GPU hiện tại hay không
        - rough runtime cho 1 epoch full dataset
        - decode -> FEN có hợp lệ cho Stockfish hay không
        - Stockfish proxy có deterministic ở mức self-check hay không
        - checkpoint config và dataset layout có khớp expectation hay không
        """
    ),
    code(
        r"""
        benchmark = lab.benchmark_single_train_step(
            init_ckpt_path=CHECKPOINTS["best"],
            data_root=paths["data_root"],
            device=DEVICE,
            batch_size=TRAIN_CFG.batch_size,
            num_shards=TRAIN_CFG.train_num_shards,
        )
        lab.save_json(benchmark, paths["reports_dir"] / "runtime_benchmark.json")
        stockfish_benchmark = lab.benchmark_stockfish_proxy(WEIGHT_CFG)
        lab.save_json(stockfish_benchmark, paths["reports_dir"] / "stockfish_proxy_benchmark.json")
        roundtrip = lab.base_lab.validate_encode_decode_roundtrip(
            data_root=paths["data_root"],
            split="train",
            sample_count=64,
        )
        lab.save_json(roundtrip, paths["reports_dir"] / "encode_decode_roundtrip.json")
        stockfish_decode = lab.validate_stockfish_compatible_decoding(
            data_root=paths["data_root"],
            split="train",
            sample_count=64,
            num_shards=1,
        )
        lab.save_json(stockfish_decode, paths["reports_dir"] / "stockfish_decode_validation.json")
        stockfish_validation = lab.validate_stockfish_proxy(WEIGHT_CFG)
        lab.save_json(stockfish_validation, paths["reports_dir"] / "stockfish_proxy_validation.json")
        stockfish_dataset_validation = lab.validate_stockfish_proxy_on_dataset_sample(
            data_root=paths["data_root"],
            split="train",
            cfg=WEIGHT_CFG,
            sample_index=0,
            num_shards=1,
        )
        lab.save_json(stockfish_dataset_validation, paths["reports_dir"] / "stockfish_proxy_dataset_validation.json")

        checkpoint_rows = []
        for label, ckpt_path in CHECKPOINTS.items():
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            checkpoint_rows.append(
                {
                    "label": label,
                    "epoch": payload.get("epoch"),
                    "loss_mode": payload.get("loss_mode"),
                    "output_mode": payload.get("output_mode"),
                    "y_loss_weight_start": payload.get("y_loss_weight_start"),
                    "y_loss_weight_end": payload.get("y_loss_weight_end"),
                    "y_loss_ramp_epochs": payload.get("y_loss_ramp_epochs"),
                    "z_loss_beta": payload.get("z_loss_beta"),
                    "z_huber_delta": payload.get("z_huber_delta"),
                    "lr": payload.get("lr"),
                }
            )
        checkpoint_df = pd.DataFrame(checkpoint_rows)
        lab.save_dataframe(checkpoint_df, paths["reports_dir"] / "checkpoint_config_table.csv")

        split_summary = pd.DataFrame(
            [
                lab.summarize_split_layout(paths["data_root"], "train", num_shards=TRAIN_CFG.train_num_shards),
                lab.summarize_split_layout(paths["data_root"], "val"),
                lab.summarize_split_layout(paths["data_root"], "test"),
            ]
        )
        lab.save_dataframe(split_summary, paths["reports_dir"] / "split_summary.csv")
        assert roundtrip["mismatches"] == 0, roundtrip
        if stockfish_decode["checked_with_python_chess"]:
            assert stockfish_decode["invalid_fens"] == 0, stockfish_decode
        assert stockfish_validation["bestmove_match"], stockfish_validation
        assert stockfish_validation["target_match"], stockfish_validation
        assert stockfish_dataset_validation["bestmove_match"], stockfish_dataset_validation
        assert stockfish_dataset_validation["target_match"], stockfish_dataset_validation
        display(pd.DataFrame([benchmark]))
        display(pd.DataFrame(stockfish_benchmark["per_query"]))
        display(pd.DataFrame([{"one_position_total_sec": stockfish_benchmark["one_position_total_sec"], "estimated_positions": stockfish_benchmark["estimated_positions"], "estimated_total_min": stockfish_benchmark["estimated_total_min"]}]))
        display(pd.DataFrame([roundtrip]))
        display(pd.DataFrame([stockfish_decode]))
        display(pd.DataFrame([stockfish_validation]))
        display(pd.DataFrame([stockfish_dataset_validation]))
        display(checkpoint_df)
        display(split_summary)
        """
    ),
    md(
        """
        ## Baseline Teacher Eval

        Đo teacher hiện tại trên đúng metric suite sẽ dùng để gate fine-tune.
        """
    ),
    code(
        r"""
        baseline_model, _ = lab.base_lab.load_model_from_checkpoint(CHECKPOINTS["best"], device=DEVICE)
        baseline_val = lab.evaluate_model_on_split(
            model=baseline_model,
            data_root=paths["data_root"],
            split="val",
            device=DEVICE,
            max_samples=TRAIN_CFG.eval_val_samples,
            num_shards=TRAIN_CFG.eval_val_num_shards,
            batch_size=max(TRAIN_CFG.batch_size, 1024),
        )
        baseline_test = lab.evaluate_model_on_split(
            model=baseline_model,
            data_root=paths["data_root"],
            split="test",
            device=DEVICE,
            max_samples=TRAIN_CFG.eval_test_samples,
            num_shards=TRAIN_CFG.eval_test_num_shards,
            batch_size=max(TRAIN_CFG.batch_size, 1024),
        )
        del baseline_model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        lab.save_json(baseline_val["metrics"], paths["reports_dir"] / "baseline_val_metrics.json")
        lab.save_json(baseline_test["metrics"], paths["reports_dir"] / "baseline_test_metrics.json")
        baseline_compare = pd.DataFrame(
            [
                lab.compare_metric_rows("baseline_val", baseline_val["metrics"]),
                lab.compare_metric_rows("baseline_test", baseline_test["metrics"]),
            ]
        )
        lab.save_dataframe(baseline_compare, paths["reports_dir"] / "baseline_compare.csv")
        display(baseline_compare)
        """
    ),
    md(
        """
        ## Build Stability Weights

        Trình tự:
        1. cache full-train teacher predictions
        2. lấy calibration subset near-zero cân bằng theo `|y|`
        3. chạy Stockfish subset oracle theo nhiều budget `nodes`
        4. fit lookup table `weight(abs_y, teacher_abs_err)`
        """
    ),
    code(
        r"""
        pred_cache = lab.precompute_train_prediction_cache(
            init_ckpt_path=CHECKPOINTS["best"],
            data_root=paths["data_root"],
            output_dir=paths["output_dir"],
            device=DEVICE,
            batch_size=WEIGHT_CFG.prediction_batch_size,
            num_shards=TRAIN_CFG.train_num_shards,
        )

        subset = lab.build_near_zero_calibration_subset(
            data_root=paths["data_root"],
            train_pred_cache_dir=paths["train_pred_cache_dir"],
            cfg=WEIGHT_CFG,
            num_shards=TRAIN_CFG.train_num_shards,
        )
        lab.save_dataframe(subset["count_table"], paths["reports_dir"] / "calibration_candidate_count_table.csv")
        lab.save_dataframe(subset["quota_table"], paths["reports_dir"] / "calibration_quota_table.csv")
        lab.save_dataframe(subset["quota_summary"], paths["reports_dir"] / "calibration_quota_summary.csv")
        display(subset["quota_summary"])

        proxy = lab.run_weight_calibration_stockfish_proxy(
            subset=subset["samples"],
            cfg=WEIGHT_CFG,
            output_dir=paths["output_dir"],
        )
        lookup = lab.build_stability_weight_lookup(
            calibration_rows=proxy["rows"],
            cfg=WEIGHT_CFG,
            output_dir=paths["output_dir"],
        )
        weight_audit = lab.audit_full_train_weight_distribution(
            data_root=paths["data_root"],
            train_pred_cache_dir=paths["train_pred_cache_dir"],
            lookup=lookup,
            cfg=WEIGHT_CFG,
            output_dir=paths["output_dir"],
            num_shards=TRAIN_CFG.train_num_shards,
        )
        lab.save_json(proxy["report"], paths["reports_dir"] / "weight_calibration_stockfish_report.json")
        display(pd.DataFrame([proxy["report"]["stable"], proxy["report"]["unstable"]], index=["stable", "unstable"]))
        display(pd.DataFrame([weight_audit["report"]]))
        display(lookup["cell_table"].head(12))
        """
    ),
    md(
        """
        ## Stability-Weighted Fine-Tune

        Đây là run kiểm tra giả thuyết chính:
        - fine-tune từ `ckpt_best`
        - dùng weighted hybrid objective trên full train split
        - đánh giá lại bằng cùng metric suite
        """
    ),
    code(
        r"""
        finetune = lab.run_stability_weighted_finetune(
            init_ckpt_path=CHECKPOINTS["best"],
            data_root=paths["data_root"],
            output_dir=paths["output_dir"],
            device=DEVICE,
            lookup=lookup,
            weight_cfg=WEIGHT_CFG,
            train_cfg=TRAIN_CFG,
        )
        display(finetune["history"])
        """
    ),
    code(
        r"""
        finetuned_val = lab.base_lab.run_teacher_eval_suite(
            ckpt_path=finetune["best_checkpoint"],
            data_root=paths["data_root"],
            split="val",
            device=DEVICE,
            output_dir=paths["output_dir"],
            max_samples=TRAIN_CFG.eval_val_samples,
            batch_size=max(TRAIN_CFG.batch_size, 1024),
            num_shards=TRAIN_CFG.eval_val_num_shards,
            prefix="stability_weighted_best_val",
        )
        finetuned_test = lab.base_lab.run_teacher_eval_suite(
            ckpt_path=finetune["best_checkpoint"],
            data_root=paths["data_root"],
            split="test",
            device=DEVICE,
            output_dir=paths["output_dir"],
            max_samples=TRAIN_CFG.eval_test_samples,
            batch_size=max(TRAIN_CFG.batch_size, 1024),
            num_shards=TRAIN_CFG.eval_test_num_shards,
            prefix="stability_weighted_best_test",
        )

        compare = pd.DataFrame(
            [
                lab.compare_metric_rows("baseline_test", baseline_test["metrics"]),
                lab.compare_metric_rows("stability_weighted_best_test", finetuned_test["metrics"]),
            ]
        )
        compare["delta_vs_baseline"] = compare["gate_score"] - compare["gate_score"].iloc[0]
        lab.save_dataframe(compare, paths["reports_dir"] / "stability_weighted_test_compare.csv")
        lab.save_json(finetuned_val["metrics"], paths["reports_dir"] / "stability_weighted_best_val_metrics.json")
        lab.save_json(finetuned_test["metrics"], paths["reports_dir"] / "stability_weighted_best_test_metrics.json")
        display(compare)
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
