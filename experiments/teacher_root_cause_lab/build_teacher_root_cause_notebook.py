from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from uuid import uuid4


NOTEBOOK_PATH = Path(__file__).resolve().parent / "teacher_root_cause_experiments.ipynb"


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
        # Teacher Root-Cause Lab

        Notebook này gom toàn bộ chuỗi thí nghiệm để chẩn đoán teacher hiện tại tại
        `C:\\Users\\USER\\Downloads\\dgrn_5m_v3_stage2_polish_run1`.

        Mục tiêu:
        - chuẩn hóa bộ metric teacher trước khi distill
        - kiểm tra post-hoc calibration để tách lỗi calibration/loss khỏi lỗi representation
        - kiểm tra near-zero stability để đo giả thuyết target/search volatility
        - chạy pilot center-safe finetune để xem regularization có sửa được false-decisive hay không

        Mọi output đều được lưu trong cùng cây thư mục:
        `C:\\Users\\USER\\Desktop\\chess_engine\\experiments\\teacher_root_cause_lab\\outputs`
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
        from IPython.display import Markdown, display

        PROJECT_ROOT = Path(r"C:\Users\USER\Desktop\chess_engine")
        EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "teacher_root_cause_lab"
        RUN_DIR = Path(r"C:\Users\USER\Downloads\dgrn_5m_v3_stage2_polish_run1")
        DATA_ROOT = PROJECT_ROOT / "data" / "process"

        if str(EXPERIMENT_DIR) not in sys.path:
            sys.path.insert(0, str(EXPERIMENT_DIR))

        import teacher_root_cause_helpers as lab

        torch.set_float32_matmul_precision("high")
        lab.set_global_seed(123)
        paths = lab.build_default_paths(run_dir=RUN_DIR, data_root=DATA_ROOT, experiment_dir=EXPERIMENT_DIR)
        lab.export_paths_json(paths, paths["output_dir"] / "paths.json")

        CHECKPOINTS = {
            "best": paths["run_dir"] / "ckpt_best.pt",
            "latest": paths["run_dir"] / "ckpt_latest.pt",
        }
        for label, ckpt_path in CHECKPOINTS.items():
            assert ckpt_path.exists(), f"Missing checkpoint: {label} -> {ckpt_path}"

        CONFIG = {
            "seed": 123,
            "roundtrip_samples": 64,
            "eval_test_samples": 200_000,
            "eval_val_samples": 100_000,
            "eval_test_num_shards": 4,
            "eval_val_num_shards": 2,
            "batch_size": 2048,
            "stability_positions": 128,
            "stability_source_samples": 50_000,
            "stability_source_num_shards": 4,
            "stability_depths": [1, 2, 3, 4],
        }
        lab.save_json(CONFIG, paths["output_dir"] / "runtime_config.json")

        DEVICE = lab.choose_device(prefer_cuda=True)
        if "envs\\chess_engine" not in sys.executable.lower():
            raise RuntimeError(
                f"Notebook is running under the wrong Python interpreter: {sys.executable}. "
                "Select the 'chess_engine' Jupyter kernel."
            )
        if DEVICE.type != "cuda":
            raise RuntimeError("CUDA is required for this notebook. Activate the chess_engine env with GPU-enabled PyTorch.")

        print("python:", sys.executable)
        print("device:", DEVICE)
        print("gpu_name:", torch.cuda.get_device_name(0))
        display(pd.DataFrame({"path_key": list(paths.keys()), "path_value": [str(v) for v in paths.values()]}))
        """
    ),
    md(
        """
        ## Self-Checks

        Phần này fail fast trước khi chạy cell nặng:
        - xác minh checkpoint config của run hiện tại
        - xác minh encode -> decode -> encode không làm sai tensor representation
        """
    ),
    code(
        r"""
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
                    "weight_decay": payload.get("weight_decay"),
                    "drop_path_rate": payload.get("drop_path_rate"),
                }
            )

        checkpoint_df = pd.DataFrame(checkpoint_rows)
        lab.save_dataframe(checkpoint_df, paths["reports_dir"] / "checkpoint_config_table.csv")
        display(checkpoint_df)

        roundtrip = lab.validate_encode_decode_roundtrip(
            data_root=paths["data_root"],
            split="test",
            sample_count=CONFIG["roundtrip_samples"],
        )
        lab.save_json(roundtrip, paths["reports_dir"] / "encode_decode_roundtrip.json")
        display(pd.DataFrame([roundtrip]))
        assert roundtrip["mismatches"] == 0, roundtrip
        """
    ),
    md(
        """
        ## Baseline Teacher Eval

        Chạy cùng một metric suite cho:
        - `ckpt_best.pt` trên validation để fit post-hoc calibrator
        - `ckpt_best.pt` trên test để làm baseline teacher
        - `ckpt_latest.pt` trên test để xác nhận best/latest có thật sự khác nhau hay không

        Các cell này dùng AMP + CUDA và ghi cache prediction về `outputs/cache`.
        """
    ),
    code(
        r"""
        best_val = lab.run_teacher_eval_suite(
            ckpt_path=CHECKPOINTS["best"],
            data_root=paths["data_root"],
            split="val",
            device=DEVICE,
            output_dir=paths["output_dir"],
            max_samples=CONFIG["eval_val_samples"],
            batch_size=CONFIG["batch_size"],
            num_shards=CONFIG["eval_val_num_shards"],
            prefix=f"best_val_{CONFIG['eval_val_samples'] // 1000}k",
        )

        best_test = lab.run_teacher_eval_suite(
            ckpt_path=CHECKPOINTS["best"],
            data_root=paths["data_root"],
            split="test",
            device=DEVICE,
            output_dir=paths["output_dir"],
            max_samples=CONFIG["eval_test_samples"],
            batch_size=CONFIG["batch_size"],
            num_shards=CONFIG["eval_test_num_shards"],
            prefix=f"best_test_{CONFIG['eval_test_samples'] // 1000}k",
        )

        latest_test = lab.run_teacher_eval_suite(
            ckpt_path=CHECKPOINTS["latest"],
            data_root=paths["data_root"],
            split="test",
            device=DEVICE,
            output_dir=paths["output_dir"],
            max_samples=CONFIG["eval_test_samples"],
            batch_size=CONFIG["batch_size"],
            num_shards=CONFIG["eval_test_num_shards"],
            prefix=f"latest_test_{CONFIG['eval_test_samples'] // 1000}k",
        )
        """
    ),
    code(
        r"""
        def compact_metric_row(label: str, metrics: dict) -> dict:
            return {
                "label": label,
                "overall_mse": metrics["overall"]["mse"],
                "overall_mae": metrics["overall"]["mae"],
                "overall_r2": metrics["overall"]["r2"],
                "mse_0.7": metrics["bands"]["0.7"]["mse"],
                "mae_0.7": metrics["bands"]["0.7"]["mae"],
                "slope_0.7": metrics["bands"]["0.7"]["slope"],
                "bias_0.7": metrics["bands"]["0.7"]["bias"],
                "mse_0.2": metrics["bands"]["0.2"]["mse"],
                "r2_0.2": metrics["bands"]["0.2"]["r2"],
                "false_0.1_0.3": metrics["false_decisive"]["y<=0.1,p>=0.3"]["rate"],
                "false_0.2_0.4": metrics["false_decisive"]["y<=0.2,p>=0.4"]["rate"],
                "false_0.2_0.5": metrics["false_decisive"]["y<=0.2,p>=0.5"]["rate"],
                "center_spread_ratio_0.05": metrics["center_spread_ratio_0.05"]["ratio"],
                "max_midband_abs_cal_gap": metrics["max_midband_abs_cal_gap"],
            }

        baseline_compare = pd.DataFrame(
            [
                compact_metric_row("teacher_best_test", best_test["metrics"]),
                compact_metric_row("teacher_latest_test", latest_test["metrics"]),
            ]
        )
        lab.save_dataframe(baseline_compare, paths["reports_dir"] / "baseline_checkpoint_compare.csv")
        display(baseline_compare)
        """
    ),
    md(
        """
        ## Post-Hoc Calibration

        Dùng validation predictions của `ckpt_best` để fit một symmetric isotonic calibrator,
        rồi đánh giá lại trên test.

        Diễn giải:
        - nếu metric improve rõ, teacher đang học ranking tương đối tốt nhưng bị lệch calibration/loss
        - nếu improve yếu, lỗi có thể nằm sâu hơn ở target semantics hoặc information limit
        """
    ),
    code(
        r"""
        posthoc = lab.run_posthoc_calibration_experiment(
            val_y=best_val["targets"],
            val_p=best_val["preds"],
            test_y=best_test["targets"],
            test_p=best_test["preds"],
            output_dir=paths["output_dir"],
            prefix="best_posthoc_calibration",
        )

        posthoc_compare = pd.DataFrame(
            [
                compact_metric_row("teacher_best_test_before", posthoc["report"]["before"]),
                compact_metric_row("teacher_best_test_after_posthoc", posthoc["report"]["after"]),
            ]
        )
        lab.save_dataframe(posthoc_compare, paths["reports_dir"] / "posthoc_compare.csv")
        display(posthoc_compare)
        display(pd.DataFrame([posthoc["report"]["delta"]]))
        """
    ),
    md(
        """
        ## Near-Zero Stability Probe

        Đây là thí nghiệm để kiểm tra giả thuyết `target/search volatility` bằng search proxy đúng unit.

        Cách làm:
        - lấy các vị trí test có `|y| <= 0.2`
        - teacher dự đoán trên các vị trí đó
        - decode tensor về board
        - chạy lại classical search theo nhiều depth để lấy `search native score`
        - đo quietness bằng ba proxy:
          `score range across depth`, `best-move changes`, `gap giữa final search score và static material score`

        Lưu ý:
        - đây là proxy chẩn đoán local, không phải relabel depth-25 gốc
        - ta không ép đổi sang centipawn giả; score được giữ ở đúng unit native của search proxy
        - mục tiêu là xem lỗi teacher trên near-zero có tập trung mạnh ở các vị trí search-unstable hay không
        """
    ),
    code(
        r"""
        stability = lab.run_near_zero_stability_experiment(
            ckpt_path=CHECKPOINTS["best"],
            data_root=paths["data_root"],
            output_dir=paths["output_dir"],
            device=DEVICE,
            split="test",
            near_zero_thr=0.2,
            sample_positions=CONFIG["stability_positions"],
            search_depths=tuple(CONFIG["stability_depths"]),
            sample_seed=CONFIG["seed"],
            max_source_samples=CONFIG["stability_source_samples"],
            source_num_shards=CONFIG["stability_source_num_shards"],
        )

        stability_table = pd.DataFrame(
            [
                {"subset": "stable", **stability["report"]["stable"]},
                {"subset": "unstable", **stability["report"]["unstable"]},
            ]
        )
        lab.save_dataframe(stability_table, paths["reports_dir"] / "near_zero_stability_compare.csv")
        display(stability_table)
        """
    ),
    md(
        """
        ## Center-Safe Finetune Pilot

        Pilot này giữ teacher hiện tại làm init checkpoint và thêm regularizer:

        `L_center = 1_{|y| <= tau} * ReLU(|pred| - margin)^2`

        Mục tiêu:
        - giảm false-decisive quanh center
        - không làm gãy hoàn toàn slope ở `|y| <= 0.7`

        Nếu muốn chạy lại pilot từ đầu với config mới, hãy đổi `pilot_output_dir` hoặc xóa checkpoint pilot cũ.
        """
    ),
    code(
        r"""
        pilot_output_dir = paths["output_dir"] / "center_safe_pilot"
        pilot_output_dir.mkdir(parents=True, exist_ok=True)

        pilot_cfg = lab.PilotTrainConfig(
            lambda_y=0.99,
            z_loss_beta=1.0,
            z_huber_delta=0.5,
            target_clamp_eps=1e-3,
            center_tau=0.10,
            center_margin=0.20,
            center_weight=0.25,
            learning_rate=5e-6,
            min_lr=1e-6,
            weight_decay=2e-4,
            grad_clip_norm=1.0,
            batch_size=1024,
            epochs=3,
            train_max_samples=200_000,
            val_max_samples=50_000,
            train_num_shards=8,
            val_num_shards=2,
            log_every=50,
            seed=CONFIG["seed"],
        )
        lab.save_json(asdict(pilot_cfg), pilot_output_dir / "pilot_train_config.json")

        pilot_best_ckpt = pilot_output_dir / "checkpoints" / "center_safe_pilot_best.pt"
        pilot_history_csv = pilot_output_dir / "reports" / "center_safe_pilot_history.csv"
        if pilot_best_ckpt.exists() and pilot_history_csv.exists():
            pilot = {
                "history": pd.read_csv(pilot_history_csv),
                "best_checkpoint": pilot_best_ckpt,
                "latest_checkpoint": pilot_output_dir / "checkpoints" / "center_safe_pilot_latest.pt",
            }
            print("Using existing pilot artifacts:", pilot_best_ckpt)
        else:
            pilot = lab.run_center_safe_finetune_pilot(
                init_ckpt_path=CHECKPOINTS["best"],
                data_root=paths["data_root"],
                output_dir=pilot_output_dir,
                device=DEVICE,
                cfg=pilot_cfg,
            )

        display(pilot["history"])
        """
    ),
    code(
        r"""
        pilot_test = lab.run_teacher_eval_suite(
            ckpt_path=pilot["best_checkpoint"],
            data_root=paths["data_root"],
            split="test",
            device=DEVICE,
            output_dir=pilot_output_dir,
            max_samples=CONFIG["eval_test_samples"],
            batch_size=CONFIG["batch_size"],
            num_shards=CONFIG["eval_test_num_shards"],
            prefix=f"center_safe_pilot_best_test_{CONFIG['eval_test_samples'] // 1000}k",
        )

        pilot_compare = pd.DataFrame(
            [
                compact_metric_row("teacher_best_test", best_test["metrics"]),
                compact_metric_row("center_safe_pilot_best_test", pilot_test["metrics"]),
            ]
        )
        lab.save_dataframe(pilot_compare, pilot_output_dir / "reports" / "pilot_compare.csv")
        display(pilot_compare)
        """
    ),
    md(
        """
        ## Gate Summary And Root-Cause Signals

        Gate distill tối thiểu đang dùng:
        - `mse_0.7 <= 0.045`
        - `slope_0.7 >= 0.80`
        - `max_midband_abs_cal_gap <= 0.20`
        - `false_0.1_0.3 <= 0.05`
        - `false_0.2_0.4 <= 0.02`
        - `center_spread_ratio_0.05 <= 2.0`

        Phần dưới tự động tổng hợp baseline, post-hoc calibration và center-safe pilot để quyết định bước tiếp theo.
        """
    ),
    code(
        r"""
        GATES = {
            "mse_0.7_max": 0.045,
            "slope_0.7_min": 0.80,
            "max_midband_abs_cal_gap_max": 0.20,
            "false_0.1_0.3_max": 0.05,
            "false_0.2_0.4_max": 0.02,
            "center_spread_ratio_0.05_max": 2.0,
        }

        def gate_row(label: str, metrics: dict) -> dict:
            row = compact_metric_row(label, metrics)
            row["pass_mse_0.7"] = row["mse_0.7"] <= GATES["mse_0.7_max"]
            row["pass_slope_0.7"] = row["slope_0.7"] >= GATES["slope_0.7_min"]
            row["pass_max_gap"] = row["max_midband_abs_cal_gap"] <= GATES["max_midband_abs_cal_gap_max"]
            row["pass_false_0.1_0.3"] = row["false_0.1_0.3"] <= GATES["false_0.1_0.3_max"]
            row["pass_false_0.2_0.4"] = row["false_0.2_0.4"] <= GATES["false_0.2_0.4_max"]
            row["pass_center_spread"] = row["center_spread_ratio_0.05"] <= GATES["center_spread_ratio_0.05_max"]
            row["passed_gate_count"] = int(
                row["pass_mse_0.7"]
                + row["pass_slope_0.7"]
                + row["pass_max_gap"]
                + row["pass_false_0.1_0.3"]
                + row["pass_false_0.2_0.4"]
                + row["pass_center_spread"]
            )
            return row

        gate_summary = pd.DataFrame(
            [
                gate_row("teacher_best_test", best_test["metrics"]),
                gate_row("teacher_latest_test", latest_test["metrics"]),
                gate_row("teacher_best_posthoc", posthoc["report"]["after"]),
                gate_row("center_safe_pilot_best_test", pilot_test["metrics"]),
            ]
        )
        lab.save_dataframe(gate_summary, paths["reports_dir"] / "teacher_gate_summary.csv")
        display(gate_summary)

        root_cause_signals = []
        if (
            posthoc["report"]["delta"]["slope_0.7"] > 0.10
            or posthoc["report"]["delta"]["false_decisive_0.1_0.3"] < -0.02
        ):
            root_cause_signals.append("Calibration/loss mismatch is a material contributor.")
        proxy_diag = stability["report"]["proxy_diagnostics"]
        if not proxy_diag["is_informative"]:
            root_cause_signals.append("Search-proxy stability probe is still not informative enough to support a causal claim about label volatility.")
        elif stability["report"]["unstable"]["mse"] > stability["report"]["stable"]["mse"] + 0.01:
            root_cause_signals.append("Near-zero search volatility is a likely root cause in the target semantics.")
        pilot_delta_false = (
            pilot_test["metrics"]["false_decisive"]["y<=0.1,p>=0.3"]["rate"]
            - best_test["metrics"]["false_decisive"]["y<=0.1,p>=0.3"]["rate"]
        )
        pilot_delta_slope = (
            pilot_test["metrics"]["bands"]["0.7"]["slope"]
            - best_test["metrics"]["bands"]["0.7"]["slope"]
        )
        if pilot_delta_false < -0.01 and pilot_delta_slope > -0.05:
            root_cause_signals.append("Center-safe regularization reduces dangerous center overreach without collapsing mid-band slope.")
        if not root_cause_signals:
            root_cause_signals.append("Current experiments are not yet enough to isolate a dominant root cause.")

        summary_payload = {
            "gates": GATES,
            "root_cause_signals": root_cause_signals,
            "posthoc_delta": posthoc["report"]["delta"],
            "stability_report": stability["report"],
            "pilot_delta_false_0.1_0.3": pilot_delta_false,
            "pilot_delta_slope_0.7": pilot_delta_slope,
        }
        lab.save_json(summary_payload, paths["reports_dir"] / "root_cause_summary.json")

        display(Markdown("### Root-Cause Signals"))
        for line in root_cause_signals:
            print("-", line)
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


NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote notebook to {NOTEBOOK_PATH}")
