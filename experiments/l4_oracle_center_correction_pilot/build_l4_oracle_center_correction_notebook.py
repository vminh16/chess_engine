from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from uuid import uuid4


NOTEBOOK_PATH = Path(__file__).resolve().parent / "l4_oracle_center_correction_pilot.ipynb"


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
        # L4 Oracle Center Correction Pilot

        Notebook này kiểm tra trực tiếp hướng:

        - giữ `L4` làm main objective để bảo toàn lời giải hiện có cho Failure A
        - xây một `oracle-corrected ambiguous-center auxiliary set` hẹp từ train split
        - fine-tune ngắn 1-3 epoch từ checkpoint `L4`
        - chỉ mở `head + last blocks` để pilot giống một late-stage correction hơn là full retrain

        Success criteria:

        1. `Failure B` phải cải thiện rõ so với `L4`
        2. `oracle_midband_mae_sum_stable` và `oracle_stable_0.7_slope` không được rơi mạnh khỏi `L4`
        """
    ),
    code(
        r"""
        from dataclasses import asdict, replace
        from pathlib import Path
        import sys

        import pandas as pd
        import torch
        from IPython.display import display

        PROJECT_ROOT = Path(r"C:\Users\USER\Desktop\chess_engine")
        EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "l4_oracle_center_correction_pilot"
        DATA_ROOT = PROJECT_ROOT / "data" / "process"
        RUN_DIR = Path(r"C:\Users\USER\Downloads\dgrn_5m_v3_stage2_polish_run1")

        if str(EXPERIMENT_DIR) not in sys.path:
            sys.path.insert(0, str(EXPERIMENT_DIR))

        import l4_oracle_center_correction_helpers as lab

        torch.set_float32_matmul_precision("high")
        lab.set_global_seed(123)
        DEVICE = lab.choose_device(prefer_cuda=True)
        if DEVICE.type != "cuda":
            raise RuntimeError("CUDA is required for this notebook.")
        if "envs\\chess_engine" not in sys.executable.lower():
            raise RuntimeError(f"Wrong interpreter: {sys.executable}")

        paths = lab.build_default_paths(run_dir=RUN_DIR, data_root=DATA_ROOT, experiment_dir=EXPERIMENT_DIR)
        lab.export_paths_json(paths, paths["output_dir"] / "paths.json")

        ORACLE_CFG = lab.OracleSubsetConfig()
        PILOT_CFG = lab.PilotTrainConfig()
        runtime_check = lab.validate_runtime_paths(paths, ORACLE_CFG)
        if not runtime_check["ok"]:
            raise RuntimeError(runtime_check["issues"])

        L4_CHECKPOINT = paths["objective_output_dir"] / "runs" / "L4_A1_plus_A2" / "checkpoints" / "L4_A1_plus_A2_best.pt"
        REFERENCE = lab.build_reference_context(DATA_ROOT, paths, refresh=False)

        AUTOTUNE = lab.obj_lab.autotune_train_batch_size(
            init_ckpt_path=L4_CHECKPOINT,
            data_root=DATA_ROOT,
            device=DEVICE,
            preferred_batch_size=PILOT_CFG.main_batch_size,
            min_batch_size=PILOT_CFG.min_batch_size,
            step=PILOT_CFG.batch_step,
            max_mem_ratio=PILOT_CFG.max_mem_ratio,
        )
        PILOT_CFG = replace(PILOT_CFG, main_batch_size=int(AUTOTUNE["selected_batch_size"]))
        lab.save_json(asdict(ORACLE_CFG), paths["output_dir"] / "oracle_cfg.json")
        lab.save_json(asdict(PILOT_CFG), paths["output_dir"] / "pilot_cfg.json")
        lab.save_json(runtime_check, paths["reports_dir"] / "runtime_check.json")
        lab.save_json(AUTOTUNE, paths["reports_dir"] / "batch_autotune.json")

        display(pd.DataFrame({"path_key": list(paths.keys()), "path_value": [str(v) for v in paths.values()]}))
        display(pd.DataFrame([runtime_check]))
        display(pd.DataFrame([AUTOTUNE]))
        """
    ),
    md(
        """
        ## Pre-Flight Runtime Scope Check

        Cell này hiển thị rõ:

        - Python interpreter đang dùng
        - GPU thực tế đang dùng
        - batch size sau autotune
        - đúng trainable scope trước khi chạy pilot

        Nếu scope không đúng ý định `head + last blocks` thì dừng ngay tại đây.
        """
    ),
    code(
        r"""
        runtime_summary = pd.DataFrame(
            [
                {
                    "python_executable": sys.executable,
                    "device": str(DEVICE),
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 3),
                    "main_batch_size": int(PILOT_CFG.main_batch_size),
                    "aux_batch_size": int(PILOT_CFG.aux_batch_size),
                    "main_train_num_shards": int(PILOT_CFG.main_train_num_shards),
                    "oracle_train_num_shards": int(ORACLE_CFG.train_num_shards),
                    "epochs": int(PILOT_CFG.epochs),
                }
            ]
        )

        preview_model, _ = lab.base_lab.load_model_from_checkpoint(L4_CHECKPOINT, device=DEVICE)
        preview_scope = lab.configure_trainable_scope(preview_model, PILOT_CFG.freeze_last_blocks)
        del preview_model
        torch.cuda.empty_cache()

        display(runtime_summary)
        display(preview_scope)
        """
    ),
    md(
        """
        ## Context From Previous Suites

        Cell này chỉ đọc lại các bảng chốt từ suite cũ để notebook mới luôn đứng trên đúng baseline:

        - `baseline / A2 / L0 / L4`
        - metric ưu tiên của `Failure B`
        - metric ưu tiên của `Failure A`
        """
    ),
    code(
        r"""
        failure_b_primary = pd.read_csv(paths["failure_b_reports_dir"] / "combined_failure_b_primary_metrics.csv")
        objective_primary = pd.read_csv(paths["objective_output_dir"] / "reports" / "full_primary_metrics.csv")

        display(
            failure_b_primary[
                failure_b_primary["label"].isin(["baseline", "A2_band_balanced", "L0_control_hybrid", "L4_A1_plus_A2"])
            ].sort_values("failure_b_score")
        )
        display(
            objective_primary[
                objective_primary["label"].isin(["baseline", "A2_band_balanced", "L0_control_hybrid", "L4_A1_plus_A2"])
            ][["label", "oracle_midband_mae_sum_stable", "oracle_stable_0.7_slope", "oracle_center_amp_ratio", "selection_score_v2"]]
            .sort_values("oracle_midband_mae_sum_stable")
        )
        """
    ),
    md(
        """
        ## Build Candidate Pool From L4 Train Predictions

        Candidate pool:

        - chỉ lấy vùng `raw |y| <= 0.10`
        - stratified theo `raw |y|`
        - trong mỗi band, ưu tiên nửa quota có `|pred_L4|` lớn để tập trung vào false-decisive risk
        """
    ),
    code(
        r"""
        pred_cache_manifest = lab.fb_lab.precompute_train_prediction_cache(
            checkpoint_path=L4_CHECKPOINT,
            data_root=DATA_ROOT,
            split="train",
            num_shards=ORACLE_CFG.train_num_shards,
            paths=paths,
            device=DEVICE,
            batch_size=ORACLE_CFG.prediction_cache_batch_size,
            refresh=False,
        )

        candidate_bundle = lab.build_oracle_candidate_bundle(
            checkpoint_path=L4_CHECKPOINT,
            pred_cache_manifest=pred_cache_manifest,
            data_root=DATA_ROOT,
            cfg=ORACLE_CFG,
            paths=paths,
            refresh=False,
        )

        display(pd.DataFrame([candidate_bundle["manifest"]]))
        display(candidate_bundle["quota_summary"])
        display(candidate_bundle["rows"].head(12))
        """
    ),
    md(
        """
        ## Multi-Budget Oracle Audit On Candidate Pool

        Với mỗi candidate:

        - decode về board
        - chạy Stockfish theo nhiều fixed-node budgets
        - đo `oracle_target_range`, `oracle_target_std`, `bestmove_changes`, `sign_flips`
        - giữ lại `stable` rows để làm auxiliary correction set
        """
    ),
    code(
        r"""
        oracle_audit = lab.run_oracle_candidate_audit(
            candidate_bundle=candidate_bundle,
            cfg=ORACLE_CFG,
            paths=paths,
            refresh=False,
        )

        display(pd.DataFrame([oracle_audit["report"]]))
        display(oracle_audit["summary"])
        display(
            oracle_audit["rows"][
                [
                    "candidate_id",
                    "band_label",
                    "raw_target_y",
                    "init_pred",
                    "oracle_final_y",
                    "oracle_target_range",
                    "oracle_bestmove_changes",
                    "oracle_sign_flips",
                    "is_stable",
                    "is_center_clean",
                    "aux_keep",
                ]
            ].head(20)
        )
        """
    ),
    md(
        """
        ## Build Oracle Auxiliary Bundle

        Auxiliary bundle dùng:

        - mọi sample `stable` và `|oracle_final_y| <= aux_oracle_abs_max`
        - boost sampling cho `center_clean`
        - về sau margin penalty chỉ áp cho subset `center_clean`
        """
    ),
    code(
        r"""
        aux_bundle = lab.build_oracle_aux_bundle(
            candidate_bundle=candidate_bundle,
            oracle_audit=oracle_audit,
            pilot_cfg=PILOT_CFG,
            paths=paths,
            refresh=False,
        )

        display(pd.DataFrame([aux_bundle["manifest"]]))
        display(aux_bundle["rows"]["aux_role"].value_counts(dropna=False).rename_axis("aux_role").reset_index(name="count"))
        display(aux_bundle["rows"].head(12))
        """
    ),
    md(
        """
        ## Run Pilot Fine-Tune

        Schedule:

        - main step: replay `L4` objective trên train shards, nhưng downweight raw-center band
        - aux step: oracle correction trên aux bundle
        - trainable scope: `head + last blocks`
        - duration: 2 epochs mặc định
        """
    ),
    code(
        r"""
        pilot = lab.run_l4_oracle_center_correction_pilot(
            init_ckpt_path=L4_CHECKPOINT,
            data_root=DATA_ROOT,
            pilot_cfg=PILOT_CFG,
            aux_bundle=aux_bundle,
            oracle_bundle=REFERENCE["oracle_bundle"],
            pooled_center_bundle=REFERENCE["pooled_center_bundle"],
            paths=paths,
            device=DEVICE,
        )

        display(pilot["trainable_scope"])
        display(pilot["history"])
        """
    ),
    md(
        """
        ## Final Comparison Against Existing Checkpoints

        Cell này chấm lại:

        - `baseline`
        - `A2`
        - `L0`
        - `L4`
        - pilot mới `OC1_L4_oracle_center_correction`
        """
    ),
    code(
        r"""
        comparison = lab.compare_registry_with_pilot(
            registry=REFERENCE["registry"],
            pilot_checkpoint=pilot["best_checkpoint"],
            data_root=DATA_ROOT,
            pooled_center_bundle=REFERENCE["pooled_center_bundle"],
            oracle_bundle=REFERENCE["oracle_bundle"],
            eval_cfg=REFERENCE["eval_cfg"],
            paths=paths,
            device=DEVICE,
            prefix="combined_oracle_center_pilot",
        )
        display(comparison["primary"])

        l4_row = comparison["primary"][comparison["primary"]["label"] == "L4_A1_plus_A2"].iloc[0]
        pilot_row = comparison["primary"][comparison["primary"]["label"] == "OC1_L4_oracle_center_correction"].iloc[0]
        go_no_go = pd.DataFrame(
            [
                {
                    "pilot_failure_b_better_than_l4": float(pilot_row["failure_b_score"]) < float(l4_row["failure_b_score"]),
                    "pilot_midband_mae_delta_vs_l4": float(pilot_row["oracle_midband_mae_sum_stable"]) - float(l4_row["oracle_midband_mae_sum_stable"]),
                    "pilot_slope_delta_vs_l4": float(pilot_row["oracle_stable_0.7_slope"]) - float(l4_row["oracle_stable_0.7_slope"]),
                    "pilot_center_false_0.1_delta_vs_l4": float(pilot_row["pooled_center_false_0.1eq"]) - float(l4_row["pooled_center_false_0.1eq"]),
                    "pilot_center_amp_delta_vs_l4": float(pilot_row["pooled_center_amp_ratio"]) - float(l4_row["pooled_center_amp_ratio"]),
                }
            ]
        )
        display(go_no_go)
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (chess_engine)",
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
