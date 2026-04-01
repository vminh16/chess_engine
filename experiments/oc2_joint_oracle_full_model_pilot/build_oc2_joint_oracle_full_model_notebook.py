from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from uuid import uuid4


NOTEBOOK_PATH = Path(__file__).resolve().parent / "oc2_joint_oracle_full_model_pilot.ipynb"


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
        # OC2 Joint Oracle Full-Model Pilot

        Notebook này là thí nghiệm cuối cùng để kiểm tra câu hỏi còn mở:

        - `L4` có còn plastic đủ để sửa Failure B nếu dùng `joint objective` đúng không?
        - Việc sửa B có giữ được lời giải cho Failure A dưới hard gate midband hay không?

        Thiết kế:

        - giữ `L4` làm main objective
        - downweight mượt vùng `raw-center ambiguous` trong main loss
        - xây `trusted oracle center bank` theo stratification 2D:
          - `raw |y|`
          - `|pred_L4|`
        - fine-tune ngắn từ `L4` trên **toàn model**, không chỉ head
        - dùng **một optimizer step duy nhất** cho `main + aux`
        - chọn checkpoint bằng:
          - hard gate cho midband
          - rồi tối ưu center-only score
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
        EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "oc2_joint_oracle_full_model_pilot"
        DATA_ROOT = PROJECT_ROOT / "data" / "process"
        RUN_DIR = Path(r"C:\Users\USER\Downloads\dgrn_5m_v3_stage2_polish_run1")

        if str(EXPERIMENT_DIR) not in sys.path:
            sys.path.insert(0, str(EXPERIMENT_DIR))

        import oc2_joint_oracle_full_model_helpers as lab

        torch.set_float32_matmul_precision("high")
        lab.set_global_seed(123)
        DEVICE = lab.choose_device(prefer_cuda=True)
        if DEVICE.type != "cuda":
            raise RuntimeError("CUDA is required for this notebook.")
        if "envs\\chess_engine" not in sys.executable.lower():
            raise RuntimeError(f"Wrong interpreter: {sys.executable}")

        paths = lab.build_default_paths(run_dir=RUN_DIR, data_root=DATA_ROOT, experiment_dir=EXPERIMENT_DIR)
        lab.export_paths_json(paths, paths["output_dir"] / "paths.json")

        ORACLE_CFG = lab.OracleMineConfig()
        PILOT_CFG = lab.PilotTrainConfig()
        GATE_CFG = lab.MidbandGateConfig()
        lab.validate_oracle_mine_config(ORACLE_CFG)
        lab.validate_pilot_train_config(PILOT_CFG)
        lab.validate_gate_config(GATE_CFG)
        runtime_check = lab.validate_runtime_paths(paths, ORACLE_CFG)
        if not runtime_check["ok"]:
            raise RuntimeError(runtime_check["issues"])

        L4_CHECKPOINT = paths["objective_output_dir"] / "runs" / "L4_A1_plus_A2" / "checkpoints" / "L4_A1_plus_A2_best.pt"
        REFERENCE = lab.build_reference_context(DATA_ROOT, paths, refresh=False)

        lab.save_json(asdict(ORACLE_CFG), paths["output_dir"] / "oracle_cfg_initial.json")
        lab.save_json(asdict(PILOT_CFG), paths["output_dir"] / "pilot_cfg_initial.json")
        lab.save_json(asdict(GATE_CFG), paths["output_dir"] / "gate_cfg.json")
        lab.save_json(runtime_check, paths["reports_dir"] / "runtime_check.json")

        display(pd.DataFrame({"path_key": list(paths.keys()), "path_value": [str(v) for v in paths.values()]}))
        display(pd.DataFrame([runtime_check]))
        """
    ),
    md(
        """
        ## Context From Existing Suites

        Cell này đọc lại đúng các bảng đã chốt trước khi chạy OC2:

        - `baseline / A2 / L0 / L4`
        - Failure B score hiện tại
        - midband metrics hiện tại
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
        ## Build L4 Train Prediction Cache

        Cache này phục vụ mining 2D theo `raw |y| x |pred_L4|`.
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

        display(pd.DataFrame(pred_cache_manifest["shards"]).head())
        """
    ),
    md(
        """
        ## Mine 2D Candidate Bundle

        Candidate pool được chia đều theo hai trục:

        - `raw |y|` band
        - `|pred_L4|` band

        Mục tiêu là tránh bias kiểu `OC1` chỉ tập trung vào các false-decisive hard case.
        """
    ),
    code(
        r"""
        candidate_bundle = lab.build_2d_candidate_bundle(
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
        ## Oracle Audit

        Với mỗi candidate:

        - decode về board
        - chạy Stockfish multi-budget fixed-node
        - lọc stable subset
        - gán trusted center vs ambiguous roles theo oracle
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
                    "raw_band_label",
                    "pred_band_label",
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
        ## Build Role Bundle

        Role bundle tách 3 nhóm:

        - `center_anchor`: oracle sạch gần 0 và `|pred_L4|` thấp
        - `center_hard`: oracle sạch gần 0 nhưng `|pred_L4|` cao
        - `center_ambiguous`: raw-center nhưng oracle không thật sự gần 0
        """
    ),
    code(
        r"""
        role_bundle = lab.build_role_bundle(
            candidate_bundle=candidate_bundle,
            oracle_audit=oracle_audit,
            cfg=ORACLE_CFG,
            paths=paths,
            refresh=False,
        )

        display(pd.DataFrame([role_bundle["manifest"]]))
        display(role_bundle["summary"])
        display(role_bundle["rows"].head(12))
        """
    ),
    md(
        """
        ## Autotune Joint Full-Model Batch Size

        Khác với `OC1`, batch autotune ở đây dùng **joint step thật**:

        - forward/backward main
        - forward/backward aux
        - một optimizer step duy nhất
        """
    ),
    code(
        r"""
        AUTOTUNE = lab.autotune_joint_batch_size(
            init_ckpt_path=L4_CHECKPOINT,
            data_root=DATA_ROOT,
            role_bundle=role_bundle,
            pilot_cfg=PILOT_CFG,
            device=DEVICE,
            preferred_batch_size=PILOT_CFG.main_batch_size,
            min_batch_size=PILOT_CFG.min_batch_size,
            step=PILOT_CFG.batch_step,
            max_mem_ratio=PILOT_CFG.max_mem_ratio,
        )

        PILOT_CFG = replace(PILOT_CFG, main_batch_size=int(AUTOTUNE["selected_batch_size"]))
        REFERENCE["eval_cfg"] = replace(
            REFERENCE["eval_cfg"],
            batch_size=int(min(REFERENCE["eval_cfg"].batch_size, max(PILOT_CFG.main_batch_size, 128))),
        )
        lab.validate_pilot_train_config(PILOT_CFG)
        lab.save_json(AUTOTUNE, paths["reports_dir"] / "joint_batch_autotune.json")
        lab.save_json(asdict(PILOT_CFG), paths["output_dir"] / "pilot_cfg_final.json")

        preview_model, _ = lab.base_lab.load_model_from_checkpoint(L4_CHECKPOINT, device=DEVICE)
        preview_scope = lab.configure_full_model_trainable(preview_model)
        del preview_model
        torch.cuda.empty_cache()

        runtime_summary = pd.DataFrame(
            [
                {
                    "python_executable": sys.executable,
                    "device": str(DEVICE),
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 3),
                    "main_batch_size": int(PILOT_CFG.main_batch_size),
                    "anchor_batch_size": int(PILOT_CFG.anchor_batch_size),
                    "hard_batch_size": int(PILOT_CFG.hard_batch_size),
                    "ambiguous_batch_size": int(PILOT_CFG.ambiguous_batch_size),
                    "epochs": int(PILOT_CFG.epochs),
                }
            ]
        )

        display(runtime_summary)
        display(pd.DataFrame(AUTOTUNE["attempts"]))
        display(preview_scope)
        """
    ),
    md(
        """
        ## Fresh L4 Reference Under OC2 Eval Setup

        Cell này chấm lại `L4` bằng đúng pipeline sẽ dùng để gate OC2.
        """
    ),
    code(
        r"""
        l4_reference = lab.evaluate_l4_reference(
            checkpoint_path=L4_CHECKPOINT,
            data_root=DATA_ROOT,
            pooled_center_bundle=REFERENCE["pooled_center_bundle"],
            oracle_bundle=REFERENCE["oracle_bundle"],
            eval_cfg=REFERENCE["eval_cfg"],
            paths=paths,
            device=DEVICE,
            prefix="l4_reference",
        )

        display(pd.DataFrame([l4_reference["primary"]]))
        display(pd.DataFrame([l4_reference["center_eval"]]))
        display(pd.DataFrame([{"center_score": l4_reference["center_score"], "legacy_failure_b_score": l4_reference["legacy_failure_b_score"]}]))
        """
    ),
    md(
        """
        ## Run OC2 Pilot

        Training logic:

        - full-model trainable
        - `main_loss = L4 + smooth raw-center downweight`
        - `aux_loss = oracle anchor + oracle hard + oracle ambiguous + center margin`
        - accumulate gradients from `main` và `aux`, rồi mới `optimizer.step()`
        - epoch-end eval luôn ở `model.eval()`
        """
    ),
    code(
        r"""
        pilot = lab.run_oc2_joint_oracle_full_model_pilot(
            init_ckpt_path=L4_CHECKPOINT,
            data_root=DATA_ROOT,
            pilot_cfg=PILOT_CFG,
            gate_cfg=GATE_CFG,
            role_bundle=role_bundle,
            oracle_bundle=REFERENCE["oracle_bundle"],
            pooled_center_bundle=REFERENCE["pooled_center_bundle"],
            l4_reference=l4_reference,
            paths=paths,
            device=DEVICE,
        )

        display(pilot["trainable_scope"])
        display(pilot["history"])
        display(pd.DataFrame([pilot["decision_summary"]]))
        """
    ),
    md(
        """
        ## Final Comparison Against Existing Checkpoints

        So sánh:

        - `baseline`
        - `A2`
        - `L0`
        - `L4`
        - `OC2_best_any_center`
        - `OC2_best_gate` nếu có
        """
    ),
    code(
        r"""
        compare = lab.evaluate_registry_with_oc2(
            registry=REFERENCE["registry"],
            oc2_best_any_checkpoint=pilot["best_any_checkpoint"],
            oc2_best_gate_checkpoint=pilot["best_gate_checkpoint"],
            data_root=DATA_ROOT,
            pooled_center_bundle=REFERENCE["pooled_center_bundle"],
            oracle_bundle=REFERENCE["oracle_bundle"],
            eval_cfg=REFERENCE["eval_cfg"],
            l4_reference=l4_reference,
            gate_cfg=GATE_CFG,
            paths=paths,
            device=DEVICE,
            prefix="combined_oc2_final_pilot",
        )

        display(compare["primary"])
        display(compare["pooled_center"])
        """
    ),
    md(
        """
        ## Decision Rule

        Thí nghiệm được xem là thành công nếu:

        1. `OC2_best_gate` tồn tại
        2. `center_score` giảm rõ rệt so với `L4`
        3. `oracle_midband_mae_sum_stable` và `oracle_stable_0.7_slope` vẫn pass hard gate

        Nếu `OC2_best_any_center` tốt hơn center nhưng không có `best_gate`, điều đó nghĩa là tín hiệu sửa B có tồn tại nhưng vẫn phá A quá mức.
        """
    ),
    code(
        r"""
        compare_primary = compare["primary"].copy()
        decision_frame = compare_primary[
            [
                "label",
                "center_score",
                "legacy_failure_b_score",
                "oracle_midband_mae_sum_stable",
                "oracle_stable_0.7_slope",
                "midband_gate_pass",
            ]
        ].sort_values(["center_score", "oracle_midband_mae_sum_stable"], ascending=[True, True])

        display(decision_frame)
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
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
print(f"Wrote {NOTEBOOK_PATH}")
