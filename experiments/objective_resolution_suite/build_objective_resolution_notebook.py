from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from uuid import uuid4


NOTEBOOK_PATH = Path(__file__).resolve().parent / "objective_resolution_suite.ipynb"


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
        # Objective Resolution Suite

        Notebook nay dung de tra loi 3 cau hoi con mo:

        1. Ket luan hien tai co ben vung hay chi do mot oracle subset duy nhat?
        2. A1 cai thien vi inverse-curvature weighting that su, hay chi do tang vai tro z-space?
        3. Co objective ket hop nao sua duoc ca `mid-band compression` va `center failure` hay khong?

        Nguyen tac danh gia:

        - khong dung MSE tong lam metric chinh
        - moi compare deu quy ve `y600` canonical space
        - metric gate uu tien:
          - `oracle_stable_0.7_slope`
          - `oracle_midband_mae_sum_stable`
          - `oracle_center_amp_ratio`
          - `oracle_center_false_0.1eq`
          - `oracle_center_false_0.2eq`
          - `oracle_center_spread_ratio`
          - `oracle_center_wrong_sign_0.1eq`
          - `oracle_center_wrong_sign_0.2eq`
          - `oracle_band_sign_0.05_0.2_stable`
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
        EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "objective_resolution_suite"
        ROOT_CAUSE_DIR = PROJECT_ROOT / "experiments" / "root_cause_ablation_suite" / "outputs"
        RUN_DIR = Path(r"C:\Users\USER\Downloads\dgrn_5m_v3_stage2_polish_run1")
        DATA_ROOT = PROJECT_ROOT / "data" / "process"

        if str(EXPERIMENT_DIR) not in sys.path:
            sys.path.insert(0, str(EXPERIMENT_DIR))

        import objective_resolution_helpers as lab

        torch.set_float32_matmul_precision("high")
        lab.set_global_seed(123)
        paths = lab.build_default_paths(run_dir=RUN_DIR, data_root=DATA_ROOT, experiment_dir=EXPERIMENT_DIR)
        lab.export_paths_json(paths, paths["output_dir"] / "paths.json")

        if "envs\\chess_engine" not in sys.executable.lower():
            raise RuntimeError(f"Wrong interpreter: {sys.executable}")
        DEVICE = lab.choose_device(prefer_cuda=True)
        if DEVICE.type != "cuda":
            raise RuntimeError("CUDA is required.")

        CHECKPOINT = paths["run_dir"] / "ckpt_best.pt"
        EXISTING_LABELS = [
            "A1_curvature_compensated",
            "A2_band_balanced",
            "C1_scale800",
            "C2_scale1200",
        ]
        NEW_VARIANT_NAMES = [
            "L0_control_hybrid",
            "L1_z_strong_hybrid",
            "L2_curvature_y_only",
            "L3_full_A1",
            "L4_A1_plus_A2",
            "S1_A1_center_w020_m010",
        ]
        REPLICATE_LABELS_INITIAL = ["baseline", "A1_curvature_compensated", "A2_band_balanced"]
        REPLICATE_LABELS_FINAL = ["baseline", "A1_curvature_compensated", "A2_band_balanced", "L4_A1_plus_A2", "S1_A1_center_w020_m010"]

        TRAIN_CFG = lab.ab_lab.TrainConfig(
            batch_size=640,
            epochs=1,
            learning_rate=3e-6,
            min_lr=1e-6,
            weight_decay=2e-4,
            grad_clip_norm=1.0,
            seed=123,
            log_every_steps=200,
            train_num_shards=None,
            val_max_samples=100_000,
            test_max_samples=200_000,
            val_num_shards=2,
            test_num_shards=4,
            benchmark_num_shards=1,
        )
        ORACLE_CFG = lab.ab_lab.OracleEvalConfig()
        BOOT_CFG = lab.BootstrapConfig(n_bootstrap=4000, seed=123, ci_alpha=0.05)
        REPL_CFG = lab.ReplicateOracleConfig(
            split="test",
            sample_abs_y_edges=(0.0, 0.05, 0.20, 0.50, 0.70, 1.00),
            sample_per_band=24,
            num_replicates=2,
            base_seed=777,
        )
        VARIANT_CATALOG = lab.build_variant_catalog()

        lab.ab_lab.validate_train_config(TRAIN_CFG)
        lab.ab_lab.validate_oracle_eval_config(ORACLE_CFG)
        lab.validate_bootstrap_config(BOOT_CFG)
        lab.validate_replicate_config(REPL_CFG)
        for name in NEW_VARIANT_NAMES:
            lab.ab_lab.validate_variant_config(VARIANT_CATALOG[name])
        lab.validate_runtime_paths(
            paths=paths,
            baseline_ckpt=CHECKPOINT,
            existing_suite_output_dir=ROOT_CAUSE_DIR,
            existing_labels=EXISTING_LABELS,
            stockfish_path=REPL_CFG.stockfish_path,
        )
        AUTOTUNE = lab.autotune_train_batch_size(
            init_ckpt_path=CHECKPOINT,
            data_root=DATA_ROOT,
            device=DEVICE,
            preferred_batch_size=TRAIN_CFG.batch_size,
            min_batch_size=128,
            step=64,
            max_mem_ratio=0.85,
        )
        TRAIN_CFG = replace(TRAIN_CFG, batch_size=int(AUTOTUNE["selected_batch_size"]))
        lab.save_json(AUTOTUNE, paths["reports_dir"] / "train_batch_autotune.json")
        lab.ab_lab.validate_checkpoint_model(CHECKPOINT, device=DEVICE)
        lab.ab_lab.validate_target_remap_logic([600.0, 800.0, 1200.0])
        lab.ab_lab.validate_train_config(TRAIN_CFG)

        ORACLE_BUNDLE = lab.build_primary_oracle_bundle(DATA_ROOT)
        display(pd.DataFrame({"path_key": list(paths.keys()), "path_value": [str(v) for v in paths.values()]}))
        display(pd.DataFrame([AUTOTUNE]))
        display(pd.DataFrame({"new_variant": NEW_VARIANT_NAMES}))
        """
    ),
    md(
        """
        ## Re-Evaluate Baseline And Existing Ablations

        Cell nay evaluate lai baseline va cac run cu bang metric/gate moi.
        Tu diem nay tro di, notebook tach rieng:

        - center amplitude / dispersion
        - center direction / wrong-sign
        """
    ),
    code(
        r"""
        existing_registry = lab.build_checkpoint_registry(
            baseline_ckpt=CHECKPOINT,
            existing_suite_output_dir=ROOT_CAUSE_DIR,
            existing_labels=EXISTING_LABELS,
            followup_output_dir=paths["output_dir"],
            followup_labels=[],
        )
        existing_primary_df, existing_results = lab.evaluate_registry(
            registry=existing_registry,
            data_root=DATA_ROOT,
            oracle_bundle=ORACLE_BUNDLE,
            oracle_cfg=ORACLE_CFG,
            train_cfg=TRAIN_CFG,
            device=DEVICE,
            output_dir=paths["output_dir"],
            prefix="existing",
        )
        display(existing_primary_df.sort_values("selection_score_v2", ascending=True))
        """
    ),
    md(
        """
        ## Bootstrap Existing Conclusions

        Muc tieu:

        - kiem tra xem delta so voi baseline co ben vung tren oracle subset hien tai hay khong
        - tach improvement that su khoi nhiễu bootstrap
        - bootstrap ca nhom metric amplitude lan direction
        """
    ),
    code(
        r"""
        existing_bootstrap = lab.bootstrap_registry(existing_results, BOOT_CFG, paths["output_dir"], prefix="existing")
        existing_bootstrap_summary = lab.summarize_bootstrap(existing_bootstrap)
        if not existing_bootstrap_summary.empty:
            lab.save_dataframe(existing_bootstrap_summary, paths["reports_dir"] / "existing_bootstrap_summary.csv")
        display(existing_bootstrap_summary)
        """
    ),
    md(
        """
        ## Replicate Oracle Subsets For Existing Runs

        Cell nay rerun Stockfish tren cac subset moi de kiem tra xem ket luan hien tai co lap lai duoc hay khong.
        Muc tieu dac biet o day la xem A2 co that su cai thien center direction, hay chi giam center decisiveness.
        """
    ),
    code(
        r"""
        initial_rep_registry = [row for row in existing_registry if row["label"] in REPLICATE_LABELS_INITIAL]
        existing_rep_per, existing_rep_agg = lab.run_oracle_replicates(
            registry=initial_rep_registry,
            baseline_ckpt=CHECKPOINT,
            data_root=DATA_ROOT,
            cfg=REPL_CFG,
            device=DEVICE,
            output_dir=paths["output_dir"],
        )
        display(existing_rep_agg.sort_values("oracle_midband_mae_sum_stable_mean", ascending=True))
        """
    ),
    md(
        """
        ## Run Follow-Up Objective Experiments

        Muc tieu:

        - `L1`: test xem tang vai tro z-space co du giai thich improvement hay khong
        - `L2`: test pure inverse-curvature weighting
        - `L3`: full A1
        - `L4`: A1 + band-balanced sampler
        - `S1`: A1 + center penalty
        """
    ),
    code(
        r"""
        followup_runs = lab.run_variant_group(
            variant_names=NEW_VARIANT_NAMES,
            variant_catalog=VARIANT_CATALOG,
            init_ckpt_path=CHECKPOINT,
            data_root=DATA_ROOT,
            train_cfg=TRAIN_CFG,
            oracle_cfg=ORACLE_CFG,
            oracle_bundle=ORACLE_BUNDLE,
            paths=paths,
            device=DEVICE,
        )
        """
    ),
    md(
        """
        ## Evaluate Full Registry With New Gate

        Bang ket qua nay can doc theo 3 lop:

        - `improves_midband`
        - `improves_center_amplitude`
        - `improves_center_direction`

        `dominates_baseline` chi dung khi candidate thang ca 3 lop.
        """
    ),
    code(
        r"""
        full_registry = lab.build_checkpoint_registry(
            baseline_ckpt=CHECKPOINT,
            existing_suite_output_dir=ROOT_CAUSE_DIR,
            existing_labels=EXISTING_LABELS,
            followup_output_dir=paths["output_dir"],
            followup_labels=NEW_VARIANT_NAMES,
        )
        full_primary_df, full_results = lab.evaluate_registry(
            registry=full_registry,
            data_root=DATA_ROOT,
            oracle_bundle=ORACLE_BUNDLE,
            oracle_cfg=ORACLE_CFG,
            train_cfg=TRAIN_CFG,
            device=DEVICE,
            output_dir=paths["output_dir"],
            prefix="full",
        )
        display(full_primary_df.sort_values("selection_score_v2", ascending=True))
        """
    ),
    md(
        """
        ## Bootstrap Full Registry

        Sau cell nay, uu tien cac candidate co:

        - `supports_improvement = True` o `stable_0.7_slope` va `midband_teacher_vs_oracle_mae_sum_stable`
        - dong thoi khong xau di ro ret o `center_amp_ratio_eq_0.05`, `center_false_pred0.1eq`, `center_spread_ratio_eq_0.05`
        - va neu co the, khong xau di o `center_wrong_sign_pred0.1eq`, `center_wrong_sign_pred0.2eq`, `stable_sign_match_0.05_0.2`
        """
    ),
    code(
        r"""
        full_bootstrap = lab.bootstrap_registry(full_results, BOOT_CFG, paths["output_dir"], prefix="full")
        full_bootstrap_summary = lab.summarize_bootstrap(full_bootstrap)
        if not full_bootstrap_summary.empty:
            lab.save_dataframe(full_bootstrap_summary, paths["reports_dir"] / "full_bootstrap_summary.csv")
        display(full_bootstrap_summary)
        """
    ),
    md(
        """
        ## Replicate Oracle Subsets For Final Candidates
        """
    ),
    code(
        r"""
        final_rep_registry = [row for row in full_registry if row["label"] in REPLICATE_LABELS_FINAL]
        final_rep_per, final_rep_agg = lab.run_oracle_replicates(
            registry=final_rep_registry,
            baseline_ckpt=CHECKPOINT,
            data_root=DATA_ROOT,
            cfg=REPL_CFG,
            device=DEVICE,
            output_dir=paths["output_dir"] / "final_replicates",
        )
        display(final_rep_agg.sort_values("oracle_midband_mae_sum_stable_mean", ascending=True))
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python (chess_engine)",
            "language": "python",
            "name": "chess_engine",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote notebook to {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
