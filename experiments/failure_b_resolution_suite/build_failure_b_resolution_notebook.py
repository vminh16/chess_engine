from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from uuid import uuid4


NOTEBOOK_PATH = Path(__file__).resolve().parent / "failure_b_resolution_suite.ipynb"


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
        # Failure B Resolution Suite

        Notebook nay chi tap trung vao `Failure B`:

        - `ultra-center over-confidence`
        - tach rieng:
          - center amplitude / dispersion
          - center direction / wrong-sign

        Muc tieu cua suite:

        1. `D1`: audit do sach cua `raw train-center labels` so voi `oracle-center`
        2. `D2`: audit gradient interference de xem update tu non-center bands co lam center logits no len hay khong
        3. `R1`: replicate control cho `baseline / L0 / L1`
        4. chay 2 pilot sua Failure B:
           - `P_B2_raw_center_strong`
           - `P_B1_proxy_center_weighted`

        Nguyen tac:

        - khong dung `MSE` tong lam metric chon winner
        - toan bo output deu ghi ra file rieng va co the tai su dung neu notebook bi crash
        - notebook nay chay tren repo day du o `C:\\Users\\USER\\Desktop\\chess_engine`
        """
    ),
    code(
        r"""
        from dataclasses import replace
        from pathlib import Path
        import sys

        import pandas as pd
        import torch
        from IPython.display import display

        PROJECT_ROOT = Path(r"C:\Users\USER\Desktop\chess_engine")
        EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "failure_b_resolution_suite"
        DATA_ROOT = PROJECT_ROOT / "data" / "process"
        RUN_DIR = Path(r"C:\Users\USER\Downloads\dgrn_5m_v3_stage2_polish_run1")

        if str(EXPERIMENT_DIR) not in sys.path:
            sys.path.insert(0, str(EXPERIMENT_DIR))

        import failure_b_resolution_helpers as lab

        torch.set_float32_matmul_precision("high")
        lab.set_global_seed(123)
        DEVICE = lab.choose_device(prefer_cuda=True)
        if DEVICE.type != "cuda":
            raise RuntimeError("CUDA is required.")
        if "envs\\chess_engine" not in sys.executable.lower():
            raise RuntimeError(f"Wrong interpreter: {sys.executable}")

        paths = lab.build_default_paths(run_dir=RUN_DIR, data_root=DATA_ROOT, experiment_dir=EXPERIMENT_DIR)
        lab.export_paths_json(paths, paths["output_dir"] / "paths.json")
        runtime_check = lab.validate_runtime_paths(paths)
        CHECKPOINT = paths["run_dir"] / "ckpt_best.pt"
        L3_CHECKPOINT = paths["objective_output_dir"] / "runs" / "L3_full_A1" / "checkpoints" / "L3_full_A1_best.pt"

        CENTER_CFG = lab.CenterPurityConfig()
        GRAD_CFG = lab.GradientAuditConfig(
            train_split="train",
            train_num_shards=8,
            batch_size=256,
            batches_per_band=3,
            sample_seed=123,
            influence_step_l2=5e-4,
            probe_center_max=128,
            probe_midband_max=128,
        )
        CONTROL_REPL_CFG = lab.obj_lab.ReplicateOracleConfig(
            split="test",
            sample_abs_y_edges=(0.0, 0.05, 0.20, 0.50, 0.70, 1.00),
            sample_per_band=24,
            num_replicates=3,
            base_seed=1777,
        )
        PILOT_CFG = lab.PilotTrainConfig(
            batch_size=576,
            epochs=1,
            learning_rate=3e-6,
            min_lr=1e-6,
            weight_decay=2e-4,
            grad_clip_norm=1.0,
            seed=123,
            train_num_shards=8,
            test_max_samples=200_000,
            test_num_shards=4,
            log_every_steps=200,
            max_mem_ratio=0.85,
            min_batch_size=128,
            batch_step=64,
        )
        ORACLE_CFG = lab.ab_lab.OracleEvalConfig()
        EVAL_CFG = lab.ab_lab.TrainConfig(
            batch_size=PILOT_CFG.batch_size,
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

        AUTOTUNE = lab.obj_lab.autotune_train_batch_size(
            init_ckpt_path=L3_CHECKPOINT,
            data_root=DATA_ROOT,
            device=DEVICE,
            preferred_batch_size=PILOT_CFG.batch_size,
            min_batch_size=PILOT_CFG.min_batch_size,
            step=PILOT_CFG.batch_step,
            max_mem_ratio=PILOT_CFG.max_mem_ratio,
        )
        PILOT_CFG = replace(PILOT_CFG, batch_size=int(AUTOTUNE["selected_batch_size"]))
        EVAL_CFG = replace(EVAL_CFG, batch_size=int(AUTOTUNE["selected_batch_size"]))
        lab.save_json(AUTOTUNE, paths["reports_dir"] / "pilot_batch_autotune.json")

        ORACLE_BUNDLE = lab.ab_lab.load_oracle_subset_bundle(ORACLE_CFG, data_root=DATA_ROOT)
        EXISTING_REGISTRY_DF = lab.load_checkpoint_registry(paths)
        EXISTING_REGISTRY = EXISTING_REGISTRY_DF.to_dict("records")
        PILOT_VARIANTS = lab.build_pilot_variants(paths)

        display(pd.DataFrame({"path_key": list(paths.keys()), "path_value": [str(v) for v in paths.values()]}))
        display(pd.DataFrame([runtime_check]))
        display(pd.DataFrame([AUTOTUNE]))
        display(EXISTING_REGISTRY_DF)
        """
    ),
    md(
        """
        ## D1: Center Label Purity Audit

        Cell nay gop tat ca oracle subset da co, deduplicate theo `(split, shard_id, local_index)`,
        roi tra loi 2 cau hoi:

        1. `|y_train| <= tau` co thuc su map vao `oracle-center stable` hay khong?
        2. co the xay `trusted-center proxy` tu `(abs_y, abs_pred, abs_err)` hay khong?
        """
    ),
    code(
        r"""
        center_pool = lab.build_oracle_center_pool(paths, CENTER_CFG, refresh=False)
        purity_audit = lab.run_center_label_purity_audit(center_pool["unique_rows"], CENTER_CFG, paths)
        center_lookup = lab.build_center_purity_lookup(center_pool["unique_rows"], CENTER_CFG, paths)

        display(pd.DataFrame([center_pool["report"]]))
        display(purity_audit["summary"])
        display(purity_audit["band_summary"])
        display(center_lookup["lookup"].sort_values("smoothed_clean_rate", ascending=False).head(20))
        """
    ),
    md(
        """
        ## D2: Gradient Interference Audit

        Cell nay khong train.

        No do 3 lop thong tin:

        - gradient norm theo band
        - cosine similarity giua gradient center va gradient mid-band
        - one-step influence:
          mot buoc update tu batch nao lam `|pred|` tren `oracle-center clean probe` tang bao nhieu
        """
    ),
    code(
        r"""
        probe_sets = lab.build_probe_sets(center_pool["unique_rows"], DATA_ROOT, GRAD_CFG, paths, refresh=False)
        grad_manifest = lab.sample_gradient_batches(DATA_ROOT, GRAD_CFG, paths, refresh=False)
        gradient_batches = lab.load_gradient_batches(DATA_ROOT, grad_manifest, paths)
        grad_audit = lab.run_gradient_interference_audit(
            checkpoint_path=CHECKPOINT,
            gradient_batches=gradient_batches,
            probe_sets=probe_sets,
            cfg=GRAD_CFG,
            paths=paths,
            device=DEVICE,
            refresh=False,
        )

        display(pd.DataFrame([grad_audit["summary"]]))
        display(grad_audit["norms"])
        display(grad_audit["cosines"])
        display(grad_audit["influence"].sort_values(["probe_name", "delta_mean_abs_pred"], ascending=[True, False]))
        """
    ),
    md(
        """
        ## R1: Replicate Controls For Baseline / L0 / L1

        Cell nay chi tra loi attribution:

        - train them 1 epoch voi objective cu (`L0`) co tu no cai thien khong?
        - neu co, `L1` co them contribution rieng tu z-space hay khong?
        """
    ),
    code(
        r"""
        control_repl = lab.run_control_replicate_l0_l1(
            baseline_ckpt=CHECKPOINT,
            data_root=DATA_ROOT,
            cfg=CONTROL_REPL_CFG,
            paths=paths,
            device=DEVICE,
            refresh=False,
        )
        display(control_repl["aggregate"].sort_values("oracle_teacher_mae_mean", ascending=True))
        """
    ),
    md(
        """
        ## Evaluate Existing Runs On Pooled Center Bundle

        Cell nay tao mot pooled stable-center bundle tu tat ca oracle subsets da co,
        roi chấm lai cac checkpoint hien co theo metric Failure B.
        """
    ),
    code(
        r"""
        pooled_center_bundle = lab.build_pooled_center_bundle(
            pooled_unique=center_pool["unique_rows"],
            data_root=DATA_ROOT,
            center_thr=CENTER_CFG.oracle_center_thr,
            refresh=False,
            paths=paths,
        )
        existing_eval = lab.evaluate_failure_b_registry(
            registry=EXISTING_REGISTRY,
            pooled_center_bundle=pooled_center_bundle,
            oracle_bundle=ORACLE_BUNDLE,
            data_root=DATA_ROOT,
            train_cfg=EVAL_CFG,
            oracle_cfg=ORACLE_CFG,
            paths=paths,
            device=DEVICE,
            prefix="existing_failure_b",
        )
        display(existing_eval["primary"])
        """
    ),
    md(
        """
        ## Pilot Fixes For Failure B

        Hai pilot nay test hai huong khac nhau:

        - `P_B2_raw_center_strong`: neu Failure B chu yeu do objective pressure / gradient interference
        - `P_B1_proxy_center_weighted`: neu raw center labels bi ban, chi nen regularize tren trusted-center proxy

        Cell nay resume-safe:

        - train prediction cache ghi theo shard
        - pilot nao da co `best checkpoint` hop le se duoc skip
        """
    ),
    code(
        r"""
        pred_cache_manifest = lab.precompute_train_prediction_cache(
            checkpoint_path=L3_CHECKPOINT,
            data_root=DATA_ROOT,
            split="train",
            num_shards=PILOT_CFG.train_num_shards,
            paths=paths,
            device=DEVICE,
            batch_size=max(PILOT_CFG.batch_size, 1024),
            refresh=False,
        )

        pilot_results = {}
        for variant_name in ["P_B2_raw_center_strong", "P_B1_proxy_center_weighted"]:
            pilot_results[variant_name] = lab.run_failure_b_pilot(
                variant=PILOT_VARIANTS[variant_name],
                pilot_cfg=PILOT_CFG,
                data_root=DATA_ROOT,
                oracle_cfg=ORACLE_CFG,
                oracle_bundle=ORACLE_BUNDLE,
                pooled_center_bundle=pooled_center_bundle,
                center_lookup=center_lookup,
                pred_cache_manifest=pred_cache_manifest,
                paths=paths,
                device=DEVICE,
                refresh=False,
            )
        pilot_history = pd.concat(
            [result["history"].assign(variant=name) for name, result in pilot_results.items()],
            ignore_index=True,
        )
        display(pilot_history)
        """
    ),
    md(
        """
        ## Final Comparison

        Tong hop:

        - existing checkpoints
        - pilot checkpoints moi

        Va xep hang theo `failure_b_score`, khong theo `MSE`.
        """
    ),
    code(
        r"""
        pilot_registry = [
            {"label": name, "checkpoint": str(result["best_checkpoint"]), "target_scale": 600.0}
            for name, result in pilot_results.items()
        ]
        combined_registry = EXISTING_REGISTRY + pilot_registry
        combined_eval = lab.evaluate_failure_b_registry(
            registry=combined_registry,
            pooled_center_bundle=pooled_center_bundle,
            oracle_bundle=ORACLE_BUNDLE,
            data_root=DATA_ROOT,
            train_cfg=EVAL_CFG,
            oracle_cfg=ORACLE_CFG,
            paths=paths,
            device=DEVICE,
            prefix="combined_failure_b",
        )
        display(combined_eval["primary"])
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
