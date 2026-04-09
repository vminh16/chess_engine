from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from uuid import uuid4


NOTEBOOK_PATH = Path(__file__).resolve().parent / "root_cause_ablation_suite.ipynb"


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
        # Root-Cause Ablation Suite

        Notebook nay hien thuc truc tiep roadmap trong `docs/research/root_cause/teacher_root_cause_spec_2026-03-28.md`.

        Muc tieu:

        - khoa lai cac self-check de tranh sai mapping du lieu, remap target, oracle subset, va sampler
        - evaluate baseline bang cung metric scale-aware + oracle-aligned
        - chay cac ablation khuyen nghi:
          - `A1_curvature_compensated`
          - `A2_band_balanced`
          - `C1_scale800`
          - `C2_scale1200`
        - xuat comparison table de so sanh truoc khi quyet dinh run tiep theo

        Quy uoc quan trong:

        - moi phep so sanh giua cac run co `target_scale` khac nhau deu duoc quy ve cung mien chuan `y600`
        - metric trong "variant space" van duoc luu rieng de debug, nhung khong dung de chon winner
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
        EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "root_cause_ablation_suite"
        RUN_DIR = Path(r"C:\Users\USER\Downloads\dgrn_5m_v3_stage2_polish_run1")
        DATA_ROOT = PROJECT_ROOT / "data" / "process"

        if str(EXPERIMENT_DIR) not in sys.path:
            sys.path.insert(0, str(EXPERIMENT_DIR))

        import root_cause_ablation_helpers as lab

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

        ORACLE_CFG = lab.OracleEvalConfig()
        TRAIN_CFG = lab.TrainConfig(
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

        VARIANT_DICT = {variant.name: variant for variant in lab.build_recommended_variants()}
        SELECTED_VARIANTS = [
            "A1_curvature_compensated",
            "A2_band_balanced",
            "C1_scale800",
            "C2_scale1200",
        ]

        NOTEBOOK_CONFIG = {
            "oracle_cfg": asdict(ORACLE_CFG),
            "train_cfg": asdict(TRAIN_CFG),
            "selected_variants": SELECTED_VARIANTS,
        }
        lab.save_json(NOTEBOOK_CONFIG, paths["output_dir"] / "runtime_config.json")

        train_cfg_validation = lab.validate_train_config(TRAIN_CFG)
        oracle_cfg_validation = lab.validate_oracle_eval_config(ORACLE_CFG)
        variant_validations = [lab.validate_variant_config(VARIANT_DICT[name]) for name in SELECTED_VARIANTS]
        lab.save_json(train_cfg_validation, paths["reports_dir"] / "train_cfg_validation.json")
        lab.save_json(oracle_cfg_validation, paths["reports_dir"] / "oracle_cfg_validation.json")
        lab.save_json({"variants": variant_validations}, paths["reports_dir"] / "variant_cfg_validation.json")

        print("python:", sys.executable)
        print("device:", DEVICE)
        print("gpu_name:", torch.cuda.get_device_name(0))
        display(pd.DataFrame({"path_key": list(paths.keys()), "path_value": [str(v) for v in paths.values()]}))
        display(pd.DataFrame(variant_validations))
        """
    ),
    md(
        """
        ## Self-Check And Gradient Audit

        Cell nay fail-fast cho cac loi so dang:

        - checkpoint model khong con `forward_logits`
        - head khong dung `tanh`
        - remap target `600 -> c` sai tinh chat monotonic / identity
        - `oracle_subset_rows.csv` khong map dung ve shard goc
        - balanced sampler duplicate hoac bo mat sample

        Sau do no xuat `gradient_mass_audit` cho baseline va cac variant.
        """
    ),
    code(
        r"""
        benchmark = lab.benchmark_single_train_step(
            init_ckpt_path=CHECKPOINT,
            data_root=paths["data_root"],
            device=DEVICE,
            batch_size=TRAIN_CFG.batch_size,
            num_shards=TRAIN_CFG.benchmark_num_shards,
        )
        checkpoint_validation = lab.validate_checkpoint_model(CHECKPOINT, device=DEVICE)
        remap_validation = lab.validate_target_remap_logic([600.0, 800.0, 1200.0])
        oracle_mapping_validation = lab.validate_oracle_subset_mapping(ORACLE_CFG, data_root=paths["data_root"])
        sampler_validation = lab.validate_band_balanced_sampler(batch_size=TRAIN_CFG.batch_size)
        split_rows = [
            lab.summarize_split_layout(paths["data_root"], split_name)
            for split_name in ("train", "val", "test")
        ]

        lab.save_json(benchmark, paths["reports_dir"] / "runtime_benchmark.json")
        lab.save_json(checkpoint_validation, paths["reports_dir"] / "checkpoint_validation.json")
        lab.save_json(remap_validation, paths["reports_dir"] / "target_remap_validation.json")
        lab.save_json(oracle_mapping_validation, paths["reports_dir"] / "oracle_subset_mapping_validation.json")
        lab.save_json(sampler_validation, paths["reports_dir"] / "band_sampler_validation.json")
        lab.save_dataframe(pd.DataFrame(split_rows), paths["reports_dir"] / "split_summary.csv")

        gradient_tables = []
        for variant_name in ["A1_curvature_compensated", "A2_band_balanced", "C1_scale800", "C2_scale1200", "B1_center_penalty"]:
            variant = VARIANT_DICT[variant_name]
            grad_df = lab.compute_gradient_mass_profile(
                data_root=paths["data_root"],
                split="train",
                variant=variant,
                num_shards=None,
            )
            grad_df.insert(0, "variant", variant.name)
            gradient_tables.append(grad_df)
            lab.save_dataframe(grad_df, paths["reports_dir"] / f"gradient_mass_{variant.name}.csv")
        gradient_summary = pd.concat(gradient_tables, axis=0, ignore_index=True)
        lab.save_dataframe(gradient_summary, paths["reports_dir"] / "gradient_mass_summary.csv")

        display(pd.DataFrame([benchmark]))
        display(pd.DataFrame([checkpoint_validation]))
        display(pd.DataFrame([remap_validation]))
        display(pd.DataFrame([oracle_mapping_validation]))
        display(pd.DataFrame([sampler_validation]))
        display(pd.DataFrame(split_rows))
        display(gradient_summary)
        """
    ),
    md(
        """
        ## Baseline Snapshot

        Baseline duoc evaluate lai bang:

        - scale-aware val/test metrics trong mien chuan `y600`
        - oracle-subset metrics tren cung subset diagnostic, cung quy ve `y600` de so sanh cong bang voi cac run scale khac

        Tat ca run ablation sau do se duoc so voi baseline nay.
        """
    ),
    code(
        r"""
        ORACLE_BUNDLE = lab.load_oracle_subset_bundle(ORACLE_CFG, data_root=paths["data_root"])

        baseline = lab.baseline_snapshot(
            init_ckpt_path=CHECKPOINT,
            data_root=paths["data_root"],
            oracle_bundle=ORACLE_BUNDLE,
            oracle_cfg=ORACLE_CFG,
            train_cfg=TRAIN_CFG,
            paths=paths,
            device=DEVICE,
            target_scale=600.0,
        )

        baseline_preview = pd.DataFrame(
            [
                {
                    "label": "baseline",
                    "val_mse_0.7eq": baseline["val_eval"]["metrics"]["bands"]["0.70"]["mse"],
                    "test_mse_0.7eq": baseline["test_eval"]["metrics"]["bands"]["0.70"]["mse"],
                    "oracle_stable_0.7_slope": baseline["oracle_eval"]["summary"]["stable_0.7_slope"],
                    "oracle_center_amp_ratio": baseline["oracle_eval"]["summary"]["center_amp_ratio_eq_0.05"],
                    "oracle_center_false_0.1eq": baseline["oracle_eval"]["summary"]["center_false_pred0.1eq"],
                    "oracle_center_false_0.2eq": baseline["oracle_eval"]["summary"]["center_false_pred0.2eq"],
                    "oracle_midband_mae_sum_stable": baseline["oracle_eval"]["summary"]["midband_teacher_vs_oracle_mae_sum_stable"],
                    "oracle_gate_score": lab.oracle_gate_score(baseline["oracle_eval"]["summary"]),
                }
            ]
        )
        display(baseline_preview)
        """
    ),
    md(
        """
        ## Run Selected Variants

        Cell nay chay fine-tune ngan cho cac variant da chon.

        Luu y:
        - `A1` nham kiem tra curvature-compensated loss
        - `A2` nham kiem tra density skew / sampler
        - `C1/C2` nham kiem tra scale mismatch
        - `B1` da duoc implement trong helper, nhung khong chay mac dinh o notebook nay
        """
    ),
    code(
        r"""
        variant_runs = []
        for variant_name in SELECTED_VARIANTS:
            variant = VARIANT_DICT[variant_name]
            print(f"\n===== RUN {variant.name} =====")
            print(variant.description)
            result = lab.run_variant_finetune(
                init_ckpt_path=CHECKPOINT,
                data_root=paths["data_root"],
                variant=variant,
                train_cfg=TRAIN_CFG,
                oracle_cfg=ORACLE_CFG,
                oracle_bundle=ORACLE_BUNDLE,
                paths=paths,
                device=DEVICE,
            )
            variant_runs.append(result)
        """
    ),
    md(
        """
        ## Compare Results

        Bieu bang cuoi cung dat nhung metric gate quan trong can doi:

        - tat ca cot compare deu o mien chuan `y600`, khong do truc tiep tren thang rieng cua tung variant
        - `oracle_stable_0.7_slope`
        - `oracle_midband_mae_sum_stable`
        - `oracle_center_amp_ratio`
        - `oracle_center_false_0.1eq`
        - `oracle_center_false_0.2eq`
        - `oracle_gate_score`
        """
    ),
    code(
        r"""
        compare_df = lab.compare_runs_table(baseline, variant_runs)
        lab.save_dataframe(compare_df, paths["reports_dir"] / "compare_runs.csv")
        display(compare_df.sort_values("oracle_gate_score", ascending=True))
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
