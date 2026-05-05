from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.phase_b_offline_benchmark.offline_benchmark_helpers import (
    OfflineBenchmarkConfig,
    resolve_config_from_run,
    run_offline_benchmark,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase B offline benchmark on a selected checkpoint.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory containing reports/ and checkpoints/.")
    parser.add_argument(
        "--checkpoint-policy",
        choices=["best_gate", "best_any", "best_pareto_a", "best_pareto_b", "latest"],
        default="best_gate",
        help="Checkpoint policy when resolving from --run-dir.",
    )
    parser.add_argument("--candidate-checkpoint", type=Path, default=None, help="Explicit candidate checkpoint path.")
    parser.add_argument("--candidate-label", type=str, default="candidate", help="Candidate label in reports.")
    parser.add_argument("--reference-checkpoint", type=Path, default=None, help="Explicit reference checkpoint path.")
    parser.add_argument("--reference-label", type=str, default="reference", help="Reference label in reports.")
    parser.add_argument("--data-root", type=Path, default=None, help="Root of data/process.")
    parser.add_argument("--oracle-role-bundle-dir", type=Path, default=None, help="Path to oracle_role_bundle cache.")
    parser.add_argument("--pooled-center-bundle-dir", type=Path, default=None, help="Path to pooled_center_bundle cache.")
    parser.add_argument("--outputs-root", type=Path, default=None, help="Benchmark outputs root.")
    parser.add_argument("--benchmark-name", type=str, default="", help="Output benchmark name.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Override eval batch size.")
    parser.add_argument("--test-max-samples", type=int, default=None, help="Override test max samples.")
    parser.add_argument("--test-num-shards", type=int, default=None, help="Override test shard count.")
    parser.add_argument("--midband-rel-tol", type=float, default=0.05, help="Relative tolerance for oracle midband gate.")
    parser.add_argument("--slope-abs-tol", type=float, default=0.02, help="Absolute tolerance for oracle slope gate.")
    parser.add_argument("--center-strong-threshold", type=float, default=None, help="Optional strong-pass center threshold.")
    parser.add_argument("--no-bootstrap", action="store_true", help="Disable bootstrap compare vs reference.")
    parser.add_argument("--bootstrap-n", type=int, default=2000, help="Bootstrap replicate count.")
    parser.add_argument("--bootstrap-seed", type=int, default=123, help="Bootstrap RNG seed.")
    parser.add_argument("--bootstrap-ci-alpha", type=float, default=0.05, help="Bootstrap CI alpha.")
    parser.add_argument("--abs-calibration-bins", type=int, default=20, help="Number of |target| bins for calibration plots.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device. 'auto' selects cuda when available.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.run_dir is None and args.candidate_checkpoint is None:
        raise SystemExit("Either --run-dir or --candidate-checkpoint is required.")

    if args.run_dir is not None:
        config = resolve_config_from_run(
            run_dir=args.run_dir,
            checkpoint_policy=args.checkpoint_policy,
            benchmark_name=(args.benchmark_name or None),
            candidate_label=args.candidate_label,
            reference_label=args.reference_label,
            data_root=args.data_root,
            reference_checkpoint=args.reference_checkpoint,
            oracle_role_bundle_dir=args.oracle_role_bundle_dir,
            pooled_center_bundle_dir=args.pooled_center_bundle_dir,
            outputs_root=args.outputs_root,
            eval_batch_size=args.eval_batch_size,
            test_max_samples=args.test_max_samples,
            test_num_shards=args.test_num_shards,
            midband_rel_tol=args.midband_rel_tol,
            slope_abs_tol=args.slope_abs_tol,
            center_strong_threshold=args.center_strong_threshold,
            run_bootstrap_compare=not bool(args.no_bootstrap),
            bootstrap_n=args.bootstrap_n,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_ci_alpha=args.bootstrap_ci_alpha,
            abs_calibration_bins=args.abs_calibration_bins,
        )
    else:
        config = OfflineBenchmarkConfig(
            benchmark_name=(args.benchmark_name or "phase_b_offline_benchmark"),
            candidate_checkpoint=args.candidate_checkpoint,
            candidate_label=args.candidate_label,
            reference_checkpoint=args.reference_checkpoint,
            reference_label=args.reference_label,
            data_root=args.data_root or (REPO_ROOT / "data" / "process"),
            outputs_root=args.outputs_root or (REPO_ROOT / "evaluation" / "phase_b_offline_benchmark" / "outputs"),
            eval_batch_size=args.eval_batch_size,
            test_max_samples=args.test_max_samples,
            test_num_shards=args.test_num_shards,
            midband_rel_tol=args.midband_rel_tol,
            slope_abs_tol=args.slope_abs_tol,
            center_strong_threshold=args.center_strong_threshold,
            run_bootstrap_compare=not bool(args.no_bootstrap),
            bootstrap_n=args.bootstrap_n,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_ci_alpha=args.bootstrap_ci_alpha,
            abs_calibration_bins=args.abs_calibration_bins,
        )
        if args.oracle_role_bundle_dir is not None:
            config.oracle_role_bundle_dir = args.oracle_role_bundle_dir
        if args.pooled_center_bundle_dir is not None:
            config.pooled_center_bundle_dir = args.pooled_center_bundle_dir

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False.")

    artifacts = run_offline_benchmark(config, device=device)
    print(json.dumps(artifacts["decision_summary"], indent=2, ensure_ascii=False))
    print(json.dumps(artifacts["paths"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
