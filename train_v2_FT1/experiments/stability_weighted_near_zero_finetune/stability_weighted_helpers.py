from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

try:
    import chess as pychess
except Exception:  # pragma: no cover
    pychess = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OLD_LAB_DIR = PROJECT_ROOT / "experiments" / "teacher_root_cause_lab"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(OLD_LAB_DIR) not in sys.path:
    sys.path.insert(0, str(OLD_LAB_DIR))

import teacher_root_cause_helpers as base_lab  # noqa: E402


@dataclass
class ExperimentPaths:
    project_root: str
    run_dir: str
    data_root: str
    experiment_dir: str
    output_dir: str
    plots_dir: str
    reports_dir: str
    checkpoints_dir: str
    cache_dir: str
    train_pred_cache_dir: str


@dataclass
class StabilityWeightConfig:
    near_zero_thr: float = 0.20
    calibration_abs_y_sample_edges: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20)
    sample_per_abs_y_band: int = 24
    stockfish_path: str = r"D:\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
    stockfish_threads: int = 1
    stockfish_hash_mb: int = 32
    stockfish_node_budgets: Tuple[int, ...] = (2_000, 8_000, 32_000)
    stockfish_command_pause_ms: int = 50
    stockfish_timeout_sec: float = 15.0
    calibration_seed: int = 123
    prediction_batch_size: int = 2560
    weight_abs_y_edges: Tuple[float, ...] = (0.0, 0.025, 0.05, 0.10, 0.15, 0.20)
    teacher_abs_err_quantiles: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
    smoothing_prior: float = 3.0
    weight_strength: float = 0.45
    weight_min: float = 0.55


@dataclass
class FineTuneConfig:
    lambda_y: float = 0.99
    z_loss_beta: float = 1.0
    z_huber_delta: float = 0.5
    target_clamp_eps: float = 1e-3
    learning_rate: float = 3e-6
    min_lr: float = 1e-6
    weight_decay: float = 2e-4
    grad_clip_norm: float = 1.0
    batch_size: int = 640
    epochs: int = 1
    log_every_steps: int = 200
    seed: int = 123
    eval_val_samples: int = 100_000
    eval_test_samples: int = 200_000
    eval_val_num_shards: int = 2
    eval_test_num_shards: int = 4
    train_num_shards: Optional[int] = None


def build_default_paths(
    run_dir: str | Path = r"C:\Users\USER\Downloads\dgrn_5m_v3_stage2_polish_run1",
    data_root: str | Path = r"C:\Users\USER\Desktop\chess_engine\data\process",
    experiment_dir: str | Path | None = None,
) -> Dict[str, Path]:
    if experiment_dir is None:
        experiment_dir = PROJECT_ROOT / "experiments" / "stability_weighted_near_zero_finetune"
    paths = {
        "project_root": PROJECT_ROOT,
        "run_dir": Path(run_dir),
        "data_root": Path(data_root),
        "experiment_dir": Path(experiment_dir),
    }
    paths["output_dir"] = paths["experiment_dir"] / "outputs"
    paths["plots_dir"] = paths["output_dir"] / "plots"
    paths["reports_dir"] = paths["output_dir"] / "reports"
    paths["checkpoints_dir"] = paths["output_dir"] / "checkpoints"
    paths["cache_dir"] = paths["output_dir"] / "cache"
    paths["train_pred_cache_dir"] = paths["cache_dir"] / "train_init_preds"
    for key in (
        "experiment_dir",
        "output_dir",
        "plots_dir",
        "reports_dir",
        "checkpoints_dir",
        "cache_dir",
        "train_pred_cache_dir",
    ):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def export_paths_json(paths: Dict[str, Path], out_path: Path) -> None:
    payload = ExperimentPaths(**{key: str(value) for key, value in paths.items()})
    out_path.write_text(json.dumps(asdict(payload), indent=2), encoding="utf-8")


save_json = base_lab.save_json
save_dataframe = base_lab.save_dataframe
set_global_seed = base_lab.set_global_seed
choose_device = base_lab.choose_device


def _is_strictly_increasing(values: Sequence[float | int]) -> bool:
    arr = list(values)
    return all(arr[idx] < arr[idx + 1] for idx in range(len(arr) - 1))


def validate_stability_weight_config(cfg: StabilityWeightConfig) -> Dict[str, object]:
    calibration_edges = tuple(float(x) for x in cfg.calibration_abs_y_sample_edges)
    weight_edges = tuple(float(x) for x in cfg.weight_abs_y_edges)
    node_budgets = tuple(int(x) for x in cfg.stockfish_node_budgets)
    err_quantiles = tuple(float(x) for x in cfg.teacher_abs_err_quantiles)
    issues: List[str] = []

    if cfg.near_zero_thr <= 0.0:
        issues.append("near_zero_thr must be positive")
    if len(calibration_edges) < 2 or not _is_strictly_increasing(calibration_edges):
        issues.append("calibration_abs_y_sample_edges must be strictly increasing")
    if len(weight_edges) < 2 or not _is_strictly_increasing(weight_edges):
        issues.append("weight_abs_y_edges must be strictly increasing")
    if calibration_edges and calibration_edges[0] != 0.0:
        issues.append("calibration_abs_y_sample_edges must start at 0.0")
    if weight_edges and weight_edges[0] != 0.0:
        issues.append("weight_abs_y_edges must start at 0.0")
    if calibration_edges and cfg.near_zero_thr > calibration_edges[-1] + 1e-12:
        issues.append("near_zero_thr exceeds calibration_abs_y_sample_edges upper bound")
    if weight_edges and cfg.near_zero_thr > weight_edges[-1] + 1e-12:
        issues.append("near_zero_thr exceeds weight_abs_y_edges upper bound")
    if len(node_budgets) < 2 or not _is_strictly_increasing(node_budgets):
        issues.append("stockfish_node_budgets must contain at least two strictly increasing budgets")
    if cfg.stockfish_threads != 1:
        issues.append("stockfish_threads must be 1 for deterministic proxy validation")
    if cfg.stockfish_hash_mb <= 0:
        issues.append("stockfish_hash_mb must be positive")
    if cfg.stockfish_command_pause_ms < 0:
        issues.append("stockfish_command_pause_ms must be non-negative")
    if cfg.stockfish_timeout_sec <= 0.0:
        issues.append("stockfish_timeout_sec must be positive")
    if cfg.sample_per_abs_y_band <= 0:
        issues.append("sample_per_abs_y_band must be positive")
    if cfg.prediction_batch_size <= 0:
        issues.append("prediction_batch_size must be positive")
    if not err_quantiles or not _is_strictly_increasing(err_quantiles):
        issues.append("teacher_abs_err_quantiles must be strictly increasing")
    if err_quantiles and (err_quantiles[0] <= 0.0 or err_quantiles[-1] >= 1.0):
        issues.append("teacher_abs_err_quantiles must stay inside (0, 1)")
    if not (0.0 < cfg.weight_min <= 1.0):
        issues.append("weight_min must be in (0, 1]")
    if not (0.0 <= cfg.weight_strength <= 1.0):
        issues.append("weight_strength must be in [0, 1]")
    if cfg.smoothing_prior < 0.0:
        issues.append("smoothing_prior must be non-negative")
    if not Path(cfg.stockfish_path).exists():
        issues.append(f"stockfish_path does not exist: {cfg.stockfish_path}")
    report = {
        "is_valid": bool(not issues),
        "near_zero_thr": float(cfg.near_zero_thr),
        "calibration_abs_y_sample_edges": [float(x) for x in calibration_edges],
        "weight_abs_y_edges": [float(x) for x in weight_edges],
        "stockfish_node_budgets": [int(x) for x in node_budgets],
        "teacher_abs_err_quantiles": [float(x) for x in err_quantiles],
        "issues": issues,
    }
    if issues:
        raise ValueError("Invalid StabilityWeightConfig: " + "; ".join(issues))
    return report


def validate_finetune_config(cfg: FineTuneConfig) -> Dict[str, object]:
    issues: List[str] = []
    if not (0.0 <= cfg.lambda_y <= 1.0):
        issues.append("lambda_y must be in [0, 1]")
    if cfg.z_loss_beta < 0.0:
        issues.append("z_loss_beta must be non-negative")
    if cfg.z_huber_delta <= 0.0:
        issues.append("z_huber_delta must be positive")
    if not (0.0 < cfg.target_clamp_eps < 0.5):
        issues.append("target_clamp_eps must be in (0, 0.5)")
    if cfg.learning_rate <= 0.0:
        issues.append("learning_rate must be positive")
    if cfg.min_lr <= 0.0:
        issues.append("min_lr must be positive")
    if cfg.min_lr > cfg.learning_rate:
        issues.append("min_lr must be <= learning_rate")
    if cfg.weight_decay < 0.0:
        issues.append("weight_decay must be non-negative")
    if cfg.grad_clip_norm is not None and cfg.grad_clip_norm <= 0.0:
        issues.append("grad_clip_norm must be positive when provided")
    if cfg.batch_size <= 0:
        issues.append("batch_size must be positive")
    if cfg.epochs <= 0:
        issues.append("epochs must be positive")
    if cfg.log_every_steps <= 0:
        issues.append("log_every_steps must be positive")
    if cfg.eval_val_samples <= 0 or cfg.eval_test_samples <= 0:
        issues.append("eval sample counts must be positive")
    for name in ("eval_val_num_shards", "eval_test_num_shards"):
        value = getattr(cfg, name)
        if value is not None and value <= 0:
            issues.append(f"{name} must be positive when provided")
    if cfg.train_num_shards is not None and cfg.train_num_shards <= 0:
        issues.append("train_num_shards must be positive when provided")
    report = {
        "is_valid": bool(not issues),
        "issues": issues,
        "batch_size": int(cfg.batch_size),
        "epochs": int(cfg.epochs),
        "learning_rate": float(cfg.learning_rate),
        "min_lr": float(cfg.min_lr),
    }
    if issues:
        raise ValueError("Invalid FineTuneConfig: " + "; ".join(issues))
    return report


def parse_shard_id(x_path: Path) -> int:
    return int(x_path.stem.split("_")[1])


def resolve_split_shards(
    data_root: str | Path,
    split: str,
    num_shards: Optional[int] = None,
) -> List[Tuple[int, Path, Path]]:
    pairs = base_lab.resolve_split_pairs(data_root, split)
    pairs = base_lab.select_pairs_evenly(pairs, num_shards)
    return [(parse_shard_id(x_path), x_path, y_path) for x_path, y_path in pairs]


def summarize_split_layout(
    data_root: str | Path,
    split: str,
    num_shards: Optional[int] = None,
) -> Dict[str, int]:
    shard_rows = resolve_split_shards(data_root, split, num_shards=num_shards)
    total_samples = int(sum(int(np.load(y_path, mmap_mode="r").shape[0]) for _, _, y_path in shard_rows))
    return {
        "split": split,
        "num_shards": int(len(shard_rows)),
        "samples": int(total_samples),
    }


def shard_prediction_path(cache_dir: Path, shard_id: int) -> Path:
    return cache_dir / f"pred_{shard_id:05d}.npy"


def benchmark_single_train_step(
    init_ckpt_path: str | Path,
    data_root: str | Path,
    device: torch.device,
    batch_size: int = 640,
    num_shards: Optional[int] = None,
) -> Dict[str, float]:
    model, _ = base_lab.load_model_from_checkpoint(init_ckpt_path, device=device)
    model.train()
    shard_rows = resolve_split_shards(data_root, "train", num_shards=num_shards)
    x_path, y_path = shard_rows[0][1:]
    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    xb = torch.from_numpy(np.array(X[:batch_size], dtype=np.float32, copy=True)).to(device)
    yb = torch.from_numpy(np.array(y[:batch_size], dtype=np.float32, copy=True)).to(device).view(-1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-6)
    opt.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        logits = model.forward_logits(xb).view(-1)
        pred = torch.tanh(logits)
        loss = F.mse_loss(pred, yb)
    loss.backward()
    opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    total_samples = int(sum(int(np.load(path, mmap_mode="r").shape[0]) for _, _, path in shard_rows))
    steps_per_epoch = math.ceil(total_samples / batch_size)
    return {
        "batch_size": int(batch_size),
        "step_time_sec": float(elapsed),
        "peak_mem_gb": float(torch.cuda.max_memory_allocated() / 1024**3) if device.type == "cuda" else 0.0,
        "steps_per_epoch": int(steps_per_epoch),
        "train_total_samples": int(total_samples),
        "train_num_shards": int(len(shard_rows)),
        "epoch_hours_estimate": float(elapsed * steps_per_epoch / 3600.0),
    }


def precompute_train_prediction_cache(
    init_ckpt_path: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    device: torch.device,
    batch_size: int,
    num_shards: Optional[int] = None,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    cache_dir = output_dir / "cache" / "train_init_preds"
    reports_dir = output_dir / "reports"
    cache_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model, _ = base_lab.load_model_from_checkpoint(init_ckpt_path, device=device)
    manifest_rows: List[dict] = []
    shard_rows = resolve_split_shards(data_root, "train", num_shards=num_shards)
    init_ckpt_path = str(Path(init_ckpt_path).resolve())
    manifest_path = reports_dir / "train_pred_cache_manifest.json"
    existing_manifest: Dict[str, object] = {}
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_compatible = bool(existing_manifest.get("init_ckpt_path") == init_ckpt_path)
    t0 = time.time()
    for rank, (shard_id, x_path, y_path) in enumerate(shard_rows):
        X = np.load(x_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r")
        pred_path = shard_prediction_path(cache_dir, shard_id)
        need_write = True
        if cache_compatible and pred_path.exists():
            cached = np.load(pred_path, mmap_mode="r")
            if cached.shape[0] == y.shape[0]:
                need_write = False
        if need_write:
            preds = base_lab.predict_array(
                model,
                np.asarray(X),
                device=device,
                batch_size=batch_size,
                use_amp=True,
                progress_name=f"train_pred_cache_{shard_id:05d}",
            )
            np.save(pred_path, preds.astype(np.float16))
        manifest_rows.append(
            {
                "shard_id": int(shard_id),
                "samples": int(y.shape[0]),
                "pred_path": str(pred_path),
                "cached": bool(not need_write),
            }
        )
        manifest = {
            "init_ckpt_path": init_ckpt_path,
            "num_shards": int(-1 if num_shards is None else num_shards),
            "batch_size": int(batch_size),
            "num_cached_shards": int(len(manifest_rows)),
            "cache_compatible_with_existing_manifest": bool(cache_compatible),
            "elapsed_sec": float(time.time() - t0),
        }
        save_dataframe(pd.DataFrame(manifest_rows), reports_dir / "train_pred_cache_manifest.csv")
        save_json(manifest, reports_dir / "train_pred_cache_manifest.json")
        if rank % 8 == 0:
            print(f"[train-pred-cache] shards={rank + 1}/{len(shard_rows)} elapsed={time.time() - t0:.1f}s")

    return {"manifest": manifest, "rows": pd.DataFrame(manifest_rows), "cache_dir": cache_dir}


def proportional_allocation(counts: Sequence[int], total: int) -> np.ndarray:
    counts_arr = np.asarray(counts, dtype=np.int64)
    total = int(min(total, int(counts_arr.sum())))
    if total <= 0 or counts_arr.sum() <= 0:
        return np.zeros_like(counts_arr)
    raw = counts_arr.astype(np.float64) * float(total) / float(counts_arr.sum())
    base = np.floor(raw).astype(np.int64)
    remainder = total - int(base.sum())
    if remainder > 0:
        frac = raw - base.astype(np.float64)
        order = np.argsort(-frac)
        for idx in order:
            if remainder <= 0:
                break
            if counts_arr[idx] > base[idx]:
                base[idx] += 1
                remainder -= 1
    if remainder > 0:
        for idx in np.argsort(-counts_arr):
            if remainder <= 0:
                break
            if counts_arr[idx] > base[idx]:
                base[idx] += 1
                remainder -= 1
    return base


def band_labels_from_edges(edges: Sequence[float]) -> List[str]:
    arr = list(edges)
    return [f"[{left:.3f},{right:.3f}]" for left, right in zip(arr[:-1], arr[1:])]


def sanitize_board_for_stockfish(board) -> object:
    square = base_lab.Square
    color = base_lab.Color
    piece_type = base_lab.PieceType

    def has_piece_at(sq: int, piece_color, ptype) -> bool:
        piece = board.get_piece(sq)
        return piece is not None and piece.color == piece_color and piece.type == ptype

    wk = has_piece_at(square.E1, color.WHITE, piece_type.KING)
    bk = has_piece_at(square.E8, color.BLACK, piece_type.KING)
    wr_h = has_piece_at(square.H1, color.WHITE, piece_type.ROOK)
    wr_a = has_piece_at(square.A1, color.WHITE, piece_type.ROOK)
    br_h = has_piece_at(square.H8, color.BLACK, piece_type.ROOK)
    br_a = has_piece_at(square.A8, color.BLACK, piece_type.ROOK)

    board.castling_rights[color.WHITE]["kingside"] = bool(board.castling_rights[color.WHITE]["kingside"] and wk and wr_h)
    board.castling_rights[color.WHITE]["queenside"] = bool(board.castling_rights[color.WHITE]["queenside"] and wk and wr_a)
    board.castling_rights[color.BLACK]["kingside"] = bool(board.castling_rights[color.BLACK]["kingside"] and bk and br_h)
    board.castling_rights[color.BLACK]["queenside"] = bool(board.castling_rights[color.BLACK]["queenside"] and bk and br_a)

    if pychess is not None:
        fen = board.to_fen()
        parsed = pychess.Board(fen)
        if parsed.status() & pychess.STATUS_BAD_CASTLING_RIGHTS:
            board.castling_rights[color.WHITE]["kingside"] = False
            board.castling_rights[color.WHITE]["queenside"] = False
            board.castling_rights[color.BLACK]["kingside"] = False
            board.castling_rights[color.BLACK]["queenside"] = False
        fen = board.to_fen()
        parsed = pychess.Board(fen)
        if parsed.status() & pychess.STATUS_INVALID_EP_SQUARE:
            board.en_passant_target = None
    return board


SCORE_RE = re.compile(r"score (cp|mate) (-?\d+)")


def cp_to_target(cp: float, clip_cp: float = 1200.0) -> float:
    clipped = float(np.clip(cp, -clip_cp, clip_cp))
    return float(np.tanh(clipped / 600.0))


def parse_stockfish_search_output(lines: Sequence[str]) -> Dict[str, object]:
    last_score_kind: Optional[str] = None
    last_score_value: Optional[int] = None
    bestmove: Optional[str] = None
    pv_move: Optional[str] = None
    for line in lines:
        match = SCORE_RE.search(line)
        if match:
            last_score_kind = str(match.group(1))
            last_score_value = int(match.group(2))
            pv_match = re.search(r"\spv\s+([a-h][1-8][a-h][1-8][nbrq]?)", line)
            if pv_match:
                pv_move = pv_match.group(1)
        if line.startswith("bestmove"):
            parts = line.split()
            if len(parts) >= 2:
                bestmove = parts[1]
    if last_score_kind is None or last_score_value is None or bestmove is None:
        raise RuntimeError(f"Failed to parse Stockfish output tail: {list(lines)[-12:]}")
    if last_score_kind == "mate":
        cp_equivalent = 1200.0 if last_score_value > 0 else -1200.0
    else:
        cp_equivalent = float(last_score_value)
    return {
        "score_kind": last_score_kind,
        "score_value": int(last_score_value),
        "cp_equivalent": float(cp_equivalent),
        "target_value": cp_to_target(cp_equivalent),
        "bestmove": bestmove,
        "pv_move": pv_move,
    }


def run_stockfish_probe_once(fen: str, node_budget: int, cfg: StabilityWeightConfig) -> Dict[str, object]:
    engine_path = Path(cfg.stockfish_path)
    if not engine_path.exists():
        raise FileNotFoundError(f"Missing Stockfish binary: {engine_path}")
    if pychess is not None:
        try:
            parsed_fen = pychess.Board(fen)
        except Exception as exc:
            raise ValueError(f"Invalid FEN for Stockfish probe: {fen}") from exc
        if not parsed_fen.is_valid():
            raise ValueError(f"Stockfish probe received invalid FEN status={parsed_fen.status()}: {fen}")
    pause_sec = float(cfg.stockfish_command_pause_ms) / 1000.0
    p = subprocess.Popen(
        [str(engine_path)],
        cwd=str(engine_path.parent),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    cmds = [
        "uci",
        "isready",
        f"setoption name Threads value {int(cfg.stockfish_threads)}",
        f"setoption name Hash value {int(cfg.stockfish_hash_mb)}",
        "setoption name MultiPV value 1",
        "isready",
        "ucinewgame",
        f"position fen {fen}",
        f"go nodes {int(node_budget)}",
    ]
    try:
        for cmd in cmds:
            assert p.stdin is not None
            p.stdin.write(cmd + "\n")
            p.stdin.flush()
            time.sleep(pause_sec)
        lines: List[str] = []
        t0 = time.time()
        assert p.stdout is not None
        while True:
            line = p.stdout.readline()
            if not line:
                break
            lines.append(line.rstrip("\r\n"))
            if line.startswith("bestmove"):
                break
            if time.time() - t0 > cfg.stockfish_timeout_sec:
                raise TimeoutError(f"Timed out waiting bestmove for nodes={node_budget}")
        parsed = parse_stockfish_search_output(lines)
        parsed["stdout_tail"] = lines[-12:]
        parsed["node_budget"] = int(node_budget)
        return parsed
    finally:
        if p.poll() is None:
            p.kill()


def run_stockfish_probe_curve(fen: str, cfg: StabilityWeightConfig) -> Dict[str, object]:
    rows: List[dict] = []
    for node_budget in cfg.stockfish_node_budgets:
        result = run_stockfish_probe_once(fen, int(node_budget), cfg)
        rows.append(result)
    target_values = np.asarray([row["target_value"] for row in rows], dtype=np.float64)
    cp_values = np.asarray([row["cp_equivalent"] for row in rows], dtype=np.float64)
    bestmoves = [str(row["bestmove"]) for row in rows]
    return {
        "rows": rows,
        "target_values": target_values,
        "cp_values": cp_values,
        "bestmoves": bestmoves,
        "target_range": float(np.max(target_values) - np.min(target_values)),
        "target_std": float(np.std(target_values)),
        "cp_range": float(np.max(cp_values) - np.min(cp_values)),
        "sign_flips": int(np.sum(np.sign(target_values[1:]) != np.sign(target_values[:-1]))),
        "bestmove_changes": int(sum(1 for prev, curr in zip(bestmoves[:-1], bestmoves[1:]) if prev != curr)),
    }


def benchmark_stockfish_proxy(cfg: StabilityWeightConfig, probe_fen: str | None = None) -> Dict[str, object]:
    if probe_fen is None:
        probe_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    per_query: List[dict] = []
    t0 = time.time()
    for node_budget in cfg.stockfish_node_budgets:
        q0 = time.time()
        result = run_stockfish_probe_once(probe_fen, int(node_budget), cfg)
        per_query.append(
            {
                "node_budget": int(node_budget),
                "elapsed_sec": float(time.time() - q0),
                "target_value": float(result["target_value"]),
                "bestmove": str(result["bestmove"]),
            }
        )
    total_elapsed = float(time.time() - t0)
    estimated_positions = int(len(cfg.calibration_abs_y_sample_edges) - 1) * int(cfg.sample_per_abs_y_band)
    estimated_total_sec = total_elapsed * estimated_positions
    return {
        "per_query": per_query,
        "one_position_total_sec": total_elapsed,
        "estimated_positions": estimated_positions,
        "estimated_total_min": float(estimated_total_sec / 60.0),
    }


def validate_stockfish_proxy(cfg: StabilityWeightConfig, probe_fen: str | None = None) -> Dict[str, object]:
    if probe_fen is None:
        probe_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    first = run_stockfish_probe_curve(probe_fen, cfg)
    second = run_stockfish_probe_curve(probe_fen, cfg)
    first_targets = first["target_values"].tolist()
    second_targets = second["target_values"].tolist()
    return {
        "probe_fen": probe_fen,
        "node_budgets": [int(x) for x in cfg.stockfish_node_budgets],
        "first_bestmoves": list(first["bestmoves"]),
        "second_bestmoves": list(second["bestmoves"]),
        "first_targets": [float(x) for x in first_targets],
        "second_targets": [float(x) for x in second_targets],
        "bestmove_match": bool(list(first["bestmoves"]) == list(second["bestmoves"])),
        "target_match": bool(np.allclose(first["target_values"], second["target_values"], atol=1e-9)),
    }


def validate_stockfish_compatible_decoding(
    data_root: str | Path,
    split: str,
    sample_count: int = 32,
    num_shards: int = 1,
) -> Dict[str, object]:
    if pychess is None:
        return {"sample_count": int(sample_count), "invalid_fens": None, "checked_with_python_chess": False}
    X, _ = base_lab.load_split_arrays(data_root, split, max_samples=sample_count, num_shards=num_shards)
    invalid_rows: List[dict] = []
    for idx, x in enumerate(X):
        board = sanitize_board_for_stockfish(base_lab.decode_tensor_to_board(x))
        fen = board.to_fen()
        try:
            parsed = pychess.Board(fen)
            if not parsed.is_valid():
                invalid_rows.append({"index": int(idx), "fen": fen, "status": int(parsed.status())})
        except Exception as exc:
            invalid_rows.append({"index": int(idx), "fen": fen, "error": str(exc)})
    return {
        "sample_count": int(sample_count),
        "invalid_fens": int(len(invalid_rows)),
        "checked_with_python_chess": True,
        "examples": invalid_rows[:5],
    }


def validate_stockfish_proxy_on_dataset_sample(
    data_root: str | Path,
    split: str,
    cfg: StabilityWeightConfig,
    sample_index: int = 0,
    num_shards: int = 1,
) -> Dict[str, object]:
    X, _ = base_lab.load_split_arrays(data_root, split, max_samples=max(sample_index + 1, 1), num_shards=num_shards)
    board = sanitize_board_for_stockfish(base_lab.decode_tensor_to_board(X[sample_index]))
    result = validate_stockfish_proxy(cfg, probe_fen=board.to_fen())
    result["split"] = split
    result["sample_index"] = int(sample_index)
    return result


def build_near_zero_calibration_subset(
    data_root: str | Path,
    train_pred_cache_dir: str | Path,
    cfg: StabilityWeightConfig,
    num_shards: Optional[int] = None,
) -> Dict[str, object]:
    edges = np.asarray(cfg.calibration_abs_y_sample_edges, dtype=np.float64)
    labels = band_labels_from_edges(edges)
    shard_rows = resolve_split_shards(data_root, "train", num_shards=num_shards)
    count_rows: List[dict] = []
    for shard_id, _, y_path in shard_rows:
        y = np.load(y_path, mmap_mode="r").astype(np.float32, copy=False)
        abs_y = np.abs(y.astype(np.float64))
        for band_idx, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            if band_idx == len(labels) - 1:
                mask = (abs_y >= left) & (abs_y <= right)
            else:
                mask = (abs_y >= left) & (abs_y < right)
            count_rows.append(
                {
                    "shard_id": int(shard_id),
                    "band_idx": int(band_idx),
                    "band_label": labels[band_idx],
                    "count": int(np.sum(mask)),
                }
            )
    counts_df = pd.DataFrame(count_rows)
    quota_rows: List[pd.DataFrame] = []
    for band_idx, band_label in enumerate(labels):
        band_df = counts_df[counts_df["band_idx"] == band_idx].copy().sort_values("shard_id")
        quotas = proportional_allocation(band_df["count"].to_numpy(dtype=np.int64), cfg.sample_per_abs_y_band)
        band_df["quota"] = quotas
        quota_rows.append(band_df)
    quota_df = pd.concat(quota_rows, ignore_index=True)
    rng = np.random.default_rng(cfg.calibration_seed)

    samples: List[dict] = []
    for shard_id, x_path, y_path in shard_rows:
        quota_slice = quota_df[quota_df["shard_id"] == shard_id]
        if quota_slice["quota"].sum() <= 0:
            continue
        X = np.load(x_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r").astype(np.float32, copy=False)
        pred_path = shard_prediction_path(Path(train_pred_cache_dir), shard_id)
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing prediction cache: {pred_path}")
        pred = np.load(pred_path, mmap_mode="r").astype(np.float32, copy=False)
        abs_y = np.abs(y.astype(np.float64))
        for _, row in quota_slice.iterrows():
            quota = int(row["quota"])
            if quota <= 0:
                continue
            band_idx = int(row["band_idx"])
            left = float(edges[band_idx])
            right = float(edges[band_idx + 1])
            if band_idx == len(labels) - 1:
                idx = np.flatnonzero((abs_y >= left) & (abs_y <= right))
            else:
                idx = np.flatnonzero((abs_y >= left) & (abs_y < right))
            if idx.size == 0:
                continue
            chosen = rng.choice(idx, size=min(quota, idx.size), replace=False)
            for local_idx in np.sort(chosen):
                samples.append(
                    {
                        "shard_id": int(shard_id),
                        "local_index": int(local_idx),
                        "band_idx": int(band_idx),
                        "band_label": row["band_label"],
                        "x": np.array(X[local_idx], dtype=np.uint8, copy=True),
                        "target_y": float(y[local_idx]),
                        "teacher_pred": float(pred[local_idx]),
                        "teacher_abs_err": float(abs(float(pred[local_idx]) - float(y[local_idx]))),
                    }
                )
    quota_summary = quota_df.groupby(["band_idx", "band_label"], as_index=False)[["count", "quota"]].sum()
    return {
        "samples": samples,
        "count_table": counts_df,
        "quota_table": quota_df,
        "quota_summary": quota_summary,
    }


def normalized_rank(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1 or np.allclose(arr, arr[0]):
        return np.zeros(arr.shape[0], dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(arr.shape[0], dtype=np.float64)
    return ranks / float(arr.shape[0] - 1)


def run_weight_calibration_stockfish_proxy(
    subset: Sequence[dict],
    cfg: StabilityWeightConfig,
    output_dir: str | Path,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    reports_dir = output_dir / "reports"
    plots_dir = output_dir / "plots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    for pos, sample in enumerate(subset):
        board = sanitize_board_for_stockfish(base_lab.decode_tensor_to_board(sample["x"]))
        curve = run_stockfish_probe_curve(board.to_fen(), cfg=cfg)
        row = {
            "shard_id": int(sample["shard_id"]),
            "local_index": int(sample["local_index"]),
            "band_idx": int(sample["band_idx"]),
            "band_label": sample["band_label"],
            "target_y": float(sample["target_y"]),
            "teacher_pred": float(sample["teacher_pred"]),
            "teacher_abs_err": float(sample["teacher_abs_err"]),
            "fen": board.to_fen(),
            "sf_final_target": float(curve["target_values"][-1]),
            "sf_final_cp": float(curve["cp_values"][-1]),
            "sf_target_range": float(curve["target_range"]),
            "sf_target_std": float(curve["target_std"]),
            "sf_cp_range": float(curve["cp_range"]),
            "sf_bestmove_changes": int(curve["bestmove_changes"]),
            "sf_sign_flips": int(curve["sign_flips"]),
            "sf_final_gap_to_train_target": float(abs(curve["target_values"][-1] - float(sample["target_y"]))),
        }
        for probe_row in curve["rows"]:
            node_budget = int(probe_row["node_budget"])
            row[f"sf_target_n{node_budget}"] = float(probe_row["target_value"])
            row[f"sf_cp_n{node_budget}"] = float(probe_row["cp_equivalent"])
            row[f"sf_bestmove_n{node_budget}"] = probe_row["bestmove"]
        rows.append(row)
        if pos % 16 == 0:
            print(f"[weight-calibration-stockfish] processed={pos + 1}/{len(subset)}")

    df = pd.DataFrame(rows)
    df["rank_target_range"] = normalized_rank(df["sf_target_range"].to_numpy(dtype=np.float64))
    df["rank_target_std"] = normalized_rank(df["sf_target_std"].to_numpy(dtype=np.float64))
    df["rank_bestmove_changes"] = normalized_rank(df["sf_bestmove_changes"].to_numpy(dtype=np.float64))
    df["rank_sign_flips"] = normalized_rank(df["sf_sign_flips"].to_numpy(dtype=np.float64))
    df["instability_score"] = df[["rank_target_range", "rank_target_std", "rank_bestmove_changes", "rank_sign_flips"]].mean(axis=1)
    df = df.sort_values("instability_score", ascending=True).reset_index(drop=True)
    third = max(1, df.shape[0] // 3)
    stable_df = df.head(third).copy()
    unstable_df = df.tail(third).copy()

    def subset_metrics(frame: pd.DataFrame) -> Dict[str, float]:
        y = frame["target_y"].to_numpy(dtype=np.float64)
        p = frame["teacher_pred"].to_numpy(dtype=np.float64)
        return {
            "n": int(frame.shape[0]),
            "mse": float(np.mean((p - y) ** 2)),
            "mae": float(np.mean(np.abs(p - y))),
            "false_decisive_0.3": float(np.mean(np.abs(p) >= 0.3)),
            "mean_instability_score": float(frame["instability_score"].mean()),
            "mean_sf_target_range": float(frame["sf_target_range"].mean()),
            "mean_sf_target_std": float(frame["sf_target_std"].mean()),
            "mean_sf_bestmove_changes": float(frame["sf_bestmove_changes"].mean()),
            "mean_sf_sign_flips": float(frame["sf_sign_flips"].mean()),
            "mean_sf_final_gap_to_train_target": float(frame["sf_final_gap_to_train_target"].mean()),
        }

    proxy_report = {
        "score_unit": "stockfish_cp_to_tanh_cp600",
        "stockfish_path": str(cfg.stockfish_path),
        "stockfish_threads": int(cfg.stockfish_threads),
        "stockfish_hash_mb": int(cfg.stockfish_hash_mb),
        "stockfish_node_budgets": [int(x) for x in cfg.stockfish_node_budgets],
        "num_rows": int(df.shape[0]),
        "stable": subset_metrics(stable_df),
        "unstable": subset_metrics(unstable_df),
        "proxy_diagnostics": {
            "target_range_unique_count": int(df["sf_target_range"].nunique()),
            "target_std_unique_count": int(df["sf_target_std"].nunique()),
            "bestmove_changes_nonzero_count": int(np.sum(df["sf_bestmove_changes"] > 0)),
            "sign_flip_nonzero_count": int(np.sum(df["sf_sign_flips"] > 0)),
            "is_informative": bool(
                (df["sf_target_range"].nunique() > 1)
                or (df["sf_target_std"].nunique() > 1)
                or np.any(df["sf_bestmove_changes"] > 0)
                or np.any(df["sf_sign_flips"] > 0)
            ),
        },
    }

    save_dataframe(df, reports_dir / "weight_calibration_stockfish_rows.csv")
    save_json(proxy_report, reports_dir / "weight_calibration_stockfish_report.json")

    plt.figure(figsize=(7, 4))
    plt.scatter(df["teacher_abs_err"], df["instability_score"], s=12, alpha=0.65)
    plt.xlabel("teacher abs error")
    plt.ylabel("instability score")
    plt.title("Stockfish instability score vs teacher abs error")
    plt.tight_layout()
    plt.savefig(plots_dir / "weight_calibration_stockfish_err_vs_instability.png", dpi=180)
    plt.close()
    return {"rows": df, "report": proxy_report}


def make_err_edges(values: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    edges = [0.0]
    for q in quantiles:
        edges.append(float(np.quantile(values, q)))
    edges.append(float(np.max(values) + 1e-6))
    out = []
    for value in edges:
        if not out or value > out[-1] + 1e-9:
            out.append(value)
    if len(out) < 2:
        out = [0.0, float(max(1e-6, np.max(values) + 1e-6))]
    return np.asarray(out, dtype=np.float64)


def build_stability_weight_lookup(
    calibration_rows: pd.DataFrame,
    cfg: StabilityWeightConfig,
    output_dir: str | Path,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    reports_dir = output_dir / "reports"
    plots_dir = output_dir / "plots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    abs_y_edges = np.asarray(cfg.weight_abs_y_edges, dtype=np.float64)
    err_edges = make_err_edges(calibration_rows["teacher_abs_err"].to_numpy(dtype=np.float64), cfg.teacher_abs_err_quantiles)
    n_y = len(abs_y_edges) - 1
    n_e = len(err_edges) - 1
    rows = calibration_rows.copy()
    rows["abs_y"] = np.abs(rows["target_y"].to_numpy(dtype=np.float64))
    rows["abs_y_bin_id"] = np.clip(
        np.digitize(rows["abs_y"].to_numpy(dtype=np.float64), abs_y_edges[1:-1], right=False),
        0,
        n_y - 1,
    )
    rows["err_bin_id"] = np.clip(
        np.digitize(rows["teacher_abs_err"].to_numpy(dtype=np.float64), err_edges[1:-1], right=False),
        0,
        n_e - 1,
    )
    global_mean = float(rows["instability_score"].mean())
    band_means = rows.groupby("abs_y_bin_id")["instability_score"].mean().to_dict()
    cell_means = rows.groupby(["abs_y_bin_id", "err_bin_id"])["instability_score"].mean().to_dict()
    cell_counts = rows.groupby(["abs_y_bin_id", "err_bin_id"]).size().to_dict()

    weight_matrix = np.ones((n_y, n_e), dtype=np.float32)
    lookup_rows: List[dict] = []
    for y_idx in range(n_y):
        band_prior = float(band_means.get(y_idx, global_mean))
        for e_idx in range(n_e):
            raw_mean = cell_means.get((y_idx, e_idx))
            count = int(cell_counts.get((y_idx, e_idx), 0))
            if raw_mean is None:
                smoothed = band_prior
            else:
                smoothed = float((count * raw_mean + cfg.smoothing_prior * band_prior) / (count + cfg.smoothing_prior))
            weight = float(np.clip(1.0 - cfg.weight_strength * smoothed, cfg.weight_min, 1.0))
            weight_matrix[y_idx, e_idx] = weight
            lookup_rows.append(
                {
                    "abs_y_bin_id": int(y_idx),
                    "abs_y_left": float(abs_y_edges[y_idx]),
                    "abs_y_right": float(abs_y_edges[y_idx + 1]),
                    "err_bin_id": int(e_idx),
                    "err_left": float(err_edges[e_idx]),
                    "err_right": float(err_edges[e_idx + 1]),
                    "count": count,
                    "raw_instability_mean": None if raw_mean is None else float(raw_mean),
                    "band_prior_instability": band_prior,
                    "smoothed_instability": smoothed,
                    "weight": weight,
                }
            )

    lookup_df = pd.DataFrame(lookup_rows)
    save_dataframe(lookup_df, reports_dir / "stability_weight_lookup.csv")
    report = {
        "near_zero_thr": float(cfg.near_zero_thr),
        "abs_y_edges": [float(x) for x in abs_y_edges],
        "teacher_abs_err_edges": [float(x) for x in err_edges],
        "global_instability_mean": global_mean,
        "weight_strength": float(cfg.weight_strength),
        "weight_min": float(cfg.weight_min),
        "smoothing_prior": float(cfg.smoothing_prior),
    }
    save_json(report, reports_dir / "stability_weight_lookup_report.json")

    heat = lookup_df.pivot(index="abs_y_bin_id", columns="err_bin_id", values="weight")
    plt.figure(figsize=(7, 4))
    plt.imshow(heat.to_numpy(dtype=np.float64), aspect="auto", cmap="viridis", vmin=cfg.weight_min, vmax=1.0)
    plt.colorbar(label="sample weight")
    plt.xlabel("teacher_abs_err bin")
    plt.ylabel("abs_y bin")
    plt.title("Stability weight lookup")
    plt.tight_layout()
    plt.savefig(plots_dir / "stability_weight_lookup_heatmap.png", dpi=180)
    plt.close()
    return {
        "weight_matrix": weight_matrix,
        "abs_y_edges": abs_y_edges,
        "teacher_abs_err_edges": err_edges,
        "cell_table": lookup_df,
        "report": report,
    }


def compute_sample_weights(y: np.ndarray, init_pred: np.ndarray, lookup: Dict[str, object], near_zero_thr: float) -> np.ndarray:
    weights = np.ones_like(y, dtype=np.float32)
    abs_y = np.abs(y.astype(np.float64))
    near_mask = abs_y <= near_zero_thr
    if not np.any(near_mask):
        return weights
    abs_err = np.abs(init_pred.astype(np.float64) - y.astype(np.float64))
    abs_y_edges = np.asarray(lookup["abs_y_edges"], dtype=np.float64)
    err_edges = np.asarray(lookup["teacher_abs_err_edges"], dtype=np.float64)
    matrix = np.asarray(lookup["weight_matrix"], dtype=np.float32)
    y_bins = np.clip(np.digitize(abs_y[near_mask], abs_y_edges[1:-1], right=False), 0, matrix.shape[0] - 1)
    e_bins = np.clip(np.digitize(abs_err[near_mask], err_edges[1:-1], right=False), 0, matrix.shape[1] - 1)
    weights[near_mask] = matrix[y_bins, e_bins]
    return weights


def audit_full_train_weight_distribution(
    data_root: str | Path,
    train_pred_cache_dir: str | Path,
    lookup: Dict[str, object],
    cfg: StabilityWeightConfig,
    output_dir: str | Path,
    num_shards: Optional[int] = None,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    reports_dir = output_dir / "reports"
    shard_rows = resolve_split_shards(data_root, "train", num_shards=num_shards)
    summary_rows: List[dict] = []
    total_samples = 0
    total_near_zero = 0
    weighted_sum = 0.0
    near_weighted_sum = 0.0
    for shard_id, _, y_path in shard_rows:
        y = np.load(y_path, mmap_mode="r").astype(np.float32, copy=False)
        pred = np.load(shard_prediction_path(Path(train_pred_cache_dir), shard_id), mmap_mode="r").astype(np.float32, copy=False)
        weights = compute_sample_weights(y, pred, lookup, near_zero_thr=cfg.near_zero_thr)
        near_mask = np.abs(y.astype(np.float64)) <= cfg.near_zero_thr
        total_samples += int(y.shape[0])
        total_near_zero += int(np.sum(near_mask))
        weighted_sum += float(np.sum(weights))
        near_weighted_sum += float(np.sum(weights[near_mask]))
        summary_rows.append(
            {
                "shard_id": int(shard_id),
                "samples": int(y.shape[0]),
                "near_zero_samples": int(np.sum(near_mask)),
                "mean_weight": float(np.mean(weights)),
                "mean_weight_near_zero": float(np.mean(weights[near_mask])) if np.any(near_mask) else None,
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    report = {
        "total_samples": int(total_samples),
        "total_near_zero_samples": int(total_near_zero),
        "near_zero_fraction": float(total_near_zero / max(total_samples, 1)),
        "mean_weight_all": float(weighted_sum / max(total_samples, 1)),
        "mean_weight_near_zero": float(near_weighted_sum / max(total_near_zero, 1)),
    }
    save_dataframe(summary_df, reports_dir / "full_train_weight_audit.csv")
    save_json(report, reports_dir / "full_train_weight_audit.json")
    return {"rows": summary_df, "report": report}


def target_to_logits(y: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.atanh(torch.clamp(y, -1.0 + eps, 1.0 - eps))


def huber_per_sample(residual: torch.Tensor, delta: float) -> torch.Tensor:
    abs_res = residual.abs()
    return torch.where(abs_res <= delta, 0.5 * residual * residual, delta * (abs_res - 0.5 * delta))


def compute_weighted_hybrid_terms(
    logits: torch.Tensor,
    y: torch.Tensor,
    sample_weight: torch.Tensor,
    cfg: FineTuneConfig,
) -> Dict[str, torch.Tensor]:
    pred = torch.tanh(logits)
    mse_per = (pred - y) ** 2
    y_logits = target_to_logits(y, eps=cfg.target_clamp_eps)
    residual = logits - y_logits
    y_clamped = torch.clamp(y, -1.0 + cfg.target_clamp_eps, 1.0 - cfg.target_clamp_eps)
    z_weights = torch.pow(torch.clamp(1.0 - y_clamped * y_clamped, min=cfg.target_clamp_eps), cfg.z_loss_beta)
    z_huber_per = z_weights * huber_per_sample(residual, cfg.z_huber_delta)
    objective_per = cfg.lambda_y * mse_per + (1.0 - cfg.lambda_y) * z_huber_per
    denom = torch.clamp(sample_weight.sum(), min=1e-6)
    return {
        "objective": torch.sum(sample_weight * objective_per) / denom,
        "weighted_mse": torch.sum(sample_weight * mse_per) / denom,
        "weighted_z_huber": torch.sum(sample_weight * z_huber_per) / denom,
        "plain_mse": torch.mean(mse_per),
        "pred": pred,
    }


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lower = name.lower()
        if param.ndim <= 1 or name.endswith(".bias") or ".bn" in lower or "norm" in lower:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def evaluate_model_on_split(
    model: nn.Module,
    data_root: str | Path,
    split: str,
    device: torch.device,
    max_samples: int,
    num_shards: Optional[int],
    batch_size: int,
) -> Dict[str, object]:
    model.eval()
    X, y = base_lab.load_split_arrays(data_root, split, max_samples=max_samples, num_shards=num_shards)
    pred = base_lab.predict_array(model, X, device=device, batch_size=batch_size, use_amp=True, progress_name=f"eval_{split}")
    metrics = base_lab.summarize_teacher_metrics(y.astype(np.float64), pred.astype(np.float64))
    return {"targets": y, "preds": pred, "metrics": metrics}


def gate_score_from_metrics(metrics: Dict[str, object]) -> float:
    return float(
        metrics["bands"]["0.7"]["mse"]
        + 0.5 * metrics["false_decisive"]["y<=0.1,p>=0.3"]["rate"]
        + 0.25 * metrics["max_midband_abs_cal_gap"]
        + 0.25 * max(0.0, 0.8 - metrics["bands"]["0.7"]["slope"])
    )


def run_stability_weighted_finetune(
    init_ckpt_path: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    device: torch.device,
    lookup: Dict[str, object],
    weight_cfg: StabilityWeightConfig,
    train_cfg: FineTuneConfig,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    reports_dir = output_dir / "reports"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(train_cfg.seed)
    model, init_ckpt = base_lab.load_model_from_checkpoint(init_ckpt_path, device=device)
    optimizer = build_optimizer(model, lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    shard_rows = resolve_split_shards(data_root, "train", num_shards=train_cfg.train_num_shards)
    total_samples = sum(int(np.load(y_path, mmap_mode="r").shape[0]) for _, _, y_path in shard_rows)
    total_steps = math.ceil(total_samples / train_cfg.batch_size) * max(train_cfg.epochs, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=train_cfg.min_lr)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    best_ckpt_path = checkpoints_dir / "stability_weighted_best.pt"
    latest_ckpt_path = checkpoints_dir / "stability_weighted_latest.pt"
    shard_audit_rows: List[dict] = []
    history_rows: List[dict] = []
    best_gate_score = float("inf")
    global_step = 0
    rng = np.random.default_rng(train_cfg.seed)

    for epoch in range(train_cfg.epochs):
        model.train()
        running = {"objective": 0.0, "weighted_mse": 0.0, "weighted_z_huber": 0.0, "plain_mse": 0.0, "n": 0}
        t0 = time.time()
        for shard_rank, (shard_id, x_path, y_path) in enumerate(shard_rows):
            X = np.load(x_path, mmap_mode="r")
            y = np.load(y_path, mmap_mode="r").astype(np.float32, copy=False)
            pred_path = shard_prediction_path(output_dir / "cache" / "train_init_preds", shard_id)
            init_pred = np.load(pred_path, mmap_mode="r").astype(np.float32, copy=False)
            weights = compute_sample_weights(y, init_pred, lookup, near_zero_thr=weight_cfg.near_zero_thr)
            near_mask = np.abs(y.astype(np.float64)) <= weight_cfg.near_zero_thr
            shard_audit_rows.append(
                {
                    "epoch": int(epoch),
                    "shard_id": int(shard_id),
                    "samples": int(y.shape[0]),
                    "near_zero_samples": int(np.sum(near_mask)),
                    "mean_weight": float(np.mean(weights)),
                    "mean_weight_near_zero": float(np.mean(weights[near_mask])) if np.any(near_mask) else None,
                }
            )
            order = rng.permutation(y.shape[0])
            for start in range(0, y.shape[0], train_cfg.batch_size):
                idx = order[start : start + train_cfg.batch_size]
                xb = torch.from_numpy(np.array(X[idx], dtype=np.float32, copy=True)).to(device, non_blocking=True)
                yb = torch.from_numpy(np.array(y[idx], dtype=np.float32, copy=True)).to(device, non_blocking=True).view(-1)
                wb = torch.from_numpy(np.array(weights[idx], dtype=np.float32, copy=True)).to(device, non_blocking=True).view(-1)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    logits = model.forward_logits(xb).view(-1)
                    terms = compute_weighted_hybrid_terms(logits, yb, wb, train_cfg)
                scaler.scale(terms["objective"]).backward()
                if train_cfg.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                bs = int(yb.numel())
                running["objective"] += float(terms["objective"].item()) * bs
                running["weighted_mse"] += float(terms["weighted_mse"].item()) * bs
                running["weighted_z_huber"] += float(terms["weighted_z_huber"].item()) * bs
                running["plain_mse"] += float(terms["plain_mse"].item()) * bs
                running["n"] += bs
                global_step += 1
                if global_step % train_cfg.log_every_steps == 0:
                    print(
                        f"[stability-ft][epoch={epoch}] step={global_step}/{total_steps} "
                        f"obj={running['objective'] / running['n']:.6f} "
                        f"weighted_mse={running['weighted_mse'] / running['n']:.6f} "
                        f"plain_mse={running['plain_mse'] / running['n']:.6f}"
                    )
            print(f"[stability-ft] finished shard {shard_rank + 1}/{len(shard_rows)}")

        val_eval = evaluate_model_on_split(
            model,
            data_root=data_root,
            split="val",
            device=device,
            max_samples=train_cfg.eval_val_samples,
            num_shards=train_cfg.eval_val_num_shards,
            batch_size=max(train_cfg.batch_size, 1024),
        )
        val_metrics = val_eval["metrics"]
        gate_score = gate_score_from_metrics(val_metrics)
        row = {
            "epoch": int(epoch),
            "train_objective": running["objective"] / running["n"],
            "train_weighted_mse": running["weighted_mse"] / running["n"],
            "train_weighted_z_huber": running["weighted_z_huber"] / running["n"],
            "train_plain_mse": running["plain_mse"] / running["n"],
            "val_mse_0.7": float(val_metrics["bands"]["0.7"]["mse"]),
            "val_slope_0.7": float(val_metrics["bands"]["0.7"]["slope"]),
            "val_false_0.1_0.3": float(val_metrics["false_decisive"]["y<=0.1,p>=0.3"]["rate"]),
            "val_false_0.2_0.4": float(val_metrics["false_decisive"]["y<=0.2,p>=0.4"]["rate"]),
            "val_center_spread_ratio_0.05": float(val_metrics["center_spread_ratio_0.05"]["ratio"]),
            "val_max_midband_abs_cal_gap": float(val_metrics["max_midband_abs_cal_gap"]),
            "gate_score": gate_score,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_time_sec": float(time.time() - t0),
        }
        history_rows.append(row)
        print(json.dumps(row, indent=2))

        payload = {
            "epoch": int(epoch),
            "history": history_rows,
            "weight_config": asdict(weight_cfg),
            "train_config": asdict(train_cfg),
            "lookup_report": lookup["report"],
            "init_checkpoint": str(init_ckpt_path),
            "base_checkpoint_epoch": init_ckpt.get("epoch"),
            "config": init_ckpt.get("config"),
            "model": model.state_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_metrics": val_metrics,
        }
        torch.save(payload, latest_ckpt_path)
        if gate_score < best_gate_score:
            best_gate_score = gate_score
            torch.save(payload, best_ckpt_path)

    history_df = pd.DataFrame(history_rows)
    shard_audit_df = pd.DataFrame(shard_audit_rows)
    save_dataframe(history_df, reports_dir / "stability_weighted_finetune_history.csv")
    save_json({"history": history_rows, "best_gate_score": best_gate_score}, reports_dir / "stability_weighted_finetune_history.json")
    save_dataframe(shard_audit_df, reports_dir / "stability_weighted_train_shard_audit.csv")
    return {
        "history": history_df,
        "best_checkpoint": best_ckpt_path,
        "latest_checkpoint": latest_ckpt_path,
        "best_gate_score": best_gate_score,
    }


def compare_metric_rows(label: str, metrics: Dict[str, object]) -> dict:
    return {
        "label": label,
        "overall_mse": metrics["overall"]["mse"],
        "overall_mae": metrics["overall"]["mae"],
        "mse_0.7": metrics["bands"]["0.7"]["mse"],
        "mae_0.7": metrics["bands"]["0.7"]["mae"],
        "slope_0.7": metrics["bands"]["0.7"]["slope"],
        "bias_0.7": metrics["bands"]["0.7"]["bias"],
        "mse_0.2": metrics["bands"]["0.2"]["mse"],
        "r2_0.2": metrics["bands"]["0.2"]["r2"],
        "false_0.1_0.3": metrics["false_decisive"]["y<=0.1,p>=0.3"]["rate"],
        "false_0.2_0.4": metrics["false_decisive"]["y<=0.2,p>=0.4"]["rate"],
        "center_spread_ratio_0.05": metrics["center_spread_ratio_0.05"]["ratio"],
        "max_midband_abs_cal_gap": metrics["max_midband_abs_cal_gap"],
        "gate_score": gate_score_from_metrics(metrics),
    }
