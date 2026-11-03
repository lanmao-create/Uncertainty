"""Keep shared UQ visualizations synced while an already-running train job runs.

The normal training entry point now exports maps immediately on each new best.
Use this helper only for a Python process that was started before that change.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

import uq_config as cfg
from run_tri_uq import promote_visuals


def process_alive(pid: int) -> bool:
    if os.name == "nt":
        # Never use os.kill(pid, 0) on Windows: unlike POSIX, os.kill may call
        # TerminateProcess for signals other than the console control events.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
        kernel32.GetExitCodeProcess.restype = ctypes.c_bool
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        kernel32.CloseHandle.restype = ctypes.c_bool
        handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            return bool(ok and exit_code.value == 259)  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_history(run_dir: Path, mean_classes: int = 5) -> tuple[float | None, int]:
    history = run_dir / "history.jsonl"
    if not history.is_file():
        return None, 0
    baseline = None
    rows = 0
    with history.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows += 1
            if item.get(f"baseline_miou_{mean_classes}") is not None:
                baseline = float(item[f"baseline_miou_{mean_classes}"])
    return baseline, rows


def candidates(model: str, weight_dir: Path, mean_classes: int = 5):
    pattern = re.compile(rf"^{re.escape(model)}_UQ_epoch(\d+)_mIoU([0-9]+\.[0-9]+)(?:_|\.pth$)")
    found = []
    for path in weight_dir.glob(f"{model}_UQ_epoch*_mIoU*.pth"):
        match = pattern.match(path.name)
        if not match:
            continue
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
            score = float(state.get(f"val_miou_{mean_classes}", state.get("val_miou", match.group(2))))
            found.append((score, int(match.group(1)), path))
        except (OSError, RuntimeError, ValueError, TypeError, EOFError):
            continue  # May be observed during the short interval while torch.save writes.
    return sorted(found, reverse=True, key=lambda row: row[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('Vaihingen','Potsdam','LoveDA'), default=cfg.DATASET)
    parser.add_argument("--model", required=True, choices=("ABCNet", "CMTFNet", "UNetformer", "FTUNetformer", "DeepKANSeg"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--training-pid", required=True, type=int)
    parser.add_argument("--interval", type=int, default=20)
    args = parser.parse_args()

    run_dir = cfg.OUTPUT / args.dataset / args.model / "uq" / args.run_name
    if not run_dir.is_dir():
        parser.error(f'Run directory does not exist: {run_dir}')
    import utils as baseline_cfg
    baseline_cfg.configure_dataset(args.dataset)
    mean_classes = baseline_cfg.MEAN_CLASS_COUNT
    weight_dir = cfg.RESULTS_UQ / args.dataset / args.model
    visual_dir = run_dir / "test"
    shared_dir = cfg.ROOT / "UQV" / args.dataset / args.model
    log_path = run_dir / "live_visual_watch.log"
    promoted_score = -1.0
    existing_record = shared_dir / 'Record' / 'best_selection.json'
    if existing_record.is_file():
        previous = json.loads(existing_record.read_text(encoding='utf-8'))
        promoted_score = float(previous.get(f'validation_miou_{mean_classes}', -1.0))
    idle_after_exit = 0

    def log(message: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"[{stamp}] {message}\n")

    log(f"Watching PID {args.training_pid}; model={args.model}; shared maps={shared_dir}")
    while True:
        baseline, history_rows = read_history(run_dir, mean_classes)
        eligible = [item for item in candidates(args.model, weight_dir, mean_classes)
                    if baseline is not None and item[0] >= baseline]
        if eligible and eligible[0][0] > promoted_score:
            score, epoch, checkpoint = eligible[0]
            stable = checkpoint.stat().st_size > 0
            if stable:
                log(f"Exporting epoch {epoch}, mean MIoU={score:.6f}: {checkpoint.name}")
                command = [sys.executable, str(cfg.ROOT / "infer_uq.py"), "test",
                           "--dataset", args.dataset,
                           "--model", args.model, "--run-name", args.run_name,
                           "--checkpoint", str(checkpoint.resolve()), "--samples", str(cfg.EVAL_SAMPLES)]
                result = subprocess.run(command, cwd=cfg.ROOT, check=False)
                metrics_path = run_dir / "test_metrics.json"
                if result.returncode == 0 and metrics_path.is_file():
                    record_dir = visual_dir / "Record"
                    record_dir.mkdir(parents=True, exist_ok=True)
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    (record_dir / "best_selection.json").write_text(json.dumps({
                        "model": args.model,
                        "run_name": args.run_name,
                        "epoch": epoch,
                        "checkpoint": str(checkpoint.resolve()),
                        "dataset": args.dataset,
                        f"validation_miou_{mean_classes}": score,
                        f"baseline_miou_{mean_classes}": baseline,
                        f"map_metrics_miou_{mean_classes}": metrics.get(f"miou_{mean_classes}"),
                        "tiles": baseline_cfg.test_ids,
                    }, indent=2), encoding="utf-8")
                    promote_visuals(visual_dir, shared_dir)
                    promoted_score = score
                    log(f"Shared maps atomically replaced; score={score:.6f}")
                else:
                    log(f"Inference failed (exit={result.returncode}); will retry next poll")

        alive = process_alive(args.training_pid)
        if not alive:
            idle_after_exit += 1
            if idle_after_exit >= 3:
                log(f"Training process ended after {history_rows} recorded epochs; watcher stopped")
                return
        else:
            idle_after_exit = 0
        time.sleep(max(5, args.interval))


if __name__ == "__main__":
    main()
