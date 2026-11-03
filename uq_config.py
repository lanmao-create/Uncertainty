"""Edit only this file when switching the default model or local data paths."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent

# MODEL = "UNetformer"
# MODEL = "FTUNetformer"
MODEL = "ABCNet"
# MODEL = "CMTFNet"
# MODEL = "DeepKANSeg"

DATASET = "Vaihingen"  # "Vaihingen", "Potsdam", or "LoveDA"

MODE = "Train"  # Train the UQ adapter; Test uses selected_checkpoint.json by default.

# Each model uses its highest-scoring native baseline checkpoint in this folder.
# Scores are read from the unchanged filename pattern: MODEL_epochN_SCORE[.pth].
# The folder is project-relative; no model's UQ weights are used for initialization.
BASELINE_CHECKPOINT_DIR = ROOT / "results_vaihingen"
BASELINE_MODELS = ("ABCNet", "CMTFNet", "UNetformer", "FTUNetformer", "DeepKANSeg")


def resolve_baseline_checkpoint(model: str, override: str | Path | None = None, dataset: str = DATASET) -> Path:
    """Return the highest-mIoU native checkpoint for ``model``."""
    if dataset not in ("Vaihingen", "Potsdam", "LoveDA"):
        raise ValueError(f"Unsupported UQ dataset: {dataset}")
    directory = BASELINE_CHECKPOINT_DIR / model if dataset == "Vaihingen" else ROOT / ("results_potsdam" if dataset == "Potsdam" else "results_loveda") / model
    if model not in BASELINE_MODELS:
        raise ValueError(f"Unknown baseline model: {model}")
    if override is not None:
        path = Path(override).expanduser()
        paths = [path] if path.is_absolute() else [path, ROOT / path]
        for candidate in paths:
            if candidate.is_file():
                return candidate.resolve()
            for suffix in (".pth", ".pt", ".ckpt"):
                suffixed = Path(str(candidate) + suffix)
                if suffixed.is_file():
                    return suffixed.resolve()
        raise FileNotFoundError(f"Explicit baseline checkpoint does not exist: {override}")

    pattern = re.compile(
        rf"^{re.escape(model)}_epoch(\d+)_([0-9]+(?:\.[0-9]+)?)(?:\.(?:pth|pt|ckpt))?$",
        re.IGNORECASE,
    )
    candidates = []
    if directory.is_dir():
        for path in directory.iterdir():
            if not path.is_file():
                continue
            match = pattern.match(path.name)
            if match:
                candidates.append((float(match.group(2)), int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(
            f"No native {model} baseline checkpoint found in {directory}. "
            f"Expected {model}_epochNN_SCORE with an optional .pth/.pt/.ckpt suffix. "
            "Copy the matching dataset/model baseline weight into this folder or pass --baseline-checkpoint."
        )
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2].resolve()

# Supply external images outside the Vaihingen dataset for OOD exposure.
# If empty, U_dist is still computed but its anomaly behavior is not trained.
OOD_DIR = ""
OUTPUT = ROOT / "uq_runs"
RESULTS_UQ = ROOT / "resultsUQ"
RUN_NAME = "s42"
SEED = 42

EPOCHS = 50
# Match the baseline dataset length: BATCH_SIZE * 1000 = 10,000 patches/epoch.
STEPS_PER_EPOCH = 1000
BATCH = 10
EVAL_BATCH = 10
PATCH = 256
EVAL_STRIDE = None  # Dataset default: Vaihingen=32, Potsdam/LoveDA=128
HEAD_WARMUP_EPOCHS = 5
HEAD_LR = 1e-3
BACKBONE_LR = 1e-4
TRAIN_SAMPLES = 2
EVAL_SAMPLES = 16
LOG_EVERY = 1  # Like the original train.py: print loss/accuracy for each batch.
DATA_WEIGHT_WARMUP = 10  # Separate from the 5-epoch head-only warmup.
DATA_WEIGHT_RAMP_END = 20
ETA_MAX = 0.5
LAMBDA_ID = 0.01
LAMBDA_OOD = 0.01
DIST_FOCUS_WEIGHT = 0.5  # Udist-based hard-pixel mining strength, ramped with eta.
MODEL_FOCUS_WEIGHT = 0.25  # Umodel-based MC disagreement mining strength.
PREVIEW_EVERY = 5  # Also export epoch 1 and the final epoch; 0 disables previews.
PREVIEW_TILE = None  # A training tile, not a selection/calibration/test tile.
PREVIEW_ROW = 256
PREVIEW_COL = 256

