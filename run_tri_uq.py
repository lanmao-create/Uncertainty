"""Train/evaluate matched baseline and tri-source UQ on Vaihingen, Potsdam or LoveDA.

Run from any directory. Splits and label templates match baseline utils.py.
Validation and final metric tiles are shared, so the final metric is not independent.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import tifffile
from PIL import Image
from tqdm import tqdm

from tri_uq import LossConfig, PlugInSegmenter, build_backbone, objective, predict, predict_native_baseline
from uq_visualize import display_maximum, heatmap_scale, training_preview, save_uncertainty_heatmap
from uq_reporting import Reporter, confusion_metrics, epoch_report, evaluation_report, setup_report, stage
import baseline_geoseg as geoseg
import baseline_loveda as loveda
from uq_runtime import resolve as resolve_runtime


ROOT = Path(__file__).resolve().parent
RESULTS_UQ = ROOT / "resultsUQ"
UQ_VISUALS = ROOT / "UQV"
# Reuse baseline split, modality and filename templates without importing train.py.
import utils as baseline_cfg
DATASET = "Vaihingen"


def configure_uq_dataset(name="Vaihingen", data_root=None):
    global DATASET, DATA, TRAIN_IDS, SELECT_IDS, CAL_IDS, TEST_IDS, PALETTE, N_CLASSES, MEAN_CLASSES, METRIC_KEY
    if name not in ("Vaihingen", "Potsdam", "LoveDA"):
        raise ValueError("Supported UQ datasets: Vaihingen, Potsdam, LoveDA")
    baseline_cfg.configure_dataset(name, data_root)
    DATASET = name
    N_CLASSES = baseline_cfg.N_CLASSES
    MEAN_CLASSES = baseline_cfg.MEAN_CLASS_COUNT
    METRIC_KEY = f"miou_{MEAN_CLASSES}"
    DATA = Path(baseline_cfg.MAIN_FOLDER)
    TRAIN_IDS = tuple(baseline_cfg.train_ids)
    SELECT_IDS = tuple(baseline_cfg.test_ids)
    TEST_IDS = SELECT_IDS
    # Preserve historical Vaihingen calibration; Potsdam uses training ID tiles,
    # never claims an independent held-out calibration split.
    CAL_IDS = ("38",) if name == "Vaihingen" else TRAIN_IDS
    PALETTE = np.asarray([baseline_cfg.palette[i] for i in range(baseline_cfg.N_CLASSES)], dtype=np.uint8)


def image_path(tile: str) -> Path:
    return Path(baseline_cfg.DATA_FOLDER.format(tile))


def label_path(tile: str) -> Path:
    participant = Path(baseline_cfg.LABEL_FOLDER.format(tile))
    if DATASET == "Vaihingen" and not participant.is_file():
        return DATA / "ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE" / participant.name
    return participant


configure_uq_dataset()


def best_archived_uq_score(model_name: str, weight_dir: Path) -> float:
    """Read the highest score represented by this model's archived UQ weights."""
    pattern = re.compile(rf"^{re.escape(model_name)}_UQ_epoch\d+_mIoU([0-9]+\.[0-9]+)(?:_|\.pth$)")
    scores = []
    if weight_dir.is_dir():
        for path in weight_dir.glob(f"{model_name}_UQ_epoch*_mIoU*.pth"):
            match = pattern.match(path.name)
            if match:
                scores.append((float(match.group(1)), path))
    if not scores:
        return -1.0
    top_display_score = max(score for score, _ in scores)
    # Filename scores are rounded for readability. Recover the exact metric
    # from only the top-rounded checkpoint so strict comparisons stay correct.
    exact = []
    for display_score, path in scores:
        if display_score != top_display_score:
            continue
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            exact.append(float(checkpoint.get(f"val_miou_{MEAN_CLASSES}", checkpoint.get("val_miou", display_score))))
        except (OSError, RuntimeError, ValueError, TypeError):
            exact.append(display_score)
    return max(exact, default=top_display_score)


def promote_visuals(source: Path, destination: Path) -> None:
    """Replace the shared visualization set with one complete best-score snapshot."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + ".new")
    backup = destination.with_name(destination.name + ".old")
    for path in (staging, backup):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    staging.mkdir(parents=True)
    for path in source.rglob("*"):
        if path.is_file():
            target = staging / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    organize_visual_layout(staging)
    moved_old = False
    try:
        if destination.exists():
            destination.rename(backup)
            moved_old = True
        staging.rename(destination)
    except Exception:
        if moved_old and backup.exists() and not destination.exists():
            backup.rename(destination)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    if backup.exists():
        shutil.rmtree(backup)


def organize_visual_layout(folder: Path) -> None:
    """Keep predictions, numeric records, and the three heatmap families tidy."""
    prediction_dir = folder / "Prediction"
    record_dir = folder / "Record"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)
    for name in ("Udata", "Udist", "Umodel"):
        (folder / name).mkdir(parents=True, exist_ok=True)

    def place(source: Path, destination_dir: Path) -> None:
        destination = destination_dir / source.name
        if destination.exists():
            destination.unlink()
        shutil.move(str(source), str(destination))

    record_patterns = ("*_uq.npz", "final_metrics.txt", "final_metrics.json",
                       "heatmap_scales.json", "*_anomaly.png")
    for source in folder.glob("area*_prediction.png"):
        place(source, prediction_dir)
    for pattern in record_patterns:
        for source in folder.glob(pattern):
            place(source, record_dir)

    # Migrate files produced by older runs, and drop their redundant grayscale
    # quicklooks: the colored Udata/Udist/Umodel maps are the user-facing views.
    old_diagnostics = folder / "diagnostics"
    candidates = list(folder.glob("*_Udata.png")) + list(folder.glob("*_Udist.png"))
    candidates += list(folder.glob("*_Umodel.png")) + list(folder.glob("*_Utotal.png"))
    if old_diagnostics.is_dir():
        candidates.extend(path for path in old_diagnostics.iterdir() if path.is_file())
    for source in candidates:
        if source.name.endswith(("_Udata.png", "_Udist.png", "_Umodel.png", "_Utotal.png")):
            source.unlink(missing_ok=True)
        elif source.name.endswith("_prediction.png"):
            place(source, prediction_dir)
        else:
            place(source, record_dir)
    if old_diagnostics.is_dir() and not any(old_diagnostics.iterdir()):
        old_diagnostics.rmdir()


def read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in (".tif", ".tiff"):
        return tifffile.imread(path)
    return np.asarray(Image.open(path))


def encode_label(rgb: np.ndarray) -> np.ndarray:
    out = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for i, color in enumerate(PALETTE):
        out[np.all(rgb[..., :3] == color, axis=-1)] = i
    return out


def load_tile(tile: str) -> tuple[np.ndarray, np.ndarray]:
    x = read_image(image_path(tile))[..., :3].astype(np.float32) / 255.0
    y = baseline_cfg.read_label(label_path(tile))
    if x.shape[:2] != y.shape:
        raise ValueError(f"Image/label mismatch on tile {tile}")
    return x, y


def load_eval_tile(tile: str) -> tuple[np.ndarray, np.ndarray]:
    """Use the same complete eroded labels as baseline train.py evaluation."""
    x = read_image(image_path(tile))[..., :3].astype(np.float32) / 255.0
    label = Path(baseline_cfg.ERODED_FOLDER.format(tile))
    if not label.is_file():
        raise FileNotFoundError(f"Missing baseline evaluation label: {label}")
    y = baseline_cfg.read_label(label)
    if x.shape[:2] != y.shape:
        raise ValueError(f"Image/eroded-label mismatch on tile {tile}")
    return x, y


def crop(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if h < size or w < size:
        raise ValueError(f"Image {arr.shape} smaller than {size}")
    top = random.randrange(h - size + 1)
    left = random.randrange(w - size + 1)
    return arr[top:top+size, left:left+size]


class Patches(torch.utils.data.Dataset):
    def __init__(self, ids: tuple[str, ...], steps: int, batch: int,
                 size: int = 256, ood_dir: str | None = None, cache: bool = True):
        self.ids, self.length, self.size = ids, steps * batch, size
        self.cache = cache
        self.tiles: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for tile in ids:
            if not image_path(tile).is_file() or not label_path(tile).is_file():
                raise FileNotFoundError(f"Missing input or label for tile {tile}")
        self.ood_files: list[Path] = []
        if ood_dir:
            directory = Path(ood_dir).resolve()
            self.ood_files = sorted(p for p in directory.iterdir()
                                    if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"))
            if not self.ood_files:
                raise ValueError(f"No OOD images in {directory}")
            for path in self.ood_files:
                if DATA.resolve() in path.parents:
                    raise ValueError("OOD images must be external to the selected dataset")

    def __len__(self):
        return self.length

    def __getitem__(self, _index):
        tile = random.choice(self.ids)
        if self.cache and tile in self.tiles:
            x, y = self.tiles[tile]
        else:
            x, y = load_tile(tile)
            if self.cache:
                self.tiles[tile] = (x, y)
        h, w = y.shape
        r, c = random.randrange(h - self.size + 1), random.randrange(w - self.size + 1)
        x = x[r:r+self.size, c:c+self.size].copy()
        y = y[r:r+self.size, c:c+self.size].copy()
        if random.random() < 0.5:
            x, y = x[:, ::-1].copy(), y[:, ::-1].copy()
        if random.random() < 0.5:
            x, y = x[::-1].copy(), y[::-1].copy()
        ood = np.zeros_like(y, dtype=np.bool_)
        if self.ood_files and random.random() < 0.5:
            z = read_image(random.choice(self.ood_files))
            if z.ndim != 3 or z.shape[2] < 3:
                raise ValueError("OOD image must have at least 3 channels")
            z = z[..., :3]
            side = random.randint(self.size // 4, self.size // 2)
            z = crop(z, side).astype(np.float32) / 255.0
            top = random.randrange(self.size - side + 1)
            left = random.randrange(self.size - side + 1)
            x[top:top+side, left:left+side] = z
            # Exclude a seam band from both ID and OOD supervision.
            band = 4
            y[max(0,top-band):min(self.size,top+side+band),
              max(0,left-band):min(self.size,left+side+band)] = 255
            ood[top+band:top+side-band, left+band:left+side-band] = True
            y[ood] = 0  # valid placeholder, excluded from ID by ood mask
        return (torch.from_numpy(x.transpose(2,0,1)),
                torch.from_numpy(y.astype(np.int64)), torch.from_numpy(ood))


class GeoSegUQPatches(geoseg.GeoSegPatches):
    """Reuse the restored baseline's RGB augmentation and normalization."""
    def __init__(self, ids, steps, batch, size, ood_dir=None, cache=True):
        super().__init__(ids, cache=cache, window_size=(size,size),
                         steps_per_epoch=steps, batch_size=batch, loss='SEG')
        self.size = size
        # Reuse the legacy validator, including the external-data restriction.
        self.ood_files = Patches(ids,steps,batch,size,ood_dir,cache=False).ood_files

    def baseline_item(self, index):
        return super().__getitem__(index)

    def __getitem__(self, index):
        x, _, _, y = self.baseline_item(index)
        ood = torch.zeros_like(y,dtype=torch.bool)
        if self.ood_files and random.random() < .5:
            side = random.randint(self.size//4,self.size//2)
            image = read_image(random.choice(self.ood_files))
            if image.ndim != 3 or image.shape[2] < 3:
                raise ValueError('OOD image must have at least three channels')
            rgb = crop(image[...,:3],side).astype(np.float32)/255.0
            patch = torch.from_numpy(rgb.transpose(2,0,1).copy())[None]
            z = geoseg.normalize(patch)[0]
            top,left = random.randrange(self.size-side+1),random.randrange(self.size-side+1)
            x[:,top:top+side,left:left+side] = z
            y[max(0,top-4):min(self.size,top+side+4),max(0,left-4):min(self.size,left+side+4)] = 255
            ood[top+4:top+side-4,left+4:left+side-4] = True
            y[ood] = 0
        return x,y,ood


class LoveDAUQPatches(GeoSegUQPatches):
    """Use the LoveDA augmentation, retaining the existing optional OOD mixer."""
    def __init__(self, ids, steps, batch, size, ood_dir=None, cache=True):
        self.dataset = loveda.LoveDAPatches(ids,cache=cache,window_size=(size,size),
                                          steps_per_epoch=steps,batch_size=batch,loss='SEG')
        self.size = size
        self.ood_files = Patches(ids,steps,batch,size,ood_dir,cache=False).ood_files

    def __len__(self):
        return len(self.dataset)

    def baseline_item(self, index):
        return self.dataset[index]


def make_model(args):
    backbone = build_backbone(args.model, classes=N_CLASSES, pretrained=args.pretrained, recipe=args.recipe)
    if args.method == "uq":
        model = PlugInSegmenter(backbone, args.model, classes=N_CLASSES, dropout=args.dropout)
        model.baseline_recipe = args.recipe
        return model
    backbone.baseline_recipe = args.recipe
    return backbone


def run_dir(args) -> Path:
    path = Path(args.output) / args.dataset / args.model / args.method
    return path / args.run_name if args.run_name else path


def load_checkpoint(model, path: str, device: torch.device, allow_native: bool = False, log=print,
                    require_native: bool = False):
    obj = torch.load(path, map_location=device, weights_only=True)
    if isinstance(obj, dict) and obj.get("dataset", DATASET) != DATASET:
        raise ValueError(f"Checkpoint dataset {obj['dataset']} does not match selected dataset {DATASET}")
    if isinstance(obj,dict) and 'protocol' in obj and obj['protocol']['recipe'] != getattr(model,'baseline_recipe','legacy'):
        raise ValueError('Checkpoint recipe does not match the constructed UQ model')
    if isinstance(model, PlugInSegmenter) and isinstance(obj, dict) and "model" in obj:
        if obj.get("format_version") != 2:
            raise ValueError("This is a legacy standalone-head UQ checkpoint and is incompatible with the baseline-residual head. Start a new run-name and retrain.")
    state = obj
    if isinstance(obj, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                state = obj[key]
                break
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint {path} does not contain a state_dict")
    if state and all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    if require_native and any(key.startswith('head.') for key in state):
        raise ValueError('UQ training must start from a native baseline checkpoint, not a UQ or packaged baseline_fallback archive. Use results_<dataset>/<model> weights.')
    if allow_native and isinstance(model, PlugInSegmenter) and not any(
            key.startswith("head.") for key in state):
        missing, unexpected = model.load_state_dict(
            {"backbone." + key: value for key, value in state.items()}, strict=False)
        if unexpected or set(missing) != set(model.head.state_dict(prefix="head.").keys()):
            raise ValueError(f"Native checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        log("Backbone checkpoint loaded. Missing keys: 0, Unexpected keys: 0; new stochastic head initialized for training")
    else:
        if isinstance(model, PlugInSegmenter):
            # Older UQ checkpoints predate the entropy-guided context refiner.
            # Preserve their learned weights and initialize only the new
            # PUG-inspired branch when loading them for evaluation.
            missing, unexpected = model.load_state_dict(state, strict=False)
            allowed_missing = {key for key in missing
                               if key.startswith("head.context_branches.")
                               or key.startswith("head.context_fuse.")}
            if set(missing) != allowed_missing or unexpected:
                raise ValueError(f"UQ checkpoint mismatch: missing={missing}, unexpected={unexpected}")
            if allowed_missing:
                log("Loaded prior UQ checkpoint; initialized the new entropy-guided context refiner")
        else:
            model.load_state_dict(state, strict=True)
        if not (isinstance(model, PlugInSegmenter) and allowed_missing):
            log("Checkpoint loaded. Missing keys: 0, Unexpected keys: 0")
    return obj


def probabilities(model, x, args):
    if args.normalization == 'imagenet':
        x = geoseg.normalize(x)
    if getattr(args, 'native_baseline_mc', False):
        result = predict_native_baseline(model, x, samples=args.samples,
                                        numerics=getattr(args, 'uq_numerics', 'checkpoint'))
        return result['mean_p'], result
    if args.method == "uq":
        result = predict(model, x, samples=args.samples, pilot=args.pilot,
                         delta2=args.delta2, min_samples=args.min_samples,
                         max_samples=args.max_samples,
                         numerics=getattr(args, 'uq_numerics', 'checkpoint'))
        return result["mean_p"], result
    logits = model(x)
    p = logits.softmax(1)
    return p, {"mean_logits": logits, "mean_p": p,
               "prediction": p.argmax(1), "samples": 1}


def uncertainty_blend_window(size, kind='uniform'):
    """Positive overlap weights; normalize per image after accumulation."""
    if kind == 'uniform':
        return np.ones((size, size), dtype=np.float32)
    if kind == 'gaussian':
        axis = np.linspace(-1, 1, size, dtype=np.float32)
        return np.maximum(np.exp(-(axis[:, None]**2 + axis[None, :]**2) / (2*.35**2)), 1e-4)
    if kind == 'hann':
        # Near-zero edges, smoothly rising center. Positive floor covers the
        # outer image boundary without dividing by zero; never blur the maps.
        axis = np.maximum(np.hanning(size).astype(np.float32), 1e-6)
        return axis[:, None] * axis[None, :]
    raise ValueError(f'Unknown uncertainty blend: {kind}')


def eval_tiles(model, ids: tuple[str, ...], args, output: Path | None = None, report=None):
    # Keep MC-Dropout validation reproducible and match Test exactly. Restoring
    # RNG state also prevents validation from changing the training data/MC stream.
    devices = []
    if args.device.type == "cuda":
        devices = [args.device.index if args.device.index is not None
                   else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(getattr(args, "seed", 0))
        return _eval_tiles_impl(model, ids, args, output=output, report=report)


def export_tiles(model, ids, args, output, report=None):
    """Retain standard metrics/Prediction/NPZ; render Hann heatmaps separately.

    Each eval_tiles call restores its RNG stream. The extra rendering pass never
    changes training gradients, checkpoint selection, or the saved numeric maps.
    """
    from copy import copy
    metrics, scores = eval_tiles(model, ids, args, output=output, report=report)
    if args.method != 'uq' or getattr(args, 'heatmaps_only', False):
        return metrics, scores
    render = copy(args)
    render.heatmaps_only = True
    render.heatmap_components = ('data', 'dist', 'model')
    render.flat_heatmap_output = False
    render.uq_blend = 'hann'
    render.display_sigma = 0.0
    render.display_sharpness = 0.25
    render.eval_stride = min(args.eval_stride, args.eval_patch // 4)
    if report:
        report(f'Heatmap rendering only: Hann overlap, stride={render.eval_stride}, display sharpening=0.25; standard metrics and NPZ unchanged')
    eval_tiles(model, ids, render, output=output, report=report)
    record = Path(output) / 'Record'
    record.mkdir(parents=True, exist_ok=True)
    (record / 'heatmap_rendering.json').write_text(json.dumps({
        'blend': 'hann', 'display_sigma': 0, 'display_sharpness': 0.25,
        'sharpening_method': 'unsharp sigma=1px, clipped to existing 5x5 neighborhood range',
        'heatmap_stride': render.eval_stride, 'standard_eval_stride': args.eval_stride,
        'patch': args.eval_patch, 'samples': args.samples,
        'note': 'PNG heatmaps use a separate overlapping Hann rendering pass. Prediction, selection metrics and NPZ retain the standard evaluation protocol; PNGs are not pixelwise colorizations of those NPZ maps.',
    }, indent=2), encoding='utf-8')
    return metrics, scores


def _eval_tiles_impl(model, ids: tuple[str, ...], args, output: Path | None = None, report=None):
    model.eval()
    confusion = np.zeros((N_CLASSES,N_CLASSES), dtype=np.int64)
    nll_sum = brier_sum = valid_count = 0.0
    score_list = []
    inference_seconds = 0.0
    sample_counts_all = []
    if output:
        output.mkdir(parents=True, exist_ok=True)
        prediction_dir = output / "Prediction"
        record_dir = output / "Record"
        if not getattr(args, 'heatmaps_only', False):
            prediction_dir.mkdir(parents=True, exist_ok=True)
            record_dir.mkdir(parents=True, exist_ok=True)
    else:
        prediction_dir = record_dir = None
    threshold_path = output.parent / "threshold.json" if output else None
    threshold = json.loads(threshold_path.read_text(encoding="utf-8"))["threshold"] if (
        threshold_path and threshold_path.is_file()) else None
    with torch.no_grad():
        for tile_index, tile in enumerate(ids, 1):
            rgb, target = load_eval_tile(tile)
            h, w = target.shape
            prediction = np.zeros((h,w), dtype=np.uint8)
            score_sum = np.zeros((h,w,N_CLASSES), dtype=np.float32)
            probs = np.zeros((h,w,N_CLASSES), dtype=np.float32)
            overlap_count = np.zeros((h,w), dtype=np.uint16)
            uq_weight_sum = np.zeros((h,w), dtype=np.float32)
            blend = uncertainty_blend_window(args.eval_patch, getattr(args, 'uq_blend', 'uniform'))
            uq_maps = {name: np.zeros((h,w), dtype=np.float32)
                       for name in ("data", "dist", "model", "total")}
            sample_counts = []
            if h < args.eval_patch or w < args.eval_patch:
                raise ValueError(f"Tile {tile} is smaller than evaluation patch {args.eval_patch}")
            top_positions = [min(i, h - args.eval_patch) for i in range(0, h, args.eval_stride)]
            left_positions = [min(i, w - args.eval_patch) for i in range(0, w, args.eval_stride)]
            windows = [(top, left) for top in top_positions for left in left_positions]
            total_batches = math.ceil(len(windows) / args.eval_batch)
            with tqdm(total=total_batches, desc=f"Validation {tile_index}/{len(ids)}",
                      unit="batch", leave=False, dynamic_ncols=True) as progress:
                for start in range(0, len(windows), args.eval_batch):
                    coords = windows[start:start+args.eval_batch]
                    patches = np.stack([rgb[top:top+args.eval_patch, left:left+args.eval_patch]
                                        for top, left in coords])
                    x = torch.from_numpy(patches.transpose(0,3,1,2).copy()).to(args.device)
                    if args.device.type == "cuda":
                        torch.cuda.synchronize(args.device)
                    started = time.perf_counter()
                    p, result = probabilities(model, x, args)
                    if args.device.type == "cuda":
                        torch.cuda.synchronize(args.device)
                    inference_seconds += time.perf_counter() - started
                    for index, (top, left) in enumerate(coords):
                        score_sum[top:top+args.eval_patch,left:left+args.eval_patch] += (
                            result["mean_logits"][index].permute(1,2,0).cpu().numpy())
                        probs[top:top+args.eval_patch,left:left+args.eval_patch] += (
                            p[index].permute(1,2,0).cpu().numpy())
                        overlap_count[top:top+args.eval_patch,left:left+args.eval_patch] += 1
                        uq_weight_sum[top:top+args.eval_patch,left:left+args.eval_patch] += blend
                        sample_counts.append(result["samples"])
                        sample_counts_all.append(result["samples"])
                        for name in uq_maps:
                            if name in result:
                                uq_maps[name][top:top+args.eval_patch,left:left+args.eval_patch] += result[name][index].cpu().numpy() * blend
                    progress.update(1)
            covered = overlap_count > 0
            probs[covered] /= overlap_count[covered, None]
            for name in uq_maps:
                uq_maps[name][covered] /= uq_weight_sum[covered]
            prediction[:] = score_sum.argmax(2)
            valid = target < N_CLASSES
            flat = target[valid] * N_CLASSES + prediction[valid]
            confusion += np.bincount(flat, minlength=N_CLASSES*N_CLASSES).reshape(N_CLASSES,N_CLASSES)
            chosen = np.take_along_axis(probs, target.clip(0,N_CLASSES-1)[...,None], 2)[...,0]
            nll_sum += -np.log(np.clip(chosen[valid], 1e-12, 1)).sum()
            brier_sum += ((probs[valid] ** 2).sum(1) + 1 - 2 * chosen[valid]).sum()
            valid_count += valid.sum()
            if args.method == "uq" and args.mode == "calibrate":
                values = (uq_maps["dist"][valid] / math.log(N_CLASSES)).astype(np.float32)
                # Only calibration needs these scores. Bound LoveDA calibration
                # memory using a reproducible, evenly spaced per-image sample.
                if DATASET == "LoveDA" and values.size > 4096:
                    values = values[np.linspace(0, values.size - 1, 4096, dtype=np.int64)]
                score_list.append(values)
            if output:
                if not getattr(args, 'heatmaps_only', False):
                    Image.fromarray(PALETTE[prediction]).save(prediction_dir / f"area{tile}_prediction.png")
                if args.method == "uq":
                    if getattr(args, 'heatmaps_only', False):
                        # Preserve numeric evidence behind independently scaled
                        # PNGs: colors from different components are not an
                        # absolute comparison of uncertainty magnitude.
                        summary_path = output / 'heatmap_statistics.json'
                        summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.is_file() else {
                            'checkpoint': str(getattr(args, 'checkpoint', '')),
                            'protocol': getattr(args, 'protocol', {}),
                            'samples': args.samples,
                            'uq_numerics': getattr(args, 'uq_numerics', 'checkpoint'),
                            'predictor': 'native_baseline_terminal_dropout' if getattr(args, 'native_baseline_mc', False) else 'uq_head',
                            'dirichlet_concentration_calibrated': False if getattr(args, 'native_baseline_mc', False) else None,
                            'eval_patch': args.eval_patch, 'eval_stride': args.eval_stride,
                            'blend': getattr(args, 'uq_blend', 'uniform'),
                            'scale_policy': getattr(args, 'heatmap_scale', 'per-map'),
                            'reference_profile': getattr(args, 'heatmap_reference_profile', None),
                            'scale_note': 'Reference scales are fixed across models for each component; different components may have different maximums. Per-map scales change with each image.',
                            'units': 'nat', 'tiles': {}}
                        summary['tiles'][str(tile)] = {
                            name: {'minimum': float(v.min()), 'mean': float(v.mean()),
                                   'maximum': float(v.max()),
                                   'p50': float(np.percentile(v, 50)),
                                   'p99_5': display_maximum(v),
                                   'display_maximum_nat': heatmap_scale(v, name, args, N_CLASSES)[0],
                                   'display_upper_clipped_fraction': float(np.mean(v > heatmap_scale(v, name, args, N_CLASSES)[0])),
                                   'nonfinite_count': int((~np.isfinite(v)).sum())}
                            for name, v in uq_maps.items()}
                        summary['tiles'][str(tile)]['decomposition_max_abs_error'] = float(np.max(np.abs(
                            uq_maps['data'] + uq_maps['dist'] + uq_maps['model'] - uq_maps['total'])))
                        summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
                    if not getattr(args, 'heatmaps_only', False):
                        np.savez_compressed(record_dir / f"area{tile}_uq.npz", mean_p=probs,
                                            **uq_maps, samples=np.asarray(sample_counts))
                    for name in uq_maps:
                        if name in getattr(args, 'heatmap_components', ("data", "dist", "model")):
                            visual_scale, scale_label = heatmap_scale(uq_maps[name], name, args, N_CLASSES)
                            heatmap_dir = output if getattr(args, 'flat_heatmap_output', False) else output / f"U{name}"
                            heatmap_dir.mkdir(parents=True, exist_ok=True)
                            save_uncertainty_heatmap(
                                uq_maps[name], heatmap_dir / f"area{tile}_U{name}_heatmap.png",
                                maximum=visual_scale, title=f"U_{name}",
                                display_sigma=getattr(args, 'display_sigma', 0.0),
                                display_sharpness=getattr(args, 'display_sharpness', 0.0),
                                scale_label=scale_label)
                    if threshold is not None and not getattr(args, 'heatmaps_only', False):
                        anomaly = (uq_maps["dist"] / math.log(N_CLASSES) > threshold).astype(np.uint8) * 255
                        Image.fromarray(anomaly).save(record_dir / f"area{tile}_anomaly.png")
    metrics = {**confusion_metrics(confusion),
               "nll": float(nll_sum / max(1,valid_count)),
               "brier": float(brier_sum / max(1,valid_count)),
               "confusion": confusion.tolist(),
               "model_seconds": inference_seconds,
               "mean_samples": float(np.mean(sample_counts_all))}
    return metrics, score_list


def train(args):
    if args.model == "ABCNet" and args.batch < 2:
        raise ValueError("ABCNet's pooled BatchNorm requires --batch >= 2 for training")
    out = run_dir(args)
    # A fresh run currently has no resume path. Refuse to truncate valuable
    # logs/checkpoints when the default or a previously used run name collides.
    if any((out / name).exists() for name in
           ("history.jsonl", "best.pt", "best_uq_candidate.pt", "training.log")):
        raise FileExistsError(
            f"Run directory already contains training artifacts: {out}. "
            "Choose a new --run-name; this trainer does not resume runs."
        )
    out.mkdir(parents=True, exist_ok=True)
    uq_weight_dir = Path(args.results_uq).resolve() / args.dataset / args.model
    if args.method == "uq":
        uq_weight_dir.mkdir(parents=True, exist_ok=True)
        organize_visual_layout(UQ_VISUALS / args.dataset / args.model)
    archived_uq_best = best_archived_uq_score(args.model, uq_weight_dir)
    report = Reporter(out / "training.log")
    run_config = vars(args) | {"device":str(args.device),
        "train_ids":TRAIN_IDS,"selection_ids":SELECT_IDS,"calibration_ids":CAL_IDS,
        "test_ids":TEST_IDS,"ood_supervision":bool(args.ood_dir),
        "label_source":str(baseline_cfg.LABEL_FOLDER),
        "evaluation_label_source":str(baseline_cfg.ERODED_FOLDER),
        "input_channels":3,"input_modality":"IRRG" if args.dataset == "Vaihingen" else "RGB",
        "num_classes":N_CLASSES,"class_names":list(baseline_cfg.LABELS),"selection_metric":METRIC_KEY}
    dataset_class = GeoSegUQPatches if args.recipe=='geoseg' else LoveDAUQPatches if args.recipe=='loveda' else Patches
    dataset = dataset_class(TRAIN_IDS, args.steps_per_epoch, args.batch, args.patch,
                      args.ood_dir, cache=not args.no_cache and args.dataset != "LoveDA")
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True,
                                         num_workers=0)
    model = make_model(args).to(args.device)
    backbone = model.backbone if args.method == "uq" else model
    report(setup_report(run_config, sum(p.numel() for p in backbone.parameters()),
                        sum(p.numel() for p in model.head.parameters()) if args.method == "uq" else 0))
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.device, allow_native=True, log=report,
                        require_native=args.method == 'uq')
    baseline_miou = None
    if args.method == "uq":
        args.method = "baseline"
        baseline_metrics, _ = eval_tiles(model.backbone, SELECT_IDS, args, report=report)
        baseline_miou = baseline_metrics[f"miou_{MEAN_CLASSES}"]
        args.method = "uq"
        report(evaluation_report(baseline_metrics, "Baseline evaluation"))
    preview_crop = None
    if args.method == "uq" and args.preview_every:
        if args.preview_tile not in TRAIN_IDS:
            raise ValueError("Training previews must use a TRAIN_IDS tile")
        preview_rgb, preview_y = load_tile(args.preview_tile)
        r, c = args.preview_row, args.preview_col
        if r < 0 or c < 0 or r + args.patch > preview_y.shape[0] or c + args.patch > preview_y.shape[1]:
            raise ValueError("Training preview crop lies outside its tile")
        preview_crop = (preview_rgb[r:r+args.patch, c:c+args.patch].copy(),
                        preview_y[r:r+args.patch, c:c+args.patch].copy())
        del preview_rgb, preview_y
    if args.recipe in ('geoseg','loveda'):
        groups = {}
        wd = .01 if args.model=='UNetformer' else .00025
        for name,param in model.named_parameters():
            if args.method=='uq':
                lr = args.lr if name.startswith('head.') else (args.backbone_lr if name.startswith('backbone.backbone.') else args.decoder_lr)
            else:
                lr = args.backbone_lr if name.startswith('backbone.') else args.decoder_lr
            decay = 0.0 if name.endswith('bias') else wd
            groups.setdefault((lr,decay),[]).append(param)
        optimizer = geoseg.AdamWLookahead([{'params':p,'lr':lr,'weight_decay':decay}
                                          for (lr,decay),p in groups.items()], lr=args.lr)
    elif args.method == "uq":
        optimizer = torch.optim.SGD([
            {"params": model.backbone.parameters(), "lr": args.backbone_lr},
            {"params": model.head.parameters(), "lr": args.lr}],
            momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                    weight_decay=5e-4)
    milestones = sorted(set(max(1, round(args.epochs * ratio))
                            for ratio in (0.5, 0.7, 0.9)))
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=1e-6)
                 if args.recipe=='loveda' else torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=15,T_mult=2)
                 if args.recipe=='geoseg' else torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1))
    dice_loss = geoseg.DiceLoss(smooth=.05,ignore_index=255) if args.recipe in ('geoseg','loveda') else None
    aux_loss = geoseg.SoftCrossEntropyLoss(smooth_factor=.05,ignore_index=255) if args.recipe in ('geoseg','loveda') else None
    config = LossConfig(beta_boost=args.beta_boost, lambda_id=args.lambda_id,
                        plain_ce=args.plain_ce,
                        lambda_ood=args.lambda_ood,
                        dist_focus_weight=args.dist_focus_weight,
                        model_focus_weight=args.model_focus_weight, gamma=args.gamma,
                        eta_max=args.eta_max, warmup_epochs=args.warmup,
                        ramp_end_epoch=args.ramp_end)
    (out / "config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")
    (out / "history.jsonl").write_text("", encoding="utf-8")
    # Archive only improving UQ weights. The baseline remains in results_vaihingen;
    # selected_checkpoint.json points to it if no UQ candidate meets its score.
    best = baseline_miou if args.method == "uq" else -1.0
    best_epoch = 0 if args.method == "uq" else None
    candidate_best = -1.0
    candidate_best_epoch = None
    candidate_named = None
    fallback_named = None
    global_best = max(archived_uq_best, baseline_miou if baseline_miou is not None else -1.0)
    for epoch in range(1,args.epochs+1):
        model.train()
        if args.method == "uq":
            frozen = epoch <= args.head_warmup_epochs
            model.backbone.requires_grad_(not frozen)
            if frozen:
                model.backbone.eval()
        report(f"\nTrain (epoch {epoch}/{args.epochs})\nStage: {stage(epoch, run_config)}")
        report("Learning rates: " + ", ".join(f"group{i}={group['lr']:.2e}" for i,group in enumerate(optimizer.param_groups)) + ("; baseline frozen" if args.method=="uq" and frozen else ""))
        totals = []
        detail = []
        accuracies = []
        for batch_index, (x,y,ood) in enumerate(loader, 1):
            x,y,ood = x.to(args.device),y.to(args.device),ood.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            if args.method == "uq":
                logits = model(x, samples=args.train_samples)
                losses = objective(logits, y, ood, epoch, config)
                if dice_loss is not None:
                    target = y.masked_fill(ood,255)
                    losses['dice'] = dice_loss(logits.mean(0),target)
                    losses['aux'] = .4*aux_loss(model.aux_logits,target) if model.aux_logits is not None else logits.new_zeros(())
                    losses['total'] = losses['total'] + losses['dice'] + losses['aux']
                loss = losses["total"]
                prediction = logits.detach().float().clamp(-12,12).softmax(2).mean(0).argmax(1)
                detail.append({key:float(value.detach()) for key,value in losses.items()
                               if key != "total"})
            else:
                outputs = model(x)
                logits = outputs[0] if isinstance(outputs,tuple) else outputs
                loss = geoseg.SegmentationLoss()(outputs,y) if args.recipe in ('geoseg','loveda') else torch.nn.functional.cross_entropy(logits,y,ignore_index=255)
                prediction = logits.detach().argmax(1)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; inspect concentration and learning rate")
            loss.backward()
            optimizer.step()
            totals.append(float(loss.item()))
            valid = (y < N_CLASSES) & ~ood
            batch_accuracy = (prediction[valid] == y[valid]).float().mean().item() if valid.any() else 0.0
            accuracies.append(batch_accuracy)
            if batch_index == 1 or batch_index % args.log_every == 0 or batch_index == len(loader):
                seg = detail[-1]["seg"] if detail else totals[-1]
                id_loss = args.lambda_id * detail[-1]["id_kl"] if detail else 0
                ood_loss = args.lambda_ood * detail[-1]["ood_kl"] if detail else 0
                if ood_loss == 0:
                    ood_loss = 0.0
                extra = (f"\tLoss_Dice: {detail[-1]['dice']:.6f}\tLoss_aux: {detail[-1]['aux']:.6f}"
                         if detail and args.recipe in ('geoseg','loveda') else '')
                report(f"Train (epoch {epoch}/{args.epochs}) [{batch_index}/{len(loader)} ({100*batch_index/len(loader):.0f}%)]\tLoss_seg: {seg:.6f}\tLoss_ID_KL: {id_loss:.6f}\tLoss_OOD_KL: {ood_loss:.6f}{extra}\tLoss: {totals[-1]:.6f}\tAccuracy: {100*batch_accuracy:.2f}%")
        metrics, _ = eval_tiles(model, SELECT_IDS, args, report=report)
        if preview_crop and (epoch == 1 or epoch == args.epochs or epoch % args.preview_every == 0):
            # Exporting a preview must not alter the subsequent training RNG.
            cuda_devices = [args.device.index if args.device.index is not None else torch.cuda.current_device()] if args.device.type == "cuda" else []
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(args.seed)
                training_preview(model, *preview_crop, PALETTE,
                    out / "preview" / f"epoch{epoch:03d}", epoch, args.samples, bool(args.ood_dir),
                    normalize=geoseg.normalize if args.recipe in ("geoseg","loveda") else None)
            report(f"  - Saved training preview: {out / 'preview' / f'epoch{epoch:03d}' / 'overview.png'}")
        log = {"epoch":epoch,"lr":optimizer.param_groups[-1]["lr"],
               "backbone_lr":optimizer.param_groups[0]["lr"] if args.method == "uq" else optimizer.param_groups[0]["lr"],
               "backbone_frozen":epoch <= args.head_warmup_epochs if args.method == "uq" else False,
               "train_loss":float(np.mean(totals)),"train_accuracy":float(np.mean(accuracies)),**metrics}
        if detail:
            log.update({"train_" + key:float(np.mean([row[key] for row in detail]))
                        for key in detail[0]})
        log[f"baseline_miou_{MEAN_CLASSES}"] = baseline_miou
        with (out / "history.jsonl").open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(log) + "\n")
        score = metrics[f"miou_{MEAN_CLASSES}"]
        report(epoch_report(log, run_config, max(best, score)))
        if args.method == "uq" and epoch == 1 and score <= baseline_miou:
            # Export the baseline without disturbing learned parameters or RNG.
            training_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            cuda_devices = [args.device.index if args.device.index is not None else torch.cuda.current_device()] if args.device.type == "cuda" else []
            try:
                with torch.random.fork_rng(devices=cuda_devices):
                    load_checkpoint(model, args.checkpoint, args.device, allow_native=True, log=report)
                    torch.nn.init.zeros_(model.head.classifier.weight)
                    torch.nn.init.zeros_(model.head.classifier.bias)
                    fallback_named = uq_weight_dir / f"{args.model}_UQ_epoch1_mIoU{baseline_miou:.4f}.pth"
                    if fallback_named.exists():
                        fallback_named = uq_weight_dir / f"{args.model}_UQ_epoch1_mIoU{baseline_miou:.4f}_{args.run_name}_{time.time_ns()}.pth"
                    torch.save({"format_version": 2, "protocol": args.protocol,
                                "dataset": args.dataset, "model": model.state_dict(),
                                "epoch": 1, "val_miou": baseline_miou,
                                "checkpoint_kind": "baseline_fallback",
                                "source_checkpoint": str(args.checkpoint),
                                "uncertainty_note": "Zero residual baseline; uncalibrated Dirichlet concentration; model uncertainty is zero."}, fallback_named)
                    visual_dir = out / "visuals"
                    fallback_metrics, _ = export_tiles(model, TEST_IDS, args, output=visual_dir, report=report)
                    selection = {"checkpoint": str(fallback_named.resolve()),
                                 "checkpoint_kind": "baseline_fallback", "epoch": 1,
                                 "source_checkpoint": str(args.checkpoint),
                                 f"miou_{MEAN_CLASSES}": baseline_miou,
                                 "metrics": fallback_metrics,
                                 "uncertainty_note": "Zero residual baseline; model uncertainty is zero; Dirichlet concentration is not calibrated."}
                    record = visual_dir / "Record"
                    record.mkdir(parents=True, exist_ok=True)
                    (record / "best_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
                    (record / "final_metrics.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
                    (record / "final_metrics.txt").write_text(evaluation_report(baseline_metrics, "Baseline fallback metrics") + "\n", encoding="utf-8")
                    shared = UQ_VISUALS / args.dataset / args.model
                    if baseline_miou >= archived_uq_best:
                        promote_visuals(visual_dir, shared)
                    (out / "selected_checkpoint.json").write_text(json.dumps({
                        **{k: v for k, v in selection.items() if k != "metrics"},
                        "kind": "baseline_fallback", "protocol": args.protocol,
                        "dataset": args.dataset}, indent=2), encoding="utf-8")
                    report(f"  - Epoch 1 below baseline; saved baseline fallback: {fallback_named}")
            finally:
                model.load_state_dict(training_state, strict=True)
                del training_state
        if args.method == "uq" and score > candidate_best:
            candidate_best = score
            candidate_best_epoch = epoch
            # Keep the learned probabilistic head even if baseline still wins.
            # This diagnostic archive never changes official checkpoint selection.
            trained_candidate = out / 'best_trained_uq.pth'
            trained_candidate_temp = out / 'best_trained_uq.pth.tmp'
            torch.save({"format_version": 2, "protocol": args.protocol,
                        "dataset": args.dataset, "model": model.state_dict(),
                        "epoch": epoch, "val_miou": score,
                        "checkpoint_kind": "trained_uq_candidate",
                        "source_checkpoint": str(args.checkpoint),
                        "beats_baseline": baseline_miou is None or score > baseline_miou}, trained_candidate_temp)
            trained_candidate_temp.replace(trained_candidate)
            (out / 'best_trained_uq.json').write_text(json.dumps({
                "checkpoint": str(trained_candidate.resolve()),
                "epoch": epoch, f"miou_{MEAN_CLASSES}": score,
                f"baseline_miou_{MEAN_CLASSES}": baseline_miou,
                "beats_baseline": baseline_miou is None or score > baseline_miou,
                "note": "Learned UQ head for diagnosis; separate from baseline-protected selected_checkpoint.json.",
            }, indent=2), encoding='utf-8')
            report(f"  - Best trained UQ candidate retained: {trained_candidate} (mean MIoU {score:.4f})")
            # Keep the readable epoch/best-score checkpoint format, but do not
            # archive a candidate that already fails the baseline floor.
            candidate_meets_floor_now = baseline_miou is None or score > baseline_miou
            if candidate_meets_floor_now:
                candidate_checkpoint = {"format_version": 2, "protocol":args.protocol, "dataset":args.dataset, "model": model.state_dict(), "epoch": epoch,
                                        "val_miou": score, f"val_miou_{MEAN_CLASSES}": score}
                base_name = f"{args.model}_UQ_epoch{epoch}_mIoU{score:.4f}.pth"
                candidate_named = uq_weight_dir / base_name
                if candidate_named.exists():
                    run_tag = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in args.run_name)
                    candidate_named = uq_weight_dir / f"{args.model}_UQ_epoch{epoch}_mIoU{score:.4f}_{run_tag}.pth"
                    suffix = 2
                    while candidate_named.exists():
                        candidate_named = uq_weight_dir / f"{args.model}_UQ_epoch{epoch}_mIoU{score:.4f}_{run_tag}_{suffix}.pth"
                        suffix += 1
                torch.save(candidate_checkpoint, candidate_named)
                (out / "selected_checkpoint.json").write_text(json.dumps({
                    "checkpoint": str(candidate_named.resolve()), "protocol": args.protocol,
                    "dataset": args.dataset, "kind": "uq", "epoch": epoch,
                    f"miou_{MEAN_CLASSES}": score,
                    f"baseline_miou_{MEAN_CLASSES}": baseline_miou,
                }, indent=2), encoding="utf-8")
                report(f"  - New best UQ score: {score:.4f}; saved {candidate_named}")
            else:
                candidate_named = None
        if score > best:
            best = score
            best_epoch = epoch
            report(f"  - Updated best score: {best:.4f} (epoch {epoch})")
            # Keep the shared maps synchronized with the best model while
            # training is still running. The baseline score is the floor, so
            # only a UQ checkpoint that beats both baseline and the previous
            # project-wide best is exported.
            if (args.method == "uq" and candidate_named is not None
                    and score > global_best):
                visual_dir = out / "visuals"
                visual_dir.mkdir(parents=True, exist_ok=True)
                report(f"  - Exporting new best checkpoint maps during training to {visual_dir}")
                interim_metrics, _ = export_tiles(model, TEST_IDS, args,
                                                output=visual_dir, report=report)
                interim_record = visual_dir / "Record"
                interim_record.mkdir(parents=True, exist_ok=True)
                (interim_record / "best_selection.json").write_text(json.dumps({
                    "model": args.model,
                    "run_name": args.run_name,
                    "epoch": epoch,
                    "checkpoint": str(candidate_named.resolve()),
                    f"validation_miou_{MEAN_CLASSES}": score,
                    f"baseline_miou_{MEAN_CLASSES}": baseline_miou,
                    f"map_metrics_miou_{MEAN_CLASSES}": interim_metrics[f"miou_{MEAN_CLASSES}"],
                    "tiles": TEST_IDS,
                }, indent=2), encoding="utf-8")
                (interim_record / "final_metrics.txt").write_text(
                    evaluation_report(interim_metrics, "Current best checkpoint metrics") + "\n",
                    encoding="utf-8")
                (interim_record / "final_metrics.json").write_text(json.dumps({
                    "model": args.model,
                    "run_name": args.run_name,
                    "checkpoint": str(candidate_named.resolve()),
                    "checkpoint_kind": "uq",
                    f"baseline_miou_{MEAN_CLASSES}": baseline_miou,
                    f"selected_miou_{MEAN_CLASSES}": interim_metrics[f"miou_{MEAN_CLASSES}"],
                    "tiles": TEST_IDS,
                    "metrics": interim_metrics,
                    "uncertainty_heatmap_scale_nat": [0.0, math.log(N_CLASSES)],
                    "heatmap_palette": "deep blue low -> cyan -> yellow -> bright red high",
                }, indent=2), encoding="utf-8")
                shared_visual_dir = UQ_VISUALS / args.dataset / args.model
                promote_visuals(visual_dir, shared_visual_dir)
                global_best = score
                (uq_weight_dir / "best_uq.json").write_text(json.dumps({
                    "model": args.model,
                    "checkpoint": str(candidate_named.resolve()),
                    "epoch": epoch,
                    f"miou_{MEAN_CLASSES}": score,
                    f"baseline_miou_{MEAN_CLASSES}": baseline_miou,
                    "run_name": args.run_name,
                    "visuals": str(shared_visual_dir.resolve()),
                }, indent=2), encoding="utf-8")
                report(f"  - Shared best maps replaced at epoch {epoch} (mean MIoU {score:.6f})")
        else:
            report(f"  - Best checkpoint unchanged; score={best:.4f}")
        scheduler.step()
    if best_epoch is not None:
        meets_baseline_floor = (baseline_miou is None or best >= baseline_miou)
        if args.method == "uq":
            candidate_meets_floor = (baseline_miou is None or candidate_best >= baseline_miou)
            if candidate_meets_floor and candidate_named:
                selected_named = candidate_named
                checkpoint_kind = "uq"
                selected_score = candidate_best
                selected_epoch = candidate_best_epoch
            else:
                selected_named = fallback_named or Path(args.checkpoint).resolve()
                checkpoint_kind = "baseline_fallback"
                selected_score = baseline_miou
                selected_epoch = 1 if fallback_named else 0
        else:
            selected_named = out / f"{args.model}_BASELINE_epoch{best_epoch}_mIoU{best:.4f}.pth"
            candidate_best, candidate_best_epoch = None, None
            candidate_named, candidate_meets_floor = None, None
        floor_status = {f"baseline_miou_{MEAN_CLASSES}": baseline_miou,
                        f"selected_miou_{MEAN_CLASSES}": selected_score if args.method == "uq" else best,
                        "selected_epoch": selected_epoch if args.method == "uq" else best_epoch,
                        "meets_baseline_floor_on_shared_tiles": meets_baseline_floor,
                        f"best_uq_candidate_miou_{MEAN_CLASSES}": candidate_best,
                        "best_uq_candidate_epoch": candidate_best_epoch,
                        "best_trained_uq_checkpoint": str(out / 'best_trained_uq.pth') if args.method == 'uq' else None,
                        "best_uq_candidate_meets_baseline_floor": candidate_meets_floor,
                        "selected_checkpoint": str(selected_named),
                        "selected_checkpoint_kind": checkpoint_kind if args.method == "uq" else "baseline",
                        "evaluation_tiles": SELECT_IDS,
                        "warning": "The same tiles were used for selection and final metrics; this is not independent generalization."}
        (out / "baseline_floor.json").write_text(json.dumps(floor_status, indent=2), encoding="utf-8")
        if args.method == "uq":
            try:
                relative_checkpoint = selected_named.resolve().relative_to(ROOT).as_posix()
            except ValueError:
                relative_checkpoint = str(selected_named.resolve())
            (out / "selected_checkpoint.json").write_text(json.dumps({
                "checkpoint": relative_checkpoint, "protocol":args.protocol, "dataset":args.dataset,
                "kind": checkpoint_kind,
                "epoch": selected_epoch,
                f"miou_{MEAN_CLASSES}": selected_score,
                f"baseline_miou_{MEAN_CLASSES}": baseline_miou,
            }, indent=2), encoding="utf-8")
            # Every run gets its own readable outputs. The shared UQV folder is
            # the project-wide best and is promoted only on a strict new high.
            if checkpoint_kind == "baseline_fallback":
                torch.nn.init.zeros_(model.head.classifier.weight)
                torch.nn.init.zeros_(model.head.classifier.bias)
            load_checkpoint(model, str(selected_named), args.device,
                            allow_native=True, log=report)
            visual_dir = out / "visuals"
            report(f"\nExporting selected checkpoint predictions and uncertainty maps to {visual_dir}")
            final_metrics, _ = export_tiles(model, TEST_IDS, args, output=visual_dir, report=report)
            final_text = evaluation_report(final_metrics, "Selected checkpoint final metrics")
            diagnostics_dir = visual_dir / "diagnostics"
            record_dir = visual_dir / "Record"
            (record_dir / "final_metrics.txt").write_text(final_text + "\n", encoding="utf-8")
            (record_dir / "final_metrics.json").write_text(json.dumps({
                "model": args.model,
                "run_name": args.run_name,
                "checkpoint": str(selected_named),
                "checkpoint_kind": checkpoint_kind,
                f"baseline_miou_{MEAN_CLASSES}": baseline_miou,
                f"selected_miou_{MEAN_CLASSES}": final_metrics[f"miou_{MEAN_CLASSES}"],
                "tiles": TEST_IDS,
                "metrics": final_metrics,
                "uncertainty_heatmap_scale_nat": [0.0, math.log(N_CLASSES)],
                "heatmap_palette": "deep blue low -> cyan -> yellow -> bright red high",
            }, indent=2), encoding="utf-8")
            report(final_text)
            shared_visual_dir = UQ_VISUALS / args.dataset / args.model
            if shared_visual_dir.is_dir():
                organize_visual_layout(shared_visual_dir)
            missing_shared_maps = not any((shared_visual_dir / "Udata").glob("*_heatmap.png"))
            if (checkpoint_kind == "uq" and selected_score > global_best) or (checkpoint_kind == "baseline_fallback" and missing_shared_maps):
                promote_visuals(visual_dir, shared_visual_dir)
                selection_record = {
                    "model": args.model,
                    "checkpoint": str(selected_named.resolve()),
                    "epoch": selected_epoch,
                    f"miou_{MEAN_CLASSES}": selected_score,
                    f"baseline_miou_{MEAN_CLASSES}": baseline_miou,
                    "run_name": args.run_name,
                    "visuals": str(shared_visual_dir.resolve()),
                    "checkpoint_kind": checkpoint_kind,
                    "uncertainty_note": "Zero residual baseline: uncalibrated Dirichlet concentration; model uncertainty is zero." if checkpoint_kind == "baseline_fallback" else "Trained stochastic head",
                }
                (shared_visual_dir / "Record" / "best_selection.json").write_text(json.dumps(selection_record, indent=2), encoding="utf-8")
                if checkpoint_kind == "uq":
                    (uq_weight_dir / "best_uq.json").write_text(json.dumps(selection_record, indent=2), encoding="utf-8")
                report(f"Selected {checkpoint_kind} maps exported to {shared_visual_dir}")
            else:
                report(f"Shared best outputs unchanged (current best={global_best:.6f}; run selected={selected_score:.6f})")
            report(f"Prediction: {visual_dir / 'Prediction'}\nRecords: {visual_dir / 'Record'}\nHeatmap folders: {visual_dir / 'Udata'}, {visual_dir / 'Udist'}, {visual_dir / 'Umodel'}\nRun outputs: {visual_dir}")
        floor_text = "PASS on shared tiles" if candidate_meets_floor else "trained UQ candidate below baseline; baseline fallback selected"
        candidate_text = f"\nBest trained UQ candidate: {candidate_best:.4f} at epoch {candidate_best_epoch}\nLatest-improvement checkpoint: {candidate_named}\nAll incremental UQ checkpoints: {uq_weight_dir}\nSelected checkpoint: {selected_named}" if args.method == "uq" else ""
        report(f"\nTraining complete. Selected mean MIoU: {best:.4f}; baseline mean MIoU: {baseline_miou if baseline_miou is not None else float('nan'):.4f}; UQ candidate status: {floor_text}{candidate_text}\nCheckpoint selection metadata: {out / 'selected_checkpoint.json' if args.method == 'uq' else out / 'baseline_floor.json'}\nReadable log: {out / 'training.log'}\nNumeric history: {out / 'history.jsonl'}")
    else:
        raise RuntimeError("No valid baseline-style mean IoU was produced; no best checkpoint was selected")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train","calibrate","test"))
    parser.add_argument("--dataset", choices=("Vaihingen", "Potsdam", "LoveDA"), default="Vaihingen")
    parser.add_argument("--data-root")
    parser.add_argument("--model", required=True,
                        choices=("ABCNet","CMTFNet","UNetformer","FTUNetformer","DeepKANSeg"))
    parser.add_argument("--method", choices=("baseline","uq"), default="uq")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", default=str(ROOT / "uq_runs"))
    parser.add_argument("--results-uq", type=Path, default=RESULTS_UQ)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--ood-dir")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--recipe", choices=("auto","legacy","geoseg","loveda"), default="auto")
    parser.add_argument("--eval-patch", type=int, default=None)
    parser.add_argument("--decoder-lr", type=float, default=None)
    parser.add_argument("--patch", type=int, default=None)
    parser.add_argument("--eval-stride", type=int, default=None)
    parser.add_argument("--eval-batch", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--head-warmup-epochs", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--preview-every", type=int, default=5)
    parser.add_argument("--preview-tile", default=None)
    parser.add_argument("--preview-row", type=int, default=256)
    parser.add_argument("--preview-col", type=int, default=256)
    parser.add_argument("--train-samples", type=int, default=2)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--pilot", type=int, default=0)
    parser.add_argument("--delta2", type=float, default=0.001)
    parser.add_argument("--min-samples", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--beta-boost", type=float, default=100.0)
    parser.add_argument('--plain-ce', action='store_true', help='Diagnostic ablation: unsmoothed CE; Dirichlet target unchanged')
    parser.add_argument("--lambda-id", type=float, default=0.01)
    parser.add_argument("--lambda-ood", type=float, default=0.01)
    parser.add_argument("--dist-focus-weight", type=float, default=0.5)
    parser.add_argument("--model-focus-weight", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--eta-max", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--ramp-end", type=int, default=20)
    parser.add_argument("--false-positive-rate", type=float, default=0.05)
    args = parser.parse_args(argv)
    configure_uq_dataset(args.dataset, args.data_root)
    if args.dataset == "LoveDA":
        args.no_cache = True
    resolve_runtime(args)
    if args.preview_tile is None:
        args.preview_tile = TRAIN_IDS[0]
    if args.preview_every < 0:
        raise ValueError("--preview-every must be >= 0")
    if args.log_every < 1 or args.epochs < 1 or args.steps_per_epoch < 1:
        raise ValueError("--log-every, --epochs, --steps-per-epoch must be >= 1")
    if args.ramp_end <= args.warmup:
        raise ValueError("--ramp-end must exceed --warmup")
    if not 0.0 <= args.dist_focus_weight <= 1.0:
        raise ValueError("--dist-focus-weight must be in [0, 1]")
    if not 0.0 <= args.model_focus_weight <= 1.0:
        raise ValueError("--model-focus-weight must be in [0, 1]")
    if args.run_name and (not args.run_name.replace("_", "").replace("-", "").isalnum()):
        raise ValueError("--run-name must contain only letters, digits, _ or -")
    if min(args.batch,args.eval_batch,args.patch,args.eval_patch) < 1 or args.patch % 32 or args.eval_patch % 32:
        raise ValueError("Batch sizes must be positive; patch and eval-patch must be positive multiples of 32")
    if args.eval_stride < 1 or args.eval_stride > args.eval_patch:
        raise ValueError("--eval-stride must be in [1, --patch]")
    if args.eval_batch < 1:
        raise ValueError("--eval-batch must be >=1")
    args.device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    if args.mode == "train":
        train(args)
        return
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for calibrate/test")
    model = make_model(args).to(args.device)
    out = run_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    report = Reporter(out / ("calibration.log" if args.mode == "calibrate" else "test.log"))
    report(f"{args.model}, {'Calibrate' if args.mode == 'calibrate' else 'Test'}, {args.dataset}, {args.method.upper()}\nCheckpoint: {args.checkpoint}")
    # A native baseline checkpoint is a valid zero-residual fallback; normal
    # UQ archives carry a head and are loaded strictly by load_checkpoint.
    load_checkpoint(model,args.checkpoint,args.device, allow_native=True, log=report)
    if args.mode == "calibrate":
        if args.method != "uq":
            raise ValueError("Calibration requires --method uq")
        report(f"Calibrating anomaly threshold on {len(CAL_IDS)} training images..." if args.dataset == "LoveDA"
               else f"Calibrating anomaly threshold on tiles {','.join(CAL_IDS)}...")
        _, scores = eval_tiles(model,CAL_IDS,args, report=report)
        threshold = float(np.quantile(np.concatenate(scores),1-args.false_positive_rate))
        out.mkdir(parents=True,exist_ok=True)
        (out / "threshold.json").write_text(json.dumps({"threshold":threshold,
            "false_positive_rate":args.false_positive_rate,"calibration_ids":CAL_IDS,
            "score_sampling":"up to 4096 evenly spaced valid pixels per image" if args.dataset == "LoveDA" else "all valid pixels"}),encoding="utf-8")
        report(f"Calibration complete: Udist/log(K) threshold={threshold:.6f}; target ID false-positive rate={100*args.false_positive_rate:.1f}%; saved {out / 'threshold.json'}")
    else:
        report(f"Evaluation images={len(TEST_IDS)}; label=baseline labels; patch={args.eval_patch}; stride={args.eval_stride}" if args.dataset == "LoveDA"
               else f"Evaluation tiles={','.join(TEST_IDS)}; label=baseline evaluation labels; patch={args.eval_patch}; stride={args.eval_stride}")
        metrics,_ = export_tiles(model,TEST_IDS,args,out / "test", report=report)
        (out / "test_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
        text_report = evaluation_report(metrics, "Final metrics (tiles reused for validation; not independent)")
        (out / "test_metrics.txt").write_text(text_report + "\n", encoding="utf-8")
        report(text_report)
        report(f"Metrics: {out / 'test_metrics.txt'}\nNumeric metrics: {out / 'test_metrics.json'}\nPredictions and uncertainty maps: {out / 'test'}")


if __name__ == "__main__":
    main()
