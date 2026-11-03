"""Training snapshots and contrast-enhanced heatmaps for UQ maps."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from tri_uq import predict


def display_maximum(values: np.ndarray, quantile: float = 99.5) -> float:
    """Robust per-map color scale; raw uncertainty values remain untouched."""
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return 1.0
    maximum = float(np.percentile(finite, quantile))
    if maximum <= 1e-12:
        maximum = float(finite.max())
    return max(maximum, 1e-12)


def colorize(values: np.ndarray, maximum: float) -> np.ndarray:
    """Blue-cyan-yellow-red scale: deep blue is low, bright red is high."""
    colors = np.array([(0, 0, 70), (0, 0, 255), (0, 255, 255),
                       (255, 255, 0), (255, 0, 0)])
    scaled = np.clip(values / max(float(maximum), 1e-12), 0, 1)
    return np.stack([np.interp(scaled, np.linspace(0, 1, len(colors)), colors[:, c])
                     for c in range(3)], axis=-1).astype(np.uint8)


def heatmap_scale(values, component, args, classes):
    """Display mapping only; never rescales the uncertainty tensor itself."""
    policy = getattr(args, 'heatmap_scale', 'per-map')
    if policy == 'reference':
        return float(args.heatmap_reference_maximums.get(component, math.log(classes))), 'Fixed reference'
    if policy == 'theoretical':
        return math.log(classes), 'Theoretical maximum'
    return display_maximum(values), 'P99.5'


def save_uncertainty_heatmap(values: np.ndarray, path: Path, maximum: float | None = None,
                             title: str = "Uncertainty", display_sigma: float = 0.0,
                             display_sharpness: float = 0.0,
                             scale_label: str = 'P99.5') -> None:
    """Save a full-resolution heatmap with a robust per-map nat scale."""
    values = np.asarray(values, dtype=np.float32)
    maximum = display_maximum(values) if maximum is None else float(maximum)
    if not math.isfinite(display_sigma) or display_sigma < 0:
        raise ValueError('display_sigma must be finite and nonnegative')
    if not math.isfinite(display_sharpness) or display_sharpness < 0:
        raise ValueError('display_sharpness must be finite and nonnegative')
    display_values = values
    if display_sigma > 0:
        from scipy.ndimage import gaussian_filter
        # Filter a display copy only. Keep the original scale, data and metrics.
        display_values = gaussian_filter(values, sigma=display_sigma, mode='reflect')
    if display_sharpness > 0:
        from scipy.ndimage import gaussian_filter, minimum_filter, maximum_filter
        # Display-only local-range-clipped unsharp, matching the approved trial.
        detail = display_values - gaussian_filter(display_values, sigma=1., mode='reflect')
        lower = minimum_filter(display_values, size=5, mode='reflect')
        upper = maximum_filter(display_values, size=5, mode='reflect')
        display_values = np.clip(display_values + display_sharpness*detail, lower, upper)
    heat = colorize(display_values, maximum)
    h, w = values.shape
    footer = 54
    canvas = Image.new("RGB", (w, h + footer), "white")
    canvas.paste(Image.fromarray(heat), (0, 0))
    draw = ImageDraw.Draw(canvas)
    note = f"; display smoothing sigma={display_sigma:g}px" if display_sigma > 0 else ""
    if display_sharpness > 0:
        note += f"; display sharpening={display_sharpness:g}"
    draw.text((10, h + 3), f"{title}: uncertainty (nat){note}", fill="black")
    bar_left, bar_right = 12, max(13, w - 12)
    bar_width = bar_right - bar_left
    gradient = np.linspace(0, maximum, bar_width, dtype=np.float32)[None, :]
    gradient = np.repeat(colorize(gradient, maximum), 12, axis=0)
    canvas.paste(Image.fromarray(gradient), (bar_left, h + 22))
    draw.rectangle((bar_left, h + 22, bar_right - 1, h + 33), outline="black")
    draw.text((bar_left, h + 37), "Low (0.00)", fill="black")
    high_text = f"{scale_label} ({maximum:.4g} nat; upper clipped)"
    high_width = draw.textbbox((0, 0), high_text)[2]
    draw.text((max(bar_left, bar_right - high_width), h + 37), high_text, fill="black")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


@torch.no_grad()
def training_preview(model, rgb: np.ndarray, target: np.ndarray, palette: np.ndarray,
                     output: Path, epoch: int, samples: int, ood_supervision: bool, normalize=None):
    """No threshold is inferred from this training crop; it is a visualization."""
    output.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    x = torch.from_numpy(rgb.transpose(2, 0, 1)[None].copy()).to(device)
    result = predict(model, normalize(x) if normalize else x, samples=samples)
    maps = {key: result[key][0].cpu().numpy() for key in ("data", "dist", "model", "total")}
    p = result["mean_p"][0].permute(1, 2, 0).cpu().numpy()
    prediction = result["prediction"][0].cpu().numpy()
    max_entropy = math.log(len(palette))
    np.savez_compressed(output / "uq.npz", mean_p=p, prediction=prediction,
                        dist_score=maps["dist"] / max_entropy, **maps)
    input_rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    label_rgb = np.zeros_like(input_rgb)
    valid = target < len(palette)
    label_rgb[valid] = palette[target[valid]]
    prediction_rgb = palette[prediction]
    for name, values in (("input", input_rgb), ("label", label_rgb),
                         ("prediction", prediction_rgb)):
        Image.fromarray(values).save(output / f"{name}.png")
    panels = [("Input", input_rgb), ("Label", label_rgb), ("Prediction", prediction_rgb)]
    display_scales = {}
    for name, values in maps.items():
        scale = display_maximum(values)
        display_scales[name] = scale
        grey = (np.clip(values / scale, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(grey).save(output / f"U{name}.png")
        heat = colorize(values, scale)
        save_uncertainty_heatmap(values, output / f"U{name}_heatmap.png", scale,
                                 title=f"U_{name}")
        if name != "total":
            panels.append((f"U{name}", heat))
    h, w = target.shape
    caption, footer = 24, 28
    canvas = Image.new("RGB", (3 * w, 2 * (h + caption) + footer), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (name, panel) in enumerate(panels):
        left, top = (i % 3) * w, (i // 3) * (h + caption)
        draw.text((left + 6, top + 6), name, fill="black")
        canvas.paste(Image.fromarray(panel), (left, top + caption))
    scale_note = ", ".join(f"U{k} P99.5={v:.3g}" for k, v in display_scales.items()
                            if k in ("data", "dist", "model"))
    draw.text((6, 2 * (h + caption) + 6),
              f"Epoch {epoch}; per-map display scale in nat: {scale_note}", fill="black")
    canvas.save(output / "overview.png")
    metadata = {"epoch": epoch, "split": "training", "samples": samples,
                "ood_supervision": ood_supervision, "is_ood_detection_result": False,
                "uncertainty_units": "nat", "visual_scale_policy": "per-map P99.5; raw values unchanged",
                "visual_scale_maximums": display_scales,
                "dist_score_definition": "U_dist/log(K); no OOD threshold fitted on training crop",
                "mean_uncertainty": {key: float(value.mean()) for key, value in maps.items()},
                "training_crop_accuracy": float((prediction[valid] == target[valid]).mean())}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

