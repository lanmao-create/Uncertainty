"""Check real baseline weights with two training updates and UQ crop exports.

Outputs are integration checks on training tile 1, not trained UQ results or
independent test metrics. Each model starts from its configured native weights.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

import uq_config as cfg
from run_tri_uq import PALETTE, load_checkpoint, load_tile
from tri_uq import LossConfig, MODEL_SPECS, PlugInSegmenter, build_backbone, objective, predict


def update(model, optimizer, x, y, epoch, frozen):
    model.train()
    model.backbone.requires_grad_(not frozen)
    if frozen:
        model.backbone.eval()
    optimizer.zero_grad(set_to_none=True)
    head_before = model.head.project.weight.detach().clone()
    losses = objective(model(x, samples=cfg.TRAIN_SAMPLES), y, None, epoch, LossConfig())
    if not all(torch.isfinite(value).all() for value in losses.values()):
        raise FloatingPointError("Nonfinite loss")
    losses["total"].backward()
    grads = [(name, p) for name, p in model.named_parameters() if p.grad is not None]
    if not all(torch.isfinite(p.grad).all() for _, p in grads):
        raise FloatingPointError("Nonfinite training gradient")
    backbone_grads = [(name, p) for name, p in grads if name.startswith("backbone.")]
    if frozen and backbone_grads:
        raise AssertionError("Frozen backbone received gradients")
    if not frozen and not backbone_grads:
        raise AssertionError("Joint training did not reach the backbone")
    tracked = next(((name, p) for name, p in backbone_grads if p.grad.abs().max() > 0), None)
    backbone_before = tracked[1].detach().clone() if tracked else None
    optimizer.step()
    head_delta = (model.head.project.weight.detach() - head_before).abs().max().item()
    if head_delta == 0:
        raise AssertionError("New head did not update")
    backbone_delta = ((tracked[1].detach() - backbone_before).abs().max().item()
                      if tracked else 0.0)
    if not frozen and backbone_delta == 0:
        raise AssertionError("Backbone weights did not update")
    return {"frozen": frozen, "losses": {k: float(v.detach()) for k, v in losses.items()},
            "head_max_update": head_delta, "backbone_max_update": backbone_delta,
            "tracked_backbone_parameter": tracked[0] if tracked else None,
            "backbone_parameters_with_grad": len(backbone_grads)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_SPECS)
    parser.add_argument("--output", type=Path, default=cfg.OUTPUT / "_verification")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.SEED)
    rgb, target = load_tile("1")
    positions = ((256, 256), (512, 512))
    x = torch.from_numpy(np.stack([rgb[r:r+256, c:c+256].transpose(2, 0, 1)
                                 for r, c in positions])).to(device)
    y = torch.from_numpy(np.stack([target[r:r+256, c:c+256]
                                 for r, c in positions]).astype(np.int64)).to(device)
    report = {"formal_experiment": False, "purpose": "real checkpoint integration check",
              "device": str(device), "input_tile": "1", "crop_positions": positions,
              "updates_per_model": 2, "ood_supervision": False, "models": {}}
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ([args.model] if args.model else MODEL_SPECS):
        checkpoint = cfg.resolve_baseline_checkpoint(name)
        print(f"{name}: loading {Path(checkpoint).name}", flush=True)
        model = PlugInSegmenter(build_backbone(name, pretrained=False), name)
        load_checkpoint(model, str(checkpoint), torch.device("cpu"), allow_native=True)
        model = model.to(device).eval()
        with torch.no_grad():
            native_prediction = model.backbone(x[:1]).argmax(1)[0].cpu().numpy()
        optimizer = torch.optim.SGD([
            {"params": model.backbone.parameters(), "lr": cfg.BACKBONE_LR},
            {"params": model.head.parameters(), "lr": cfg.HEAD_LR}],
            momentum=0.9, weight_decay=5e-4)
        warmup = update(model, optimizer, x, y, epoch=1, frozen=True)
        joint = update(model, optimizer, x, y, epoch=20, frozen=False)
        model.eval()
        with torch.no_grad():
            post_update_losses = objective(model(x, samples=cfg.TRAIN_SAMPLES), y,
                                           None, epoch=20, config=LossConfig())
        if not torch.isfinite(post_update_losses["total"]):
            raise FloatingPointError("Nonfinite post-update uncertainty-guided loss")
        result = predict(model, x[:1], samples=cfg.EVAL_SAMPLES)
        error = (result["total"] - result["data"] - result["dist"] - result["model"]).abs().max().item()
        if error > 2e-5:
            raise AssertionError(f"Entropy decomposition error {error}")
        out = args.output / name
        out.mkdir(parents=True, exist_ok=True)
        Image.fromarray((rgb[256:512, 256:512] * 255).astype(np.uint8)).save(out / "input.png")
        Image.fromarray(PALETTE[target[256:512, 256:512]]).save(out / "label.png")
        Image.fromarray(PALETTE[native_prediction]).save(out / "baseline_prediction.png")
        Image.fromarray(PALETTE[result["prediction"][0].cpu().numpy()]).save(out / "verification_prediction.png")
        maps = {key: result[key][0].cpu().numpy() for key in ("data", "dist", "model", "total")}
        np.savez_compressed(out / "verification_uq.npz", **maps,
                            mean_p=result["mean_p"][0].permute(1, 2, 0).cpu().numpy())
        for key, value in maps.items():
            if not np.isfinite(value).all() or value.min() < -2e-5:
                raise AssertionError(f"Invalid {key} uncertainty")
            Image.fromarray((np.clip(value / math.log(6), 0, 1) * 255).astype(np.uint8)).save(out / f"U{key}.png")
        report["models"][name] = {"checkpoint": str(checkpoint),
            "strict_load": True, "warmup_update": warmup, "joint_update": joint,
            "post_update_focus": {key: float(value) for key, value in post_update_losses.items()
                                  if "focus_weight" in key},
            "decomposition_max_error": error,
            "map_ranges": {key: [float(value.min()), float(value.max())] for key, value in maps.items()}}
        (args.output / "checkpoint_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"{name}: PASS (native load, frozen/joint updates, Udist/Umodel focus, three maps)", flush=True)
        del model, optimizer, result, maps
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"Report: {args.output / 'checkpoint_report.json'}", flush=True)


if __name__ == "__main__":
    main()
