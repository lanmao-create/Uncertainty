"""Baseline-style English reports for tri-source uncertainty training."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

LABELS = ("roads", "buildings", "low veg.", "trees", "cars", "clutter")


def configure_console():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def confusion_metrics(confusion):
    cm = np.asarray(confusion, dtype=np.int64)
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(1), cm.sum(0)
    union = support + predicted - tp
    total = int(cm.sum())
    if total == 0:
        raise ValueError("No valid evaluation pixels")
    iou = tp / np.maximum(union, 1)
    recall = [float(tp[i] / support[i]) if support[i] else None for i in range(len(tp))]
    f1 = [float(2 * tp[i] / (support[i] + predicted[i]))
          if support[i] + predicted[i] else None for i in range(len(tp))]
    present = union > 0
    mean_count = 7 if len(tp) == 7 else 5
    selected_present = present[:mean_count]
    pa = float(tp.sum() / total)
    pe = float(np.dot(support.astype(float), predicted.astype(float)) / total ** 2)
    f1_5 = [v for v in f1[:mean_count] if v is not None]
    return {"miou": float(iou[present].mean()),
            f"miou_{mean_count}": float(iou[:mean_count][selected_present].mean()) if selected_present.any() else None,
            "mean_miou": float(iou[:mean_count][selected_present].mean()) if selected_present.any() else None,
            "class_iou": iou.tolist(), "class_recall": recall, "class_f1": f1,
            "class_support": support.tolist(), "class_present": present.tolist(),
            f"mean_f1_{mean_count}": float(np.mean(f1_5)) if f1_5 else None,
            "accuracy": pa, "kappa": float((pa - pe) / (1 - pe)) if pe < 1 else None,
            "valid_pixels": total, "confusion": cm.tolist()}


def number(value, digits=4):
    return f"{value:.{digits}f}" if value is not None else "n/a"


def weight_eta(epoch, config):
    warmup, end = config.get("warmup", 10), config.get("ramp_end", 20)
    return config.get("eta_max", 0.5) * min(1.0, max(0.0, (epoch - warmup) / (end - warmup)))


def stage(epoch, config):
    if config.get("method", "uq") == "baseline":
        return "Baseline fine-tuning"
    if epoch <= config.get("head_warmup_epochs", 5):
        return "Head warmup; backbone frozen"
    eta = weight_eta(epoch, config)
    if eta == 0:
        return "Joint fine-tuning; uncertainty weighting inactive"
    if epoch < config.get("ramp_end", 20):
        return f"Joint fine-tuning; Udata/Udist/Umodel weighting ramp, eta={eta:.3f}"
    return f"Joint fine-tuning; Udata/Udist/Umodel weighting active, eta={eta:.3f}"


def setup_report(config, backbone_params=None, head_params=None):
    uq = config.get("method", "uq") == "uq"
    lines = [f"{config['model']}, Train, {config.get('dataset', 'Vaihingen')}, {'SEG+ID_KL+UQ' if uq else 'SEG'}"]
    if config.get('dataset') == 'LoveDA':
        config = dict(config)
        for key in ('train_ids', 'selection_ids', 'calibration_ids', 'test_ids'):
            config[key] = [f"{len(config.get(key, []))} images"]
    if backbone_params is not None:
        lines.append(f"Model parameters: backbone={backbone_params:,}, new head={head_params or 0:,}")
    lines += [f"Checkpoint: {config.get('checkpoint', 'random initialization')}",
              f"Train tiles: {','.join(config.get('train_ids', []))}",
              f"Validation/selection tiles: {','.join(config.get('selection_ids', []))}",
              f"Calibration tiles: {','.join(config.get('calibration_ids', []))}",
              f"Final metric tiles: {','.join(config.get('test_ids', []))} (shared with validation)",
              f"Training budget: {config['epochs']} epochs, {config['steps_per_epoch']} steps/epoch, batch={config['batch']}, patch={config.get('patch', 256)}x{config.get('patch', 256)}"]
    if uq:
        lines += [f"Training: freeze backbone for {config.get('head_warmup_epochs', 5)} epochs, then jointly fine-tune",
                  f"Data weighting: warmup={config.get('warmup', 10)}, ramp_end={config.get('ramp_end', 20)}, eta_max={config.get('eta_max', .5)}",
                  f"Udist/Umodel hard-pixel focus: strengths={config.get('dist_focus_weight', .5)}/{config.get('model_focus_weight', .25)}; detached weights ramp with eta",
                  f"MC samples: train={config.get('train_samples', 2)}, eval={config.get('samples', 16)}",
                  f"OOD supervision: {'enabled' if config.get('ood_dir') else 'disabled (OOD KL=0; Udist still guides ID hard-pixel weights)'}",
                  "Loss: UQ-weighted Loss_seg + lambda_ID * KL_ID + lambda_OOD * KL_OOD"]
    if config.get('recipe') in ('geoseg','loveda'):
        lines += [f"Recipe: {config.get('recipe')}; ImageNet normalization; eval patch={config.get('eval_patch',256)}",
                  f"Optimizer: AdamW+Lookahead; encoder_lr={config.get('backbone_lr')}; decoder_lr={config.get('decoder_lr')}; adapter_lr={config.get('lr')}",
                  f"Scheduler: {'CosineAnnealingLR' if config.get('recipe')=='loveda' else 'CosineAnnealingWarmRestarts(15,2)'}",
                  "Additional baseline supervision: Dice + 0.4 auxiliary CE when an auxiliary head is active"]
    if config.get("eta_max", .5) == 0:
        lines = ["Data weighting: disabled (eta_max=0)" if line.startswith("Data weighting:") else line for line in lines]
    lines += [f"Evaluation: baseline labels, patch={config.get('eval_patch',config.get('patch',256))}, stride={config.get('eval_stride',32)}",
              f"Selection metric: baseline nanmean over {7 if config.get('dataset') == 'LoveDA' else 5} classes",
              "Output: " + str(Path(config['output']) / config.get('dataset', 'Vaihingen') / config['model'] / config.get('method', 'uq') / config.get('run_name', '')),
              "---"]
    return "\n".join(lines)


def evaluation_report(metrics, title="Validation"):
    m = confusion_metrics(metrics["confusion"])
    count = len(m["class_iou"])
    mean_count = 7 if count == 7 else 5
    labels = ("background", "building", "road", "water", "barren", "forest", "agriculture") if count == 7 else LABELS
    lines = [title, "Confusion matrix :", str(np.asarray(m["confusion"], dtype=np.int64)),
             f"{m['valid_pixels']} pixels processed", f"Total accuracy : {100*m['accuracy']:.2f}%",
             "Class accuracy / Recall :"]
    lines += [f"{name}: {number(value)}" for name, value in zip(labels, m["class_recall"])]
    lines += ["---", "F1Score :"]
    lines += [f"{name}: {number(value)}" for name, value in zip(labels, m["class_f1"])]
    lines += [f"mean F1Score: {number(m[f'mean_f1_{mean_count}'])}", "---",
              f"Kappa: {number(m['kappa'])}", "IoU :"]
    lines += [f"{name}: {number(value) if present else 'n/a'}"
              for name, value, present in zip(labels, m["class_iou"], m["class_present"])]
    lines += [f"mean MIoU : {number(m['mean_miou'])}"]
    if "nll" in metrics:
        lines += ["---", f"NLL: {metrics['nll']:.6f}", f"Brier: {metrics['brier']:.6f}",
                  f"Inference time: {metrics['model_seconds']:.2f}s",
                  f"Mean MC samples: {metrics['mean_samples']:.1f}"]
    return "\n".join(lines + ["---"])


def epoch_report(record, config, best):
    epoch = record["epoch"]
    lines = [f"Epoch {epoch}/{config['epochs']}", f"Stage: {stage(epoch, config)}",
             f"Mean total loss: {record['train_loss']:.6f}"]
    if "train_seg" in record:
        lines += [f"Loss_seg: {record['train_seg']:.6f}",
                  f"Loss_ID_KL (weighted): {config.get('lambda_id', .01)*record['train_id_kl']:.6f}",
                  f"Loss_OOD_KL (weighted): {config.get('lambda_ood', .01)*record['train_ood_kl']:.6f}",
                  f"Mean pixel weight: {record['train_mean_weight']:.4f}",
                  f"Peak Udist focus weight: {record.get('train_max_dist_focus_weight', 1.0):.3f}",
                  f"Peak Umodel focus weight: {record.get('train_max_model_focus_weight', 1.0):.3f}",
                  f"Mean Dirichlet concentration: {record['train_mean_concentration']:.2f}",
                  f"Logit clamp fraction: {100*record['train_clamp_fraction']:.4f}%"]
    if 'train_dice' in record:
        lines += [f"Loss_Dice: {record['train_dice']:.6f}", f"Loss_aux (weighted): {record['train_aux']:.6f}"]
    if "train_accuracy" in record:
        lines.append(f"Training batch accuracy: {100*record['train_accuracy']:.2f}%")
    if "train_mean_u_data" in record:
        lines.append(f"Mean uncertainty (ID pixels, nat): Udata={record['train_mean_u_data']:.4f}, Udist={record['train_mean_u_dist']:.4f}, Umodel={record['train_mean_u_model']:.3e}")
    lines += [evaluation_report(record, f"Validation: {len(config.get('selection_ids', []))} images" if config.get('dataset') == 'LoveDA' else f"Validation tiles: {','.join(config.get('selection_ids', []))}"),
              f"Best checkpoint score: {best:.4f}"]
    return "\n".join(lines)


class Reporter:
    def __init__(self, path):
        configure_console()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def __call__(self, message):
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(message + "\n")
