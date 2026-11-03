"""Common baseline entry point: five models, Vaihingen/Potsdam/LoveDA Urban.

Defaults live in utils.py. CLI examples (from this directory):
    python train.py --dataset Potsdam --model ABCNet
    python train.py --dataset Vaihingen --model UNetformer
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import time
import traceback

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import utils as cfg
import baseline_geoseg as geoseg
import baseline_loveda as loveda

MODELS = ('ABCNet', 'CMTFNet', 'UNetformer', 'FTUNetformer', 'DeepKANSeg')
DEFAULT_MODEL, DEFAULT_DATASET, DEFAULT_MODE, DEFAULT_LOSS = cfg.MODEL, cfg.DATASET, cfg.MODE, cfg.LOSS


class Tee:
    def __init__(self, console, file):
        self.console, self.file = console, file
    def write(self, message):
        self.console.write(message)
        self.file.write(message)
        self.file.flush()
    def flush(self):
        self.console.flush()
        self.file.flush()
    def isatty(self):
        return False


def build_model(name, pretrained=True, pretrain=None, recipe="legacy"):
    """Use local single-input networks and their native state_dict keys."""
    if recipe == 'loveda' and (name != 'UNetformer' or cfg.DATASET != 'LoveDA'):
        raise ValueError('LoveDA recipe supports LoveDA UNetformer only')
    if recipe in ("geoseg", "loveda"):
        model = geoseg.create_model(name, cfg.N_CLASSES)
        if pretrained:
            geoseg.load_pretrain(model, name, pretrain)
        return model
    # Local classification weights make the baseline entry portable/offline.
    path = Path(pretrain).expanduser() if pretrain else cfg.PRETRAIN_FILES[name]
    if not path.is_absolute() and not path.is_file():
        path = cfg.ROOT / path
    if pretrained and not path.is_file():
        raise FileNotFoundError(f'{name} encoder pretrain missing: {path}. Copy the project pretrain folder, use --pretrain PATH, or explicitly select --no-pretrained.')
    if name == 'ABCNet':
        from model.ABCNet import ABCNet
        model = ABCNet(num_classes=cfg.N_CLASSES, pretrained=False)
        encoder = model.cp.resnet
    elif name == 'CMTFNet':
        from model.CMTFNet.CMTFNet import CMTFNet
        from model.CMTFNet.ResNet import ResNet50
        model = CMTFNet(num_classes=cfg.N_CLASSES, backbone=lambda: ResNet50(pretrained=False))
        encoder = model.backbone
    elif name == 'UNetformer':
        from model.UNetFormer import UNetFormer
        model = UNetFormer(num_classes=cfg.N_CLASSES, pretrained=False)
        encoder = model.backbone
    elif name == 'FTUNetformer':
        from model.FTUNetFormer import ft_unetformer
        model = ft_unetformer(num_classes=cfg.N_CLASSES, pretrained=False)
        encoder = model.backbone
    elif name == 'DeepKANSeg':
        from model.deepKAN_ResNet import UNetFormer as DeepKANSeg
        model = DeepKANSeg(num_classes=cfg.N_CLASSES, pretrained=False)
        encoder = model.backbone
    else:
        raise ValueError(name)
    if pretrained:
        state = torch.load(path, map_location='cpu', weights_only=True)
        for key in ('state_dict','model','model_state_dict'):
            if isinstance(state,dict) and key in state and isinstance(state[key],dict):
                state = state[key]
                break
        expected = encoder.state_dict()
        matched = {}
        for key,value in state.items():
            if not isinstance(value,torch.Tensor):
                continue
            for prefix in ('module.','cp.resnet.','backbone.'):
                if key.startswith(prefix):
                    key = key[len(prefix):]
            if name == 'FTUNetformer':
                # Actual forward uses patch_embed. The local backbone also
                # registers an unused patch_embedd; populate its checkpoint
                # entries without renaming the active input layer.
                if key.startswith('patch_embed.'):
                    alias = 'patch_embedd.' + key[len('patch_embed.'):]
                    if alias in expected and value.shape == expected[alias].shape:
                        matched[alias] = value
                if key.startswith('norm.'):
                    key = 'norm3.' + key[len('norm.'):]
            if key in expected and value.shape == expected[key].shape:
                matched[key] = value
        missing_parameters = set(dict(encoder.named_parameters())) - set(matched)
        # Classification has only the final output norm; intermediate output
        # norms are segmentation additions initialized to identity by the model.
        allowed = {f'norm{i}.{suffix}' for i in range(3) for suffix in ('weight','bias')} if name == 'FTUNetformer' else set()
        if missing_parameters - allowed:
            raise ValueError(f'{name} encoder pretrain mismatch: {len(missing_parameters)} learned parameters missing, e.g. {sorted(missing_parameters)[:5]}')
        encoder.load_state_dict(matched, strict=False)
        print(f'Encoder pretrain: {path.resolve()}; tensors={len(matched)}/{len(expected)}; new output-norm parameters={len(missing_parameters)}')
    return model


def output_paths(dataset, model):
    if dataset == 'Potsdam':
        return cfg.ROOT / 'results_potsdam' / model, cfg.ROOT / 'resultsp' / model
    if dataset in ('LoveDA', 'Urban'):
        return cfg.ROOT / 'results_loveda' / model, cfg.ROOT / 'resultsl' / model
    return cfg.ROOT / 'results_vaihingen' / model, cfg.ROOT / 'resultsv' / model


def check_data(training=True):
    paths = [Path(cfg.DATA_FOLDER.format(tile)) for tile in cfg.test_ids]
    paths += [Path(cfg.ERODED_FOLDER.format(tile)) for tile in cfg.test_ids]
    if training:
        paths += [Path(pattern.format(tile)) for tile in cfg.train_ids
                  for pattern in (cfg.DATA_FOLDER, cfg.LABEL_FOLDER)]
        if 'BDY' in cfg.LOSS:
            paths += [Path(cfg.BOUNDARY_FOLDER.format(t)) for t in cfg.train_ids] if cfg.BOUNDARY_FOLDER else [Path(cfg.MAIN_FOLDER) / 'MISSING_BOUNDARY_ANNOTATIONS']
        if 'OBJ' in cfg.LOSS:
            paths += [Path(cfg.OBJECT_FOLDER.format(t)) for t in cfg.train_ids] if cfg.OBJECT_FOLDER else [Path(cfg.MAIN_FOLDER) / 'MISSING_OBJECT_ANNOTATIONS']
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} dataset files missing (first 20 shown):\n' + '\n'.join(missing[:20])
                                + f'\nDataset root: {cfg.MAIN_FOLDER}; use --data-root to override it.')
    if cfg.DATASET == 'LoveDA':
        # Catch misplaced RGB images before constructing/loading the network.
        samples = [(cfg.ERODED_FOLDER, cfg.test_ids[0])]
        if training:
            samples.append((cfg.LABEL_FOLDER, cfg.train_ids[0]))
        for pattern, tile in samples:
            cfg.read_label(pattern.format(tile))
    return len(paths)


def forward_single_input(net, images, allow_aux=False):
    """Enforce the same single-TOP input contract in training and validation."""
    if images.ndim != 4 or images.shape[1] != cfg.IN_CHANNELS:
        raise ValueError(f'Expected [B,3,H,W] single-input TOP tensor, received {tuple(images.shape)}')
    logits = net(images)
    expected = (images.shape[0], cfg.N_CLASSES, images.shape[2], images.shape[3])
    if allow_aux and isinstance(logits, tuple):
        if len(logits) != 2 or any(tuple(value.shape) != expected for value in logits):
            raise RuntimeError("Invalid main/aux output shapes")
        return logits
    if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != expected:
        actual = tuple(logits.shape) if isinstance(logits, torch.Tensor) else type(logits).__name__
        raise RuntimeError(f'Expected baseline logits {expected}, received {actual}')
    return logits


def test(net, test_ids, all=False, stride=None, batch_size=None,
         window_size=None, return_details=False):
    """Sum logits; evaluate ISPRS first 5 classes or all 7 LoveDA classes."""
    stride = stride or cfg.Stride_Size
    batch_size = batch_size or cfg.BATCH_SIZE
    window_size = tuple(window_size or getattr(net, 'eval_window_size', cfg.WINDOW_SIZE))
    device = next(net.parameters()).device
    was_training = net.training
    net.eval()
    confusion = np.zeros((cfg.N_CLASSES, cfg.N_CLASSES), dtype=np.int64)
    predictions, targets = [], []
    try:
        with torch.inference_mode():
            for tile_index, tile in enumerate(test_ids, 1):
                # Potsdam is RGBIR on disk, but models receive RGB only.
                image = cfg.read_image(cfg.DATA_FOLDER.format(tile))[..., :3]
                target = cfg.read_label(cfg.ERODED_FOLDER.format(tile))
                if image.shape[:2] != target.shape:
                    raise ValueError(f'Image/evaluation-label mismatch: tile {tile}')
                if any(a < b for a, b in zip(target.shape, window_size)):
                    raise ValueError(f'Tile {tile} is smaller than window_size={window_size}')
                scores = np.zeros(target.shape + (cfg.N_CLASSES,), dtype=np.float32)
                total = math.ceil(cfg.count_sliding_window(image, step=stride, window_size=window_size) / batch_size)
                batches = cfg.grouper(batch_size, cfg.sliding_window(image, step=stride, window_size=window_size))
                for coords in tqdm(batches, total=total, desc=f'Validation {tile_index}/{len(test_ids)}',
                                   unit='batch', leave=False, dynamic_ncols=True):
                    patches = np.stack([image[r:r+h, c:c+w].transpose(2,0,1) for r,c,h,w in coords])
                    x = torch.from_numpy(patches).to(device=device, dtype=torch.float32).div_(255.0)
                    if getattr(net, "baseline_recipe", "legacy") in ("geoseg", "loveda"):
                        x = geoseg.normalize(x)
                    logits = forward_single_input(net, x)
                    outputs = logits.float().cpu().numpy()
                    for value, (r,c,h,w) in zip(outputs, coords):
                        if value.shape[-2:] != (h,w):
                            raise RuntimeError(f'Output size {value.shape} does not match patch {(h,w)}')
                        scores[r:r+h, c:c+w] += value.transpose(1,2,0)
                prediction = scores.argmax(axis=2).astype(np.uint8)
                valid = target < cfg.N_CLASSES
                encoded = target[valid].astype(np.int64) * cfg.N_CLASSES + prediction[valid]
                confusion += np.bincount(encoded, minlength=cfg.N_CLASSES**2).reshape(cfg.N_CLASSES,cfg.N_CLASSES)
                if all or return_details:
                    predictions.append(prediction)
                if all:
                    targets.append(target)
    finally:
        net.train(was_training)
    score = cfg.metrics(confusion=confusion)
    if return_details:
        return score, predictions, confusion
    return (score, predictions, targets) if all else score


def relative_path(path):
    try:
        return str(Path(path).resolve().relative_to(cfg.ROOT).as_posix())
    except ValueError:
        return str(Path(path).resolve())


def selection_metric_name():
    return f'miou_{cfg.MEAN_CLASS_COUNT}'


def new_test_directory(checkpoint):
    """Reserve a fresh, dataset/model/epoch-labelled test directory."""
    match = re.search(r'_epoch(\d+)(?:_|\.|$)', Path(checkpoint).name, re.IGNORECASE)
    epoch = f'{int(match[1]):02d}' if match else 'Unknown'
    root = cfg.ROOT / 'test_results'
    root.mkdir(parents=True, exist_ok=True)
    name = f'{cfg.DATASET}_{cfg.MODEL}_epoch{epoch}'
    index = 1
    while True:
        destination = root / (name if index == 1 else f'{name}_{index:02d}')
        try:
            destination.mkdir()
            return destination
        except FileExistsError:
            index += 1


def save_predictions(predictions, ids, result_dir):
    """Publish the complete best prediction set; keep Logs outside this bundle."""
    result_dir = Path(result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.prediction_', dir=result_dir) as temporary:
        stage = Path(temporary) / 'Prediction'
        stage.mkdir()
        for prediction, tile in zip(predictions, ids):
            Image.fromarray(cfg.convert_to_color(prediction)).save(stage / f'inference_{cfg.MODEL}_tile_{tile}.png')
        destination = result_dir / 'Prediction'
        previous = Path(temporary) / 'previous'
        if destination.exists():
            destination.rename(previous)
        try:
            stage.rename(destination)
        except Exception:
            if previous.exists() and not destination.exists():
                previous.rename(destination)
            raise


def load_checkpoint(net, path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    if isinstance(state, dict):
        for key in ('model', 'state_dict', 'model_state_dict'):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if state and all(k.startswith('module.') for k in state):
        state = {k[7:]:v for k,v in state.items()}
    net.load_state_dict(state, strict=True)
    print(f'Checkpoint loaded: {path}; missing keys=0; unexpected keys=0')


def find_best_checkpoint(folder, model, recipe=None):
    pattern = re.compile(rf'^{re.escape(model)}_epoch(\d+)_(?:mIoU)?([0-9]+(?:\.[0-9]+)?)(?:\.pth)?$')
    candidates = []
    for path in folder.glob(f'{model}_epoch*'):
        match = pattern.match(path.name)
        if path.is_file() and match and (recipe is None or geoseg.read_metadata(path)["recipe"] == recipe):
            candidates.append((float(match[2]), int(match[1]), path))
    if not candidates:
        raise FileNotFoundError(f'No {model} native baseline checkpoint in {folder}')
    return max(candidates, key=lambda item: item[:2])[2]


def train(net, optimizer, epochs, scheduler=None, weights=None, save_epoch=1,
          args=None, train_loader=None, weight_dir=None, result_dir=None):
    if args is None or train_loader is None:
        raise ValueError('Use main() or python train.py so dataset and output settings are explicit')
    device = next(net.parameters()).device
    history = result_dir / 'Logs' / f'{args.run_name}.jsonl'
    best, best_epoch, best_step, shared_best = -1.0, 0, 0, -1.0
    metric_key = selection_metric_name()
    previous_record = result_dir / 'best.json'
    if previous_record.is_file():
        try:
            previous = json.loads(previous_record.read_text(encoding='utf-8'))
            path = Path(previous['checkpoint'])
            path = path if path.is_absolute() else cfg.ROOT / path
            if not path.is_file():
                # Users may have organized historical flat weights by model.
                path = weight_dir / path.name
            if (path.is_file() and previous.get('validation_tiles') == cfg.test_ids
                    and previous.get('selection_metric', 'miou_5') == metric_key):
                shared_best = float(previous[metric_key])
        except (OSError, ValueError, KeyError, TypeError):
            print('Previous best record unavailable; this run will establish a new prediction record')
    geoseg_loss = geoseg.SegmentationLoss() if args.recipe in ("geoseg", "loveda") else None
    criterionb = cfg.BoundaryLoss() if 'BDY' in cfg.LOSS else None
    criteriono = cfg.ObjectLoss() if 'OBJ' in cfg.LOSS else None
    val_every = getattr(args, 'val_every_steps', 0)
    global_step, last_validation_step = 0, -1

    def validate_and_save(epoch, epoch_step, loss_sum, accuracy_sum, current_lr):
        nonlocal best, best_epoch, best_step, shared_best, last_validation_step
        print(f'Validation (epoch {epoch}/{epochs}; step {epoch_step}/{len(train_loader)}; global step {global_step})')
        score, predictions, confusion = test(net, cfg.test_ids, stride=args.eval_stride,
            batch_size=args.eval_batch, return_details=True)
        if not np.isfinite(score):
            raise FloatingPointError('Non-finite validation mean MIoU')
        last_validation_step = global_step
        if score > best:
            best, best_epoch, best_step = score, epoch, global_step
            score_text = str(score) if 'e' not in str(score).lower() else format(score, '.17f')
            checkpoint = weight_dir / f'{cfg.MODEL}_epoch{epoch}_{score_text}.pth'
            while checkpoint.exists():
                # Extra trailing decimal zeros preserve the original score parser.
                score_text += '0'
                checkpoint = weight_dir / f'{cfg.MODEL}_epoch{epoch}_{score_text}.pth'
            temporary = checkpoint.with_suffix('.pth.tmp')
            torch.save(net.state_dict(), temporary)
            os.replace(temporary, checkpoint)
            geoseg.write_metadata(checkpoint, args)
            print(f'New best checkpoint: {checkpoint}')
            if score > shared_best:
                save_predictions(predictions, cfg.test_ids, result_dir)
                record = {'dataset':cfg.DATASET, 'model':cfg.MODEL, 'epoch':epoch,
                          'step':global_step, 'epoch_step':epoch_step,
                          metric_key:score, 'mean_miou':score, 'selection_metric':metric_key,
                          'num_classes':cfg.N_CLASSES, 'class_names':cfg.LABELS,
                          'recipe':args.recipe, 'normalization':'imagenet' if args.recipe in ('geoseg','loveda') else 'divide_255',
                          'checkpoint':relative_path(checkpoint),
                          'eval_stride':args.eval_stride,'window_size':list(net.eval_window_size),
                          'train_window_size':list(cfg.WINDOW_SIZE),
                          'validation_tiles':cfg.test_ids, 'test_tiles':cfg.test_ids,
                          'run_name':args.run_name, 'config':relative_path(result_dir / 'Logs' / f'{args.run_name}.json')}
                record_tmp = result_dir / 'best.json.tmp'
                record_tmp.write_text(json.dumps(record,indent=2),encoding='utf-8')
                os.replace(record_tmp, result_dir / 'best.json')
                shared_best = score
                print(f'Updated best predictions: {result_dir / "Prediction"}')
        else:
            print(f'Best mean MIoU : {best:.4f} (epoch {best_epoch}; step {best_step})')
        with history.open('a',encoding='utf-8') as fp:
            fp.write(json.dumps({'epoch':epoch,'step':global_step,'epoch_step':epoch_step,
                'lr':current_lr,'train_loss':loss_sum/epoch_step,
                'train_accuracy':accuracy_sum/epoch_step,'recipe':args.recipe,metric_key:score,f'best_{metric_key}':best,
                'mean_miou':score,'selection_metric':metric_key,
                'confusion':confusion.tolist()})+'\n')

    for epoch in range(1, epochs + 1):
        net.train()
        loss_sum, accuracy_sum = 0.0, 0.0
        current_lr = optimizer.param_groups[0]['lr']
        for batch_idx, (data, boundary, objects, target) in enumerate(train_loader, 1):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = forward_single_input(net, data, allow_aux=args.recipe in ("geoseg", "loveda"))
            output = outputs[0] if isinstance(outputs, tuple) else outputs
            if geoseg_loss:
                loss_ce, loss_dice, loss_aux = geoseg_loss.components(outputs, target)
                loss_seg = loss_ce + loss_dice + 0.4 * loss_aux
            else:
                loss_ce = cfg.loss_calc(output, target, weights)
                loss_seg = loss_ce
            loss_boundary = criterionb(output, boundary) if criterionb else output.new_zeros(())
            loss_object = criteriono(output, objects.to(device)) if criteriono else output.new_zeros(())
            loss = loss_seg + 0.1 * loss_boundary.to(device) + loss_object
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss: epoch={epoch}, step={batch_idx}')
            loss.backward()
            optimizer.step()
            global_step += 1
            valid = target < cfg.N_CLASSES
            acc = (output.detach().argmax(1)[valid] == target[valid]).float().mean().item() if valid.any() else 0.0
            loss_sum += loss.item()
            accuracy_sum += acc
            if batch_idx == 1 or batch_idx % args.log_every == 0 or batch_idx == len(train_loader):
                prefix = f'Train (epoch {epoch}/{epochs}) [{batch_idx}/{len(train_loader)} ({100*batch_idx/len(train_loader):.0f}%)]'
                detail = (f'Loss_ce: {loss_ce.item():.6f}\tLoss_dice: {loss_dice.item():.6f}\tLoss_aux: {float(0.4*loss_aux):.6f}' if geoseg_loss
                          else f'Loss_ce: {loss_ce.item():.6f}\tLoss_boundary: {float(loss_boundary):.6f}\tLoss_object: {float(loss_object):.6f}')
                print(f'{prefix}\t{detail}\tLoss: {loss.item():.6f}\tAccuracy: {100*acc:.2f}', flush=True)
            if val_every and global_step % val_every == 0:
                validate_and_save(epoch, batch_idx, loss_sum, accuracy_sum, current_lr)
        # Always validate at epoch end, but never repeat an interval validation
        # already performed on the same optimizer step.
        if epoch % save_epoch == 0 and last_validation_step != global_step:
            validate_and_save(epoch, len(train_loader), loss_sum, accuracy_sum, current_lr)
        if scheduler is not None:
            scheduler.step()
    print(f'Training complete. Best mean MIoU : {best:.4f} (epoch {best_epoch}; step {best_step})')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('Vaihingen','Potsdam','LoveDA','Urban'), default=DEFAULT_DATASET)
    parser.add_argument('--model', choices=MODELS, default=DEFAULT_MODEL)
    parser.add_argument('--mode', choices=('Train','Test'), default=DEFAULT_MODE)
    parser.add_argument('--data-root', help='Override the dataset directory, not its parent')
    parser.add_argument('--recipe', choices=('auto','legacy','geoseg','loveda'), default='auto', help='auto: GeoSeg for Potsdam UNetformer/FTUNetformer, LoveDA recipe for LoveDA UNetformer; legacy otherwise')
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--eval-crop-size', type=int, default=None)
    parser.add_argument('--backbone-lr', type=float, default=None)
    parser.add_argument('--optimizer', choices=('auto','sgd','adamw'), default='auto',
                        help='Legacy recipe only: adamw uses separate encoder/decoder learning rates')
    parser.add_argument('--checkpoint', help='Native checkpoint for Test; highest-scoring one is chosen by default')
    parser.add_argument('--run-name', default='')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--steps-per-epoch', type=int, default=None)
    parser.add_argument('--batch', type=int, default=None)
    parser.add_argument('--eval-batch', type=int, default=None)
    parser.add_argument('--eval-stride', type=int, default=None,
                        help='Sliding-window stride; Potsdam/LoveDA default to 128, Vaihingen to 32')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--loss', choices=('SEG','SEG+BDY','SEG+OBJ','SEG+BDY+OBJ'), default=DEFAULT_LOSS)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-pretrained', action='store_true')
    parser.add_argument('--pretrain', help='Override the local classification/encoder pretrain path')
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--log-every', type=int, default=1)
    parser.add_argument('--val-every-steps', type=int, default=cfg.VAL_EVERY_STEPS,
                        help='0: epoch end only (default); N: extra validation every N optimizer updates')
    parser.add_argument('--check-data', action='store_true', help='Check all split paths without constructing a model')
    args = parser.parse_args(argv)
    cfg.MODEL, cfg.MODE, cfg.LOSS = args.model,args.mode,args.loss
    cfg.configure_dataset(args.dataset,args.data_root)
    selected_test_checkpoint = None
    if args.mode == 'Test':
        weights, _ = output_paths(cfg.DATASET, args.model)
        selected_test_checkpoint = Path(args.checkpoint).expanduser() if args.checkpoint else find_best_checkpoint(weights,args.model, None if args.recipe=='auto' else args.recipe)
        if not selected_test_checkpoint.is_absolute() and not selected_test_checkpoint.is_file():
            selected_test_checkpoint = cfg.ROOT/selected_test_checkpoint
        if not selected_test_checkpoint.is_file():
            parser.error(f'checkpoint not found: {selected_test_checkpoint}')
        meta = geoseg.read_metadata(selected_test_checkpoint)
        if meta.get('dataset', cfg.DATASET) != cfg.DATASET or meta.get('model', args.model) != args.model:
            parser.error('Selected checkpoint metadata does not match --dataset / --model')
        if args.recipe == 'auto':
            args.recipe = meta['recipe']
        elif args.recipe != meta['recipe']:
            parser.error('Selected checkpoint recipe differs from --recipe; use auto or choose a matching checkpoint')
        if args.crop_size is None:
            args.crop_size = meta.get('crop_size',256)
        if args.eval_crop_size is None:
            args.eval_crop_size = meta.get('eval_crop_size',meta.get('crop_size',256))
        if args.eval_stride is None:
            args.eval_stride = meta.get('eval_stride', cfg.Stride_Size)
        if args.eval_batch is None and 'eval_batch' in meta:
            args.eval_batch = meta['eval_batch']
    elif args.recipe == 'auto':
        args.recipe = ('geoseg' if cfg.DATASET=='Potsdam' and args.model in ('UNetformer','FTUNetformer')
                       else 'loveda' if cfg.DATASET=='LoveDA' and args.model=='UNetformer' else 'legacy')
    if args.recipe == 'loveda' and (cfg.DATASET!='LoveDA' or args.model!='UNetformer'):
        parser.error('LoveDA recipe supports LoveDA UNetformer only')
    if args.recipe == 'geoseg' and args.model not in ('UNetformer','FTUNetformer'):
        parser.error('GeoSeg recipe supports UNetformer/FTUNetformer only')
    if args.recipe in ('geoseg','loveda') and args.loss != 'SEG':
        parser.error('Restored recipes use --loss SEG (main CE+Dice, plus UNetformer auxiliary CE)')
    args.epochs = args.epochs if args.epochs is not None else cfg.EPOCHS
    args.batch = args.batch if args.batch is not None else ((8 if args.model=='UNetformer' else 4) if args.recipe=='geoseg' else cfg.BATCH_SIZE)
    args.steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch is not None else ((243 if args.model=='UNetformer' else 486) if args.recipe=='geoseg' else cfg.STEPS_PER_EPOCH)
    args.lr = args.lr if args.lr is not None else (6e-4 if args.recipe in ('geoseg','loveda') else cfg.BASE_LR)
    if args.optimizer != 'auto' and args.recipe != 'legacy':
        parser.error('--optimizer override is supported only for the legacy recipe')
    if args.optimizer == 'adamw' and '--lr' not in (argv if argv is not None else sys.argv[1:]) and not any(x.startswith('--lr=') for x in (argv if argv is not None else sys.argv[1:])):
        args.lr = 6e-4
    args.backbone_lr = args.backbone_lr if args.backbone_lr is not None else 6e-5
    size = args.crop_size or (768 if args.recipe=='geoseg' else 512 if args.recipe=='loveda' else 256)
    if size < 256 or size % 32:
        parser.error('crop-size must be >=256 and divisible by 32')
    if args.recipe=='legacy' and args.model=='FTUNetformer' and size!=256:
        parser.error('Legacy SwinV2 requires crop-size 256; use the geoseg recipe for variable-size Swin v1')
    cfg.WINDOW_SIZE = (size,size)
    args.eval_crop_size = args.eval_crop_size or (1024 if args.recipe=='geoseg' else 256 if args.recipe=='loveda' else size)
    if args.eval_crop_size < 256 or args.eval_crop_size % 32:
        parser.error('eval-crop-size must be >=256 and divisible by 32')
    if args.recipe=='legacy' and args.model=='FTUNetformer' and args.eval_crop_size!=256:
        parser.error('Legacy SwinV2 evaluation requires eval-crop-size 256')
    args.eval_stride = (args.eval_crop_size if args.recipe=='geoseg' else cfg.Stride_Size) if args.eval_stride is None else args.eval_stride
    args.eval_batch = args.batch if args.eval_batch is None else args.eval_batch
    if min(args.epochs,args.steps_per_epoch,args.batch,args.eval_batch,args.log_every) < 1:
        parser.error('epochs, steps, batch, eval-batch and log-every must be positive')
    if args.val_every_steps < 0:
        parser.error('val-every-steps must be nonnegative')
    if not 1 <= args.eval_stride <= args.eval_crop_size:
        parser.error('eval-stride must be in 1..patch_size')
    if args.model == 'ABCNet' and args.mode == 'Train' and args.batch < 2:
        parser.error('ABCNet training BatchNorm requires --batch >= 2')
    if args.run_name and not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_name):
        parser.error('run-name may contain letters, numbers, _ and -')
    files = check_data(training=args.mode=='Train')
    if args.check_data:
        print(f'{args.model}, {cfg.DATASET}: {files} paths OK; training={len(cfg.train_ids)}; validation/test={len(cfg.test_ids)}; eval_stride={args.eval_stride}')
        return
    weight_dir, result_dir = output_paths(cfg.DATASET,cfg.MODEL)
    if args.mode == 'Test':
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        device = torch.device(args.device)
        destination = new_test_directory(selected_test_checkpoint)
        log_path = destination / 'test.log'
        with log_path.open('w', encoding='utf-8') as logfile, redirect_stdout(Tee(sys.stdout, logfile)):
            try:
                print(f'{args.model}, Test, {cfg.DATASET}; recipe={args.recipe}')
                print(f'Checkpoint: {selected_test_checkpoint.resolve()}')
                print(f'Evaluation: images={len(cfg.test_ids)}; classes={cfg.N_CLASSES}; mean_classes={cfg.MEAN_CLASS_COUNT}; patch={args.eval_crop_size}; stride={args.eval_stride}; batch={args.eval_batch}; seed={args.seed}')
                net = build_model(args.model, pretrained=False, recipe=args.recipe).to(device)
                net.baseline_recipe = args.recipe
                net.eval_window_size = (args.eval_crop_size,args.eval_crop_size)
                load_checkpoint(net, selected_test_checkpoint, device)
                _, predictions, _ = test(net, cfg.test_ids, stride=args.eval_stride,
                                         batch_size=args.eval_batch, return_details=True)
                if len(predictions) != len(cfg.test_ids):
                    raise RuntimeError('Prediction count does not match the evaluation split')
                for prediction, tile in zip(predictions, cfg.test_ids):
                    Image.fromarray(cfg.convert_to_color(prediction)).save(
                        destination / f'inference_{cfg.MODEL}_tile_{tile}.png')
                print(f'Test results: {destination}\nLog: {log_path}')
            except Exception:
                traceback.print_exc(file=logfile)
                raise
        return
    for folder in (weight_dir,result_dir/'Logs'):
        folder.mkdir(parents=True,exist_ok=True)
    args.run_name = args.run_name or f'{cfg.MODEL.lower()}_{cfg.DATASET.lower()}_{args.mode.lower()}_s{args.seed}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    log_path = result_dir/'Logs'/f'{args.run_name}.log'
    config_path = result_dir/'Logs'/f'{args.run_name}.json'
    if log_path.exists() or config_path.exists():
        parser.error('Run name already exists; omit --run-name to generate a fresh name automatically')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    started = time.perf_counter()
    with log_path.open('w',encoding='utf-8') as logfile:
        tee = Tee(sys.stdout,logfile)
        with redirect_stdout(tee):
            try:
                print(f'{cfg.MODEL}, {cfg.MODE}, {cfg.DATASET}, {cfg.LOSS}')
                print(f'Input: {"native TOP" if cfg.DATASET == "Vaihingen" else "RGB"}; channels=3; classes={cfg.N_CLASSES}; single-input model')
                print(f'Parameters: recipe={args.recipe}; epochs={args.epochs}; steps/epoch={args.steps_per_epoch}; batch={args.batch}; patch={cfg.WINDOW_SIZE}; lr={args.lr}')
                if args.recipe in ('geoseg','loveda'):
                    schedule = f'CosineAnnealingLR(T_max={args.epochs},eta_min=1e-6)' if args.recipe=='loveda' else 'CosineAnnealingWarmRestarts(15,2)'
                    print(f'Optimizer: AdamW+Lookahead; backbone_lr={args.backbone_lr}; scheduler={schedule}')
                    print('Loss: smooth CE + Dice' + (' + 0.4 auxiliary CE' if args.model=='UNetformer' else ''))
                label_name = 'LoveDA index masks' if cfg.DATASET == 'LoveDA' else 'eroded labels'
                print(f'Evaluation: {label_name}; patch={args.eval_crop_size}; stride={args.eval_stride}; batch={args.eval_batch}; mean MIoU over {cfg.MEAN_CLASS_COUNT} classes')
                print(f'Validation schedule: every {args.val_every_steps} optimizer steps plus epoch end' if args.val_every_steps else 'Validation schedule: epoch end')
                print(f'Weights: {weight_dir}\nPrediction: {result_dir / "Prediction"}\nLog: {log_path}')
                config = vars(args) | {'train_ids':cfg.train_ids,'validation_ids':cfg.test_ids,'test_ids':cfg.test_ids,
                    'data_root':relative_path(Path(cfg.MAIN_FOLDER)),
                    'normalization':'imagenet' if args.recipe in ('geoseg','loveda') else 'divide_255',
                    'label_encoding':cfg.LABEL_ENCODING,'ignore_index':cfg.IGNORE_INDEX,
                    'num_classes':cfg.N_CLASSES,'class_names':cfg.LABELS,
                    'window_size':cfg.WINDOW_SIZE,'eval_window_size':[args.eval_crop_size,args.eval_crop_size],'milestones':None if args.recipe=='loveda' else cfg.LR_MILESTONES,
                    'training_augmentation':'random scale + SmartCrop512 + horizontal flip' if args.recipe=='loveda' else 'existing recipe',
                    'input_channels':3,'input_modality':'IRRG' if cfg.DATASET=='Vaihingen' else 'RGB',
                    'selection_metric':selection_metric_name(),'optimizer':'AdamW+Lookahead' if args.recipe in ('geoseg','loveda') else 'SGD',
                    'scheduler':'CosineAnnealingLR' if args.recipe=='loveda' else 'CosineAnnealingWarmRestarts(15,2)' if args.recipe=='geoseg' else 'MultiStepLR',
                    'scheduler_T_max':args.epochs if args.recipe=='loveda' else None,
                    'architecture':('swin_v1_base' if args.model=='FTUNetformer' else 'unetformer_aux') if args.recipe in ('geoseg','loveda') else ('deepkan_resnet18' if args.model=='DeepKANSeg' else 'legacy'),
                    'training_loss':'smooth CE + Dice + auxiliary CE when present' if args.recipe in ('geoseg','loveda') else args.loss,
                    'pretrained':not args.no_pretrained if args.mode=='Train' else False,
                    'encoder_pretrain':relative_path(args.pretrain or (geoseg.pretrain_path(args.model) if args.recipe in ('geoseg','loveda') else cfg.PRETRAIN_FILES[args.model])) if args.mode=='Train' and not args.no_pretrained else None,
                    'cache':cfg.CACHE and not args.no_cache,
                    'weight_decay':(0.01 if args.model=='UNetformer' else 0.00025) if args.recipe in ('geoseg','loveda') else 0.0005,
                    'momentum':None if args.recipe in ('geoseg','loveda') else 0.9,
                    'note':'Validation and test use identical tiles by experiment design.'}
                config_path.write_text(json.dumps(config,indent=2),encoding='utf-8')
                if args.recipe == 'legacy' and args.optimizer == 'adamw':
                    config['optimizer'] = 'AdamW'
                    config['optimizer_grouping'] = 'backbone.* encoder; all other parameters decoder; unchanged weight_decay=0.0005'
                    config_path.write_text(json.dumps(config,indent=2),encoding='utf-8')
                    print(f'Optimizer: AdamW; encoder_lr={args.backbone_lr}; decoder_lr={args.lr}; weight_decay=0.0005; scheduler=MultiStepLR')
                net = build_model(args.model,pretrained=not args.no_pretrained and args.mode=='Train',pretrain=args.pretrain,recipe=args.recipe).to(device)
                net.baseline_recipe = args.recipe
                net.eval_window_size = (args.eval_crop_size,args.eval_crop_size)
                print('Model parameters: ',sum(p.numel() for p in net.parameters()))
                if args.checkpoint:
                    parser.error('--checkpoint belongs to Test; fresh baseline Train uses encoder pretraining only')
                dataset_class = geoseg.GeoSegPatches if args.recipe=='geoseg' else loveda.LoveDAPatches if args.recipe=='loveda' else cfg.ISPRS_dataset
                dataset = dataset_class(cfg.train_ids,cache=cfg.CACHE and not args.no_cache,
                    window_size=cfg.WINDOW_SIZE,steps_per_epoch=args.steps_per_epoch,batch_size=args.batch,loss=cfg.LOSS)
                loader = torch.utils.data.DataLoader(dataset,batch_size=args.batch,num_workers=0,drop_last=True)
                if args.recipe=='loveda':
                    optimizer,scheduler = loveda.optimizer_and_scheduler(net,args.lr,args.backbone_lr,args.epochs)
                elif args.recipe=='geoseg':
                    optimizer,scheduler = geoseg.optimizer_and_scheduler(net,args.model,args.lr,args.backbone_lr)
                elif args.optimizer == 'adamw':
                    encoder_params = [p for n,p in net.named_parameters() if n.startswith('backbone.') and p.requires_grad]
                    decoder_params = [p for n,p in net.named_parameters() if not n.startswith('backbone.') and p.requires_grad]
                    if not encoder_params or not decoder_params:
                        raise ValueError('Grouped AdamW requires backbone.* encoder parameters and decoder parameters')
                    optimizer = torch.optim.AdamW([
                        {'params':encoder_params,'lr':args.backbone_lr},
                        {'params':decoder_params,'lr':args.lr}], lr=args.lr,weight_decay=0.0005)
                    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=cfg.LR_MILESTONES,gamma=0.1)
                else:
                    optimizer = torch.optim.SGD(net.parameters(),lr=args.lr,momentum=0.9,weight_decay=0.0005)
                    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=cfg.LR_MILESTONES,gamma=0.1)
                train(net,optimizer,args.epochs,scheduler,weights=cfg.WEIGHTS,args=args,train_loader=loader,
                      weight_dir=weight_dir,result_dir=result_dir)
                print(f'Elapsed: {(time.perf_counter()-started)/3600:.2f} h; Log: {log_path}')
            except Exception:
                traceback.print_exc(file=logfile)
                raise


if __name__ == '__main__':
    main()
