"""Resolve the exact baseline architecture and preprocessing for UQ runs."""
import json
from pathlib import Path

import torch
import uq_config as cfg


def omit_unspecified(options):
    result = []
    index = 0
    while index < len(options):
        if options[index].startswith('--') and index+1 < len(options) and options[index+1]=='None':
            index += 2
        else:
            result.append(options[index])
            index += 1
    return result


def checkpoint_protocol(path, dataset, model):
    if not path:
        return {}
    path = Path(path)
    sidecar = Path(str(path) + '.json')
    if sidecar.is_file():
        meta = json.loads(sidecar.read_text(encoding='utf-8'))
    else:
        archive = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        meta = archive.get('protocol', {}) if isinstance(archive, dict) else {}
        if not meta and (dataset == 'Potsdam' and model in ('UNetformer', 'FTUNetformer')
                         or dataset == 'LoveDA' and model == 'UNetformer'):
            state = archive.get('model', archive.get('state_dict', archive))
            restored = any('aux_head.' in k for k in state) if model == 'UNetformer' else any(
                'layers.0.blocks.0.attn.relative_position_bias_table' in k for k in state)
            if restored:
                from baseline_geoseg import read_metadata
                meta = read_metadata(path)
    if meta.get('dataset', dataset) != dataset or meta.get('model', model) != model:
        raise ValueError('Checkpoint dataset/model differs from the requested dataset/model')
    return meta


def resolve(args):
    meta = checkpoint_protocol(args.checkpoint, args.dataset, args.model)
    recorded = meta.get('recipe', 'legacy')
    if recorded not in ('legacy','geoseg','loveda'):
        raise ValueError(f'Unsupported checkpoint recipe: {recorded}')
    requested = args.recipe
    if requested != 'auto' and requested != recorded and meta:
        raise ValueError(f'Requested recipe {requested} differs from checkpoint recipe {recorded}')
    args.recipe = recorded if requested == 'auto' else requested
    classes = 7 if args.dataset == 'LoveDA' else 6
    if meta.get('num_classes',classes) != classes:
        raise ValueError('Checkpoint class count differs from the selected dataset')
    if args.recipe == 'geoseg' and args.model not in ('UNetformer', 'FTUNetformer'):
        raise ValueError('GeoSeg UQ supports UNetformer/FTUNetformer only')
    love = args.recipe == 'loveda'
    if love and (args.dataset!='LoveDA' or args.model!='UNetformer'):
        raise ValueError('LoveDA recipe supports LoveDA UNetformer only')
    geo = args.recipe in ('geoseg','loveda')
    expected_normalization = 'imagenet' if geo else 'divide_255'
    if meta.get('normalization', expected_normalization) != expected_normalization:
        raise ValueError('Checkpoint normalization is inconsistent with its architecture recipe')
    defaults = {
        'patch':meta.get('patch', meta.get('crop_size', 512 if love else 768 if geo else 256)),
        'eval_patch':meta.get('eval_patch', meta.get('eval_crop_size', 1024 if geo and not love else 256)),
        'eval_stride':meta.get('eval_stride', 1024 if geo and not love else (32 if args.dataset=='Vaihingen' else 128)),
        'batch':(8 if args.model=='UNetformer' else 4) if geo and not love else cfg.BATCH,
        'eval_batch':meta.get('uq_eval_batch', 1 if geo and not love else cfg.EVAL_BATCH),
        'steps_per_epoch':(243 if args.model=='UNetformer' else 486) if geo and not love else cfg.STEPS_PER_EPOCH,
        'lr':6e-4 if geo else cfg.HEAD_LR,
        'backbone_lr':6e-6 if geo else cfg.BACKBONE_LR,
        'decoder_lr':6e-5 if geo else cfg.BACKBONE_LR,
    }
    for name, value in defaults.items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)
    args.normalization = expected_normalization
    if not geo and args.model=='FTUNetformer' and (args.patch!=256 or args.eval_patch!=256):
        raise ValueError('Legacy SwinV2 requires patch=eval-patch=256')
    args.protocol = {'version':1, 'dataset':args.dataset, 'model':args.model, 'num_classes':classes, 'recipe':args.recipe,
                     'normalization':args.normalization, 'patch':args.patch,
                     'eval_patch':args.eval_patch, 'eval_stride':args.eval_stride,
                     'uq_eval_batch':args.eval_batch}
    return args
