"""Single-input GeoSeg baseline recipe, independent of a GeoSeg installation.

Models and loss helpers are vendored from WangLibo1995/GeoSeg (GPL-3.0).
The original 18/6 Potsdam split and this project's output formats are retained.
"""
from pathlib import Path
import json
import random
import math

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

import utils as cfg
from vendor_geoseg.dice import DiceLoss
from vendor_geoseg.soft_ce import SoftCrossEntropyLoss

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
PRETRAIN = cfg.ROOT / 'pretrain' / 'stseg_base.pth'


def normalize(images):
    """Input is RGB /255; same transform for training and validation."""
    return (images - images.new_tensor(MEAN)[None, :, None, None]) / images.new_tensor(STD)[None, :, None, None]


def create_model(name, classes):
    if name == 'UNetformer':
        from model.UNetFormerGeoSeg import UNetFormer
        return UNetFormer(num_classes=classes, pretrained=False)
    if name == 'FTUNetformer':
        from model.FTUNetFormerGeoSeg import ft_unetformer
        return ft_unetformer(num_classes=classes, pretrained=False)
    raise ValueError('GeoSeg recipe supports UNetformer and FTUNetformer')


def pretrain_path(name):
    return PRETRAIN if name == 'FTUNetformer' else cfg.PRETRAIN_FILES[name]


def load_pretrain(model, name, path=None):
    path = Path(path).expanduser() if path else pretrain_path(name)
    if not path.is_absolute() and not path.is_file():
        path = cfg.ROOT / path
    state = torch.load(path, map_location='cpu', weights_only=True)
    for key in ('state_dict', 'model', 'model_state_dict'):
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break
    expected = model.backbone.state_dict()
    matched = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        for prefix in ('module.', 'net.', 'backbone.'):
            if key.startswith(prefix):
                key = key[len(prefix):]
        if key in expected and value.shape == expected[key].shape:
            matched[key] = value
    missing = set(dict(model.backbone.named_parameters())) - set(matched)
    if missing:
        raise ValueError(f'{name} GeoSeg pretrain mismatch: {len(missing)} learned tensors missing: {sorted(missing)[:8]}. Swin v1 requires matching v1 weights.')
    model.backbone.load_state_dict(matched, strict=False)
    print(f'Encoder pretrain: {path.resolve()}; tensors={len(matched)}/{len(expected)}; missing learned tensors=0')


class SegmentationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=255)
        self.dice = DiceLoss(smooth=0.05, ignore_index=255)

    def components(self, outputs, target):
        main, aux = outputs if isinstance(outputs, tuple) else (outputs, None)
        ce, dice = self.ce(main, target), self.dice(main, target)
        auxiliary = self.ce(aux, target) if aux is not None else ce.new_zeros(())
        return ce, dice, auxiliary

    def forward(self, outputs, target):
        ce, dice, auxiliary = self.components(outputs, target)
        return ce + dice + 0.4 * auxiliary


class AdamWLookahead(torch.optim.AdamW):
    """AdamW plus the original GeoSeg k=5/alpha=.5 synchronization schedule.

    A native Optimizer subclass keeps scheduler hooks compatible with modern torch.
    Slow weights are initialized at the first fast step, as in GeoSeg tools/utils.py.
    """
    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)
        for group in self.param_groups:
            counter = group.get('lookahead_counter', 0)
            if counter == 0:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if 'slow_param' not in state:
                        state['slow_param'] = p.detach().clone()
                    state['slow_param'].lerp_(p, 0.5)
                    p.copy_(state['slow_param'])
            group['lookahead_counter'] = (counter + 1) % 5
        return loss


def optimizer_and_scheduler(model, name, lr=6e-4, backbone_lr=6e-5):
    wd = 0.01 if name == 'UNetformer' else 2.5e-4
    groups = {}
    for key, param in model.named_parameters():
        if not param.requires_grad:
            continue
        rate = backbone_lr if key.startswith('backbone.') else lr
        # GeoSeg process_model_params excludes bias parameters from decay.
        decay = 0.0 if key.endswith('bias') else wd
        groups.setdefault((rate, decay), []).append(param)
    optimizer = AdamWLookahead([{'params':p, 'lr':rate, 'weight_decay':decay}
                               for (rate, decay), p in groups.items()], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
    return optimizer, scheduler


class GeoSegPatches(cfg.ISPRS_dataset):
    """Online 1024-grid source patches, mosaic, random scale and smart 768 crop.

    This avoids materializing the old offline flipped patch dataset. Sampling is
    stochastic; it does not promise bit-identical old patch order or augmentation.
    """
    def source_patch(self):
        i = random.randrange(len(self.data_files))
        image = self._read(i, self.data_files, self.data_cache_, 'image')
        label = self._read(i, self.label_files, self.label_cache_, 'label')
        h, w = label.shape
        if image.shape[-2:] != (h, w):
            raise ValueError(f'Image/label size mismatch: {self.data_files[i]}')
        top = random.randrange(math.ceil(h / 1024)) * 1024
        left = random.randrange(math.ceil(w / 1024)) * 1024
        rgb = np.zeros((1024, 1024, 3), dtype=np.uint8)
        mask = np.full((1024, 1024), 255, dtype=np.uint8)
        ih, iw = min(1024, h-top), min(1024, w-left)
        rgb[:ih, :iw] = image[:, top:top+ih, left:left+iw].transpose(1, 2, 0)
        mask[:ih, :iw] = label[top:top+ih, left:left+iw]
        variant = random.randrange(3)  # original / horizontal / vertical offline variants
        if variant:
            axis = 1 if variant == 1 else 0
            rgb, mask = np.flip(rgb, axis), np.flip(mask, axis)
        return np.ascontiguousarray(rgb), np.ascontiguousarray(mask)

    def __getitem__(self, index):
        rgb, mask = self.source_patch()
        if random.random() < 0.25:
            cy, cx = random.randint(256, 768), random.randint(256, 768)
            for n, (t,b,l,r) in enumerate(((0,cy,0,cx),(0,cy,cx,1024),(cy,1024,0,cx),(cy,1024,cx,1024))):
                other, labels = (rgb.copy(), mask.copy()) if n == 0 else self.source_patch()
                sy, sx = random.randint(0,1024-(b-t)), random.randint(0,1024-(r-l))
                rgb[t:b,l:r] = other[sy:sy+b-t,sx:sx+r-l]
                mask[t:b,l:r] = labels[sy:sy+b-t,sx:sx+r-l]
        scale = random.choice((0.75, 1.0, 1.25, 1.5))
        side = round(1024*scale)
        side = max(side, self.window_size[0])
        rgb = np.asarray(Image.fromarray(rgb).resize((side,side), Image.Resampling.BICUBIC))
        mask = np.asarray(Image.fromarray(mask).resize((side,side), Image.Resampling.NEAREST))
        size = self.window_size[0]
        for _ in range(11):
            top, left = random.randint(0,side-size), random.randint(0,side-size)
            labels = mask[top:top+size,left:left+size]
            _, counts = np.unique(labels[labels < cfg.N_CLASSES], return_counts=True)
            if len(counts)>1 and counts.max()/counts.sum() < .75:
                break
        rgb = rgb[top:top+size,left:left+size].copy()
        mask = labels.copy()
        if random.random() < .5:
            turns = random.randrange(4)
            rgb, mask = np.rot90(rgb,turns), np.rot90(mask,turns)
        for axis in (0,1):
            if random.random() < .5:
                rgb,mask = np.flip(rgb,axis),np.flip(mask,axis)
        x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).float().div_(255)
        x = normalize(x[None])[0]
        y = torch.from_numpy(np.ascontiguousarray(mask)).long()
        return x, torch.zeros_like(y), torch.zeros_like(y), y


def metadata_path(checkpoint):
    return Path(str(checkpoint) + '.json')


def read_metadata(checkpoint):
    path = metadata_path(checkpoint)
    if path.is_file():
        return json.loads(path.read_text(encoding='utf-8'))
    archive = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if isinstance(archive, dict) and archive.get('protocol'):
        meta = dict(archive['protocol'])
        for old, new in [('patch', 'crop_size'), ('eval_patch', 'eval_crop_size')]:
            if old in meta:
                meta.setdefault(new, meta[old])
        return meta
    state = archive
    for name in ('model', 'state_dict', 'model_state_dict'):
        if isinstance(state, dict) and isinstance(state.get(name), dict):
            state = state[name]
            break
    keys = tuple(state) if isinstance(state, dict) else ()
    swin_v1 = any(k.endswith('layers.0.blocks.0.attn.relative_position_bias_table') for k in keys)
    auxiliary = any('aux_head.' in k for k in keys)
    if swin_v1 or auxiliary:
        # Tensor signatures identify the architecture, not the historical crop
        # or normalization. Use the documented project recipe defaults and make
        # this assumption visible; do not invent a historical metadata sidecar.
        recipe = 'loveda' if auxiliary and cfg.DATASET == 'LoveDA' else 'geoseg'
        love = recipe == 'loveda'
        print(f'Checkpoint metadata missing: detected {"Swin v1" if swin_v1 else "UNetformer auxiliary head"}; '
              f'using {recipe} defaults (ImageNet normalization). '
              'Copy the original .pth.json to reproduce nondefault evaluation settings.')
        return {'recipe': recipe, 'normalization': 'imagenet',
                'crop_size': 512 if love else 768,
                'eval_crop_size': 256 if love else 1024,
                'eval_stride': 128 if love else 1024,
                'eval_batch': 1 if not love else 10,
                'metadata_source': 'architecture_signature_and_recipe_defaults'}
    return {'recipe': 'legacy'}


def write_metadata(checkpoint, args):
    data = {'recipe':args.recipe, 'dataset':cfg.DATASET, 'model':args.model,
            'num_classes':cfg.N_CLASSES, 'crop_size':cfg.WINDOW_SIZE[0],
            'eval_crop_size':args.eval_crop_size,
            'normalization':'imagenet' if args.recipe in ('geoseg','loveda') else 'divide_255',
            'eval_stride':args.eval_stride, 'eval_batch':args.eval_batch,
            'architecture':('swin_v1_base' if args.model=='FTUNetformer' else 'unetformer_aux') if args.recipe in ('geoseg','loveda') else ('deepkan_resnet18' if args.model=='DeepKANSeg' else 'legacy'),
            'run_name':args.run_name}
    metadata_path(checkpoint).write_text(json.dumps(data,indent=2),encoding='utf-8')
