"""LoveDA Urban UNetFormer recipe; retains the user's split and evaluation.

Uses the restored GeoSeg auxiliary-head architecture, RGB ImageNet normalization,
random scale + SmartCrop-style 512 crops and AdamW/Lookahead + cosine schedule.
No Potsdam mosaic, no additional data or held-out images enter training.
"""
import random
from PIL import Image
import numpy as np
import torch

import utils as cfg
import baseline_geoseg as geo

normalize = geo.normalize
SegmentationLoss = geo.SegmentationLoss


class LoveDAPatches(cfg.ISPRS_dataset):
    """Random scale, <=11 smart-crop attempts and horizontal flip, no mosaic."""
    def __getitem__(self, index):
        i = random.randrange(len(self.data_files))
        rgb = self._read(i, self.data_files, self.data_cache_, 'image').transpose(1,2,0)
        mask = self._read(i, self.label_files, self.label_cache_, 'label')
        if rgb.shape[:2] != mask.shape:
            raise ValueError(f'Image/label size mismatch: {self.data_files[i]}')
        h,w = mask.shape
        scale = random.choice((.75,1.,1.25,1.5))
        nh,nw = max(1,int(h*scale)),max(1,int(w*scale))
        rgb = np.asarray(Image.fromarray(rgb).resize((nw,nh),Image.Resampling.BICUBIC))
        mask = np.asarray(Image.fromarray(mask).resize((nw,nh),Image.Resampling.NEAREST))
        ch,cw = self.window_size
        if nh < ch or nw < cw:
            ph,pw = max(0,ch-nh),max(0,cw-nw)
            rgb = np.pad(rgb,((ph//2,ph-ph//2),(pw//2,pw-pw//2),(0,0)))
            mask = np.pad(mask,((ph//2,ph-ph//2),(pw//2,pw-pw//2)),constant_values=255)
        h,w = mask.shape
        for _ in range(11):
            r,c = random.randint(0,h-ch),random.randint(0,w-cw)
            labels = mask[r:r+ch,c:c+cw]
            _,counts = np.unique(labels[labels<cfg.N_CLASSES],return_counts=True)
            if len(counts)>1 and counts.max()/counts.sum()<.75:
                break
        rgb = rgb[r:r+ch,c:c+cw]
        if self.augmentation and random.random()<.5:
            rgb,labels = np.flip(rgb,axis=1),np.flip(labels,axis=1)
        x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).float().div_(255)
        x = normalize(x[None])[0]
        y = torch.from_numpy(np.ascontiguousarray(labels)).long()
        return x,torch.zeros_like(y),torch.zeros_like(y),y


def optimizer_and_scheduler(model, lr=6e-4, backbone_lr=6e-5, epochs=50):
    optimizer,_ = geo.optimizer_and_scheduler(model,'UNetformer',lr,backbone_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=1e-6)
    return optimizer,scheduler
