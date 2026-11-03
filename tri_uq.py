"""Pluggable stochastic Dirichlet head for the local segmenters.

The three maps are a decomposition of predictive class entropy under the
specified mask/Dirichlet hierarchy, not identified physical noise sources.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


MODEL_SPECS = {
    "ABCNet": ("conv_out", 256),
    "CMTFNet": ("decoder.seg_head.head", 256),
    "UNetformer": ("decoder.segmentation_head", 64),
    "FTUNetformer": ("decoder.segmentation_head", 256),
    "DeepKANSeg": ("decoder.segmentation_head", 64),
}


def build_backbone(name: str, classes: int = 6, pretrained: bool = False, recipe: str = 'legacy'):
    """Construct local models without an implicit network download."""
    if recipe == 'loveda' and name != 'UNetformer':
        raise ValueError('LoveDA recipe supports UNetformer only')
    if recipe in ('geoseg','loveda'):
        from baseline_geoseg import create_model, load_pretrain
        model = create_model(name, classes)
        if pretrained:
            load_pretrain(model, name)
        return model
    if name == "ABCNet":
        from model.ABCNet import ABCNet
        model = ABCNet(num_classes=classes, pretrained=pretrained)
    elif name == "CMTFNet":
        from model.CMTFNet.CMTFNet import CMTFNet
        from model.CMTFNet.ResNet import ResNet50
        model = CMTFNet(num_classes=classes, backbone=lambda: ResNet50(pretrained=pretrained))
    elif name == "UNetformer":
        from model.UNetFormer import UNetFormer
        model = UNetFormer(num_classes=classes, pretrained=pretrained)
    elif name == "DeepKANSeg":
        from model.deepKAN_ResNet import UNetFormer as DeepKANSeg
        model = DeepKANSeg(num_classes=classes, pretrained=pretrained)
    elif name == "FTUNetformer":
        from model.FTUNetFormer import ft_unetformer
        weight_path = Path(__file__).resolve().parent / "pretrain" / "swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth"
        model = ft_unetformer(num_classes=classes, pretrained=pretrained,
                              weight_path=str(weight_path))
    else:
        raise ValueError(f"Unknown model {name!r}; choose {tuple(MODEL_SPECS)}")
    return model


class StochasticDirichletHead(nn.Module):
    def __init__(self, channels: int, classes: int = 6, hidden: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        if classes < 2 or not 0 < dropout < 1:
            raise ValueError("classes must be >=2 and dropout in (0,1)")
        self.classes = classes
        # PUGNet's transferable idea is to use coarse predictive ambiguity as
        # a spatial feature-refinement guide. Keep this as a separate guidance
        # signal; it is not one of the three probabilistic uncertainty terms.
        self.context_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=d, dilation=d,
                          groups=channels, bias=False),
                nn.GELU(),
            ) for d in (1, 2, 4)
        ])
        self.context_fuse = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.zeros_(self.context_fuse.weight)
        self.project = nn.Conv2d(channels, hidden, 1)
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(hidden, classes, 1)
        # Start as an exact identity adapter: retain the pretrained segmenter
        # logits and learn a stochastic residual instead of relearning classes.
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def prepare(self, features: torch.Tensor, coarse_logits: torch.Tensor) -> torch.Tensor:
        # PUG-style entropy guidance at the decoder feature resolution. Detach
        # the guide so the backbone cannot lower its loss by manipulating the
        # attention mask; the segmentation gradients still train the refiner.
        coarse_p = coarse_logits.detach().float().softmax(1)
        guide = -(coarse_p * coarse_p.clamp_min(1e-8).log()).sum(1, keepdim=True)
        guide = (guide / math.log(self.classes)).clamp(0.0, 1.0)
        guide = F.interpolate(guide, size=features.shape[-2:], mode="bilinear",
                              align_corners=False)
        context = sum(branch(features) for branch in self.context_branches) / len(self.context_branches)
        refined = features + guide.to(features.dtype) * self.context_fuse(context)
        return F.gelu(self.project(refined))

    def sample_logits(self, prepared: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        # Explicit training=True also samples at inference while the backbone stays eval().
        z = self.classifier(F.dropout2d(prepared, self.dropout.p, training=True))
        residual = F.interpolate(z, size=base_logits.shape[-2:], mode="bilinear", align_corners=False)
        return base_logits + residual


class PlugInSegmenter(nn.Module):
    def __init__(self, backbone: nn.Module, name: str, classes: int = 6,
                 hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        path, channels = MODEL_SPECS[name]
        self.backbone = backbone
        self.head = StochasticDirichletHead(channels, classes, hidden, dropout)
        self.name = name
        self._feature = None
        self.aux_logits = None
        module = self.backbone.get_submodule(path)
        self._hook = module.register_forward_pre_hook(self._capture)

    def _capture(self, _module, inputs):
        self._feature = inputs[0]

    def features_and_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._feature = None
        self.aux_logits = None
        base_logits = self.backbone(x)
        if isinstance(base_logits, tuple):
            if len(base_logits) != 2:
                raise RuntimeError('Expected main and auxiliary baseline outputs')
            base_logits, self.aux_logits = base_logits
        feature, self._feature = self._feature, None
        if feature is None or feature.ndim != 4:
            raise RuntimeError(f"{self.name} did not expose a 4-D decoder feature")
        if feature.shape[0] != x.shape[0]:
            raise RuntimeError("Feature batch size differs from input")
        if not isinstance(base_logits, torch.Tensor) or base_logits.ndim != 4:
            raise RuntimeError(f"{self.name} must return [B,K,H,W] logits")
        if base_logits.shape[1] != self.head.classifier.out_channels:
            raise RuntimeError(f"{self.name} returned {base_logits.shape[1]} classes; expected {self.head.classifier.out_channels}")
        return feature, base_logits

    def forward(self, x: torch.Tensor, samples: int = 2) -> torch.Tensor:
        if samples < 1:
            raise ValueError("samples must be positive")
        features, base_logits = self.features_and_logits(x)
        prepared = self.head.prepare(features, base_logits)
        return torch.stack([self.head.sample_logits(prepared, base_logits)
                            for _ in range(samples)])


def entropy(p: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return -(p * p.clamp_min(1e-30).log()).sum(dim)


def alpha_from_logits(logits: torch.Tensor) -> torch.Tensor:
    # Exact exp(z) parameterization in the paper. The finite clamp is a
    # documented numerical approximation, never an evidence+1 conversion.
    return logits.float().clamp(-12.0, 12.0).exp()


def dirichlet_kl(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """KL[Dir(alpha) || Dir(beta)] for tensors with class axis 1."""
    alpha, beta = alpha.float(), beta.float()
    a0, b0 = alpha.sum(1), beta.sum(1)
    return (torch.lgamma(a0) - torch.lgamma(b0)
            - torch.lgamma(alpha).sum(1) + torch.lgamma(beta).sum(1)
            + ((alpha - beta) * (torch.digamma(alpha) - torch.digamma(a0[:, None]))).sum(1))


def decompose(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """logits: [T,B,K,H,W], outputs [B,H,W] except mean_p [B,K,H,W]."""
    if logits.ndim != 5 or logits.shape[0] < 1:
        raise ValueError("Expected [T,B,K,H,W]")
    alpha = alpha_from_logits(logits)
    p = alpha / alpha.sum(2, keepdim=True)
    a0 = alpha.sum(2)
    a = torch.digamma(a0 + 1) - (p * torch.digamma(alpha + 1)).sum(2)
    hp = entropy(p, dim=2)
    mean_p = p.mean(0)
    data = a.mean(0)
    dist = (hp - a).mean(0)
    model = entropy(mean_p) - hp.mean(0)
    return {"mean_logits": logits.mean(0), "mean_p": mean_p,
            "total": entropy(mean_p), "data": data,
            "dist": dist, "model": model, "mean_concentration": a0.mean(0)}


def decompose_stable(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """Unclipped exp(z) hierarchy, float64/log-domain inference alternative.

    This explicitly differs from historical clipped training. Process one MC
    pass at a time to bound memory; no logits or probabilities are clipped.
    """
    if logits.ndim != 5 or logits.shape[0] < 1:
        raise ValueError('Expected [T,B,K,H,W]')
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite logits cannot define a Dirichlet predictor')

    def digamma_correction(log_value):
        # psi(exp(l)+1)-l; asymptotic form avoids exponent overflow/cancellation.
        inv = (-log_value.clamp_min(20)).exp()
        asymptotic = .5 * inv - inv.square() / 12
        exact = torch.digamma(log_value.clamp_max(20).exp() + 1) - log_value
        return torch.where(log_value > 20, asymptotic, exact)

    sum_p = sum_h = sum_data = sum_dist = sum_log_s = None
    for sample in logits:
        z = sample.double()
        log_s = torch.logsumexp(z, 1)
        log_p = z - log_s[:, None]
        p = log_p.exp()
        h = -(p * log_p).sum(1)
        dist = (p * digamma_correction(z)).sum(1) - digamma_correction(log_s)
        dist = torch.minimum(dist.clamp_min(0), h)  # roundoff bounds only
        data = h - dist
        if sum_p is None:
            sum_p, sum_h, sum_data, sum_dist, sum_log_s = p, h, data, dist, log_s
        else:
            sum_p = sum_p + p
            sum_h = sum_h + h
            sum_data = sum_data + data
            sum_dist = sum_dist + dist
            sum_log_s = sum_log_s + log_s
    count = logits.shape[0]
    mean_p = sum_p / count
    total = -(mean_p * mean_p.clamp_min(torch.finfo(torch.float64).tiny).log()).sum(1)
    model = (total - sum_h / count).clamp_min(0)
    return {'mean_logits': logits.mean(0), 'mean_p': mean_p.float(),
            'total': total.float(), 'data': (sum_data / count).float(),
            'dist': (sum_dist / count).float(), 'model': model.float(),
            'mean_log_concentration': sum_log_s / count}


@dataclass
class LossConfig:
    plain_ce: bool = False
    beta_boost: float = 100.0
    lambda_id: float = 0.01
    lambda_ood: float = 0.01
    dist_focus_weight: float = 0.5
    model_focus_weight: float = 0.25
    gamma: float = 1.0
    eta_max: float = 0.5
    warmup_epochs: int = 10
    ramp_end_epoch: int = 20


def objective(logits: torch.Tensor, labels: torch.Tensor,
              ood_mask: torch.Tensor | None, epoch: int,
              config: LossConfig) -> dict[str, torch.Tensor]:
    """Mean per region, with stopped-gradient data weights and ignored labels."""
    t, b, k, h, w = logits.shape
    if labels.shape != (b, h, w):
        raise ValueError("Label shape differs from logits")
    valid = (labels >= 0) & (labels < k)
    ood = torch.zeros_like(valid) if ood_mask is None else ood_mask.bool()
    if ood.shape != valid.shape:
        raise ValueError("OOD mask shape differs from labels")
    id_mask = valid & ~ood
    ood = ood & valid
    alpha = alpha_from_logits(logits)
    safe_labels = labels.clamp(0, k - 1)
    beta_id = (torch.ones_like(alpha[0]).scatter_add(
        1, safe_labels[:, None], torch.full_like(safe_labels[:, None], config.beta_boost,
                                                 dtype=alpha.dtype)))
    kl_id = torch.stack([dirichlet_kl(alpha[i], beta_id) for i in range(t)]).mean(0)
    kl_ood = torch.stack([dirichlet_kl(alpha[i], torch.ones_like(alpha[i]))
                          for i in range(t)]).mean(0)
    maps = decompose(logits)
    if config.ramp_end_epoch <= config.warmup_epochs:
        raise ValueError("ramp_end_epoch must exceed warmup_epochs")
    eta = config.eta_max * min(1.0, max(0.0,
        (epoch - config.warmup_epochs) / (config.ramp_end_epoch - config.warmup_epochs)))
    weight = 1 - eta + eta * torch.exp(-config.gamma *
                                      (maps["data"].detach() / math.log(k)))
    # U_dist and U_model guide online supervised hard-pixel mining. Normalize
    # each score per image, clip extreme ratios, and stop gradients through the
    # weights so the network cannot reduce its loss by gaming its own weights.
    # Both influences follow the same warmup/ramp as the uncertainty strategy.
    id_count = id_mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1)
    has_id_pixels = id_mask.any(dim=(-2, -1), keepdim=True)

    def relative_focus(score):
        score = score.detach().clamp_min(0)
        score_mean = (score * id_mask).sum(dim=(-2, -1), keepdim=True) / id_count
        relative = (score / score_mean.clamp_min(1e-8)).clamp(max=3.0)
        relative_mean = ((relative * id_mask).sum(dim=(-2, -1), keepdim=True)
                         / id_count).clamp_min(1e-8)
        relative = relative / relative_mean
        return torch.where((score_mean > 1e-8) & has_id_pixels,
                           relative, torch.ones_like(relative))

    dist_relative = relative_focus(maps["dist"])
    model_relative = relative_focus(maps["model"])
    focus_ramp = eta / config.eta_max if config.eta_max > 0 else 0.0
    dist_weight = 1.0 + config.dist_focus_weight * focus_ramp * (dist_relative - 1.0)
    model_weight = 1.0 + config.model_focus_weight * focus_ramp * (model_relative - 1.0)
    focus_weight = dist_weight * model_weight
    focus_mean = (focus_weight * id_mask).sum(dim=(-2, -1), keepdim=True) / id_count
    focus_weight = torch.where(has_id_pixels, focus_weight / focus_mean.clamp_min(1e-8),
                               torch.ones_like(focus_weight))
    # Smooth q=(1+b*onehot)/(K+b), as specified in the theory.
    logp = logits.float().clamp(-12.0, 12.0).log_softmax(2)
    selected = logp.gather(2, safe_labels[None, :, None].expand(t, -1, -1, -1, -1)).squeeze(2)
    ce = (-(logp.sum(2) + config.beta_boost * selected)
          / (k + config.beta_boost)).mean(0)
    if config.plain_ce:
        ce = -selected.mean(0)
    zero = logits.sum() * 0.0
    combined_weight = weight * focus_weight
    seg = (combined_weight[id_mask] * ce[id_mask]).mean() if id_mask.any() else zero
    id_kl = kl_id[id_mask].mean() if id_mask.any() else zero
    ood_kl = kl_ood[ood].mean() if ood.any() else zero
    total = seg + config.lambda_id * id_kl + config.lambda_ood * ood_kl
    return {"total": total, "seg": seg, "id_kl": id_kl, "ood_kl": ood_kl,
            "mean_u_data": maps["data"][id_mask].mean() if id_mask.any() else zero,
            "mean_u_dist": maps["dist"][id_mask].mean() if id_mask.any() else zero,
            "mean_u_model": maps["model"][id_mask].mean() if id_mask.any() else zero,
            "mean_weight": combined_weight[id_mask].mean() if id_mask.any() else zero,
            "mean_dist_focus_weight": dist_weight[id_mask].mean() if id_mask.any() else zero,
            "max_dist_focus_weight": dist_weight[id_mask].max() if id_mask.any() else zero,
            "mean_model_focus_weight": model_weight[id_mask].mean() if id_mask.any() else zero,
            "max_model_focus_weight": model_weight[id_mask].max() if id_mask.any() else zero,
            "max_combined_focus_weight": focus_weight[id_mask].max() if id_mask.any() else zero,
            "mean_concentration": maps["mean_concentration"].mean(),
            "clamp_fraction": ((logits < -12) | (logits > 12)).float().mean()}


@torch.no_grad()
def predict_native_baseline(model: nn.Module, x: torch.Tensor, samples: int = 16,
                            numerics: str = 'checkpoint') -> dict:
    """Sample the existing terminal Dropout2d, without adding/fitting a head.

    Intended for native UNetformer / FTUNetformer diagnostics. Dirichlet
    concentration is implicit in baseline logits and has not been calibrated.
    """
    if samples < 2 or numerics not in ('checkpoint', 'stable'):
        raise ValueError('Native MC requires samples>=2 and a known numeric protocol')
    model.eval()
    head = model.get_submodule('decoder.segmentation_head')
    if not isinstance(head, nn.Sequential) or len(head) != 3 or not isinstance(head[1], nn.Dropout2d) or head[1].p <= 0:
        raise ValueError('Native MC requires the existing Conv/Dropout2d/Classifier head')
    captured = {}
    hook = head[1].register_forward_pre_hook(lambda _m, inputs: captured.update(features=inputs[0]))
    try:
        baseline_logits = model(x)
    finally:
        hook.remove()
    if isinstance(baseline_logits, tuple):
        baseline_logits = baseline_logits[0]
    features = captured['features']
    logits = torch.stack([
        F.interpolate(head[2](F.dropout2d(features, p=head[1].p, training=True)),
                      size=x.shape[-2:], mode='bilinear', align_corners=False)
        for _ in range(samples)])
    out = decompose_stable(logits) if numerics == 'stable' else decompose(logits)
    out['mean_logits'] = baseline_logits  # preserve deterministic baseline metrics
    out['mean_p'] = baseline_logits.float().softmax(1)
    out['samples'] = samples
    return out


@torch.no_grad()
def predict(model: PlugInSegmenter, x: torch.Tensor, samples: int = 16,
            pilot: int = 0, delta2: float = 0.001, min_samples: int = 8,
            max_samples: int = 32, numerics: str = 'checkpoint') -> dict[str, torch.Tensor | int]:
    model.eval()
    features, base_logits = model.features_and_logits(x)
    prepared = model.head.prepare(features, base_logits)
    if pilot:
        if pilot < 2 or delta2 <= 0:
            raise ValueError("pilot >=2 and delta2 >0 required")
        p0 = torch.stack([F.softmax(model.head.sample_logits(prepared, base_logits), 1)
                          for _ in range(pilot)])
        v = ((p0 - p0.mean(0)).square().sum(2).sum(0) / (pilot - 1)).mean().item()
        samples = min(max_samples, max(min_samples, math.ceil(v / delta2)))
    # Pilot samples are deliberately excluded from this formal estimate.
    logits = torch.stack([model.head.sample_logits(prepared, base_logits)
                          for _ in range(samples)])
    if numerics not in ('checkpoint', 'stable'):
        raise ValueError('Unknown UQ numeric protocol')
    out = decompose_stable(logits) if numerics == 'stable' else decompose(logits)
    out["prediction"] = out["mean_p"].argmax(1)
    out["samples"] = samples
    return out
