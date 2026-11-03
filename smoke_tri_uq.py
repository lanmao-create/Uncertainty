"""Numerical and 256x256 interface checks for the plug-in models."""
import gc
import math

import torch

from tri_uq import (LossConfig, PlugInSegmenter, StochasticDirichletHead, build_backbone,
                    decompose, dirichlet_kl, objective, predict)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)
    z = torch.randn(2, 1, 6, 16, 16, device=device, requires_grad=True)
    y = torch.randint(0, 6, (1, 16, 16), device=device)
    mask = torch.zeros_like(y, dtype=torch.bool)
    mask[:, 4:12, 4:12] = True
    maps = decompose(z)
    assert (maps["total"] - maps["data"] - maps["dist"] - maps["model"]).abs().max() < 2e-5
    assert all(maps[k].min() > -2e-5 for k in ("data", "dist", "model"))
    losses = objective(z, y, mask, 20, LossConfig())
    assert abs(float(losses["mean_dist_focus_weight"]) - 1.0) < 1e-5
    assert abs(float(losses["mean_model_focus_weight"]) - 1.0) < 1e-5
    assert float(losses["max_dist_focus_weight"]) > 1.0
    assert float(losses["max_model_focus_weight"]) > 1.0
    losses["total"].backward()
    assert torch.isfinite(z.grad).all()
    a = torch.rand(2,6,3,4,device=device) + 0.2
    b = torch.rand_like(a) + 0.2
    reference = torch.distributions.kl_divergence(
        torch.distributions.Dirichlet(a.permute(0,2,3,1)),
        torch.distributions.Dirichlet(b.permute(0,2,3,1)))
    assert (dirichlet_kl(a,b) - reference).abs().max() < 1e-4
    print("formula, KL, OOD mask, Udist/Umodel focus weighting, gradient: OK")
    head = StochasticDirichletHead(8, 6, hidden=4)
    f = torch.randn(2, 8, 12, 12, requires_grad=True)
    coarse = torch.randn(2, 6, 24, 24, requires_grad=True)
    prepared = head.prepare(f, coarse)
    initial = head.sample_logits(prepared, coarse)
    assert torch.equal(initial, coarse), "zero-initialized adapter must preserve baseline logits"
    prepared.sum().backward(retain_graph=True)
    assert coarse.grad is None, "coarse entropy guidance is intentionally detached"
    f.grad = None
    with torch.no_grad():
        head.classifier.weight.fill_(0.01)
    head.sample_logits(head.prepare(f, coarse), coarse).square().mean().backward()
    assert head.context_fuse.weight.grad is not None
    print("entropy-guided refiner identity, gradient, and detached guide: OK")
    del z, y, mask, maps, losses, a, b, reference
    gc.collect()
    for name in ("ABCNet", "CMTFNet", "UNetformer", "FTUNetformer", "DeepKANSeg"):
        model = PlugInSegmenter(build_backbone(name, pretrained=False), name).to(device).eval()
        x = torch.randn(1,3,256,256,device=device)
        with torch.no_grad():
            result = predict(model,x,samples=2,pilot=2,min_samples=2,max_samples=2)
        assert result["mean_p"].shape == (1,6,256,256)
        assert result["samples"] == 2
        assert (result["mean_p"].sum(1) - 1).abs().max() < 1e-5
        assert (result["total"] - result["data"] - result["dist"]
                - result["model"]).abs().max() < 2e-5
        print(name, "256x256 forward: OK")
        del model, x, result
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"PASS on {device}")


if __name__ == "__main__":
    main()
