"""Short entry point: native baseline checkpoint -> stochastic UQ head training."""
from __future__ import annotations

import argparse
from pathlib import Path

import uq_config as cfg
from run_tri_uq import main as run
from tri_uq import MODEL_SPECS
from uq_runtime import omit_unspecified


def resolve_baseline_checkpoint(model: str, override: str | None = None, dataset: str = cfg.DATASET) -> Path:
    """Keep the CLI helper while resolving from the live baseline folder."""
    return cfg.resolve_baseline_checkpoint(model, override, dataset)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default=cfg.MODE, choices=("Train", "Test", "Calibrate"))
    parser.add_argument("--model", default=cfg.MODEL, choices=MODEL_SPECS)
    parser.add_argument("--dataset", choices=("Vaihingen", "Potsdam", "LoveDA"), default=cfg.DATASET)
    parser.add_argument("--data-root", help="Override the selected dataset root")
    parser.add_argument("--recipe", choices=("auto","legacy","geoseg","loveda"), default="auto")
    parser.add_argument("--patch", type=int, default=None)
    parser.add_argument("--eval-patch", type=int, default=None)
    parser.add_argument("--decoder-lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--baseline-checkpoint")
    parser.add_argument("--checkpoint", help="optional checkpoint override; otherwise uses selected_checkpoint.json")
    parser.add_argument("--ood-dir")
    parser.add_argument("--no-cache", action="store_true", help="Read tiles on demand to reduce CPU RAM use")
    parser.add_argument("--results-uq", default=str(cfg.RESULTS_UQ),
                        help="directory for incremental best UQ checkpoints")
    parser.add_argument("--run-name", default=cfg.RUN_NAME)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--head-warmup-epochs", type=int, default=cfg.HEAD_WARMUP_EPOCHS)
    parser.add_argument("--head-lr", type=float, default=None)
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--eta-max", type=float, default=cfg.ETA_MAX)
    parser.add_argument("--lambda-id", type=float, default=cfg.LAMBDA_ID)
    parser.add_argument("--lambda-ood", type=float, default=cfg.LAMBDA_OOD)
    parser.add_argument("--dist-focus-weight", type=float, default=cfg.DIST_FOCUS_WEIGHT,
                        help="strength of detached Udist-based hard-pixel weighting (0 disables)")
    parser.add_argument("--model-focus-weight", type=float, default=cfg.MODEL_FOCUS_WEIGHT,
                        help="strength of detached Umodel-based MC disagreement weighting (0 disables)")
    parser.add_argument("--samples", type=int, default=cfg.EVAL_SAMPLES)
    parser.add_argument("--pilot", type=int, default=0)
    parser.add_argument("--delta2", type=float, default=0.001)
    parser.add_argument("--min-samples", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--false-positive-rate", type=float, default=0.05)
    parser.add_argument("--weight-warmup-epochs", type=int, default=cfg.DATA_WEIGHT_WARMUP)
    parser.add_argument("--weight-ramp-end", type=int, default=cfg.DATA_WEIGHT_RAMP_END)
    parser.add_argument("--log-every", type=int, default=cfg.LOG_EVERY)
    parser.add_argument("--preview-every", type=int, default=cfg.PREVIEW_EVERY)
    parser.add_argument("--eval-stride", type=int, default=cfg.EVAL_STRIDE)
    parser.add_argument("--eval-batch", type=int, default=None)
    args = parser.parse_args(argv)
    if args.mode != "Train":
        from infer_uq import main as infer
        options = ["test" if args.mode == "Test" else "calibrate", "--model", args.model,
                   "--run-name", args.run_name, "--dataset", args.dataset,
                   "--eval-stride", str(args.eval_stride), "--eval-batch", str(args.eval_batch), "--seed", str(args.seed),
                   "--samples", str(args.samples), "--pilot", str(args.pilot),
                   "--delta2", str(args.delta2), "--min-samples", str(args.min_samples),
                   "--max-samples", str(args.max_samples),
                   "--false-positive-rate", str(args.false_positive_rate)]
        if args.checkpoint:
            options += ["--checkpoint", args.checkpoint]
        if args.data_root:
            options += ["--data-root", args.data_root]
        options += ["--recipe",args.recipe,"--patch",str(args.patch),"--eval-patch",str(args.eval_patch),"--device",str(args.device)]
        infer(omit_unspecified(options))
        return
    try:
        checkpoint = resolve_baseline_checkpoint(args.model, args.baseline_checkpoint, args.dataset)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    options = ["train", "--model", args.model, "--method", "uq",
               "--dataset", args.dataset, "--checkpoint", str(Path(checkpoint).resolve()),
               "--output", str(cfg.OUTPUT), "--run-name", args.run_name,
               "--results-uq", str(Path(args.results_uq)),
               "--seed", str(args.seed), "--epochs", str(args.epochs),
               "--steps-per-epoch", str(args.steps_per_epoch),
               "--batch", str(args.batch),
               "--head-warmup-epochs", str(args.head_warmup_epochs),
               "--lr", str(args.head_lr), "--backbone-lr", str(args.backbone_lr),
               "--eta-max", str(args.eta_max),
               "--lambda-id", str(args.lambda_id),
               "--lambda-ood", str(args.lambda_ood),
                "--dist-focus-weight", str(args.dist_focus_weight),
                "--model-focus-weight", str(args.model_focus_weight),
               "--warmup", str(args.weight_warmup_epochs),
               "--ramp-end", str(args.weight_ramp_end),
               "--log-every", str(args.log_every),
               "--train-samples", str(cfg.TRAIN_SAMPLES),
               "--samples", str(args.samples), "--pilot", str(args.pilot),
               "--delta2", str(args.delta2), "--min-samples", str(args.min_samples),
               "--max-samples", str(args.max_samples),
               "--false-positive-rate", str(args.false_positive_rate),
               "--preview-every", str(args.preview_every),
               "--preview-tile", cfg.PREVIEW_TILE or ({"Potsdam":"6_10", "LoveDA":"1366", "Vaihingen":"1"}[args.dataset]),
               "--preview-row", str(cfg.PREVIEW_ROW),
               "--preview-col", str(cfg.PREVIEW_COL)]
    options += ["--eval-stride", str(args.eval_stride), "--eval-batch", str(args.eval_batch)]
    ood_dir = args.ood_dir or cfg.OOD_DIR
    if ood_dir:
        options += ["--ood-dir", ood_dir]
    if args.data_root:
        options += ["--data-root", args.data_root]
    if args.no_cache or args.dataset == "LoveDA":
        options += ["--no-cache"]
    options += ["--recipe",args.recipe,"--patch",str(args.patch),"--eval-patch",str(args.eval_patch),
                "--decoder-lr",str(args.decoder_lr),"--device",str(args.device)]
    run(omit_unspecified(options))


if __name__ == "__main__":
    main()

