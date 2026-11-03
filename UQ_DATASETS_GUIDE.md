# Vaihingen / Potsdam / LoveDA UQ

**Potsdam 的新版 UNetformer / FTUNetformer baseline 已接入 GeoSeg UQ 配方。**
这两者的训练/验证裁块、优化器和默认步数由权重配方决定，详见
[POTSDAM_UQ_GEOSEG_GUIDE.md](POTSDAM_UQ_GEOSEG_GUIDE.md)。下文统一256裁块等设置只适用于 legacy 配方。

Baseline uses `train.py`; UQ uses `train_uq.py`. Default dataset remains Vaihingen.
All default paths are relative to the script's project folder, independent of shell location.

## Outputs

```text
resultsUQ/<dataset>/<model>/<model>_UQ_epochNN_mIoU0.xxxx.pth
UQV/<dataset>/<model>/Prediction/
UQV/<dataset>/<model>/Udata/
UQV/<dataset>/<model>/Udist/
UQV/<dataset>/<model>/Umodel/
UQV/<dataset>/<model>/Record/
uq_runs/<dataset>/<model>/uq/<run-name>/
```

Directories are created automatically. Improving weights are retained; shared visualizations
are replaced as a complete set only when a UQ candidate beats the baseline threshold and
the previous archived best. A run name already containing training artifacts is rejected;
use a new run name for a fresh experiment. Existing moved results are not rewritten.

## Baseline initialization

Vaihingen: highest score matching `<model>_epochNN_SCORE[.pth]` in `results_vaihingen/`.
Potsdam: same matching rule in `results_potsdam/<model>/`.
UQ weights are never automatically selected as baseline initialization.
Wait until the corresponding baseline experiment finishes before launching UQ.

## Dataset settings

Both datasets reuse splits and filename templates in `utils.py`. Both have six output
classes and select checkpoints with baseline-style mean IoU over the first five classes.
Both use RGB/TOP first three channels divided by 255, no DSM.

| Setting | Vaihingen | Potsdam |
|---|---|---|
| Train / validation=test tiles | 12 / 4 | 18 / 6 |
| Evaluation stride | 32 | 128 |
| Preview training tile | 1 | 6_10 |
| Optional ID threshold calibration | historical tile 38 | training tiles, not held out |

Defaults: patch 256, batch 10, eval batch 10, 1000 steps/epoch, 50 epochs,
5 head-only warmup epochs, head LR 0.001, backbone LR 0.0001, train MC 2, eval MC 16.
The existing UQ objective, heatmap names, palette, and improvement strategies are preserved.
Without external OOD data there is no OOD KL supervision. Computed ID uncertainty remains
available to the existing training strategies; this is not evidence of trained OOD detection.

## Potsdam commands (run inside Vaihingen)

```powershell
python train_uq.py --dataset Potsdam --mode Train --model ABCNet --run-name abcnet_potsdam_uq_s42_v1
```

```powershell
python train_uq.py --dataset Potsdam --mode Train --model CMTFNet --run-name cmtfnet_potsdam_uq_s42_v1
```

```powershell
python train_uq.py --dataset Potsdam --mode Train --model UNetformer --run-name unetformer_potsdam_uq_s42_v1
```

```powershell
python train_uq.py --dataset Potsdam --mode Train --model FTUNetformer --run-name ftunetformer_potsdam_uq_s42_v1
```

Vaihingen rerun: replace `--dataset Potsdam` with `--dataset Vaihingen` and choose a fresh run name.
Optional dataset override: `--data-root "E:/datasets/Potsdam"` (directly contains
`4_Ortho_RGBIR`, `5_Labels_for_participants`, and `5_Labels_for_participants_no_Boundary`).

Test example (same run name as training):

```powershell
python train_uq.py --dataset Potsdam --mode Test --model ABCNet --run-name abcnet_potsdam_uq_s42_v1
```

The baseline is reevaluated at UQ startup using the same labels, stride, and metric.
Final selection can fall back to that original baseline if UQ does not reach its score.
This protects checkpoint selection, not a guarantee that UQ training will improve accuracy
or generalize better. Validation and test remain the same user-defined split.
## LoveDA Urban

LoveDA reuses the unmodified Urban lists and palette from `utils_loveda.py`:
1156 training images and 677 validation/test images. Input is RGB /255.
Raw mask 0 is ignored (255 internally), raw 1..7 map to classes 0..6.
All four backbones and UQ heads output seven classes. Selection, baseline fallback,
and global-best comparisons use seven-class baseline-style nanmean IoU (`miou_7`).
Uncertainty normalization uses log(7); ISPRS continues to use log(6) and `miou_5`.

Initialization selects the highest native baseline in `results_loveda/<model>/`.
Outputs follow the same layout under `resultsUQ/LoveDA/<model>/`,
`UQV/LoveDA/<model>/` and `uq_runs/LoveDA/<model>/uq/<run-name>/`.
Prediction colors, three heatmap names, Record NPZ files, and checkpoint names are preserved.

Defaults remain 50 epochs, 1000 steps/epoch, batch/eval-batch 10, patch 256,
eval stride 128, head warmup 5, train MC 2, eval MC 16. Preview uses training image 1366.
LoveDA disables full-tile training cache to bound CPU RAM. Full split IDs are stored
in config.json but are not printed in the terminal.
Ordinary validation does not retain calibration score arrays. Optional calibration
uses training images and up to 4096 evenly spaced valid pixels per image; its
threshold is an approximate ID quantile, not an independently validated OOD detector.

Wait for each model's baseline to finish before using the commands below:

```powershell
python train_uq.py --dataset LoveDA --mode Train --model ABCNet --run-name abcnet_loveda_uq_s42_v1
```

```powershell
python train_uq.py --dataset LoveDA --mode Train --model CMTFNet --run-name cmtfnet_loveda_uq_s42_v1
```

```powershell
python train_uq.py --dataset LoveDA --mode Train --model UNetformer --run-name unetformer_loveda_uq_s42_v1
```

```powershell
python train_uq.py --dataset LoveDA --mode Train --model FTUNetformer --run-name ftunetformer_loveda_uq_s42_v1
```
