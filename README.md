# BCRR-Mamba-UNet

Boundary-Conditioned Reset-and-Routing Mamba U-Net for lightweight skin lesion segmentation.

This project is an independent implementation built on the EGE-UNet/LB-UNet design space. The proposed block keeps a grouped, compressed Mamba core, scans row/column directions with shared parameters, predicts spatial direction weights, and uses region uncertainty plus boundary evidence to control state mixing. It also retains multi-scale boundary supervision and adds a contour continuity loss.

## Layout

```text
BCRR-Mamba-UNet/
  bcrr/
    models.py       network and every architecture ablation
    data.py         ISIC/PH2 loading and boundary generation
    losses.py       segmentation, boundary, gate and contour losses
    metrics.py      Dice, IoU, mIoU, HD95 and boundary F1
    engine.py       training/evaluation and prediction images
    utils.py        reproducibility and efficiency measurement
  train.py          one run, checkpointing and paper-ready exports
  run_ablations.py  architecture/loss ablation suite
  summarize.py      mean/std aggregation across seeds
```

## Quick smoke test

From this directory:

```bash
python train.py --data-root ../data --model bcrr_full --epochs 2 --batch-size 2 --image-size 128 --workers 0 --save-predictions
```

The model uses official `mamba-ssm` when it is available on CUDA. Otherwise a dependency-free recurrent SSM fallback is used so the data, loss, metrics, checkpoint and visualization pipeline can be tested on CPU.

## Full controlled ablations

```bash
python run_ablations.py --data-root ../data --train-dataset isic2018 --val-dataset isic2018 --test-datasets isic2017 PH2 --epochs 300 --batch-size 8 --amp --output-root ./outputs/ablation_isic2018
python summarize.py --root ./outputs/ablation_isic2018
```

The suite includes the LB-UNet/CNN baseline, plain Mamba, single-axis Mamba, fixed-direction fusion, no boundary gate, no state reset, no grouping, no compression, multiple Mamba placements, and loss ablations. Prediction images are saved by default; use `--no-save-predictions` to disable them.

Each run writes:

- `config.json`, `loss_options.json`, `train.log`, `history.csv`;
- `best.pt` and `latest.pt`;
- `summary.csv` and per-image `samples_*.csv`;
- prediction probability, binary mask, ground-truth mask, boundary map and overlay PNGs when prediction saving is enabled.

The aggregate files are `all_runs.csv` and `ablation_mean_std.csv`.
