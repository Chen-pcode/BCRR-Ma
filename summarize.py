from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate BCRR-Mamba-UNet ablation summaries.")
    parser.add_argument("--root", default="./outputs/ablation_isic2018")
    args = parser.parse_args()
    root = Path(args.root)
    files = list(root.glob("*/summary.csv"))
    if not files:
        raise FileNotFoundError(f"No summary.csv found under {root}")
    frame = pd.concat([pd.read_csv(path).assign(run_dir=str(path.parent)) for path in files], ignore_index=True)
    frame.to_csv(root / "all_runs.csv", index=False)
    metric_columns = [column for column in ["dice", "iou", "miou", "hd95", "boundary_f1", "accuracy", "sensitivity", "specificity", "precision", "params_m", "flops_g", "fps"] if column in frame.columns]
    group_columns = [column for column in ["model", "eval_dataset", "split", "train_dataset"] if column in frame.columns]
    mean = frame.groupby(group_columns)[metric_columns].mean().add_suffix("_mean").reset_index()
    std = frame.groupby(group_columns)[metric_columns].std(ddof=0).fillna(0).add_suffix("_std").reset_index()
    merged = mean.merge(std, on=group_columns, how="left")
    merged.to_csv(root / "ablation_mean_std.csv", index=False)
    print(f"raw: {root / 'all_runs.csv'}")
    print(f"aggregate: {root / 'ablation_mean_std.csv'}")
    print(merged.to_string(index=False))


if __name__ == "__main__":
    main()

