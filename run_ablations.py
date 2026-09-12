from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from bcrr.models import ABLATIONS


LOSS_ABLATIONS = {
    "no_boundary_loss": ["--no-boundary-loss"],
    "no_gate_loss": ["--no-gate-loss"],
    "no_contour_loss": ["--no-contour-loss"],
    "segmentation_only": ["--no-boundary-loss", "--no-gate-loss", "--no-contour-loss"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete BCRR-Mamba-UNet ablation suite.")
    parser.add_argument("--data-root", default="../data")
    parser.add_argument("--train-dataset", default="isic2018")
    parser.add_argument("--val-dataset", default="isic2018")
    parser.add_argument("--test-datasets", nargs="*", default=["isic2017", "PH2"])
    parser.add_argument("--output-root", default="./outputs/ablation_isic2018")
    parser.add_argument("--models", nargs="*", default=list(ABLATIONS))
    parser.add_argument("--loss-ablations", nargs="*", default=list(LOSS_ABLATIONS))
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 1234, 2026])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-predictions", dest="save_predictions", action="store_true", default=True)
    parser.add_argument("--no-save-predictions", dest="save_predictions", action="store_false")
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    jobs = [(model, None) for model in args.models]
    jobs += [("bcrr_full", loss_name) for loss_name in args.loss_ablations]
    for model, loss_name in jobs:
        for seed in args.seeds:
            label = model if loss_name is None else f"{model}__{loss_name}"
            out = root / f"{label}__seed{seed}"
            if args.skip_completed and (out / "summary.csv").exists():
                print(f"skip {out}")
                continue
            command = [
                sys.executable, "train.py", "--data-root", args.data_root,
                "--train-dataset", args.train_dataset, "--val-dataset", args.val_dataset,
                "--output-dir", str(out), "--model", model, "--epochs", str(args.epochs),
                "--patience", str(args.patience), "--batch-size", str(args.batch_size),
                "--image-size", str(args.image_size), "--workers", str(args.workers), "--seed", str(seed),
            ]
            if args.test_datasets:
                command += ["--test-datasets", *args.test_datasets]
            if args.amp:
                command.append("--amp")
            if args.save_predictions:
                command.append("--save-predictions")
            if loss_name is not None:
                command += LOSS_ABLATIONS[loss_name]
            print("running:", " ".join(command), flush=True)
            subprocess.run(command, check=True)
    frames = []
    for summary in root.glob("*/summary.csv"):
        try:
            frames.append(pd.read_csv(summary).assign(run_dir=str(summary.parent)))
        except Exception as error:
            print(f"warning: cannot read {summary}: {error}")
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(root / "all_runs.csv", index=False)
        print(f"wrote {root / 'all_runs.csv'}")


if __name__ == "__main__":
    main()
