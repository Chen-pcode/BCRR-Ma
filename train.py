from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from bcrr.data import SkinDataset
from bcrr.engine import evaluate, train_epoch
from bcrr.models import ABLATIONS, get_model
from bcrr.utils import count_params, estimate_flops, measure_fps, model_size_mb, seed_everything, write_json


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BCRR-Mamba-UNet and export paper-ready metrics/predictions.")
    parser.add_argument("--data-root", default="../data")
    parser.add_argument("--isic2017-root", default=None)
    parser.add_argument("--isic2018-root", default=None)
    parser.add_argument("--ph2-root", default=None)
    parser.add_argument("--train-dataset", default="isic2018")
    parser.add_argument("--val-dataset", default="isic2018")
    parser.add_argument("--test-datasets", nargs="*", default=["isic2017", "PH2"])
    parser.add_argument("--model", choices=sorted(ABLATIONS), default="bcrr_full")
    parser.add_argument("--output-dir", default="./outputs/bcrr_full")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--no-boundary-loss", action="store_true")
    parser.add_argument("--no-gate-loss", action="store_true")
    parser.add_argument("--no-contour-loss", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def make_loader(dataset, args, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size if shuffle else 1,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )


def log(message: str, path: Path) -> None:
    print(message, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def main() -> None:
    args = arguments()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train.log"
    if not args.resume:
        log_path.write_text("", encoding="utf-8")
    seed_everything(args.seed, args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roots = {
        "isic2017": args.isic2017_root or str(Path(args.data_root) / "isic2017"),
        "isic2018": args.isic2018_root or str(Path(args.data_root) / "isic2018"),
        "ph2": args.ph2_root or str(Path(args.data_root) / "PH2Dataset"),
    }
    train_set = SkinDataset(args.data_root, args.train_dataset, "train", args.image_size, True, roots)
    val_set = SkinDataset(args.data_root, args.val_dataset, "val", args.image_size, False, roots)
    train_loader, val_loader = make_loader(train_set, args, True), make_loader(val_set, args, False)
    model = get_model(args.model).to(device)
    params = count_params(model)
    flops = estimate_flops(model, args.image_size, device)
    efficiency = {"params": params, "params_m": params / 1e6, "flops": flops, "flops_g": flops / 1e9, "model_size_mb": model_size_mb(model)}
    try:
        efficiency["fps"] = measure_fps(model, args.image_size, device, steps=10, warmup=2)
    except Exception as error:
        efficiency["fps_error"] = str(error)
    metadata = {**vars(args), "device": str(device), "model": args.model, "train_count": len(train_set), "val_count": len(val_set), **efficiency}
    write_json(output / "config.json", metadata)
    log(f"start model={args.model} device={device} train={len(train_set)} val={len(val_set)} params={efficiency['params_m']:.4f}M flops={efficiency['flops_g']:.4f}G", log_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    loss_options = {"use_boundary_loss": not args.no_boundary_loss, "use_gate_loss": not args.no_gate_loss, "use_contour_loss": not args.no_contour_loss}
    history: list[dict] = []
    best_dice, best_epoch, start_epoch = -1.0, -1, 1
    latest_path = output / "latest.pt"
    if args.resume:
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler.is_enabled() and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        best_dice, best_epoch, start_epoch = checkpoint["best_dice"], checkpoint["best_epoch"], checkpoint["epoch"] + 1
        if (output / "history.csv").exists():
            history = pd.read_csv(output / "history.csv").to_dict("records")
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scaler, device, args.amp, loss_options)
        scheduler.step()
        metrics, _ = evaluate(model, val_loader, device, args.threshold)
        record = {"epoch": epoch, "loss": loss, **{f"val_{key}": value for key, value in metrics.items()}}
        history.append(record)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        if metrics["dice"] > best_dice:
            best_dice, best_epoch = metrics["dice"], epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "dice": best_dice, "config": metadata}, output / "best.pt")
            marker = " [best]"
        else:
            marker = ""
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict() if scaler.is_enabled() else None, "epoch": epoch, "best_dice": best_dice, "best_epoch": best_epoch}, latest_path)
        log(f"epoch={epoch:03d}/{args.epochs} loss={loss:.4f} dice={metrics['dice']:.4f} iou={metrics['iou']:.4f} hd95={metrics['hd95']:.3f}{marker}", log_path)
        if epoch - best_epoch >= args.patience:
            log(f"early stop at epoch {epoch}; best_epoch={best_epoch}", log_path)
            break

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    evaluations = [(args.val_dataset, "val")] + [(name, "test" if name.lower() in {"ph2", "ph2dataset"} else "val") for name in args.test_datasets]
    summary_rows = []
    for dataset_name, split in evaluations:
        dataset = SkinDataset(args.data_root, dataset_name, split, args.image_size, False, roots)
        metrics, samples = evaluate(model, make_loader(dataset, args, False), device, args.threshold, output / "predictions" / f"{dataset_name}_{split}" if args.save_predictions else None)
        samples.to_csv(output / f"samples_{dataset_name}_{split}.csv", index=False)
        summary_rows.append({"model": args.model, "seed": args.seed, "train_dataset": args.train_dataset, "eval_dataset": dataset_name, "split": split, "best_epoch": best_epoch, "runtime_min": (time.time() - start_time) / 60.0, **efficiency, **metrics})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "summary.csv", index=False)
    write_json(output / "loss_options.json", loss_options)
    log("Final evaluation:\n" + summary.to_string(index=False), log_path)


if __name__ == "__main__":
    main()

