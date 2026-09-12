from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from .losses import total_loss
from .metrics import binary_metrics

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def train_epoch(model, loader, optimizer, scaler, device, amp: bool, loss_options: dict[str, bool]) -> float:
    model.train()
    total, samples = 0.0, 0
    for batch in tqdm(loader, desc="train", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        boundary = batch["boundary"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            loss = total_loss(model(image), mask, boundary, **loss_options)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += float(loss.detach()) * image.shape[0]
        samples += image.shape[0]
    return total / max(samples, 1)


def _save_prediction(sample_dir: Path, sample_id: str, image: np.ndarray, probability: np.ndarray, boundary: np.ndarray, predicted: np.ndarray, target: np.ndarray) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray((probability * 255).astype(np.uint8)).save(sample_dir / f"{sample_id}_prob.png")
    Image.fromarray((predicted * 255).astype(np.uint8)).save(sample_dir / f"{sample_id}_mask.png")
    Image.fromarray((target * 255).astype(np.uint8)).save(sample_dir / f"{sample_id}_gt.png")
    Image.fromarray((boundary * 255).astype(np.uint8)).save(sample_dir / f"{sample_id}_boundary.png")
    rgb = np.clip(image.transpose(1, 2, 0) * STD + MEAN, 0, 1)
    overlay = (rgb * 255).astype(np.uint8)
    overlay[predicted > 0, 0] = 255
    overlay[predicted > 0, 1] = (overlay[predicted > 0, 1] * 0.45).astype(np.uint8)
    overlay[predicted > 0, 2] = (overlay[predicted > 0, 2] * 0.45).astype(np.uint8)
    edge = boundary >= 0.5
    overlay[edge] = np.array([0, 255, 255], dtype=np.uint8)
    Image.fromarray(overlay).save(sample_dir / f"{sample_id}_overlay.png")


@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5, prediction_dir: Path | None = None) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    rows: list[dict[str, float | str]] = []
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    for batch in tqdm(loader, desc="evaluate", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        outputs = model(image)
        probability = outputs["logits"].sigmoid().cpu().numpy()[:, 0]
        predicted = (probability >= threshold).astype(np.uint8)
        target = batch["mask"].numpy()[:, 0].astype(np.uint8)
        boundary = outputs["boundary"].sigmoid().cpu().numpy()[:, 0]
        for index, sample_id in enumerate(batch["id"]):
            row = {"id": sample_id, **binary_metrics(predicted[index], target[index])}
            rows.append(row)
            if prediction_dir is not None:
                _save_prediction(prediction_dir, sample_id, image[index].cpu().numpy(), probability[index], boundary[index], predicted[index], target[index])
    frame = pd.DataFrame(rows)
    numeric = frame.select_dtypes(include=[np.number])
    return {column: float(numeric[column].mean()) for column in numeric.columns}, frame
