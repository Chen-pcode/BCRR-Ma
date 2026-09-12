from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, maximum_filter

EPS = 1e-7


def _confusion(pred: np.ndarray, target: np.ndarray) -> tuple[float, float, float, float]:
    pred, target = pred.astype(bool), target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    tn = np.logical_and(~pred, ~target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()
    return float(tp), float(tn), float(fp), float(fn)


def hd95(pred: np.ndarray, target: np.ndarray) -> float:
    pred, target = pred.astype(bool), target.astype(bool)
    if pred.sum() == 0 and target.sum() == 0:
        return 0.0
    if pred.sum() == 0 or target.sum() == 0:
        return float(max(pred.shape))
    pred_border = np.logical_xor(pred, binary_erosion(pred))
    target_border = np.logical_xor(target, binary_erosion(target))
    d_pred = distance_transform_edt(~pred_border)
    d_target = distance_transform_edt(~target_border)
    distances = np.concatenate((d_target[pred_border], d_pred[target_border]))
    return float(np.percentile(distances, 95)) if distances.size else 0.0


def boundary_map(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    return np.logical_xor(mask, binary_erosion(mask))


def boundary_f1(pred: np.ndarray, target: np.ndarray, tolerance: int = 2) -> tuple[float, float, float]:
    pred_edge, target_edge = boundary_map(pred), boundary_map(target)
    if not pred_edge.any() and not target_edge.any():
        return 1.0, 1.0, 1.0
    pred_near = maximum_filter(target_edge.astype(np.uint8), size=2 * tolerance + 1) > 0
    target_near = maximum_filter(pred_edge.astype(np.uint8), size=2 * tolerance + 1) > 0
    precision = float((pred_edge & pred_near).sum()) / float(pred_edge.sum() + EPS)
    recall = float((target_edge & target_near).sum()) / float(target_edge.sum() + EPS)
    f1 = 2.0 * precision * recall / (precision + recall + EPS)
    return precision, recall, f1


def binary_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    tp, tn, fp, fn = _confusion(pred, target)
    iou = (tp + EPS) / (tp + fp + fn + EPS)
    bg_iou = (tn + EPS) / (tn + fp + fn + EPS)
    dice = (2 * tp + EPS) / (2 * tp + fp + fn + EPS)
    precision = (tp + EPS) / (tp + fp + EPS)
    sensitivity = (tp + EPS) / (tp + fn + EPS)
    specificity = (tn + EPS) / (tn + fp + EPS)
    bp, br, bf = boundary_f1(pred, target)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "miou": float((iou + bg_iou) / 2.0),
        "accuracy": float((tp + tn + EPS) / (tp + tn + fp + fn + EPS)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(dice),
        "hd95": hd95(pred, target),
        "boundary_precision": bp,
        "boundary_recall": br,
        "boundary_f1": bf,
    }

