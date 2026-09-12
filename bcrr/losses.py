from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probability = torch.sigmoid(logits.float())
    probability = probability.flatten(1)
    target = target.float().flatten(1)
    intersection = (probability * target).sum(1)
    return (1.0 - (2.0 * intersection + eps) / (probability.sum(1) + target.sum(1) + eps)).mean()


def bce_dice(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logits, target = logits.float(), target.float()
    return F.binary_cross_entropy_with_logits(logits, target) + dice_loss(logits, target)


def boundary_continuity_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compare local boundary gradients without requiring a second annotation."""
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits, target = logits.float(), target.float()
        probability = torch.sigmoid(logits)
        sobel_x = logits.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8.0
        sobel_y = sobel_x.transpose(2, 3)
        pred_x = F.conv2d(probability, sobel_x, padding=1)
        pred_y = F.conv2d(probability, sobel_y, padding=1)
        true_x = F.conv2d(target, sobel_x, padding=1)
        true_y = F.conv2d(target, sobel_y, padding=1)
        weight = F.max_pool2d(target, 5, stride=1, padding=2)
        return ((pred_x - true_x).abs() + (pred_y - true_y).abs()).mul(0.5 + weight).mean()


def total_loss(
    outputs: dict[str, object],
    target: torch.Tensor,
    boundary: torch.Tensor,
    use_boundary_loss: bool = True,
    use_gate_loss: bool = True,
    use_contour_loss: bool = True,
    aux_weight: float = 0.25,
) -> torch.Tensor:
    logits = outputs["logits"]
    loss = bce_dice(logits, target)
    if use_boundary_loss:
        boundary_logits = outputs["boundary"]
        loss = loss + 0.4 * F.binary_cross_entropy_with_logits(boundary_logits.float(), boundary.float())
        for scale in outputs.get("boundaries", []):
            scaled_target = F.interpolate(boundary, size=scale.shape[-2:], mode="nearest")
            loss = loss + 0.1 * F.binary_cross_entropy_with_logits(scale.float(), scaled_target.float())
    for aux in outputs.get("aux", []):
        loss = loss + aux_weight * bce_dice(aux, target)
    if use_gate_loss:
        gates = [item["gate"] for item in outputs.get("priors", []) if "gate" in item]
        for gate in gates:
            scaled_boundary = F.interpolate(boundary, size=gate.shape[-2:], mode="nearest")
            with torch.autocast(device_type=gate.device.type, enabled=False):
                loss = loss + 0.08 * F.binary_cross_entropy(gate.float().clamp(1e-4, 1 - 1e-4), scaled_boundary.float())
    if use_contour_loss:
        loss = loss + 0.08 * boundary_continuity_loss(logits, target)
    return loss
