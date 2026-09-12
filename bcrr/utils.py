from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def count_params(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    return (count_params(model) + sum(buffer.numel() for buffer in model.buffers())) * 4 / (1024 ** 2)


def estimate_flops(model: nn.Module, image_size: int, device: torch.device) -> int:
    flops = 0
    hooks = []

    def conv_hook(module: nn.Conv2d, inputs, output):
        nonlocal flops
        batch, out_channels, height, width = output.shape
        kh, kw = module.kernel_size
        flops += int(batch * out_channels * height * width * (module.in_channels // module.groups) * kh * kw)

    def linear_hook(module: nn.Linear, inputs, output):
        nonlocal flops
        flops += int(output.numel() * module.in_features)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, image_size, image_size, device=device))
    for hook in hooks:
        hook.remove()
    model.train(was_training)
    return flops


@torch.no_grad()
def measure_fps(model: nn.Module, image_size: int, device: torch.device, steps: int = 30, warmup: int = 5) -> float:
    model.eval()
    image = torch.randn(1, 3, image_size, image_size, device=device)
    for _ in range(warmup):
        model(image)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps):
        model(image)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return float(steps / max(time.perf_counter() - start, 1e-9))

