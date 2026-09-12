from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as OfficialMamba
except Exception:  # pragma: no cover - optional CUDA dependency
    OfficialMamba = None


def _groups(channels: int) -> int:
    return next((g for g in (8, 4, 2, 1) if channels % g == 0), 1)


class ConvGNAct(nn.Sequential):
    def __init__(self, ci: int, co: int, kernel: int = 3, stride: int = 1, groups: int = 1):
        super().__init__(
            nn.Conv2d(ci, co, kernel, stride, kernel // 2, groups=groups, bias=False),
            nn.GroupNorm(_groups(co), co),
            nn.SiLU(inplace=True),
        )


class DepthwiseResidual(nn.Module):
    def __init__(self, ci: int, co: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            ConvGNAct(ci, ci, 3, stride, ci),
            ConvGNAct(ci, co, 1),
            ConvGNAct(co, co, 3, 1, co),
            ConvGNAct(co, co, 1),
        )
        self.skip = nn.Identity() if ci == co and stride == 1 else ConvGNAct(ci, co, 1, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + self.skip(x)


class FallbackSSM(nn.Module):
    """Small differentiable SSM fallback for CPU smoke tests.

    It is intentionally dependency-free. The CUDA path uses official Mamba;
    this path makes the complete training and ablation pipeline testable on
    machines where mamba-ssm cannot be installed.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv1d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.output = nn.Conv1d(channels, channels, 1, bias=False)

    def forward(self, tokens: torch.Tensor, reset: torch.Tensor | None = None) -> torch.Tensor:
        sequence = tokens.transpose(1, 2)
        result = self.output(torch.tanh(self.dw(sequence))).transpose(1, 2)
        if reset is not None:
            result = result * (0.5 + 0.5 * reset)
        return result + tokens


class SharedScanCore(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.official = OfficialMamba is not None and torch.cuda.is_available()
        if self.official:
            self.core = OfficialMamba(d_model=channels, d_state=8, d_conv=4, expand=2)
        else:
            self.core = FallbackSSM(channels)

    def forward(self, tokens: torch.Tensor, reset: torch.Tensor | None = None) -> torch.Tensor:
        if self.official:
            return self.core(tokens)
        return self.core(tokens, reset)


class BoundaryConditionedRoutingMamba(nn.Module):
    """Grouped multi-axis Mamba with boundary-conditioned state mixing.

    The block keeps one shared state-space core for all channel groups and
    directions. A cheap region/boundary prior predicts uncertainty, a reset
    gate, and spatially varying scan-direction weights. This keeps the block
    small enough for EGE/LB-style models while making global propagation
    structure-aware.
    """

    def __init__(
        self,
        channels: int,
        compression: float = 0.5,
        grouped: bool = True,
        dual_axis: bool = True,
        use_gate: bool = True,
        use_router: bool = True,
        use_state_reset: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.grouped = grouped
        self.dual_axis = dual_axis
        self.use_gate = use_gate
        self.use_router = use_router
        self.use_state_reset = use_state_reset
        inner = max(8, int(round(channels * compression)))
        group_count = 2 if grouped and inner % 2 == 0 else 1
        if inner % group_count:
            group_count = 1
        self.group_count = group_count
        self.inner = inner
        group_channels = inner // group_count
        self.reduce = nn.Conv2d(channels, inner, 1, bias=False)
        self.expand = nn.Conv2d(inner, channels, 1, bias=False)
        self.norm = nn.LayerNorm(group_channels)
        self.core = SharedScanCore(group_channels)
        self.local = ConvGNAct(channels, channels, 3, 1, channels)
        self.region_prior = nn.Conv2d(channels, 1, 1)
        self.boundary_prior = nn.Conv2d(channels, 1, 1)
        directions = 4 if dual_axis else 2
        self.direction_count = directions
        router_channels = max(8, channels // 2)
        self.router = (
            nn.Sequential(
                nn.Conv2d(channels + 2, router_channels, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(router_channels, directions, 1),
            ) if use_router else None
        )
        self.gate = (
            nn.Sequential(
                nn.Conv2d(channels + 2, router_channels, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(router_channels, 1, 1),
                nn.Sigmoid(),
            ) if use_gate else None
        )

    @staticmethod
    def _tokens(feature: torch.Tensor) -> torch.Tensor:
        return feature.flatten(2).transpose(1, 2)

    @staticmethod
    def _feature(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], height, width)

    def _scan(self, feature: torch.Tensor, reset: torch.Tensor, transpose: bool, reverse: bool) -> torch.Tensor:
        if transpose:
            feature = feature.transpose(2, 3)
            reset = reset.transpose(2, 3)
        batch, channels, height, width = feature.shape
        grouped = feature.reshape(batch * self.group_count, channels // self.group_count, height, width)
        reset_grouped = reset.repeat_interleave(self.group_count, dim=0)
        tokens = self._tokens(grouped)
        reset_tokens = self._tokens(reset_grouped)
        if reverse:
            tokens = tokens.flip(1)
            reset_tokens = reset_tokens.flip(1)
        if self.core.official and self.use_state_reset:
            control = 0.5 + 0.5 * reset_tokens
            tokens = tokens * control
            output = self.core(self.norm(tokens)) * control
        else:
            output = self.core(self.norm(tokens), reset_tokens if self.use_state_reset else None)
        if reverse:
            output = output.flip(1)
        result = self._feature(output, height, width).reshape(batch, channels, height, width)
        return result.transpose(2, 3) if transpose else result

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        region = self.region_prior(x)
        boundary = self.boundary_prior(x)
        probability = torch.sigmoid(region)
        uncertainty = 4.0 * probability * (1.0 - probability)
        evidence = torch.cat((x, uncertainty, boundary.abs()), dim=1)
        reset = self.gate(evidence) if self.gate is not None else torch.zeros_like(boundary)
        state_control = (0.25 + 0.75 * reset) if self.use_state_reset else torch.ones_like(reset)
        route = self.router(evidence).softmax(dim=1) if self.router is not None else None
        reduced = self.reduce(x)
        scans = [
            self._scan(reduced, state_control, transpose=False, reverse=False),
            self._scan(reduced, state_control, transpose=False, reverse=True),
        ]
        if self.dual_axis:
            scans.extend([
                self._scan(reduced, state_control, transpose=True, reverse=False),
                self._scan(reduced, state_control, transpose=True, reverse=True),
            ])
        if route is None:
            fused = sum(scans) / len(scans)
        else:
            fused = sum(route[:, i:i + 1] * scans[i] for i in range(len(scans)))
        mixed = self.expand(fused)
        if self.use_gate:
            mixed = x + (0.5 + 0.5 * reset) * (mixed - x)
        else:
            mixed = x + mixed
        mixed = mixed + self.local(mixed)
        return {
            "feature": mixed,
            "region": region,
            "boundary": boundary,
            "uncertainty": uncertainty,
            "gate": reset,
            "route": route if route is not None else x.new_full((x.shape[0], len(scans), x.shape[2], x.shape[3]), 1.0 / len(scans)),
        }


class BoundaryFusion(nn.Module):
    def __init__(self, high: int, skip: int, out: int):
        super().__init__()
        self.high = ConvGNAct(high, out, 1)
        self.skip = ConvGNAct(skip, out, 1)
        self.mix = DepthwiseResidual(out * 2, out)
        self.boundary = nn.Conv2d(out * 2, 1, 1)

    def forward(self, high: torch.Tensor, skip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        high = F.interpolate(self.high(high), size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.skip(skip)
        fusion = torch.cat((high, skip), dim=1)
        return self.mix(fusion), self.boundary(fusion)


class BCRRMambaUNet(nn.Module):
    """Six-stage LB-style U-Net with boundary-conditioned grouped Mamba."""

    def __init__(
        self,
        widths: Iterable[int] = (8, 16, 24, 32, 48, 64),
        use_mamba: bool = True,
        mamba_stages: tuple[int, ...] = (3, 4, 5),
        compression: float = 0.5,
        grouped: bool = True,
        dual_axis: bool = True,
        use_gate: bool = True,
        use_router: bool = True,
        use_state_reset: bool = True,
    ):
        super().__init__()
        widths = tuple(widths)
        self.widths = widths
        self.mamba_stages = tuple(mamba_stages)
        self.use_gate = use_gate
        self.use_boundary_heads = True
        self.stem = ConvGNAct(3, widths[0])
        self.encoder = nn.ModuleList()
        self.mamba_blocks = nn.ModuleDict()
        for stage in range(1, len(widths)):
            self.encoder.append(DepthwiseResidual(widths[stage - 1], widths[stage], 2))
            if use_mamba and stage in self.mamba_stages:
                self.mamba_blocks[str(stage)] = BoundaryConditionedRoutingMamba(
                    widths[stage], compression, grouped, dual_axis, use_gate, use_router, use_state_reset
                )
        self.decoder = nn.ModuleList(
            [BoundaryFusion(widths[stage], widths[stage - 1], widths[stage - 1]) for stage in range(len(widths) - 1, 0, -1)]
        )
        self.output = nn.Conv2d(widths[0], 1, 1)
        self.auxiliary = nn.ModuleList([nn.Conv2d(widths[stage], 1, 1) for stage in (2, 1, 0)])

    def forward(self, x: torch.Tensor) -> dict[str, object]:
        features = [self.stem(x)]
        priors: list[dict[str, torch.Tensor]] = []
        for stage, block in enumerate(self.encoder, start=1):
            feature = block(features[-1])
            if str(stage) in self.mamba_blocks:
                result = self.mamba_blocks[str(stage)](feature)
                feature = result["feature"]
                priors.append(result)
            features.append(feature)
        decoded = features[-1]
        decoder_features: list[torch.Tensor] = []
        boundaries: list[torch.Tensor] = []
        for block, skip in zip(self.decoder, reversed(features[:-1])):
            decoded, boundary = block(decoded, skip)
            decoder_features.append(decoded)
            boundaries.append(boundary)
        logits = self.output(decoded)
        boundary = sum(
            F.interpolate(item, size=logits.shape[-2:], mode="bilinear", align_corners=False)
            for item in boundaries
        ) / len(boundaries)
        aux = [
            F.interpolate(head(feature), size=logits.shape[-2:], mode="bilinear", align_corners=False)
            for head, feature in zip(self.auxiliary, decoder_features[-3:])
        ]
        return {"logits": logits, "boundary": boundary, "aux": aux, "priors": priors, "boundaries": boundaries}


BASE = dict(widths=(8, 16, 24, 32, 48, 64), mamba_stages=(3, 4, 5))
ABLATIONS: dict[str, dict] = {
    "bcrr_full": {},
    "lb_unet_baseline": {"use_mamba": False},
    "cnn_only": {"use_mamba": False},
    "plain_mamba": {"use_gate": False, "use_router": False, "use_state_reset": False},
    "single_axis": {"dual_axis": False},
    "fixed_direction": {"use_router": False},
    "no_boundary_gate": {"use_gate": False, "use_state_reset": False},
    "no_state_reset": {"use_state_reset": False},
    "no_grouping": {"grouped": False},
    "no_compression": {"compression": 1.0},
    "mamba_32": {"mamba_stages": (3,)},
    "mamba_16_8": {"mamba_stages": (4, 5)},
    "mamba_16": {"mamba_stages": (4,)},
}


def get_model(name: str) -> BCRRMambaUNet:
    if name not in ABLATIONS:
        raise ValueError(f"Unknown model '{name}'. Choices: {', '.join(ABLATIONS)}")
    return BCRRMambaUNet(**BASE, **ABLATIONS[name])
