from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resolve_split(data_root: str | Path, dataset: str, split: str, roots: dict[str, str | Path] | None = None) -> tuple[Path, Path]:
    root = Path(data_root)
    roots = roots or {}
    key = dataset.lower()
    if key in {"isic2017", "2017", "isic2018", "2018"}:
        name = "isic2017" if "2017" in key else "isic2018"
        base = Path(roots.get(name, root / name))
        if not base.exists():
            base = root / name
        actual = "val" if split == "test" else split
        return base / actual / "images", base / actual / "masks"
    if key in {"ph2", "ph2dataset"}:
        configured = roots.get("ph2")
        bases = []
        if configured:
            bases.append(Path(configured))
        bases.extend([root / "PH2Dataset", root / "ph2dataset", root / "ph2", root])
        candidates = []
        for base in bases:
            candidates.extend([base / "ph2" / "test", base / "PH2Dataset" / "ph2" / "test", base / "test"])
        actual = next((item for item in candidates if (item / "images").exists() and (item / "masks").exists()), candidates[0])
        return actual / "images", actual / "masks"
    raise ValueError(f"Unknown dataset: {dataset}")


def _mask_stems(image: Path) -> list[str]:
    stem = image.stem
    return [stem, f"{stem}_segmentation", stem.replace("_Dermoscopic_Image", "_lesion")]


def pair_files(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"Missing image/mask directory: {image_dir}, {mask_dir}")
    masks = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
    pairs = []
    for image in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
        mask = next((masks[s] for s in _mask_stems(image) if s in masks), None)
        if mask is None and (mask_dir / image.name).exists():
            mask = mask_dir / image.name
        if mask is not None:
            pairs.append((image, mask))
    if not pairs:
        raise RuntimeError(f"No image-mask pairs in {image_dir}")
    return pairs


def mask_to_boundary(mask: torch.Tensor, width: int = 3) -> torch.Tensor:
    dilated = F.max_pool2d(mask, width, stride=1, padding=width // 2)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, width, stride=1, padding=width // 2)
    return (dilated - eroded).clamp_(0, 1)


class SkinDataset(Dataset):
    def __init__(self, data_root: str | Path, dataset: str, split: str, image_size: int = 256, augment: bool = False, roots: dict[str, str | Path] | None = None):
        self.images, self.masks = resolve_split(data_root, dataset, split, roots)
        self.pairs = pair_files(self.images, self.masks)
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path, mask_path = self.pairs[index]
        image = np.asarray(Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR), dtype=np.float32) / 255.0
        mask = (np.asarray(Image.open(mask_path).convert("L").resize((self.image_size, self.image_size), Image.NEAREST), dtype=np.float32) > 127).astype(np.float32)
        if self.augment:
            image, mask = self._augment(image, mask)
        image = (image - MEAN) / STD
        image_t = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()
        mask_t = torch.from_numpy(mask[None].copy()).float()
        boundary_t = mask_to_boundary(mask_t.unsqueeze(0)).squeeze(0)
        return {"image": image_t, "mask": mask_t, "boundary": boundary_t, "id": image_path.stem}

    @staticmethod
    def _augment(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            image, mask = np.flip(image, 1), np.flip(mask, 1)
        if random.random() < 0.5:
            image, mask = np.flip(image, 0), np.flip(mask, 0)
        turns = random.randint(0, 3)
        if turns:
            image, mask = np.rot90(image, turns), np.rot90(mask, turns)
        if random.random() < 0.8:
            image = np.clip(image * np.random.uniform(0.85, 1.15, (1, 1, 3)) + np.random.uniform(-0.05, 0.05, (1, 1, 3)), 0, 1)
        return image.copy(), mask.copy()
