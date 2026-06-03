from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import FIELDS, FIELD_TO_ID, MODALITIES


@dataclass(frozen=True)
class VolumeRecord:
    path: Path
    modality: str
    split: str
    field: str
    field_id: int


def resolve_data_root(data_root: str | Path) -> Path:
    root = Path(data_root)
    if root.exists():
        if any((root / modality).exists() for modality in MODALITIES):
            return root
        nested = root / "Dataset"
        if nested.exists() and any((nested / modality).exists() for modality in MODALITIES):
            return nested
        return root
    here = Path(__file__).resolve().parents[2]
    candidate = (here / data_root).resolve()
    if candidate.exists():
        if any((candidate / modality).exists() for modality in MODALITIES):
            return candidate
        nested = candidate / "Dataset"
        if nested.exists() and any((nested / modality).exists() for modality in MODALITIES):
            return nested
        return candidate
    raise FileNotFoundError(f"Dataset root not found: {data_root}")


def list_records(
    data_root: str | Path,
    modality: str,
    split: str,
    fields: Sequence[str] = FIELDS,
) -> List[VolumeRecord]:
    root = resolve_data_root(data_root)
    records: List[VolumeRecord] = []
    for field in fields:
        folder = root / modality / split / field
        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")
        for path in sorted(folder.glob("*.nii.gz")):
            records.append(
                VolumeRecord(
                    path=path,
                    modality=modality,
                    split=split,
                    field=field,
                    field_id=FIELD_TO_ID[field],
                )
            )
    if not records:
        raise RuntimeError(f"No NIfTI files found under {root / modality / split}")
    return records


def group_by_field(records: Iterable[VolumeRecord]) -> Dict[int, List[VolumeRecord]]:
    grouped: Dict[int, List[VolumeRecord]] = {FIELD_TO_ID[field]: [] for field in FIELDS}
    for record in records:
        grouped[record.field_id].append(record)
    return {k: v for k, v in grouped.items() if v}


def robust_normalize(
    volume: np.ndarray,
    low_percentile: float = 0.5,
    high_percentile: float = 99.5,
) -> Tuple[np.ndarray, Dict[str, float]]:
    volume = np.asarray(volume, dtype=np.float32)
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.isfinite(volume) & (np.abs(volume) > 1e-6)
    values = volume[mask] if int(mask.sum()) > 1024 else volume.reshape(-1)
    low = float(np.percentile(values, low_percentile))
    high = float(np.percentile(values, high_percentile))
    if high <= low + 1e-6:
        low = float(np.min(values))
        high = float(np.max(values))
    if high <= low + 1e-6:
        high = low + 1.0
    clipped = np.clip(volume, low, high)
    normalized = (clipped - low) / (high - low)
    normalized = normalized * 2.0 - 1.0
    return normalized.astype(np.float32), {"low": low, "high": high}


def denormalize(volume: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    low = float(stats["low"])
    high = float(stats["high"])
    volume = (volume.astype(np.float32) + 1.0) * 0.5
    return volume * (high - low) + low


def load_nifti_normalized(path: str | Path) -> Tuple[np.ndarray, nib.Nifti1Image, Dict[str, float]]:
    image = nib.load(str(path))
    volume_xyz = np.asarray(image.dataobj, dtype=np.float32)
    normalized_xyz, stats = robust_normalize(volume_xyz)
    volume_dhw = np.transpose(normalized_xyz, (2, 1, 0)).copy()
    return volume_dhw, image, stats


def save_nifti_like(volume_dhw: np.ndarray, reference: nib.Nifti1Image, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    volume_xyz = np.transpose(volume_dhw, (2, 1, 0)).astype(np.float32)
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(volume_xyz, reference.affine, header), str(output_path))


def _pad_to_shape(volume: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    pads = []
    for dim, target in zip(volume.shape, shape):
        missing = max(0, int(target) - int(dim))
        before = missing // 2
        after = missing - before
        pads.append((before, after))
    if any(before or after for before, after in pads):
        volume = np.pad(volume, pads, mode="constant", constant_values=-1.0)
    return volume


def _foreground_bbox(volume: np.ndarray, threshold: float = -0.95) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    mask = volume > threshold
    if int(mask.sum()) < 1024:
        return None
    coords = np.where(mask)
    mins = np.array([int(c.min()) for c in coords], dtype=np.int64)
    maxs = np.array([int(c.max()) + 1 for c in coords], dtype=np.int64)
    return mins, maxs


def random_crop_3d(
    volume: np.ndarray,
    patch_size: Sequence[int],
    foreground_prob: float = 0.8,
) -> np.ndarray:
    patch = np.array(patch_size, dtype=np.int64)
    volume = _pad_to_shape(volume, patch)
    shape = np.array(volume.shape, dtype=np.int64)
    bbox = _foreground_bbox(volume)

    if bbox is not None and random.random() < foreground_prob:
        mins, maxs = bbox
        center_low = np.minimum(np.maximum(mins, patch // 2), np.maximum(shape - patch // 2, patch // 2))
        center_high = np.maximum(np.minimum(maxs, shape - patch // 2), center_low + 1)
        center = np.array(
            [random.randrange(int(lo), int(hi)) for lo, hi in zip(center_low, center_high)],
            dtype=np.int64,
        )
        start = np.clip(center - patch // 2, 0, shape - patch)
    else:
        start = np.array(
            [random.randrange(0, int(s - p + 1)) for s, p in zip(shape, patch)],
            dtype=np.int64,
        )

    d, h, w = start.tolist()
    pd, ph, pw = patch.tolist()
    return volume[d : d + pd, h : h + ph, w : w + pw].copy()


def augment_patch(patch: np.ndarray, intensity_jitter: float = 0.05) -> np.ndarray:
    for axis in range(3):
        if random.random() < 0.5:
            patch = np.flip(patch, axis=axis)
    if intensity_jitter > 0:
        scale = 1.0 + random.uniform(-intensity_jitter, intensity_jitter)
        shift = random.uniform(-intensity_jitter, intensity_jitter)
        patch = np.clip(patch * scale + shift, -1.0, 1.0)
    return np.ascontiguousarray(patch, dtype=np.float32)


class UnpairedFieldPatchDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        modality: str,
        split: str = "Training",
        fields: Sequence[str] = FIELDS,
        patch_size: Sequence[int] = (64, 64, 64),
        samples_per_epoch: int = 2000,
        same_domain_prob: float = 0.05,
        augment: bool = True,
    ) -> None:
        self.records = list_records(data_root, modality, split, fields)
        self.by_field = group_by_field(self.records)
        self.field_ids = sorted(self.by_field)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.same_domain_prob = float(same_domain_prob)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _load_patch(self, record: VolumeRecord) -> torch.Tensor:
        volume, _, _ = load_nifti_normalized(record.path)
        patch = random_crop_3d(volume, self.patch_size)
        if self.augment:
            patch = augment_patch(patch)
        return torch.from_numpy(patch[None, ...])

    def __getitem__(self, index: int):
        source = random.choice(self.records)
        if random.random() < self.same_domain_prob:
            target_field_id = source.field_id
        else:
            candidates = [field_id for field_id in self.field_ids if field_id != source.field_id]
            target_field_id = random.choice(candidates)
        target = random.choice(self.by_field[target_field_id])

        return {
            "source": self._load_patch(source),
            "source_domain": torch.tensor(source.field_id, dtype=torch.long),
            "target": self._load_patch(target),
            "target_domain": torch.tensor(target.field_id, dtype=torch.long),
            "source_path": str(source.path),
            "target_path": str(target.path),
        }


def estimate_field_intensity_stats(
    data_root: str | Path,
    modality: str,
    split: str = "Training",
    fields: Sequence[str] = FIELDS,
    max_per_field: int = 8,
    stride: int = 4,
) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    records = group_by_field(list_records(data_root, modality, split, fields))
    for field in fields:
        field_id = FIELD_TO_ID[field]
        lows: List[float] = []
        highs: List[float] = []
        for record in records.get(field_id, [])[:max_per_field]:
            image = nib.load(str(record.path))
            volume = np.asarray(image.dataobj, dtype=np.float32)
            sampled = volume[::stride, ::stride, ::stride]
            _, item_stats = robust_normalize(sampled)
            lows.append(item_stats["low"])
            highs.append(item_stats["high"])
        if lows:
            stats[field] = {"low": float(np.median(lows)), "high": float(np.median(highs))}
    return stats


def save_json(data, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
