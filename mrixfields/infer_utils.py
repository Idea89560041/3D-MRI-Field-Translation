from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch


def _starts(length: int, patch: int, stride: int) -> List[int]:
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def _pad_volume(volume: np.ndarray, patch_size: Sequence[int]) -> Tuple[np.ndarray, Tuple[slice, slice, slice]]:
    pads = []
    crop = []
    for dim, patch in zip(volume.shape, patch_size):
        missing = max(0, int(patch) - int(dim))
        before = missing // 2
        after = missing - before
        pads.append((before, after))
        crop.append(slice(before, before + dim))
    if any(a or b for a, b in pads):
        volume = np.pad(volume, pads, mode="constant", constant_values=-1.0)
    return volume, tuple(crop)  # type: ignore[return-value]


@torch.no_grad()
def sliding_window_translate(
    model,
    volume_dhw: np.ndarray,
    target_domain_id: int,
    patch_size: Sequence[int],
    overlap: float,
    device: torch.device,
    batch_size: int = 1,
    z: torch.Tensor | None = None,
) -> np.ndarray:
    model.eval()
    patch_size = tuple(int(v) for v in patch_size)
    stride = tuple(max(1, int(p * (1.0 - overlap))) for p in patch_size)
    padded, crop = _pad_volume(volume_dhw.astype(np.float32), patch_size)
    d_starts = _starts(padded.shape[0], patch_size[0], stride[0])
    h_starts = _starts(padded.shape[1], patch_size[1], stride[1])
    w_starts = _starts(padded.shape[2], patch_size[2], stride[2])

    output = np.zeros_like(padded, dtype=np.float32)
    count = np.zeros_like(padded, dtype=np.float32)
    windows = [(d, h, w) for d in d_starts for h in h_starts for w in w_starts]

    if z is None:
        z = torch.randn(1, model.latent_dim, device=device)
    z = z.to(device)

    for offset in range(0, len(windows), batch_size):
        batch_windows = windows[offset : offset + batch_size]
        patches = []
        for d, h, w in batch_windows:
            patch = padded[d : d + patch_size[0], h : h + patch_size[1], w : w + patch_size[2]]
            patches.append(patch[None, ...])
        x = torch.from_numpy(np.stack(patches, axis=0)).to(device=device, dtype=torch.float32)
        domains = torch.full((x.shape[0],), int(target_domain_id), dtype=torch.long, device=device)
        z_batch = z.expand(x.shape[0], -1)
        fake, _, _ = model.translate(x, domains, z=z_batch)
        fake_np = fake[:, 0].detach().cpu().numpy()
        for item, (d, h, w) in zip(fake_np, batch_windows):
            output[d : d + patch_size[0], h : h + patch_size[1], w : w + patch_size[2]] += item
            count[d : d + patch_size[0], h : h + patch_size[1], w : w + patch_size[2]] += 1.0

    output = output / np.maximum(count, 1e-6)
    return output[crop].astype(np.float32)

