from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from mrixfields.constants import FIELD_TO_ID, FIELDS, MODALITIES, parse_fields
from mrixfields.data import denormalize, list_records, load_nifti_normalized, save_nifti_like
from mrixfields.infer_utils import sliding_window_translate
from mrixfields.models import MURD3D


DEFAULT_DATA_ROOT = "Dataset"
DEFAULT_SOURCE_FIELD = "0.1T"
DEFAULT_TARGET_FIELDS = ("1.5T", "3T", "5T", "7T")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Task2 0.1T enhancement inference from a fine-tuned checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", default="Testing")
    parser.add_argument("--modality", choices=MODALITIES, default=None)
    parser.add_argument("--source-field", default=DEFAULT_SOURCE_FIELD, choices=FIELDS)
    parser.add_argument("--target-fields", nargs="*", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--patch-size", nargs=3, type=int, default=None)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--scale", choices=("target_stats", "source", "normalized"), default="target_stats")
    parser.add_argument("--style-seed", type=int, default=2026)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--debug-crop", nargs=3, type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def output_name(path: Path, source_field: str, target_field: str) -> str:
    name = path.name
    token = f"_{source_field}_"
    if token in name:
        return name.replace(token, f"_{target_field}_")
    return name.replace(".nii.gz", f"_{target_field}.nii.gz")


def center_crop(volume, crop_size):
    crop_size = tuple(int(v) for v in crop_size)
    starts = [max(0, (dim - crop) // 2) for dim, crop in zip(volume.shape, crop_size)]
    slices = tuple(slice(start, min(start + crop, dim)) for start, crop, dim in zip(starts, crop_size, volume.shape))
    return volume[slices]


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    train_args = checkpoint.get("args", {})
    modality = args.modality or train_args.get("modality")
    if modality is None:
        raise ValueError("Provide --modality or use a checkpoint that stores modality in args.")

    patch_size = tuple(args.patch_size or train_args.get("patch_size", (64, 64, 64)))
    model = MURD3D(
        num_domains=len(FIELDS),
        base_channels=int(train_args.get("base_channels", 16)),
        max_channels=int(train_args.get("max_channels", 256)),
        levels=int(train_args.get("levels", 2)),
        res_blocks=int(train_args.get("res_blocks", 4)),
        style_dim=int(train_args.get("style_dim", 64)),
        latent_dim=int(train_args.get("latent_dim", 16)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if args.target_fields:
        target_fields = parse_fields(args.target_fields)
    else:
        target_fields = checkpoint.get("target_fields") or train_args.get("target_fields") or list(DEFAULT_TARGET_FIELDS)
    if args.source_field in target_fields:
        raise ValueError("Task2 inference expects target fields to exclude the 0.1T source field.")
    field_stats = checkpoint.get("field_stats", {})

    out_root = Path(args.output_dir or Path("outputs") / f"task2_0p1T_enhancement_predictions_{modality}")
    generator = torch.Generator(device=device)
    generator.manual_seed(args.style_seed)
    z = torch.randn(1, model.latent_dim, generator=generator, device=device)

    records = list_records(args.data_root, modality, args.split, fields=[args.source_field])
    if args.max_cases is not None:
        records = records[: args.max_cases]
    for target_field in target_fields:
        target_domain_id = FIELD_TO_ID[target_field]
        pair_dir = out_root / "task2" / modality / f"{args.source_field}_to_{target_field}"
        for record in tqdm(records, desc=f"{modality} {args.source_field}->{target_field}"):
            volume, reference, source_stats = load_nifti_normalized(record.path)
            if args.debug_crop is not None:
                volume = center_crop(volume, args.debug_crop)
            translated = sliding_window_translate(
                model,
                volume,
                target_domain_id=target_domain_id,
                patch_size=patch_size,
                overlap=args.overlap,
                device=device,
                batch_size=args.batch_size,
                z=z,
            )
            if args.scale == "target_stats" and target_field in field_stats:
                translated = denormalize(translated, field_stats[target_field])
            elif args.scale == "source":
                translated = denormalize(translated, source_stats)
            save_nifti_like(translated, reference, pair_dir / output_name(record.path, args.source_field, target_field))


if __name__ == "__main__":
    main()
