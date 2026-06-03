from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mrixfields.constants import FIELD_TO_ID, FIELDS, MODALITIES, parse_fields
from mrixfields.data import (
    denormalize,
    estimate_field_intensity_stats,
    group_by_field,
    list_records,
    load_nifti_normalized,
    random_crop_3d,
    augment_patch,
    save_json,
    save_nifti_like,
)
from mrixfields.infer_utils import sliding_window_translate
from mrixfields.losses import gradient_l1, hinge_discriminator_loss, hinge_generator_loss
from mrixfields.models import Discriminator3d, MURD3D


DEFAULT_DATA_ROOT = "Dataset"
DEFAULT_TARGET_FIELD = "7T"


class DirectionalUnpairedPatchDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        modality: str,
        split: str,
        source_fields: Sequence[str],
        target_fields: Sequence[str],
        patch_size: Sequence[int],
        samples_per_epoch: int,
        augment: bool = True,
    ) -> None:
        self.source_records = list_records(data_root, modality, split, source_fields)
        target_records = list_records(data_root, modality, split, target_fields)
        self.target_by_field = group_by_field(target_records)
        self.target_field_ids = sorted(self.target_by_field)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _load_patch(self, path: Path) -> torch.Tensor:
        volume, _, _ = load_nifti_normalized(path)
        patch = random_crop_3d(volume, self.patch_size)
        if self.augment:
            patch = augment_patch(patch)
        return torch.from_numpy(patch[None, ...])

    def __getitem__(self, index: int):
        source = random.choice(self.source_records)
        target_field_id = random.choice(self.target_field_ids)
        target = random.choice(self.target_by_field[target_field_id])
        return {
            "source": self._load_patch(source.path),
            "source_domain": torch.tensor(source.field_id, dtype=torch.long),
            "target": self._load_patch(target.path),
            "target_domain": torch.tensor(target.field_id, dtype=torch.long),
            "source_path": str(source.path),
            "target_path": str(target.path),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Task3 MURD3D backbone for Task1 Any-to-7T.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--modality", choices=MODALITIES, required=True)
    parser.add_argument("--split", default="Training")
    parser.add_argument("--source-fields", nargs="*", default=None)
    parser.add_argument("--target-field", default=DEFAULT_TARGET_FIELD, choices=FIELDS)
    parser.add_argument("--pretrained", default=None, help="Task3 checkpoint used as initialization.")
    parser.add_argument("--resume", default=None, help="Task1 checkpoint used to resume optimizer and step.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--patch-size", nargs=3, type=int, default=(64, 64, 64))
    parser.add_argument("--batch-size", type=int, default=1, help="Per-process batch size for DDP.")
    parser.add_argument("--samples-per-epoch", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--max-channels", type=int, default=None)
    parser.add_argument("--levels", type=int, default=None)
    parser.add_argument("--res-blocks", type=int, default=None)
    parser.add_argument("--style-dim", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--adv-weight", type=float, default=1.0)
    parser.add_argument("--cycle-weight", type=float, default=12.0)
    parser.add_argument("--identity-weight", type=float, default=8.0)
    parser.add_argument("--content-weight", type=float, default=10.0)
    parser.add_argument("--style-weight", type=float, default=10.0)
    parser.add_argument("--gradient-weight", type=float, default=0.1)
    parser.add_argument("--diversity-weight", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--stats-max-per-field", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-discriminator-preload", action="store_true")

    parser.add_argument("--preview-every", type=int, default=500)
    parser.add_argument("--preview-split", default="Testing")
    parser.add_argument("--preview-max-cases", type=int, default=1)
    parser.add_argument("--preview-crop-size", nargs=3, type=int, default=(96, 96, 96))
    parser.add_argument("--preview-overlap", type=float, default=0.5)
    parser.add_argument("--preview-batch-size", type=int, default=1)
    return parser.parse_args()


def setup_distributed(args):
    if "LOCAL_RANK" not in os.environ:
        return False, 0, 0, 1, torch.device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("DDP launch detected, but CUDA is not available.")
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return True, local_rank, rank, world_size, torch.device(f"cuda:{local_rank}")


def cleanup_distributed(distributed):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def to_device(batch, device):
    return {
        "source": batch["source"].to(device, non_blocking=True),
        "source_domain": batch["source_domain"].to(device, non_blocking=True),
        "target": batch["target"].to(device, non_blocking=True),
        "target_domain": batch["target_domain"].to(device, non_blocking=True),
    }


def center_crop_dhw(volume, crop_size):
    crop = tuple(int(v) for v in crop_size)
    starts = [max(0, (dim - size) // 2) for dim, size in zip(volume.shape, crop)]
    slices = tuple(slice(start, min(start + size, dim)) for start, size, dim in zip(starts, crop, volume.shape))
    return volume[slices]


def checkpoint_model_config(args, checkpoint):
    ckpt_args = checkpoint.get("args", {}) if checkpoint else {}
    return {
        "base_channels": int(args.base_channels or ckpt_args.get("base_channels", 16)),
        "max_channels": int(args.max_channels or ckpt_args.get("max_channels", 256)),
        "levels": int(args.levels or ckpt_args.get("levels", 2)),
        "res_blocks": int(args.res_blocks or ckpt_args.get("res_blocks", 4)),
        "style_dim": int(args.style_dim or ckpt_args.get("style_dim", 64)),
        "latent_dim": int(args.latent_dim or ckpt_args.get("latent_dim", 16)),
    }


def save_checkpoint(path, step, model, discriminator, opt_g, opt_d, args, field_stats, model_config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    args_dict = vars(args).copy()
    args_dict.update(model_config)
    torch.save(
        {
            "step": step,
            "model": unwrap_model(model).state_dict(),
            "discriminator": unwrap_model(discriminator).state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "args": args_dict,
            "task": "task1_any_to_7T_finetune",
            "fields": list(FIELDS),
            "source_fields": args.source_fields_resolved,
            "target_fields": [args.target_field],
            "field_stats": field_stats,
        },
        path,
    )


@torch.no_grad()
def save_previews(step, model, args, output_dir, device, field_stats):
    if args.preview_every <= 0 or args.preview_max_cases <= 0:
        return

    model_for_preview = unwrap_model(model)
    was_training = model_for_preview.training
    model_for_preview.eval()
    target_domain_id = FIELD_TO_ID[args.target_field]

    for source_field in args.source_fields_resolved:
        records = list_records(args.data_root, args.modality, args.preview_split, fields=[source_field])
        records = records[: args.preview_max_cases]
        preview_dir = (
            output_dir
            / "previews"
            / f"step_{step:07d}"
            / args.modality
            / f"{source_field}_to_{args.target_field}"
        )
        for record in records:
            volume, reference, source_stats = load_nifti_normalized(record.path)
            if args.preview_crop_size:
                volume = center_crop_dhw(volume, args.preview_crop_size)
            translated = sliding_window_translate(
                model_for_preview,
                volume,
                target_domain_id=target_domain_id,
                patch_size=args.patch_size,
                overlap=args.preview_overlap,
                device=device,
                batch_size=args.preview_batch_size,
            )
            stats = field_stats.get(args.target_field, source_stats)
            translated = denormalize(translated, stats)
            output_name = record.path.name.replace(f"_{source_field}_", f"_{args.target_field}_")
            save_nifti_like(translated, reference, preview_dir / output_name)

    if was_training:
        model_for_preview.train()


def main():
    args = parse_args()
    args.source_fields_resolved = parse_fields(args.source_fields) if args.source_fields else [
        field for field in FIELDS if field != args.target_field
    ]
    if args.target_field in args.source_fields_resolved:
        raise ValueError("Task1 fine-tune expects non-7T source fields by default. Remove target from source fields.")

    distributed, local_rank, rank, world_size, device = setup_distributed(args)
    is_main = rank == 0
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    output_dir = Path(args.output_dir or Path("outputs") / f"task1_any_to_7T_finetune_{args.modality}")
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    init_path = args.resume or args.pretrained
    init_checkpoint = torch.load(init_path, map_location=device) if init_path else None
    model_config = checkpoint_model_config(args, init_checkpoint)

    dataset = DirectionalUnpairedPatchDataset(
        data_root=args.data_root,
        modality=args.modality,
        split=args.split,
        source_fields=args.source_fields_resolved,
        target_fields=[args.target_field],
        patch_size=args.patch_size,
        samples_per_epoch=args.samples_per_epoch,
        augment=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    batches = cycle(loader)

    model = MURD3D(num_domains=len(FIELDS), **model_config).to(device)
    discriminator = Discriminator3d(
        num_domains=len(FIELDS),
        base_channels=model_config["base_channels"],
        max_channels=model_config["max_channels"],
    ).to(device)

    start_step = 0
    field_stats = {}
    if init_checkpoint:
        model.load_state_dict(init_checkpoint["model"])
        if "discriminator" in init_checkpoint and not args.no_discriminator_preload:
            discriminator.load_state_dict(init_checkpoint["discriminator"])
        field_stats = init_checkpoint.get("field_stats", {})
        if args.resume:
            start_step = int(init_checkpoint.get("step", 0))

    if is_main:
        needed_fields = sorted(set(args.source_fields_resolved + [args.target_field]), key=list(FIELDS).index)
        if not all(field in field_stats for field in needed_fields):
            field_stats.update(
                estimate_field_intensity_stats(
                    args.data_root,
                    args.modality,
                    split=args.split,
                    fields=needed_fields,
                    max_per_field=args.stats_max_per_field,
                )
            )
        save_json(
            {
                "task": "task1_any_to_7T_finetune",
                "modality": args.modality,
                "source_fields": args.source_fields_resolved,
                "target_field": args.target_field,
                "field_stats": field_stats,
            },
            output_dir / "stats.json",
        )

    if distributed:
        object_list = [field_stats]
        dist.broadcast_object_list(object_list, src=0)
        field_stats = object_list[0]
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        discriminator = DDP(discriminator, device_ids=[local_rank], output_device=local_rank)
        if is_main:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
            print(f"Using DDP with world_size={world_size}; CUDA_VISIBLE_DEVICES={visible}.")
    elif is_main:
        print(f"Using device: {device}")

    opt_g = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    if args.resume and init_checkpoint:
        if "opt_g" in init_checkpoint:
            opt_g.load_state_dict(init_checkpoint["opt_g"])
        if "opt_d" in init_checkpoint:
            opt_d.load_state_dict(init_checkpoint["opt_d"])

    scaler = torch.amp.GradScaler("cuda", enabled=args.mixed_precision and device.type == "cuda")

    log_file = None
    writer = None
    if is_main:
        log_path = output_dir / "train_log.csv"
        write_header = not log_path.exists() or start_step == 0
        log_file = log_path.open("a", newline="", encoding="utf-8")
        writer = csv.DictWriter(
            log_file,
            fieldnames=["step", "d_loss", "g_loss", "adv", "cycle", "identity", "content", "style", "diversity"],
        )
        if write_header:
            writer.writeheader()

    steps_iter = range(start_step + 1, args.steps + 1)
    progress = tqdm(steps_iter, initial=start_step, total=args.steps) if is_main else steps_iter

    try:
        for step in progress:
            batch = to_device(next(batches), device)
            x_src = batch["source"]
            x_tgt = batch["target"]
            src_domain = batch["source_domain"]
            tgt_domain = batch["target_domain"]
            use_amp = args.mixed_precision and device.type == "cuda"

            opt_d.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                with torch.no_grad():
                    fake_tgt, _, _ = unwrap_model(model).translate(x_src, tgt_domain)
                real_logits = discriminator(x_tgt, tgt_domain)
                fake_logits = discriminator(fake_tgt.detach(), tgt_domain)
                d_loss = hinge_discriminator_loss(real_logits, fake_logits)
            scaler.scale(d_loss).backward()
            scaler.step(opt_d)

            opt_g.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                z2 = torch.randn(x_src.shape[0], model_config["latent_dim"], device=device)
                (
                    fake_tgt,
                    rec_src,
                    id_src,
                    content_src,
                    content_fake,
                    style_tgt,
                    encoded_fake_style,
                    fake_tgt_2,
                ) = model("train_g", x_src, src_domain, tgt_domain, z2=z2)
                fake_logits = discriminator(fake_tgt, tgt_domain)
                adv_loss = hinge_generator_loss(fake_logits)
                cycle_loss = F.l1_loss(rec_src, x_src) + args.gradient_weight * gradient_l1(rec_src, x_src)
                identity_loss = F.l1_loss(id_src, x_src)
                content_loss = F.l1_loss(content_fake, content_src.detach())
                style_loss = F.l1_loss(encoded_fake_style, style_tgt.detach())
                diversity_loss = -F.l1_loss(fake_tgt, fake_tgt_2)
                g_loss = (
                    args.adv_weight * adv_loss
                    + args.cycle_weight * cycle_loss
                    + args.identity_weight * identity_loss
                    + args.content_weight * content_loss
                    + args.style_weight * style_loss
                    + args.diversity_weight * diversity_loss
                )
            scaler.scale(g_loss).backward()
            scaler.step(opt_g)
            scaler.update()

            if is_main:
                progress.set_description(f"d={d_loss.item():.3f} g={g_loss.item():.3f}")
                if step % args.log_every == 0 and writer is not None:
                    writer.writerow(
                        {
                            "step": step,
                            "d_loss": float(d_loss.item()),
                            "g_loss": float(g_loss.item()),
                            "adv": float(adv_loss.item()),
                            "cycle": float(cycle_loss.item()),
                            "identity": float(identity_loss.item()),
                            "content": float(content_loss.item()),
                            "style": float(style_loss.item()),
                            "diversity": float(diversity_loss.item()),
                        }
                    )
                    log_file.flush()

                if step % args.save_every == 0 or step == args.steps:
                    save_checkpoint(
                        output_dir / f"checkpoint_{step:07d}.pt",
                        step,
                        model,
                        discriminator,
                        opt_g,
                        opt_d,
                        args,
                        field_stats,
                        model_config,
                    )
                    save_checkpoint(output_dir / "latest.pt", step, model, discriminator, opt_g, opt_d, args, field_stats, model_config)

                if args.preview_every > 0 and (step % args.preview_every == 0 or step == args.steps):
                    save_previews(step, model, args, output_dir, device, field_stats)
    finally:
        if log_file is not None:
            log_file.close()
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
