from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from mrixfields.constants import FIELD_TO_ID, FIELDS, MODALITIES
from mrixfields.data import (
    UnpairedFieldPatchDataset,
    denormalize,
    estimate_field_intensity_stats,
    list_records,
    load_nifti_normalized,
    save_json,
    save_nifti_like,
)
from mrixfields.infer_utils import sliding_window_translate
from mrixfields.losses import gradient_l1, hinge_discriminator_loss, hinge_generator_loss
from mrixfields.models import Discriminator3d, MURD3D


def parse_args():
    parser = argparse.ArgumentParser(description="Train a 3D MURD-style model for MRIxFields.")
    parser.add_argument("--data-root", default="Dataset")
    parser.add_argument("--modality", choices=MODALITIES, required=True)
    parser.add_argument("--split", default="Training")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--patch-size", nargs=3, type=int, default=(64, 64, 64))
    parser.add_argument("--batch-size", type=int, default=1, help="Per-process batch size for DDP.")
    parser.add_argument("--samples-per-epoch", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--same-domain-prob", type=float, default=0.05)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--max-channels", type=int, default=256)
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--res-blocks", type=int, default=4)
    parser.add_argument("--style-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--adv-weight", type=float, default=1.0)
    parser.add_argument("--cycle-weight", type=float, default=10.0)
    parser.add_argument("--identity-weight", type=float, default=10.0)
    parser.add_argument("--content-weight", type=float, default=10.0)
    parser.add_argument("--style-weight", type=float, default=10.0)
    parser.add_argument("--gradient-weight", type=float, default=0.1)
    parser.add_argument("--diversity-weight", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--stats-max-per-field", type=int, default=8)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--preview-every", type=int, default=500)
    parser.add_argument("--preview-split", default="Testing")
    parser.add_argument("--preview-source-field", default="0.1T", choices=FIELDS)
    parser.add_argument("--preview-target-field", default="7T", choices=FIELDS)
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


def save_checkpoint(path, step, model, discriminator, opt_g, opt_d, args, field_stats):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": unwrap_model(model).state_dict(),
            "discriminator": unwrap_model(discriminator).state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "args": vars(args),
            "fields": list(FIELDS),
            "field_stats": field_stats,
        },
        path,
    )


@torch.no_grad()
def save_preview(step, model, args, output_dir, device, field_stats):
    if args.preview_every <= 0 or args.preview_max_cases <= 0:
        return

    model_for_preview = unwrap_model(model)
    was_training = model_for_preview.training
    model_for_preview.eval()

    records = list_records(
        args.data_root,
        args.modality,
        args.preview_split,
        fields=[args.preview_source_field],
    )[: args.preview_max_cases]
    target_domain_id = FIELD_TO_ID[args.preview_target_field]
    preview_dir = (
        output_dir
        / "previews"
        / f"step_{step:07d}"
        / args.modality
        / f"{args.preview_source_field}_to_{args.preview_target_field}"
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
        stats = field_stats.get(args.preview_target_field, source_stats)
        translated = denormalize(translated, stats)
        output_name = record.path.name.replace(
            f"_{args.preview_source_field}_",
            f"_{args.preview_target_field}_",
        )
        save_nifti_like(translated, reference, preview_dir / output_name)

    if was_training:
        model_for_preview.train()


def main():
    args = parse_args()
    distributed, local_rank, rank, world_size, device = setup_distributed(args)
    is_main = rank == 0

    torch.manual_seed(args.seed + rank)
    output_dir = Path(args.output_dir or Path("outputs") / f"murd3d_{args.modality}")
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    dataset = UnpairedFieldPatchDataset(
        data_root=args.data_root,
        modality=args.modality,
        split=args.split,
        patch_size=args.patch_size,
        samples_per_epoch=args.samples_per_epoch,
        same_domain_prob=args.same_domain_prob,
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

    model = MURD3D(
        num_domains=len(FIELDS),
        base_channels=args.base_channels,
        max_channels=args.max_channels,
        levels=args.levels,
        res_blocks=args.res_blocks,
        style_dim=args.style_dim,
        latent_dim=args.latent_dim,
    ).to(device)
    discriminator = Discriminator3d(
        num_domains=len(FIELDS),
        base_channels=args.base_channels,
        max_channels=args.max_channels,
    ).to(device)

    start_step = 0
    field_stats = {}
    if is_main:
        field_stats = estimate_field_intensity_stats(
            args.data_root,
            args.modality,
            split=args.split,
            max_per_field=args.stats_max_per_field,
        )
        save_json({"modality": args.modality, "field_stats": field_stats}, output_dir / "stats.json")

    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        start_step = int(checkpoint.get("step", 0))
        field_stats = checkpoint.get("field_stats", field_stats)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        discriminator = DDP(discriminator, device_ids=[local_rank], output_device=local_rank)
        if is_main:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
            print(f"Using DDP with world_size={world_size}; CUDA_VISIBLE_DEVICES={visible}.")
    elif is_main:
        print(f"Using device: {device}")

    opt_g = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    if checkpoint is not None:
        opt_g.load_state_dict(checkpoint["opt_g"])
        opt_d.load_state_dict(checkpoint["opt_d"])
    scaler = torch.amp.GradScaler("cuda", enabled=args.mixed_precision and device.type == "cuda")

    log_file = None
    writer = None
    if is_main:
        log_path = output_dir / "train_log.csv"
        write_header = not log_path.exists() or start_step == 0
        log_file = log_path.open("a", newline="", encoding="utf-8")
        writer = csv.DictWriter(
            log_file,
            fieldnames=[
                "step",
                "d_loss",
                "g_loss",
                "adv",
                "cycle",
                "identity",
                "content",
                "style",
                "diversity",
            ],
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
                z2 = torch.randn(x_src.shape[0], args.latent_dim, device=device)
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
                    )
                    save_checkpoint(output_dir / "latest.pt", step, model, discriminator, opt_g, opt_d, args, field_stats)

                if args.preview_every > 0 and (step % args.preview_every == 0 or step == args.steps):
                    save_preview(step, model, args, output_dir, device, field_stats)
    finally:
        if log_file is not None:
            log_file.close()
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
