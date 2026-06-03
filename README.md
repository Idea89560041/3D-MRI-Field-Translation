# 3D Field-Conditioned MRI Field-Strength Translation

This repository contains a PyTorch implementation for 3D MRI field-strength translation. The code supports:

- Task 1: arbitrary field strength to 7T synthesis
- Task 2: 0.1T image enhancement
- Task 3: controllable any-to-any field-strength translation

The implementation uses unpaired 3D patch training and full-volume sliding-window inference. Data, checkpoints, and generated outputs are not included in this code package.

## Dataset

The experiments use the MRIxFields 2026 dataset:

```text
https://mrixfields.chihucloud.com/2026/
```

By default, scripts expect the dataset under `Dataset/`:

```text
Dataset/
  T1W/
    Training/
      0.1T/
      1.5T/
      3T/
      5T/
      7T/
    Validation/
      ...
    Testing/
      ...
  T2W/
    ...
  T2FLAIR/
    ...
```

Each field folder should contain NIfTI volumes (`*.nii.gz`). The loader also accepts a project root containing a nested `Dataset/` folder via `--data-root /path/to/project`.

## Environment

Python 3.10 and PyTorch with CUDA are recommended.

```bash
pip install -r requirements.txt
```

## Task 3: Any-to-Any Pretraining

Train the field-conditioned 3D backbone:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port 29513 train_murd3d.py \
  --data-root Dataset \
  --modality T1W \
  --output-dir outputs/murd3d_T1W \
  --mixed-precision
```

Repeat for `T2W` and `T2FLAIR` by changing `--modality` and `--output-dir`.

Single-GPU debugging:

```bash
CUDA_VISIBLE_DEVICES=0 python train_murd3d.py \
  --data-root Dataset \
  --modality T1W \
  --steps 10 \
  --samples-per-epoch 10 \
  --preview-every 0 \
  --mixed-precision
```

## Task 1: Any Field to 7T

Fine-tune the Task 3 backbone for arbitrary-field-to-7T synthesis:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port 29521 train_task1_finetune.py \
  --data-root Dataset \
  --modality T1W \
  --pretrained outputs/murd3d_T1W/latest.pt \
  --output-dir outputs/task1_any_to_7T_finetune_T1W \
  --mixed-precision
```

Run testing inference:

```bash
CUDA_VISIBLE_DEVICES=0 python infer_task1.py \
  --data-root Dataset \
  --checkpoint outputs/task1_any_to_7T_finetune_T1W/latest.pt \
  --modality T1W \
  --output-dir outputs/task1_T1W
```

## Task 2: 0.1T Enhancement

Fine-tune the Task 3 backbone with fixed 0.1T source images and higher-field targets:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port 29522 train_task2_finetune.py \
  --data-root Dataset \
  --modality T1W \
  --pretrained outputs/murd3d_T1W/latest.pt \
  --output-dir outputs/task2_0p1T_enhancement_finetune_T1W \
  --mixed-precision
```

Run testing inference:

```bash
CUDA_VISIBLE_DEVICES=0 python infer_task2.py \
  --data-root Dataset \
  --checkpoint outputs/task2_0p1T_enhancement_finetune_T1W/latest.pt \
  --modality T1W \
  --output-dir outputs/task2_T1W
```

To train or infer a single target field, add for example:

```bash
--target-fields 7T
```

## Task 3: Any-to-Any Inference

Use the pretrained any-to-any checkpoint directly:

```bash
CUDA_VISIBLE_DEVICES=0 python infer_murd3d.py \
  --data-root Dataset \
  --checkpoint outputs/murd3d_T1W/latest.pt \
  --modality T1W \
  --task task3 \
  --output-dir outputs/task3_T1W
```

Task 1 and Task 2 inference can also be run with the generic inference script:

```bash
python infer_murd3d.py --checkpoint outputs/murd3d_T1W/latest.pt --modality T1W --task task1
python infer_murd3d.py --checkpoint outputs/murd3d_T1W/latest.pt --modality T1W --task task2
```

## Checkpoints

No pretrained checkpoints are included. Place checkpoints under:

```text
outputs/murd3d_<MODALITY>/latest.pt
outputs/task1_any_to_7T_finetune_<MODALITY>/latest.pt
outputs/task2_0p1T_enhancement_finetune_<MODALITY>/latest.pt
```

or pass explicit `--checkpoint` / `--pretrained` paths.

## Repository Contents

```text
mrixfields/                 shared data/model/loss/inference utilities
train_murd3d.py             Task 3 any-to-any pretraining
infer_murd3d.py             generic Task 1/2/3 inference
train_task1_finetune.py     Task 1 fine-tuning
infer_task1.py              Task 1 inference
train_task2_finetune.py     Task 2 fine-tuning
infer_task2.py              Task 2 inference
examples/                   example shell commands
```

## Notes

- Training uses unpaired 3D patches.
- Testing uses full-volume sliding-window inference.
- Multi-GPU training uses `torchrun` with DistributedDataParallel.
- `--batch-size` is per GPU.
- The default output root is `outputs/`.
