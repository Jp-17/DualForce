#!/usr/bin/env python3
"""
DualForce Training Script (Accelerate + FSDP)

Usage (single node, 8 GPUs):
    accelerate launch --num_processes 8 \
        scripts/training_scripts/dualforce_train.py \
        configs/dualforce/dualforce_train_8gpu.py

Usage (with overrides):
    accelerate launch --num_processes 8 \
        scripts/training_scripts/dualforce_train.py \
        configs/dualforce/dualforce_train_8gpu.py \
        --cfg-options trainer.max_steps=50000 data.batch_size=1
"""

import os
import argparse
import torch
from mmengine.config import Config, DictAction

from mova.registry import DATASETS, DIFFUSION_PIPELINES, TRANSFORMS
from mova.engine.trainer.accelerate.accelerate_trainer import AccelerateTrainer
from mova.datasets.dualforce_dataset import dualforce_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="DualForce Training")
    parser.add_argument("config", help="Config file path")
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options'
    )
    return parser.parse_args()


def build_dataloader(cfg):
    """Build data loader for DualForce dataset."""
    transform = None
    if cfg.get("transform", None) is not None:
        transform = TRANSFORMS.build(cfg.transform)

    dataset_cfg = cfg.dataset.copy()
    if transform is not None:
        dataset_cfg["transform"] = transform
    dataset = DATASETS.build(dataset_cfg)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
        collate_fn=dualforce_collate_fn,
    )

    return dataloader


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Build dataloader
    dataloader = build_dataloader(cfg.data)
    print(f"[DualForce] Dataset size: {len(dataloader.dataset)}")
    print(f"[DualForce] Batch size per GPU: {cfg.data.batch_size}")
    print(f"[DualForce] Gradient accumulation: {cfg.trainer.get('gradient_accumulation_steps', 1)}")
    effective_batch = cfg.data.batch_size * cfg.trainer.get('gradient_accumulation_steps', 1)
    print(f"[DualForce] Effective batch size per GPU: {effective_batch}")

    # Device setup
    use_fsdp = cfg.trainer.get("use_fsdp", False)
    if use_fsdp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = "cpu"  # FSDP handles device placement
        print(f"[FSDP] Process LOCAL_RANK={local_rank}: Loading on CPU, FSDP will shard")
    else:
        device = "cpu"

    torch_dtype = torch.bfloat16

    # Build model kwargs
    model_default_args = {"device": device, "torch_dtype": torch_dtype}
    if getattr(cfg, 'diffusion_pipeline', None) is not None:
        if cfg.diffusion_pipeline.get('use_gradient_checkpointing', False):
            model_default_args['use_gradient_checkpointing'] = True
        if cfg.diffusion_pipeline.get('use_gradient_checkpointing_offload', False):
            model_default_args['use_gradient_checkpointing_offload'] = True

    # Build DualForce model
    print("[DualForce] Building model...")
    model = DIFFUSION_PIPELINES.build(
        cfg.diffusion_pipeline,
        default_args=model_default_args,
    )

    # Set scheduler timesteps
    model.scheduler.set_timesteps(
        cfg.trainer.get("num_train_timesteps", 1000),
        training=True
    )

    # Freeze non-trainable modules
    train_modules = cfg.trainer.get("train_modules", ["video_dit", "struct_dit", "dual_tower_bridge"])
    model.freeze_for_training(train_modules)

    # Logger config
    logger_kwargs = {}
    if hasattr(cfg, 'logger'):
        logger_kwargs = dict(cfg.logger)

    # Build trainer
    trainer = AccelerateTrainer(
        model=model,
        train_dataloader=dataloader,
        optimizer_cfg=dict(cfg.optimizer),
        # Training
        max_steps=cfg.trainer.max_steps,
        gradient_accumulation_steps=cfg.trainer.get("gradient_accumulation_steps", 1),
        gradient_clip_norm=cfg.trainer.get("gradient_clip_norm", 1.0),
        # Mixed precision
        mixed_precision=cfg.trainer.get("mixed_precision", "bf16"),
        # FSDP
        use_fsdp=cfg.trainer.get("use_fsdp", False),
        fsdp_config=dict(cfg.fsdp) if hasattr(cfg, 'fsdp') else None,
        # Logging
        log_interval=cfg.trainer.get("log_interval", 1),
        logger_type=cfg.trainer.get("logger_type", "tensorboard"),
        logger_kwargs=logger_kwargs,
        # Checkpointing
        save_interval=cfg.trainer.get("save_interval", 500),
        save_path=cfg.trainer.get("save_path", "./checkpoints/dualforce"),
        resume_from=cfg.trainer.get("resume_from", None),
        # LR Scheduler
        lr_scheduler_type=cfg.trainer.get("lr_scheduler_type", "cosine"),
        warmup_steps=cfg.trainer.get("warmup_steps", 1000),
        min_lr=cfg.trainer.get("min_lr", 1e-6),
        # Training modules
        train_modules=train_modules,
        # No LoRA for DualForce
        use_lora=False,
        # Context Parallel
        enable_cp=cfg.trainer.get("enable_cp", False),
    )

    print("[DualForce] Starting training...")
    trainer.train()


if __name__ == "__main__":
    main()
