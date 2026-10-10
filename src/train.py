import argparse
import os
import random
import yaml
import numpy as np
import torch

from src.datasets.build import (
    build_train_dataset,
    build_test_dataset,
    build_dataloader,
    ExplicitIndexSampler,
)
from src.models import (
    build_faster_rcnn,
    build_detr,
    build_deformable_detr,
)
from src.training.engine import (
    train_one_epoch,
    evaluate_one_epoch,
    save_checkpoint,
    load_checkpoint,
    build_optimizer_from_config,
    get_device,
)
from src.utils.wandb_logger import init_run, finish_run


MODEL_BUILDERS = {
    "faster_rcnn": build_faster_rcnn,
    "detr": build_detr,
    "deformable_detr": build_deformable_detr,
}


def set_seed(seed):
    """Set random seed across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Object Detection Models on PASCAL VOC")
    parser.add_argument(
        "--model",
        type=str,
        default="faster_rcnn",
        choices=["faster_rcnn", "detr", "deformable_detr"],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment.yaml",
        help="Path to experiment configuration YAML file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs override",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size override",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of dataloader workers override",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B logging mode",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum number of batches to process per epoch (default: None)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1,
        help="Evaluation interval in epochs (default: 1)",
    )
    parser.add_argument(
        "--recovery-dir",
        type=str,
        default="/content/checkpoints" if os.path.exists("/content") else None,
        help="Directory to save mid-epoch recovery checkpoints (default: /content/checkpoints if on Colab, else save-dir)",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=500,
        help="Batch frequency for saving mid-epoch recovery checkpoints (default: 500)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load experiment configuration
    with open(args.config, "r") as f:
        exp_config = yaml.safe_load(f)

    train_cfg = exp_config.get("training", {})
    ds_cfg = exp_config.get("dataset", {})

    seed = args.seed if args.seed is not None else train_cfg.get("seeds", [42])[0]
    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 10)
    batch_size = args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 4)
    num_workers = args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 2)
    eval_interval = args.eval_interval if args.eval_interval is not None else train_cfg.get("eval_interval", 1)
    num_classes = ds_cfg.get("num_classes", 20)

    set_seed(seed)
    device = get_device()
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Seed: {seed}, Epochs: {epochs}, Batch Size: {batch_size}, Num Workers: {num_workers}, Eval Interval: {eval_interval}, Max Batches: {args.max_batches}")
    if args.max_batches is not None:
        print(f"[Smoke Test / Partial Run] --max-batches={args.max_batches} enabled. Epochs and saved checkpoints will be partial artifacts.")

    run_name = f"{args.model}_seed{seed}"
    init_run(
        name=run_name,
        config=exp_config,
        group=args.model,
        seed=seed,
        mode=args.wandb_mode,
    )

    print("Building datasets and dataloaders...")
    train_dataset = build_train_dataset()
    test_dataset = build_test_dataset()

    use_cuda = (device.type == "cuda")
    test_loader = build_dataloader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        use_cuda=use_cuda,
    )

    print(f"Building model '{args.model}' (num_classes={num_classes})...")
    builder = MODEL_BUILDERS[args.model]
    model = builder(num_classes=num_classes, pretrained=True)

    if args.model == "deformable_detr":
        if hasattr(model, "model") and hasattr(model.model, "backbone"):
            for p in model.model.backbone.parameters():
                p.requires_grad = False
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Freezing backbone for '{args.model}'. Total parameters: {total_params:,} | Trainable parameters: {trainable_params:,}")

    optimizer = build_optimizer_from_config(model, args.model, exp_config)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    os.makedirs(args.save_dir, exist_ok=True)
    start_epoch = 1
    start_batch_idx = 0
    saved_epoch_indices = None
    best_map = 0.0

    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        resumed_epoch, last_metrics, resumed_batch_idx, saved_epoch_indices = load_checkpoint(
            args.resume, model, optimizer, device=device, scaler=scaler
        )
        is_partial = last_metrics.get("is_partial", False) or (args.max_batches is not None)
        best_map = last_metrics.get("AP", 0.0)

        if resumed_batch_idx > 0:
            start_epoch = resumed_epoch
            start_batch_idx = resumed_batch_idx
            print(
                f"[Recovery Checkpoint] Resuming epoch {start_epoch} from batch {start_batch_idx} "
                f"using exact restored epoch permutation (note: batch position is restored; sampler/RNG state is not saved). Best mAP so far: {best_map:.4f}"
            )
        elif is_partial:
            start_epoch = resumed_epoch
            start_batch_idx = 0
            print(
                f"[Smoke-Test Checkpoint] Resumed partial checkpoint at epoch {resumed_epoch}. "
                f"Re-running epoch {start_epoch} from start (note: batch-level sampler state is not preserved). Best mAP so far: {best_map:.4f}"
            )
        else:
            start_epoch = resumed_epoch + 1
            start_batch_idx = 0
            print(f"Resumed from completed epoch {resumed_epoch}. Next epoch: {start_epoch} (Best mAP so far: {best_map:.4f})")

    if args.max_batches is not None:
        latest_ckpt_name = f"{args.model}_seed{seed}_smoke_latest.pth"
        best_ckpt_name = f"{args.model}_seed{seed}_smoke_best.pth"
        recovery_ckpt_name = f"{args.model}_seed{seed}_smoke_recovery.pth"
    else:
        latest_ckpt_name = f"{args.model}_seed{seed}_latest.pth"
        best_ckpt_name = f"{args.model}_seed{seed}_best.pth"
        recovery_ckpt_name = f"{args.model}_seed{seed}_recovery.pth"

    best_ckpt_path = os.path.join(args.save_dir, best_ckpt_name)
    if os.path.exists(best_ckpt_path):
        try:
            _, best_ckpt_metrics, _, _ = load_checkpoint(best_ckpt_path, model)
            existing_best = best_ckpt_metrics.get("AP", 0.0)
            if existing_best > best_map:
                best_map = existing_best
        except Exception:
            pass

    recovery_dir = args.recovery_dir if args.recovery_dir is not None else args.save_dir
    os.makedirs(recovery_dir, exist_ok=True)
    recovery_filepath = os.path.join(recovery_dir, recovery_ckpt_name)

    print("Starting training loop...")
    for epoch in range(start_epoch, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")

        if epoch == start_epoch and saved_epoch_indices is not None:
            epoch_indices = saved_epoch_indices
        else:
            epoch_indices = torch.randperm(len(train_dataset)).tolist()

        train_sampler = ExplicitIndexSampler(epoch_indices)
        train_loader = build_dataloader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=train_sampler,
            use_cuda=use_cuda,
        )

        if use_cuda:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        current_start_batch = start_batch_idx if epoch == start_epoch else 0

        train_loss, train_time = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            model_type=args.model,
            device=device,
            epoch=epoch,
            max_batches=args.max_batches,
            scaler=scaler,
            start_batch_idx=current_start_batch,
            save_freq=args.save_freq,
            recovery_filepath=recovery_filepath,
            epoch_indices=epoch_indices,
        )

        if use_cuda:
            torch.cuda.synchronize(device)
            peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            peak_res_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

        should_eval = (epoch % eval_interval == 0) or (epoch == epochs)
        eval_metrics = {}
        eval_time = 0.0

        if should_eval:
            if use_cuda:
                torch.cuda.synchronize(device)
            eval_metrics, eval_time = evaluate_one_epoch(
                model=model,
                dataloader=test_loader,
                model_type=args.model,
                device=device,
                epoch=epoch,
                max_batches=args.max_batches,
            )
            if use_cuda:
                torch.cuda.synchronize(device)
            val_map = eval_metrics.get("AP", 0.0)
            print(f"Epoch {epoch} Results - Train Loss: {train_loss:.4f}, Val mAP: {val_map:.4f}")
        else:
            print(f"Epoch {epoch} Results - Train Loss: {train_loss:.4f} (Evaluation skipped for interval={eval_interval})")

        if args.max_batches is not None:
            eval_metrics["is_partial"] = True
            eval_metrics["max_batches"] = args.max_batches

        total_time = train_time + eval_time
        if use_cuda:
            print(
                f"Epoch {epoch} Timing - Train: {train_time:.2f}s, Eval: {eval_time:.2f}s, Total: {total_time:.2f}s | "
                f"Peak Training VRAM - Allocated: {peak_alloc_mb:.2f} MB, Reserved: {peak_res_mb:.2f} MB"
            )
        else:
            print(f"Epoch {epoch} Timing - Train: {train_time:.2f}s, Eval: {eval_time:.2f}s, Total: {total_time:.2f}s")

        # Save latest checkpoint
        checkpoint_path = os.path.join(args.save_dir, latest_ckpt_name)
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            filepath=checkpoint_path,
            metrics=eval_metrics,
            scaler=scaler,
        )

        if should_eval and val_map > best_map:
            best_map = val_map
            best_ckpt_path = os.path.join(args.save_dir, best_ckpt_name)
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                filepath=best_ckpt_path,
                metrics=eval_metrics,
                scaler=scaler,
            )
            print(f"Saved new best model checkpoint to {best_ckpt_path} (mAP: {best_map:.4f})")

        # Clean up stale mid-epoch recovery checkpoint once full-epoch latest checkpoint is saved
        if os.path.exists(recovery_filepath):
            try:
                os.remove(recovery_filepath)
            except OSError:
                pass


    finish_run()
    print("\nTraining completed successfully!")



if __name__ == "__main__":
    main()

