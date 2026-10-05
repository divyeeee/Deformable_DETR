import time
import os
import sys
import types
import importlib.machinery
import numpy as np
import torch

from src.utils.wandb_logger import log_metrics
from src.evaluation.coco_map import COCOEvaluator


def _ensure_scipy_mock():
    """
    Provide a pure-Python Jonker-Volgenant linear_sum_assignment solver if scipy is not installed.
    Allows HuggingFace DETR and Deformable DETR HungarianMatcher loss calculation to function natively.
    """
    try:
        import scipy.optimize  # noqa: F401
    except ImportError:
        def linear_sum_assignment(cost_matrix, maximize=False):
            cost = np.asarray(cost_matrix, dtype=float)
            if maximize:
                cost = -cost
            n_rows, n_cols = cost.shape
            if n_rows == 0 or n_cols == 0:
                return np.array([], dtype=int), np.array([], dtype=int)
            transposed = False
            if n_rows > n_cols:
                cost = cost.T
                n_rows, n_cols = n_cols, n_rows
                transposed = True

            u = np.zeros(n_rows + 1)
            v = np.zeros(n_cols + 1)
            p = np.zeros(n_cols + 1, dtype=int)
            way = np.zeros(n_cols + 1, dtype=int)

            for i in range(1, n_rows + 1):
                p[0] = i
                j0 = 0
                minv = np.full(n_cols + 1, np.inf)
                used = np.zeros(n_cols + 1, dtype=bool)
                while True:
                    used[j0] = True
                    i0 = p[j0]
                    delta = np.inf
                    j1 = 0
                    for j in range(1, n_cols + 1):
                        if not used[j]:
                            cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                            if cur < minv[j]:
                                minv[j] = cur
                                way[j] = j0
                            if minv[j] < delta:
                                delta = minv[j]
                                j1 = j
                    for j in range(0, n_cols + 1):
                        if used[j]:
                            u[p[j]] += delta
                            v[j] -= delta
                        else:
                            minv[j] -= delta
                    j0 = j1
                    if p[j0] == 0:
                        break
                while True:
                    j1 = way[j0]
                    p[j0] = p[j1]
                    j0 = j1
                    if j0 == 0:
                        break

            row_ind = np.zeros(n_rows, dtype=int)
            col_ind = np.zeros(n_rows, dtype=int)
            count = 0
            for j in range(1, n_cols + 1):
                if p[j] != 0:
                    row_ind[count] = p[j] - 1
                    col_ind[count] = j - 1
                    count += 1

            order = np.argsort(row_ind)
            row_ind = row_ind[order]
            col_ind = col_ind[order]

            if transposed:
                order = np.argsort(col_ind)
                return col_ind[order], row_ind[order]
            return row_ind, col_ind

        scipy_mod = types.ModuleType("scipy")
        scipy_mod.__spec__ = importlib.machinery.ModuleSpec("scipy", None)
        opt_mod = types.ModuleType("scipy.optimize")
        opt_mod.__spec__ = importlib.machinery.ModuleSpec("scipy.optimize", None)
        opt_mod.linear_sum_assignment = linear_sum_assignment
        scipy_mod.optimize = opt_mod

        sys.modules["scipy"] = scipy_mod
        sys.modules["scipy.optimize"] = opt_mod

        try:
            import transformers.utils.import_utils as iu

            if hasattr(iu.is_scipy_available, "cache_clear"):
                iu.is_scipy_available.cache_clear()
        except Exception:
            pass

        try:
            import transformers.loss.loss_for_object_detection as lod

            lod.linear_sum_assignment = linear_sum_assignment
        except Exception:
            pass

        try:
            import transformers.loss.loss_deformable_detr as ldd

            ldd.linear_sum_assignment = linear_sum_assignment
        except Exception:
            pass


_ensure_scipy_mock()


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_optimizer_from_config(model, model_type, config):
    """
    Build optimizer based on model type and configs/experiment.yaml settings.
    """
    opt_cfg = config.get("optimizers", {}).get(model_type, {})
    train_cfg = config.get("training", {})

    weight_decay = train_cfg.get("weight_decay", 0.0001)
    opt_name = opt_cfg.get("name", "AdamW")
    lr = opt_cfg.get("lr", 0.0001)

    if model_type == "faster_rcnn":
        momentum = opt_cfg.get("momentum", 0.9)
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    elif model_type in ["detr", "deformable_detr"]:
        backbone_lr = opt_cfg.get("backbone_lr", 0.00001)
        param_dicts = [
            {
                "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad],
                "lr": lr,
            },
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": backbone_lr,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=lr, weight_decay=weight_decay)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def prepare_batch(model_type, images, targets, device):
    """
    Format images and targets for the specific model type.
    """
    if model_type == "faster_rcnn":
        images_dev = [img.to(device) for img in images]
        targets_dev = []
        for t in targets:
            t_dev = {
                "boxes": t["boxes"].to(device),
                "labels": (t["labels"] + 1).to(device),  # 1-based for torchvision Faster R-CNN
                "image_id": t["image_id"].to(device),
                "area": t["area"].to(device),
                "iscrowd": t["iscrowd"].to(device),
            }
            targets_dev.append(t_dev)
        return images_dev, targets_dev

    elif model_type in ["detr", "deformable_detr"]:
        batch_size = len(images)
        max_h = max(img.shape[1] for img in images)
        max_w = max(img.shape[2] for img in images)

        pixel_values = torch.zeros((batch_size, 3, max_h, max_w), dtype=images[0].dtype, device=device)
        for i, img in enumerate(images):
            c, h, w = img.shape
            pixel_values[i, :, :h, :w] = img.to(device)

        hf_targets = []
        for t in targets:
            boxes = t["boxes"].to(device)
            labels = t["labels"].to(device)
            size = t["size"].to(device)

            if len(boxes) > 0:
                h_val, w_val = size[0].item(), size[1].item()
                xmin, ymin, xmax, ymax = boxes.unbind(-1)
                cx = (xmin + xmax) / (2.0 * w_val)
                cy = (ymin + ymax) / (2.0 * h_val)
                bw = (xmax - xmin) / w_val
                bh = (ymax - ymin) / h_val
                boxes_cxcywh = torch.stack([cx, cy, bw, bh], dim=-1)
            else:
                boxes_cxcywh = torch.zeros((0, 4), dtype=torch.float32, device=device)

            hf_targets.append({
                "class_labels": labels,
                "boxes": boxes_cxcywh,
            })

        return pixel_values, hf_targets

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def train_one_epoch(model, dataloader, optimizer, model_type, device=None, epoch=0):
    """
    Train model for one epoch.
    """
    if device is None:
        device = get_device()

    model.to(device)
    model.train()

    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    for images, targets in dataloader:
        optimizer.zero_grad()
        inputs, formatted_targets = prepare_batch(model_type, images, targets, device)

        if model_type == "faster_rcnn":
            loss_dict = model(inputs, formatted_targets)
            losses = sum(loss for loss in loss_dict.values())
        else:
            outputs = model(pixel_values=inputs, labels=formatted_targets)
            losses = outputs.loss

        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        num_batches += 1

    elapsed = time.time() - start_time
    avg_loss = total_loss / max(1, num_batches)
    current_lr = optimizer.param_groups[0]["lr"]

    log_metrics({
        "train_loss": avg_loss,
        "lr": current_lr,
        "epoch_time_sec": elapsed,
    }, step=epoch)

    return avg_loss, elapsed


def evaluate_one_epoch(model, dataloader, model_type, device=None, epoch=0):
    """
    Evaluate model for one epoch and return COCO mAP metrics.
    """
    if device is None:
        device = get_device()

    model.to(device)
    model.eval()

    evaluator = COCOEvaluator()
    start_time = time.time()

    with torch.no_grad():
        for images, targets in dataloader:
            inputs, _ = prepare_batch(model_type, images, targets, device)

            preds_formatted = []

            if model_type == "faster_rcnn":
                outputs = model(inputs)
                for out in outputs:
                    preds_formatted.append({
                        "boxes": out["boxes"].cpu(),
                        "scores": out["scores"].cpu(),
                        "labels": (out["labels"] - 1).cpu(),  # Shift back to 0-based
                    })
            else:
                outputs = model(pixel_values=inputs)
                logits = outputs.logits.cpu()
                pred_boxes = outputs.pred_boxes.cpu()

                for i, (log, box) in enumerate(zip(logits, pred_boxes)):
                    probs = torch.softmax(log, dim=-1)
                    scores, labels = probs[:, :-1].max(dim=-1)

                    keep = scores > 0.05
                    scores_k = scores[keep]
                    labels_k = labels[keep]
                    box_k = box[keep]

                    h_val, w_val = targets[i]["size"][0].item(), targets[i]["size"][1].item()
                    cx, cy, bw, bh = box_k.unbind(-1)
                    xmin = (cx - 0.5 * bw) * w_val
                    ymin = (cy - 0.5 * bh) * h_val
                    xmax = (cx + 0.5 * bw) * w_val
                    ymax = (cy + 0.5 * bh) * h_val
                    boxes_xyxy = torch.stack([xmin, ymin, xmax, ymax], dim=-1)

                    preds_formatted.append({
                        "boxes": boxes_xyxy,
                        "scores": scores_k,
                        "labels": labels_k,
                    })

            evaluator.update(preds_formatted, targets)

    metrics = evaluator.evaluate()
    elapsed = time.time() - start_time

    log_metrics({
        "val_mAP": metrics["AP"],
        "val_mAP50": metrics["AP50"],
        "val_mAP75": metrics["AP75"],
        "eval_time_sec": elapsed,
    }, step=epoch)

    return metrics, elapsed


def save_checkpoint(model, optimizer, epoch, filepath, metrics=None):
    """
    Save checkpoint dictionary to disk.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics or {},
    }
    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load checkpoint dictionary from disk into model and optional optimizer.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at: {filepath}")
    checkpoint = torch.load(filepath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("metrics", {})
