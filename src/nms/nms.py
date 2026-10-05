import torch
from src.losses.iou_losses import box_iou


def _validate_inputs(boxes, scores):
    """Ensure boxes is [N, 4] and scores is [N]."""
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    scores = torch.as_tensor(scores, dtype=torch.float32)

    if boxes.dim() == 1 and boxes.shape[0] == 4:
        boxes = boxes.unsqueeze(0)
    elif boxes.dim() != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Boxes must have shape [N, 4], got {boxes.shape}")

    if scores.dim() == 0:
        scores = scores.unsqueeze(0)
    elif scores.dim() != 1:
        raise ValueError(f"Scores must have shape [N], got {scores.shape}")

    if boxes.shape[0] != scores.shape[0]:
        raise ValueError(f"Boxes and scores length mismatch: {boxes.shape[0]} vs {scores.shape[0]}")

    return boxes, scores


def hard_nms(boxes, scores, iou_threshold=0.5):
    """
    Standard Greedy Non-Maximum Suppression (Hard NMS).
    Args:
        boxes (Tensor): Bounding boxes [N, 4] in xyxy format.
        scores (Tensor): Detection confidence scores [N].
        iou_threshold (float): IoU overlap threshold for suppression.
    Returns:
        Tensor: Indices of kept boxes [K] (int64).
    """
    boxes, scores = _validate_inputs(boxes, scores)
    if boxes.shape[0] == 0:
        return torch.tensor([], dtype=torch.int64)
    if boxes.shape[0] == 1:
        return torch.tensor([0], dtype=torch.int64)

    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        current_box = boxes[i : i + 1]
        other_boxes = boxes[order[1:]]

        ious = box_iou(current_box, other_boxes, pairwise=True).squeeze(0)
        mask = ious <= iou_threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.int64)


def soft_nms(boxes, scores, iou_threshold=0.5, sigma=0.5, score_threshold=0.001, method="gaussian"):
    """
    Soft Non-Maximum Suppression (Soft NMS).
    Args:
        boxes (Tensor): Bounding boxes [N, 4] in xyxy format.
        scores (Tensor): Detection confidence scores [N].
        iou_threshold (float): IoU threshold for linear decay.
        sigma (float): Gaussian decay variance parameter.
        score_threshold (float): Score cutoff for stopping/keeping boxes.
        method (str): Decay method ('gaussian' or 'linear').
    Returns:
        tuple: (keep_indices [K], updated_scores [K])
    """
    boxes, scores = _validate_inputs(boxes, scores)
    N = boxes.shape[0]

    if N == 0:
        return torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.float32)
    if N == 1:
        if scores[0] >= score_threshold:
            return torch.tensor([0], dtype=torch.int64), scores.clone()
        return torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.float32)

    boxes_c = boxes.clone()
    scores_c = scores.clone()
    indices = torch.arange(N, dtype=torch.int64)

    keep_indices = []
    updated_scores = []

    for i in range(N):
        max_pos = torch.argmax(scores_c[i:]).item() + i
        max_score = scores_c[max_pos].item()

        if max_score < score_threshold:
            break

        # Swap max element to current position i
        if max_pos != i:
            boxes_c[i], boxes_c[max_pos] = boxes_c[max_pos].clone(), boxes_c[i].clone()
            scores_c[i], scores_c[max_pos] = scores_c[max_pos].clone(), scores_c[i].clone()
            indices[i], indices[max_pos] = indices[max_pos].clone(), indices[i].clone()

        keep_indices.append(indices[i].item())
        updated_scores.append(scores_c[i].item())

        if i == N - 1:
            break

        current_box = boxes_c[i : i + 1]
        remaining_boxes = boxes_c[i + 1 :]
        ious = box_iou(current_box, remaining_boxes, pairwise=True).squeeze(0)

        if method == "gaussian":
            decay = torch.exp(-(ious ** 2) / sigma)
        elif method == "linear":
            decay = torch.ones_like(ious)
            mask = ious > iou_threshold
            decay[mask] = 1.0 - ious[mask]
        else:
            raise ValueError(f"Unknown soft NMS method: {method}")

        scores_c[i + 1 :] *= decay

    return torch.tensor(keep_indices, dtype=torch.int64), torch.tensor(updated_scores, dtype=torch.float32)


def diou_nms(boxes, scores, iou_threshold=0.5, beta=1.0):
    """
    Distance-IoU Non-Maximum Suppression (DIoU-NMS).
    Args:
        boxes (Tensor): Bounding boxes [N, 4] in xyxy format.
        scores (Tensor): Detection confidence scores [N].
        iou_threshold (float): DIoU threshold for suppression.
        beta (float): Exponent for center distance penalty.
    Returns:
        Tensor: Indices of kept boxes [K] (int64).
    """
    boxes, scores = _validate_inputs(boxes, scores)
    if boxes.shape[0] == 0:
        return torch.tensor([], dtype=torch.int64)
    if boxes.shape[0] == 1:
        return torch.tensor([0], dtype=torch.int64)

    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        current_box = boxes[i : i + 1]
        other_boxes = boxes[order[1:]]

        ious = box_iou(current_box, other_boxes, pairwise=True).squeeze(0)

        c1_x = (current_box[:, 0] + current_box[:, 2]) / 2.0
        c1_y = (current_box[:, 1] + current_box[:, 3]) / 2.0
        c2_x = (other_boxes[:, 0] + other_boxes[:, 2]) / 2.0
        c2_y = (other_boxes[:, 1] + other_boxes[:, 3]) / 2.0

        rho2 = (c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2

        cw1 = torch.min(current_box[:, 0], other_boxes[:, 0])
        ch1 = torch.min(current_box[:, 1], other_boxes[:, 1])
        cw2 = torch.max(current_box[:, 2], other_boxes[:, 2])
        ch2 = torch.max(current_box[:, 3], other_boxes[:, 3])

        c2 = (cw2 - cw1) ** 2 + (ch2 - ch1) ** 2 + 1e-7

        diou = ious - (rho2 / c2) ** beta
        mask = diou <= iou_threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.int64)
