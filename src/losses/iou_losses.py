import math
import torch


def _reshape_boxes(boxes):
    """Ensure boxes tensor is 2D with shape [N, 4]."""
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    if boxes.dim() == 1:
        if boxes.shape[0] != 4:
            raise ValueError(f"1D box input must have 4 elements, got shape {boxes.shape}")
        boxes = boxes.unsqueeze(0)
    elif boxes.dim() != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Boxes must have shape [N, 4], got shape {boxes.shape}")
    return boxes


def box_iou(boxes1, boxes2, pairwise=True):
    """
    Compute IoU between boxes.
    Args:
        boxes1: Tensor of shape [N, 4] or [4] in xyxy format.
        boxes2: Tensor of shape [M, 4] or [4] in xyxy format.
        pairwise (bool): If True, returns matrix [N, M]. If False, returns vector [N] (requires N == M).
    Returns:
        Tensor: IoU values.
    """
    b1 = _reshape_boxes(boxes1)
    b2 = _reshape_boxes(boxes2)

    if pairwise:
        area1 = (b1[:, 2] - b1[:, 0]).clamp(min=0) * (b1[:, 3] - b1[:, 1]).clamp(min=0)
        area2 = (b2[:, 2] - b2[:, 0]).clamp(min=0) * (b2[:, 3] - b2[:, 1]).clamp(min=0)

        x1 = torch.max(b1[:, None, 0], b2[None, :, 0])
        y1 = torch.max(b1[:, None, 1], b2[None, :, 1])
        x2 = torch.min(b1[:, None, 2], b2[None, :, 2])
        y2 = torch.min(b1[:, None, 3], b2[None, :, 3])

        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        union = area1[:, None] + area2[None, :] - inter
        iou = inter / (union + 1e-7)

        if boxes1.dim() == 1 and boxes2.dim() == 1:
            return iou.squeeze(0).squeeze(0)
        return iou
    else:
        if b1.shape[0] != b2.shape[0]:
            raise ValueError(f"Elementwise IoU requires same batch size, got {b1.shape[0]} vs {b2.shape[0]}")

        area1 = (b1[:, 2] - b1[:, 0]).clamp(min=0) * (b1[:, 3] - b1[:, 1]).clamp(min=0)
        area2 = (b2[:, 2] - b2[:, 0]).clamp(min=0) * (b2[:, 3] - b2[:, 1]).clamp(min=0)

        x1 = torch.max(b1[:, 0], b2[:, 0])
        y1 = torch.max(b1[:, 1], b2[:, 1])
        x2 = torch.min(b1[:, 2], b2[:, 2])
        y2 = torch.min(b1[:, 3], b2[:, 3])

        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        union = area1 + area2 - inter
        iou = inter / (union + 1e-7)

        if boxes1.dim() == 1 and boxes2.dim() == 1:
            return iou.squeeze(0)
        return iou


def _reduce_loss(loss, reduction):
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def generalized_iou_loss(boxes1, boxes2, reduction="mean"):
    """
    Compute Generalized IoU (GIoU) loss = 1 - GIoU.
    """
    b1 = _reshape_boxes(boxes1)
    b2 = _reshape_boxes(boxes2)

    if b1.shape[0] != b2.shape[0]:
        raise ValueError(f"Boxes shape mismatch: {b1.shape[0]} vs {b2.shape[0]}")

    area1 = (b1[:, 2] - b1[:, 0]).clamp(min=0) * (b1[:, 3] - b1[:, 1]).clamp(min=0)
    area2 = (b2[:, 2] - b2[:, 0]).clamp(min=0) * (b2[:, 3] - b2[:, 1]).clamp(min=0)

    x1 = torch.max(b1[:, 0], b2[:, 0])
    y1 = torch.max(b1[:, 1], b2[:, 1])
    x2 = torch.min(b1[:, 2], b2[:, 2])
    y2 = torch.min(b1[:, 3], b2[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    union = area1 + area2 - inter
    iou = inter / (union + 1e-7)

    # Enclosing box
    cw1 = torch.min(b1[:, 0], b2[:, 0])
    ch1 = torch.min(b1[:, 1], b2[:, 1])
    cw2 = torch.max(b1[:, 2], b2[:, 2])
    ch2 = torch.max(b1[:, 3], b2[:, 3])

    c_area = (cw2 - cw1).clamp(min=0) * (ch2 - ch1).clamp(min=0)
    giou = iou - (c_area - union) / (c_area + 1e-7)
    loss = 1.0 - giou

    return _reduce_loss(loss, reduction)


def distance_iou_loss(boxes1, boxes2, reduction="mean"):
    """
    Compute Distance IoU (DIoU) loss = 1 - DIoU.
    """
    b1 = _reshape_boxes(boxes1)
    b2 = _reshape_boxes(boxes2)

    if b1.shape[0] != b2.shape[0]:
        raise ValueError(f"Boxes shape mismatch: {b1.shape[0]} vs {b2.shape[0]}")

    area1 = (b1[:, 2] - b1[:, 0]).clamp(min=0) * (b1[:, 3] - b1[:, 1]).clamp(min=0)
    area2 = (b2[:, 2] - b2[:, 0]).clamp(min=0) * (b2[:, 3] - b2[:, 1]).clamp(min=0)

    x1 = torch.max(b1[:, 0], b2[:, 0])
    y1 = torch.max(b1[:, 1], b2[:, 1])
    x2 = torch.min(b1[:, 2], b2[:, 2])
    y2 = torch.min(b1[:, 3], b2[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    union = area1 + area2 - inter
    iou = inter / (union + 1e-7)

    # Center distance squared
    c1_x = (b1[:, 0] + b1[:, 2]) / 2.0
    c1_y = (b1[:, 1] + b1[:, 3]) / 2.0
    c2_x = (b2[:, 0] + b2[:, 2]) / 2.0
    c2_y = (b2[:, 1] + b2[:, 3]) / 2.0

    rho2 = (c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2

    # Enclosing box diagonal squared
    cw1 = torch.min(b1[:, 0], b2[:, 0])
    ch1 = torch.min(b1[:, 1], b2[:, 1])
    cw2 = torch.max(b1[:, 2], b2[:, 2])
    ch2 = torch.max(b1[:, 3], b2[:, 3])

    c2 = (cw2 - cw1) ** 2 + (ch2 - ch1) ** 2 + 1e-7

    diou = iou - (rho2 / c2)
    loss = 1.0 - diou

    return _reduce_loss(loss, reduction)


def complete_iou_loss(boxes1, boxes2, reduction="mean"):
    """
    Compute Complete IoU (CIoU) loss = 1 - CIoU.
    """
    b1 = _reshape_boxes(boxes1)
    b2 = _reshape_boxes(boxes2)

    if b1.shape[0] != b2.shape[0]:
        raise ValueError(f"Boxes shape mismatch: {b1.shape[0]} vs {b2.shape[0]}")

    area1 = (b1[:, 2] - b1[:, 0]).clamp(min=0) * (b1[:, 3] - b1[:, 1]).clamp(min=0)
    area2 = (b2[:, 2] - b2[:, 0]).clamp(min=0) * (b2[:, 3] - b2[:, 1]).clamp(min=0)

    x1 = torch.max(b1[:, 0], b2[:, 0])
    y1 = torch.max(b1[:, 1], b2[:, 1])
    x2 = torch.min(b1[:, 2], b2[:, 2])
    y2 = torch.min(b1[:, 3], b2[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    union = area1 + area2 - inter
    iou = inter / (union + 1e-7)

    # Center distance squared
    c1_x = (b1[:, 0] + b1[:, 2]) / 2.0
    c1_y = (b1[:, 1] + b1[:, 3]) / 2.0
    c2_x = (b2[:, 0] + b2[:, 2]) / 2.0
    c2_y = (b2[:, 1] + b2[:, 3]) / 2.0

    rho2 = (c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2

    # Enclosing box diagonal squared
    cw1 = torch.min(b1[:, 0], b2[:, 0])
    ch1 = torch.min(b1[:, 1], b2[:, 1])
    cw2 = torch.max(b1[:, 2], b2[:, 2])
    ch2 = torch.max(b1[:, 3], b2[:, 3])

    c2 = (cw2 - cw1) ** 2 + (ch2 - ch1) ** 2 + 1e-7

    # Aspect ratio penalty
    w1 = (b1[:, 2] - b1[:, 0]).clamp(min=1e-7)
    h1 = (b1[:, 3] - b1[:, 1]).clamp(min=1e-7)
    w2 = (b2[:, 2] - b2[:, 0]).clamp(min=1e-7)
    h2 = (b2[:, 3] - b2[:, 1]).clamp(min=1e-7)

    v = (4.0 / (math.pi ** 2)) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)) ** 2

    with torch.no_grad():
        alpha = v / (1.0 - iou + v + 1e-7)

    ciou = iou - (rho2 / c2 + alpha * v)
    loss = 1.0 - ciou

    return _reduce_loss(loss, reduction)
