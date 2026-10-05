import torch
from src.losses.iou_losses import box_iou


class COCOEvaluator:
    """
    COCO-style object detection evaluator implemented from scratch in PyTorch.
    Computes AP@[.50:.95], AP50, AP75, AP_S, AP_M, AP_L, AR@100, and per-class APs.
    """

    def __init__(self, iou_thresholds=None, area_ranges=None, max_dets=None):
        if iou_thresholds is None:
            # 0.50:0.05:0.95 (10 steps)
            self.iou_thresholds = [round(0.50 + 0.05 * i, 2) for i in range(10)]
        else:
            self.iou_thresholds = iou_thresholds

        if area_ranges is None:
            self.area_ranges = {
                "all": (0.0, 1e10),
                "small": (0.0, 32.0 ** 2),       # [0, 1024)
                "medium": (32.0 ** 2, 96.0 ** 2), # [1024, 9216)
                "large": (96.0 ** 2, 1e10),      # [9216, inf)
            }
        else:
            self.area_ranges = area_ranges

        if max_dets is None:
            self.max_dets = [1, 10, 100]
        else:
            self.max_dets = max_dets

        self.predictions = []
        self.targets = []

    def reset(self):
        self.predictions = []
        self.targets = []

    def update(self, predictions, targets):
        """
        Add batch of predictions and targets.
        Args:
            predictions (list of dicts): Each dict has 'boxes' [N,4], 'scores' [N], 'labels' [N].
            targets (list of dicts): Each dict has 'boxes' [M,4], 'labels' [M], optional 'area' [M], optional 'iscrowd' [M].
        """
        for pred, tgt in zip(predictions, targets):
            boxes_p = torch.as_tensor(pred.get("boxes", torch.zeros((0, 4))), dtype=torch.float32)
            scores_p = torch.as_tensor(pred.get("scores", torch.zeros((0,))), dtype=torch.float32)
            labels_p = torch.as_tensor(pred.get("labels", torch.zeros((0,), dtype=torch.int64)), dtype=torch.int64)

            if len(boxes_p) > 0:
                area_p = (boxes_p[:, 2] - boxes_p[:, 0]).clamp(min=0) * (boxes_p[:, 3] - boxes_p[:, 1]).clamp(min=0)
            else:
                area_p = torch.zeros((0,), dtype=torch.float32)

            boxes_t = torch.as_tensor(tgt.get("boxes", torch.zeros((0, 4))), dtype=torch.float32)
            labels_t = torch.as_tensor(tgt.get("labels", torch.zeros((0,), dtype=torch.int64)), dtype=torch.int64)

            if "area" in tgt and len(tgt["area"]) == len(boxes_t):
                area_t = torch.as_tensor(tgt["area"], dtype=torch.float32)
            else:
                if len(boxes_t) > 0:
                    area_t = (boxes_t[:, 2] - boxes_t[:, 0]).clamp(min=0) * (boxes_t[:, 3] - boxes_t[:, 1]).clamp(min=0)
                else:
                    area_t = torch.zeros((0,), dtype=torch.float32)

            if "iscrowd" in tgt and len(tgt["iscrowd"]) == len(boxes_t):
                iscrowd_t = torch.as_tensor(tgt["iscrowd"], dtype=torch.int64)
            else:
                iscrowd_t = torch.zeros(len(boxes_t), dtype=torch.int64)

            self.predictions.append({
                "boxes": boxes_p,
                "scores": scores_p,
                "labels": labels_p,
                "area": area_p,
            })

            self.targets.append({
                "boxes": boxes_t,
                "labels": labels_t,
                "area": area_t,
                "iscrowd": iscrowd_t,
            })

    @staticmethod
    def _interpolate_101_points(recalls, precisions):
        """
        Compute 101-point interpolated AP from recall and precision arrays.
        """
        if len(recalls) == 0:
            return 0.0

        r_grid = [round(i * 0.01, 2) for i in range(101)]
        p_interp = []

        for r_th in r_grid:
            mask = recalls >= r_th
            if mask.any():
                p_max = precisions[mask].max().item()
            else:
                p_max = 0.0
            p_interp.append(p_max)

        return sum(p_interp) / 101.0

    def evaluate(self):
        """
        Compute evaluation metrics across accumulated predictions and targets.
        Returns:
            dict: Evaluation results containing AP, AP50, AP75, AP_S, AP_M, AP_L, AR@100, and per-class APs.
        """
        all_classes = set()
        for tgt in self.targets:
            all_classes.update(tgt["labels"].tolist())
        for pred in self.predictions:
            all_classes.update(pred["labels"].tolist())

        classes = sorted(list(all_classes))

        if len(classes) == 0:
            return {
                "AP": 0.0, "AP50": 0.0, "AP75": 0.0,
                "AP_S": 0.0, "AP_M": 0.0, "AP_L": 0.0,
                "AR@100": 0.0, "per_class_AP": {}
            }

        eval_table = {}

        for area_name, (min_area, max_area) in self.area_ranges.items():
            eval_table[area_name] = {}
            for max_d in self.max_dets:
                eval_table[area_name][max_d] = {}

                for cls in classes:
                    eval_table[area_name][max_d][cls] = {}

                    image_gts = []
                    image_preds = []

                    for img_idx, (pred, tgt) in enumerate(zip(self.predictions, self.targets)):
                        # GT boxes for class and area range
                        gt_mask = (tgt["labels"] == cls) & (tgt["area"] >= min_area) & (tgt["area"] < max_area)
                        gt_boxes = tgt["boxes"][gt_mask]
                        gt_iscrowd = tgt["iscrowd"][gt_mask]

                        image_gts.append({
                            "boxes": gt_boxes,
                            "iscrowd": gt_iscrowd,
                            "matched": [False] * len(gt_boxes)
                        })

                        # Pred boxes for class and area range
                        p_mask = (pred["labels"] == cls) & (pred["area"] >= min_area) & (pred["area"] < max_area)
                        p_boxes = pred["boxes"][p_mask]
                        p_scores = pred["scores"][p_mask]

                        # Apply max_dets per image limit
                        if len(p_scores) > max_d:
                            topk_indices = torch.topk(p_scores, k=max_d).indices
                            p_boxes = p_boxes[topk_indices]
                            p_scores = p_scores[topk_indices]

                        for b, s in zip(p_boxes, p_scores):
                            image_preds.append({
                                "img_idx": img_idx,
                                "box": b,
                                "score": s.item()
                            })

                    # Sort all predictions across images by score descending
                    image_preds.sort(key=lambda x: x["score"], reverse=True)
                    total_gts = sum(len(g["boxes"]) for g in image_gts)

                    for iou_th in self.iou_thresholds:
                        if total_gts == 0:
                            eval_table[area_name][max_d][cls][iou_th] = {"ap": -1.0, "ar": -1.0}
                            continue

                        if len(image_preds) == 0:
                            eval_table[area_name][max_d][cls][iou_th] = {"ap": 0.0, "ar": 0.0}
                            continue

                        # Reset matched state
                        for g in image_gts:
                            g["matched"] = [False] * len(g["boxes"])

                        tp = []
                        fp = []

                        for p in image_preds:
                            img_idx = p["img_idx"]
                            p_box = p["box"].unsqueeze(0)
                            gt_info = image_gts[img_idx]
                            gt_boxes = gt_info["boxes"]

                            if len(gt_boxes) == 0:
                                tp.append(0)
                                fp.append(1)
                                continue

                            ious = box_iou(p_box, gt_boxes, pairwise=True).squeeze(0)
                            max_iou, max_idx = ious.max(dim=0)
                            max_iou_val = max_iou.item()
                            max_idx_val = max_idx.item()

                            if max_iou_val >= iou_th:
                                if not gt_info["matched"][max_idx_val]:
                                    tp.append(1)
                                    fp.append(0)
                                    gt_info["matched"][max_idx_val] = True
                                else:
                                    tp.append(0)
                                    fp.append(1)
                            else:
                                tp.append(0)
                                fp.append(1)

                        tp_tensor = torch.tensor(tp, dtype=torch.float32)
                        fp_tensor = torch.tensor(fp, dtype=torch.float32)

                        cum_tp = torch.cumsum(tp_tensor, dim=0)
                        cum_fp = torch.cumsum(fp_tensor, dim=0)

                        recalls = cum_tp / total_gts
                        precisions = cum_tp / (cum_tp + cum_fp)

                        ap = self._interpolate_101_points(recalls, precisions)
                        ar = (cum_tp[-1] / total_gts).item() if len(cum_tp) > 0 else 0.0

                        eval_table[area_name][max_d][cls][iou_th] = {"ap": ap, "ar": ar}

        def _average_ap(area_name, max_d, iou_list=None):
            if iou_list is None:
                iou_list = self.iou_thresholds
            aps = []
            for cls in classes:
                for iou_th in iou_list:
                    ap_val = eval_table[area_name][max_d][cls][iou_th]["ap"]
                    if ap_val >= 0.0:
                        aps.append(ap_val)
            return sum(aps) / len(aps) if len(aps) > 0 else 0.0

        def _average_ar(area_name, max_d, iou_list=None):
            if iou_list is None:
                iou_list = self.iou_thresholds
            ars = []
            for cls in classes:
                for iou_th in iou_list:
                    ar_val = eval_table[area_name][max_d][cls][iou_th]["ar"]
                    if ar_val >= 0.0:
                        ars.append(ar_val)
            return sum(ars) / len(ars) if len(ars) > 0 else 0.0

        ap = _average_ap("all", 100)
        ap50 = _average_ap("all", 100, iou_list=[0.50])
        ap75 = _average_ap("all", 100, iou_list=[0.75])
        ap_s = _average_ap("small", 100)
        ap_m = _average_ap("medium", 100)
        ap_l = _average_ap("large", 100)
        ar100 = _average_ar("all", 100)

        per_class_ap = {}
        for cls in classes:
            cls_aps = [eval_table["all"][100][cls][iou_th]["ap"] for iou_th in self.iou_thresholds if eval_table["all"][100][cls][iou_th]["ap"] >= 0.0]
            per_class_ap[cls] = sum(cls_aps) / len(cls_aps) if len(cls_aps) > 0 else 0.0

        return {
            "AP": ap,
            "AP50": ap50,
            "AP75": ap75,
            "AP_S": ap_s,
            "AP_M": ap_m,
            "AP_L": ap_l,
            "AR@100": ar100,
            "per_class_AP": per_class_ap,
        }
