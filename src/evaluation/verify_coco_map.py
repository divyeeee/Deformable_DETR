import os
import sys
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.evaluation.coco_map import COCOEvaluator


def get_test_cases():
    """
    Generate 3 deterministic synthetic detection sets:
    1. Perfect detections (100% IoU match, exact class labels, no duplicates).
    2. Localization / IoU differences (imperfect box overlap, e.g. IoU = 0.80).
    3. Multi-class with duplicates and missed detections.
    """
    # Case 1: Perfect Detections
    p1 = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "scores": torch.tensor([0.99]),
            "labels": torch.tensor([0]),
        },
        {
            "boxes": torch.tensor([[20.0, 20.0, 60.0, 60.0]]),
            "scores": torch.tensor([0.98]),
            "labels": torch.tensor([1]),
        },
    ]
    t1 = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "labels": torch.tensor([0]),
        },
        {
            "boxes": torch.tensor([[20.0, 20.0, 60.0, 60.0]]),
            "labels": torch.tensor([1]),
        },
    ]

    # Case 2: Localization / IoU Differences
    # Box [0,0,100,100] (area 10000) vs [0,0,100,80] (area 8000) -> IoU = 0.80
    p2 = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 100.0, 80.0]]),
            "scores": torch.tensor([0.95]),
            "labels": torch.tensor([0]),
        }
    ]
    t2 = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 100.0, 100.0]]),
            "labels": torch.tensor([0]),
        }
    ]

    # Case 3: Multiple classes with duplicates and missed detections
    p3 = [
        {
            "boxes": torch.tensor([
                [10.0, 10.0, 50.0, 50.0],
                [10.0, 10.0, 50.0, 50.0],
                [100.0, 100.0, 150.0, 150.0]
            ]),
            "scores": torch.tensor([0.95, 0.85, 0.90]),
            "labels": torch.tensor([0, 0, 1]),
        },
        {
            "boxes": torch.tensor([
                [20.0, 20.0, 70.0, 70.0],
                [50.0, 50.0, 90.0, 75.0]
            ]),
            "scores": torch.tensor([0.92, 0.88]),
            "labels": torch.tensor([0, 2]),
        },
        {
            "boxes": torch.zeros((0, 4)),
            "scores": torch.zeros((0,)),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }
    ]
    t3 = [
        {
            "boxes": torch.tensor([
                [10.0, 10.0, 50.0, 50.0],
                [100.0, 100.0, 150.0, 150.0]
            ]),
            "labels": torch.tensor([0, 1]),
        },
        {
            "boxes": torch.tensor([
                [20.0, 20.0, 70.0, 70.0],
                [50.0, 50.0, 90.0, 90.0]
            ]),
            "labels": torch.tensor([0, 2]),
        },
        {
            "boxes": torch.tensor([[30.0, 30.0, 80.0, 80.0]]),
            "labels": torch.tensor([1]),
        }
    ]

    return [
        ("Case 1: Perfect Detections", p1, t1),
        ("Case 2: Localization / IoU Differences", p2, t2),
        ("Case 3: Multi-Class with Duplicates & Misses", p3, t3),
    ]


def run_pycocotools_eval(predictions, targets):
    """Run official pycocotools COCOeval on dataset format."""
    images = []
    annotations = []
    results = []
    ann_id = 1

    all_classes = set()
    for t in targets:
        all_classes.update(t["labels"].tolist())
    for p in predictions:
        all_classes.update(p["labels"].tolist())

    categories = [{"id": int(c), "name": str(c)} for c in sorted(list(all_classes))]

    for img_idx, (pred, tgt) in enumerate(zip(predictions, targets)):
        images.append({"id": img_idx, "width": 1000, "height": 1000})

        for b, l in zip(tgt["boxes"], tgt["labels"]):
            w = (b[2] - b[0]).item()
            h = (b[3] - b[1]).item()
            area = w * h
            annotations.append({
                "id": ann_id,
                "image_id": img_idx,
                "category_id": int(l.item()),
                "bbox": [float(b[0]), float(b[1]), float(w), float(h)],
                "area": float(area),
                "iscrowd": 0,
            })
            ann_id += 1

        for b, s, l in zip(pred["boxes"], pred["scores"], pred["labels"]):
            w = (b[2] - b[0]).item()
            h = (b[3] - b[1]).item()
            results.append({
                "image_id": img_idx,
                "category_id": int(l.item()),
                "bbox": [float(b[0]), float(b[1]), float(w), float(h)],
                "score": float(s),
            })

    gt_dict = {"images": images, "categories": categories, "annotations": annotations}
    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()

    if len(results) > 0:
        coco_dt = coco_gt.loadRes(results)
    else:
        coco_dt = COCO()
        coco_dt.dataset = {"images": images, "categories": categories, "annotations": []}
        coco_dt.createIndex()

    orig_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    finally:
        sys.stdout.close()
        sys.stdout = orig_stdout

    return {
        "AP": float(coco_eval.stats[0]),
        "AP50": float(coco_eval.stats[1]),
        "AP75": float(coco_eval.stats[2]),
    }


def verify_evaluators(tol=1e-3):
    """
    Compare custom COCOEvaluator against pycocotools COCOeval.
    Asserts absolute difference <= tol (1e-3) across all cases.
    """
    cases = get_test_cases()
    all_passed = True

    print("=========================================================================")
    print(" VERIFYING CUSTOM COCO EVALUATOR AGAINST PYCOCOTOOLS COCOEVAL")
    print("=========================================================================\n")

    for case_name, preds, targets in cases:
        custom_evaluator = COCOEvaluator()
        custom_evaluator.update(preds, targets)
        custom_metrics = custom_evaluator.evaluate()

        pycoco_metrics = run_pycocotools_eval(preds, targets)

        print(f"--- {case_name} ---")
        print(f"{'Metric':<10} | {'Custom AP':<12} | {'pycocotools':<12} | {'Abs Diff':<12} | Status")
        print("-" * 65)

        for metric in ["AP", "AP50", "AP75"]:
            val_custom = custom_metrics[metric]
            val_pycoco = pycoco_metrics[metric]
            diff = abs(val_custom - val_pycoco)

            status = "PASSED" if diff <= tol else "FAILED"
            if diff > tol:
                all_passed = False

            print(f"{metric:<10} | {val_custom:<12.4f} | {val_pycoco:<12.4f} | {diff:<12.6f} | {status}")

        print()

    # Discrepancy Note:
    # 1. 101-point precision envelope interpolation:
    #    Both custom and pycocotools use 101-point recall grid [0.00:0.01:1.00] with max-precision envelope.
    # 2. Precision precision:
    #    PyTorch float32 vs Python float64 introduces minor precision rounding (< 1e-4), well within 1e-3 tolerance.
    return all_passed


if __name__ == "__main__":
    success = verify_evaluators()
    if not success:
        sys.exit(1)
