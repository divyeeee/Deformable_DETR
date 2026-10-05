import math
import torch
from src.evaluation.coco_map import COCOEvaluator


def test_perfect_detection():
    evaluator = COCOEvaluator()
    targets = [{
        "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
        "labels": torch.tensor([0]),
    }]
    predictions = [{
        "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
        "scores": torch.tensor([0.99]),
        "labels": torch.tensor([0]),
    }]

    evaluator.update(predictions, targets)
    metrics = evaluator.evaluate()

    assert torch.isclose(torch.tensor(metrics["AP"]), torch.tensor(1.0), atol=1e-4)
    assert torch.isclose(torch.tensor(metrics["AP50"]), torch.tensor(1.0), atol=1e-4)
    assert torch.isclose(torch.tensor(metrics["AP75"]), torch.tensor(1.0), atol=1e-4)


def test_no_detections():
    evaluator = COCOEvaluator()
    targets = [{
        "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
        "labels": torch.tensor([0]),
    }]
    predictions = [{
        "boxes": torch.zeros((0, 4)),
        "scores": torch.zeros((0,)),
        "labels": torch.zeros((0,), dtype=torch.int64),
    }]

    evaluator.update(predictions, targets)
    metrics = evaluator.evaluate()

    assert metrics["AP"] == 0.0
    assert metrics["AP50"] == 0.0
    assert metrics["AP75"] == 0.0


def test_wrong_class():
    evaluator = COCOEvaluator()
    targets = [{
        "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
        "labels": torch.tensor([0]),
    }]
    predictions = [{
        "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
        "scores": torch.tensor([0.95]),
        "labels": torch.tensor([1]),
    }]

    evaluator.update(predictions, targets)
    metrics = evaluator.evaluate()

    assert metrics["AP"] == 0.0


def test_duplicate_detections():
    evaluator = COCOEvaluator()
    targets = [{
        "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
        "labels": torch.tensor([0]),
    }]
    predictions = [{
        "boxes": torch.tensor([
            [10.0, 10.0, 50.0, 50.0],
            [10.0, 10.0, 50.0, 50.0]
        ]),
        "scores": torch.tensor([0.95, 0.85]),
        "labels": torch.tensor([0, 0]),
    }]

    evaluator.update(predictions, targets)
    metrics = evaluator.evaluate()

    assert metrics["AP50"] == 1.0


def test_iou_threshold_behavior():
    evaluator = COCOEvaluator()
    targets = [{
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "labels": torch.tensor([0]),
    }]
    predictions = [{
        "boxes": torch.tensor([[4.0, 0.0, 10.0, 10.0]]),
        "scores": torch.tensor([0.90]),
        "labels": torch.tensor([0]),
    }]

    evaluator.update(predictions, targets)
    metrics = evaluator.evaluate()

    assert metrics["AP50"] == 1.0
    assert metrics["AP75"] == 0.0


def test_multiple_classes():
    evaluator = COCOEvaluator()
    targets = [{
        "boxes": torch.tensor([
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 40.0, 40.0]
        ]),
        "labels": torch.tensor([0, 1]),
    }]
    predictions = [{
        "boxes": torch.tensor([
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 40.0, 40.0]
        ]),
        "scores": torch.tensor([0.9, 0.95]),
        "labels": torch.tensor([0, 1]),
    }]

    evaluator.update(predictions, targets)
    metrics = evaluator.evaluate()

    assert metrics["AP"] == 1.0
    assert 0 in metrics["per_class_AP"]
    assert 1 in metrics["per_class_AP"]
    assert metrics["per_class_AP"][0] == 1.0
    assert metrics["per_class_AP"][1] == 1.0


def test_small_medium_large_area_handling():
    evaluator = COCOEvaluator()
    targets = [{
        "boxes": torch.tensor([
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 50.0, 50.0],
            [0.0, 0.0, 100.0, 100.0]
        ]),
        "labels": torch.tensor([0, 0, 0]),
    }]
    predictions = [{
        "boxes": torch.tensor([
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 50.0, 50.0],
            [0.0, 0.0, 100.0, 100.0]
        ]),
        "scores": torch.tensor([0.9, 0.9, 0.9]),
        "labels": torch.tensor([0, 0, 0]),
    }]

    evaluator.update(predictions, targets)
    metrics = evaluator.evaluate()

    assert metrics["AP_S"] == 1.0
    assert metrics["AP_M"] == 1.0
    assert metrics["AP_L"] == 1.0


def test_101_point_interpolation():
    recalls = torch.tensor([0.5, 1.0])
    precisions = torch.tensor([1.0, 0.5])

    ap = COCOEvaluator._interpolate_101_points(recalls, precisions)
    expected = (51 * 1.0 + 50 * 0.5) / 101.0
    assert math.isclose(ap, expected, rel_tol=1e-4)


if __name__ == "__main__":
    test_perfect_detection()
    print("test_perfect_detection: PASSED")
    test_no_detections()
    print("test_no_detections: PASSED")
    test_wrong_class()
    print("test_wrong_class: PASSED")
    test_duplicate_detections()
    print("test_duplicate_detections: PASSED")
    test_iou_threshold_behavior()
    print("test_iou_threshold_behavior: PASSED")
    test_multiple_classes()
    print("test_multiple_classes: PASSED")
    test_small_medium_large_area_handling()
    print("test_small_medium_large_area_handling: PASSED")
    test_101_point_interpolation()
    print("test_101_point_interpolation: PASSED")
    print("\nAll 8 COCO mAP unit tests PASSED successfully!")
