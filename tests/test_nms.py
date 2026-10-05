import torch
from src.nms.nms import hard_nms, soft_nms, diou_nms


def test_empty_input():
    boxes = torch.zeros((0, 4), dtype=torch.float32)
    scores = torch.zeros((0,), dtype=torch.float32)

    keep_hard = hard_nms(boxes, scores)
    keep_soft, scores_soft = soft_nms(boxes, scores)
    keep_diou = diou_nms(boxes, scores)

    assert keep_hard.numel() == 0
    assert keep_soft.numel() == 0
    assert scores_soft.numel() == 0
    assert keep_diou.numel() == 0


def test_single_box():
    boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    scores = torch.tensor([0.9])

    keep_hard = hard_nms(boxes, scores)
    keep_soft, scores_soft = soft_nms(boxes, scores)
    keep_diou = diou_nms(boxes, scores)

    assert torch.equal(keep_hard, torch.tensor([0]))
    assert torch.equal(keep_soft, torch.tensor([0]))
    assert torch.isclose(scores_soft, torch.tensor([0.9]))
    assert torch.equal(keep_diou, torch.tensor([0]))


def test_overlapping_boxes():
    boxes = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],
        [1.0, 1.0, 10.0, 10.0],
        [100.0, 100.0, 110.0, 110.0]
    ])
    scores = torch.tensor([0.9, 0.8, 0.7])

    keep = hard_nms(boxes, scores, iou_threshold=0.5)
    assert torch.equal(keep, torch.tensor([0, 2]))


def test_non_overlapping_boxes():
    boxes = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],
        [20.0, 20.0, 30.0, 30.0],
        [40.0, 40.0, 50.0, 50.0]
    ])
    scores = torch.tensor([0.9, 0.85, 0.8])

    keep_hard = hard_nms(boxes, scores, iou_threshold=0.5)
    keep_soft, _ = soft_nms(boxes, scores, iou_threshold=0.5)
    keep_diou = diou_nms(boxes, scores, iou_threshold=0.5)

    assert torch.equal(keep_hard, torch.tensor([0, 1, 2]))
    assert torch.equal(keep_soft, torch.tensor([0, 1, 2]))
    assert torch.equal(keep_diou, torch.tensor([0, 1, 2]))


def test_hard_vs_soft_behavior():
    boxes = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],
        [2.0, 2.0, 10.0, 10.0]
    ])
    scores = torch.tensor([0.9, 0.85])

    keep_hard = hard_nms(boxes, scores, iou_threshold=0.5)
    assert len(keep_hard) == 1
    assert keep_hard[0].item() == 0

    keep_soft, updated_scores = soft_nms(boxes, scores, iou_threshold=0.5, score_threshold=0.01)
    assert len(keep_soft) == 2
    assert updated_scores[1].item() < 0.85


def test_diou_behavior():
    boxes = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],
        [1.0, 1.0, 10.0, 10.0]
    ])
    scores = torch.tensor([0.9, 0.8])

    keep_diou = diou_nms(boxes, scores, iou_threshold=0.5)
    assert len(keep_diou) == 1
    assert keep_diou[0].item() == 0


def test_finite_outputs():
    boxes = torch.tensor([
        [10.0, 10.0, 20.0, 20.0],
        [12.0, 12.0, 22.0, 22.0]
    ])
    scores = torch.tensor([0.95, 0.75])

    keep_h = hard_nms(boxes, scores)
    keep_s, scores_s = soft_nms(boxes, scores)
    keep_d = diou_nms(boxes, scores)

    assert torch.all(torch.isfinite(scores_s))


def test_deterministic_results():
    boxes = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],
        [1.0, 1.0, 11.0, 11.0],
        [20.0, 20.0, 30.0, 30.0]
    ])
    scores = torch.tensor([0.9, 0.8, 0.85])

    run1 = hard_nms(boxes, scores)
    run2 = hard_nms(boxes, scores)
    assert torch.equal(run1, run2)


if __name__ == "__main__":
    test_empty_input()
    print("test_empty_input: PASSED")
    test_single_box()
    print("test_single_box: PASSED")
    test_overlapping_boxes()
    print("test_overlapping_boxes: PASSED")
    test_non_overlapping_boxes()
    print("test_non_overlapping_boxes: PASSED")
    test_hard_vs_soft_behavior()
    print("test_hard_vs_soft_behavior: PASSED")
    test_diou_behavior()
    print("test_diou_behavior: PASSED")
    test_finite_outputs()
    print("test_finite_outputs: PASSED")
    test_deterministic_results()
    print("test_deterministic_results: PASSED")
    print("\nAll 8 NMS unit tests PASSED successfully!")
