import math
import torch
from src.losses.iou_losses import (
    box_iou,
    generalized_iou_loss,
    distance_iou_loss,
    complete_iou_loss,
)


def test_identical_boxes():
    b1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b2 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])

    iou = box_iou(b1, b2, pairwise=False)
    giou_l = generalized_iou_loss(b1, b2)
    diou_l = distance_iou_loss(b1, b2)
    ciou_l = complete_iou_loss(b1, b2)

    assert torch.isclose(iou, torch.tensor([1.0]), atol=1e-4)
    assert torch.isclose(giou_l, torch.tensor(0.0), atol=1e-4)
    assert torch.isclose(diou_l, torch.tensor(0.0), atol=1e-4)
    assert torch.isclose(ciou_l, torch.tensor(0.0), atol=1e-4)


def test_non_overlapping_boxes():
    b1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b2 = torch.tensor([[20.0, 20.0, 30.0, 30.0]])

    iou = box_iou(b1, b2, pairwise=False)
    giou_l = generalized_iou_loss(b1, b2)
    diou_l = distance_iou_loss(b1, b2)
    ciou_l = complete_iou_loss(b1, b2)

    assert torch.isclose(iou, torch.tensor([0.0]), atol=1e-4)
    assert giou_l.item() > 1.0
    assert diou_l.item() > 1.0
    assert ciou_l.item() > 1.0


def test_partially_overlapping_boxes():
    b1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b2 = torch.tensor([[5.0, 0.0, 15.0, 10.0]])

    iou = box_iou(b1, b2, pairwise=False)
    assert torch.isclose(iou, torch.tensor([1.0 / 3.0]), atol=1e-4)

    giou_l = generalized_iou_loss(b1, b2)
    diou_l = distance_iou_loss(b1, b2)
    ciou_l = complete_iou_loss(b1, b2)

    assert 0.0 < giou_l.item() < 1.0
    assert 0.0 < diou_l.item() < 1.0
    assert 0.0 < ciou_l.item() < 1.0


def test_contained_boxes():
    b1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b2 = torch.tensor([[2.0, 2.0, 8.0, 8.0]])

    iou = box_iou(b1, b2, pairwise=False)
    assert torch.isclose(iou, torch.tensor([0.36]), atol=1e-4)

    giou_l = generalized_iou_loss(b1, b2)
    diou_l = distance_iou_loss(b1, b2)
    ciou_l = complete_iou_loss(b1, b2)

    assert torch.isfinite(giou_l)
    assert torch.isfinite(diou_l)
    assert torch.isfinite(ciou_l)


def test_symmetry():
    b1 = torch.tensor([[1.0, 2.0, 5.0, 8.0], [0.0, 0.0, 10.0, 10.0]])
    b2 = torch.tensor([[2.0, 3.0, 6.0, 9.0], [5.0, 5.0, 15.0, 15.0]])

    iou12 = box_iou(b1, b2, pairwise=False)
    iou21 = box_iou(b2, b1, pairwise=False)
    assert torch.allclose(iou12, iou21, atol=1e-5)

    giou12 = generalized_iou_loss(b1, b2, reduction="none")
    giou21 = generalized_iou_loss(b2, b1, reduction="none")
    assert torch.allclose(giou12, giou21, atol=1e-5)

    diou12 = distance_iou_loss(b1, b2, reduction="none")
    diou21 = distance_iou_loss(b2, b1, reduction="none")
    assert torch.allclose(diou12, diou21, atol=1e-5)

    ciou12 = complete_iou_loss(b1, b2, reduction="none")
    ciou21 = complete_iou_loss(b2, b1, reduction="none")
    assert torch.allclose(ciou12, ciou21, atol=1e-5)


def test_finite_outputs_and_gradients():
    for loss_fn in [generalized_iou_loss, distance_iou_loss, complete_iou_loss]:
        b1 = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]], requires_grad=True)
        b2 = torch.tensor([[2.0, 2.0, 8.0, 8.0], [20.0, 20.0, 30.0, 30.0]], requires_grad=False)

        loss = loss_fn(b1, b2)
        assert torch.isfinite(loss)

        loss.backward()
        assert b1.grad is not None
        assert torch.all(torch.isfinite(b1.grad))


def test_known_numerical_cases():
    b1 = torch.tensor([0.0, 0.0, 10.0, 10.0])
    b2 = torch.tensor([0.0, 0.0, 10.0, 5.0])

    iou = box_iou(b1, b2, pairwise=False)
    assert torch.isclose(iou, torch.tensor([0.5]), atol=1e-4)

    giou_l = generalized_iou_loss(b1, b2)
    assert torch.isclose(giou_l, torch.tensor(0.5), atol=1e-4)


if __name__ == "__main__":
    test_identical_boxes()
    print("test_identical_boxes: PASSED")
    test_non_overlapping_boxes()
    print("test_non_overlapping_boxes: PASSED")
    test_partially_overlapping_boxes()
    print("test_partially_overlapping_boxes: PASSED")
    test_contained_boxes()
    print("test_contained_boxes: PASSED")
    test_symmetry()
    print("test_symmetry: PASSED")
    test_finite_outputs_and_gradients()
    print("test_finite_outputs_and_gradients: PASSED")
    test_known_numerical_cases()
    print("test_known_numerical_cases: PASSED")
    print("\nAll 7 IoU loss unit tests PASSED successfully!")
