import torch
from src.datasets.voc import (
    VOCDataset,
    collate_fn,
    convert_to_faster_rcnn_target,
    convert_to_detr_target,
)
from src.datasets.transforms import get_train_transforms, get_test_transforms


def test_datasets():
    root = "data/raw/VOCdevkit"

    print("--- 1. Testing individual datasets ---")
    voc2007_trainval = VOCDataset(root=root, year="2007", image_set="trainval", transforms=get_train_transforms())
    voc2012_trainval = VOCDataset(root=root, year="2012", image_set="trainval", transforms=get_train_transforms())
    voc2007_test = VOCDataset(root=root, year="2007", image_set="test", transforms=get_test_transforms())

    print(f"VOC2007 trainval length: {len(voc2007_trainval)} (expected: 5011)")
    print(f"VOC2012 trainval length: {len(voc2012_trainval)} (expected: 11540)")
    print(f"VOC2007 test length:     {len(voc2007_test)} (expected: 4952)")

    assert len(voc2007_trainval) == 5011, f"Expected 5011, got {len(voc2007_trainval)}"
    assert len(voc2012_trainval) == 11540, f"Expected 11540, got {len(voc2012_trainval)}"
    assert len(voc2007_test) == 4952, f"Expected 4952, got {len(voc2007_test)}"

    print("\n--- 2. Testing combined training dataset ---")
    train_dataset = VOCDataset(
        root=root,
        year=["2007", "2012"],
        image_set="trainval",
        transforms=get_train_transforms()
    )
    print(f"Combined train dataset length: {len(train_dataset)} (expected: 16551)")
    assert len(train_dataset) == 16551, f"Expected 16551, got {len(train_dataset)}"

    print("\n--- 3. Testing samples and target shapes ---")
    splits = {
        "VOC2007 trainval": voc2007_trainval,
        "VOC2012 trainval": voc2012_trainval,
        "VOC2007 test": voc2007_test,
        "Combined train": train_dataset,
    }

    required_keys = {"boxes", "labels", "image_id", "area", "iscrowd", "orig_size", "size"}

    for name, dataset in splits.items():
        image, target = dataset[0]

        print(f"\n[{name}] Sample 0:")
        print(f"  Image tensor shape: {image.shape}")
        print(f"  Target keys:        {sorted(list(target.keys()))}")
        print(f"  Boxes shape:        {target['boxes'].shape}")
        print(f"  Labels shape:       {target['labels'].shape}")
        print(f"  Original size:      {target['orig_size'].tolist()}")
        print(f"  Resized size:       {target['size'].tolist()}")

        # Assertions
        assert set(target.keys()) == required_keys, f"Missing or incorrect keys in target: {target.keys()}"
        assert isinstance(image, torch.Tensor), "Image is not a torch.Tensor"
        assert image.dim() == 3 and image.shape[0] == 3, f"Unexpected image shape: {image.shape}"
        assert target["boxes"].dim() == 2 and target["boxes"].shape[1] == 4, f"Boxes shape must be [N, 4], got {target['boxes'].shape}"
        assert target["labels"].dim() == 1 and target["labels"].shape[0] == target["boxes"].shape[0], "Labels length must match boxes count"
        assert target["area"].shape[0] == target["boxes"].shape[0], "Area length must match boxes count"
        assert target["iscrowd"].shape[0] == target["boxes"].shape[0], "Iscrowd length must match boxes count"

        if len(target["labels"]) > 0:
            assert target["labels"].min() >= 0 and target["labels"].max() < 20, "Labels must be 0-based in range [0, 19]"

        # Test model adapters
        frcnn_target = convert_to_faster_rcnn_target(target)
        if len(frcnn_target["labels"]) > 0:
            assert frcnn_target["labels"].min() >= 1 and frcnn_target["labels"].max() <= 20, "Faster R-CNN labels must be 1-based [1, 20]"

        detr_target = convert_to_detr_target(target)
        if len(detr_target["boxes"]) > 0:
            cxcywh = detr_target["boxes"]
            assert (cxcywh >= 0.0).all() and (cxcywh <= 1.0).all(), "DETR boxes must be normalized to [0, 1]"

    print("\nAll dataset verification tests PASSED successfully!")


if __name__ == "__main__":
    test_datasets()
