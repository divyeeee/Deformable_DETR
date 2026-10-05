import os
import yaml
from torch.utils.data import DataLoader

from src.datasets.voc import VOCDataset, collate_fn
from src.datasets.transforms import get_train_transforms, get_test_transforms


def load_voc_config(config_path="configs/voc.yaml"):
    """Load VOC dataset configuration settings from YAML."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
            return cfg.get("dataset", {})
    return {}


def build_train_dataset(config_path="configs/voc.yaml"):
    """
    Build training dataset (VOC2007 trainval + VOC2012 trainval).
    """
    cfg = load_voc_config(config_path)
    root = cfg.get("root", "data/raw/VOCdevkit")
    train_cfg = cfg.get("train", {})
    years = train_cfg.get("years", ["2007", "2012"])
    image_set = train_cfg.get("image_set", "trainval")

    return VOCDataset(
        root=root,
        year=years,
        image_set=image_set,
        transforms=get_train_transforms()
    )


def build_test_dataset(config_path="configs/voc.yaml"):
    """
    Build test dataset (VOC2007 test).
    """
    cfg = load_voc_config(config_path)
    root = cfg.get("root", "data/raw/VOCdevkit")
    test_cfg = cfg.get("test", {})
    years = test_cfg.get("years", ["2007"])
    image_set = test_cfg.get("image_set", "test")

    return VOCDataset(
        root=root,
        year=years,
        image_set=image_set,
        transforms=get_test_transforms()
    )


def build_dataloader(dataset, batch_size=4, num_workers=2, shuffle=False):
    """
    Build PyTorch DataLoader using custom collate_fn for variable box target tuples.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


if __name__ == "__main__":
    print("--- Running Dataset Builder Smoke Test ---")
    train_ds = build_train_dataset()
    test_ds = build_test_dataset()

    print(f"Train Dataset Size: {len(train_ds)}")
    print(f"Test Dataset Size:  {len(test_ds)}")

    train_loader = build_dataloader(train_ds, batch_size=4, num_workers=0, shuffle=True)
    test_loader = build_dataloader(test_ds, batch_size=4, num_workers=0, shuffle=False)

    train_images, train_targets = next(iter(train_loader))
    test_images, test_targets = next(iter(test_loader))

    print("\n--- Train Batch Smoke Test ---")
    print(f"Batch size (images): {len(train_images)}")
    print(f"Sample 0 image shape: {train_images[0].shape}")
    print(f"Batch target count:   {len(train_targets)}")
    print(f"Sample 0 boxes shape: {train_targets[0]['boxes'].shape}")

    print("\n--- Test Batch Smoke Test ---")
    print(f"Batch size (images): {len(test_images)}")
    print(f"Sample 0 image shape: {test_images[0].shape}")
    print(f"Batch target count:   {len(test_targets)}")
    print(f"Sample 0 boxes shape: {test_targets[0]['boxes'].shape}")

    print("\nSmoke test PASSED successfully!")
