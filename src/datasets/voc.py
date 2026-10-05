import os
import xml.etree.ElementTree as ET

import torch
from PIL import Image
from torch.utils.data import Dataset


VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

CLASS_TO_ID = {name: i for i, name in enumerate(VOC_CLASSES)}


class VOCDataset(Dataset):
    def __init__(self, root, year="2007", image_set="trainval", transforms=None):
        self.root = root
        self.image_set = image_set
        self.transforms = transforms

        if isinstance(year, (list, tuple)):
            self.years = [str(y) for y in year]
        elif "+" in str(year):
            self.years = [str(y) for y in year.split("+")]
        elif "," in str(year):
            self.years = [str(y) for y in year.split(",")]
        else:
            self.years = [str(year)]

        self.year = "+".join(self.years) if len(self.years) > 1 else self.years[0]
        self.samples = []

        for yr in self.years:
            voc_root = os.path.join(root, f"VOC{yr}")
            image_dir = os.path.join(voc_root, "JPEGImages")
            annotation_dir = os.path.join(voc_root, "Annotations")
            split_file = os.path.join(
                voc_root,
                "ImageSets",
                "Main",
                f"{image_set}.txt"
            )

            if not os.path.exists(split_file):
                raise FileNotFoundError(f"Split file not found: {split_file}")

            with open(split_file, "r") as f:
                ids = [line.strip() for line in f.readlines() if line.strip()]

            for img_id in ids:
                img_path = os.path.join(image_dir, f"{img_id}.jpg")
                ann_path = os.path.join(annotation_dir, f"{img_id}.xml")
                self.samples.append((yr, img_id, img_path, ann_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        yr, image_id, image_path, annotation_path = self.samples[index]

        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        orig_size = torch.tensor([h, w], dtype=torch.int64)
        size = torch.tensor([h, w], dtype=torch.int64)

        boxes = []
        labels = []
        difficult = []

        if os.path.exists(annotation_path):
            root = ET.parse(annotation_path).getroot()

            for obj in root.findall("object"):
                class_name = obj.find("name").text.lower().strip()
                if class_name not in CLASS_TO_ID:
                    continue

                bbox = obj.find("bndbox")
                xmin = float(bbox.find("xmin").text) - 1.0
                ymin = float(bbox.find("ymin").text) - 1.0
                xmax = float(bbox.find("xmax").text) - 1.0
                ymax = float(bbox.find("ymax").text) - 1.0

                xmin = max(0.0, xmin)
                ymin = max(0.0, ymin)
                xmax = min(float(w), xmax)
                ymax = min(float(h), ymax)

                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(CLASS_TO_ID[class_name])
                    diff_elem = obj.find("difficult")
                    difficult.append(int(diff_elem.text) if diff_elem is not None else 0)

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            iscrowd = torch.zeros(len(boxes), dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index]),
            "area": area,
            "iscrowd": iscrowd,
            "orig_size": orig_size,
            "size": size,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def convert_to_faster_rcnn_target(target):
    """
    Adapter helper for Faster R-CNN (torchvision).
    Shifts 0-based labels (0..19) to 1-based labels (1..20) since class 0 is reserved for background.
    """
    target_frcnn = target.copy()
    target_frcnn["labels"] = target["labels"] + 1
    return target_frcnn


def convert_to_detr_target(target):
    """
    Adapter helper for DETR / Deformable DETR.
    Converts absolute xyxy boxes to normalized cxcywh relative to resized image dimensions.
    """
    target_detr = target.copy()
    boxes = target["boxes"]
    size = target["size"]
    if len(boxes) > 0:
        h, w = size[0].item(), size[1].item()
        xmin, ymin, xmax, ymax = boxes.unbind(-1)
        cx = (xmin + xmax) / (2.0 * w)
        cy = (ymin + ymax) / (2.0 * h)
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        target_detr["boxes"] = torch.stack([cx, cy, bw, bh], dim=-1)
    else:
        target_detr["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
    return target_detr