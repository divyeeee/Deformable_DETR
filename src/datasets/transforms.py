import torch
import torchvision.transforms.functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class Resize:
    def __init__(self, min_size=800, max_size=1333):
        if isinstance(min_size, (list, tuple)):
            self.min_size = min_size
        else:
            self.min_size = (min_size,)
        self.max_size = max_size

    def __call__(self, image, target):
        old_w, old_h = image.size
        if len(self.min_size) > 1:
            idx = int(torch.randint(len(self.min_size), (1,)).item())
            min_size = float(self.min_size[idx])
        else:
            min_size = float(self.min_size[0])

        scale = min_size / min(old_w, old_h)
        if max(old_w, old_h) * scale > self.max_size:
            scale = self.max_size / max(old_w, old_h)

        new_w = int(round(old_w * scale))
        new_h = int(round(old_h * scale))

        image = F.resize(image, (new_h, new_w))

        scale_x = new_w / old_w
        scale_y = new_h / old_h

        if len(target["boxes"]) > 0:
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            target["boxes"] = boxes
            target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        else:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["area"] = torch.zeros((0,), dtype=torch.float32)

        target["size"] = torch.tensor([new_h, new_w], dtype=torch.int64)

        return image, target


class RandomHorizontalFlip:
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, image, target):
        if torch.rand(1).item() < self.probability:
            w, _ = image.size
            image = F.hflip(image)

            if len(target["boxes"]) > 0:
                boxes = target["boxes"].clone()
                xmin = w - boxes[:, 2]
                xmax = w - boxes[:, 0]
                boxes[:, 0] = xmin
                boxes[:, 2] = xmax
                target["boxes"] = boxes
                target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            else:
                target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                target["area"] = torch.zeros((0,), dtype=torch.float32)

        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Normalize:
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, self.mean, self.std)
        return image, target


def get_train_transforms(min_size=600, max_size=1333):
    return Compose([
        Resize(min_size=min_size, max_size=max_size),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(),
    ])


def get_test_transforms(min_size=800, max_size=1333):
    return Compose([
        Resize(min_size=min_size, max_size=max_size),
        ToTensor(),
        Normalize(),
    ])

