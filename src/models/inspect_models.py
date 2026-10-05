from src.models import build_faster_rcnn, build_detr, build_deformable_detr


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def main():
    models = [
        ("Faster R-CNN ResNet-50 FPN V2", build_faster_rcnn),
        ("DETR ResNet-50", build_detr),
        ("Deformable DETR ResNet-50", build_deformable_detr),
    ]

    for name, builder in models:
        model = builder(num_classes=20, pretrained=True)
        total, trainable = count_parameters(model)
        print(f"Model Name: {name}")
        print(f"  Total Parameters:     {total}")
        print(f"  Trainable Parameters: {trainable}")
        print()


if __name__ == "__main__":
    main()
