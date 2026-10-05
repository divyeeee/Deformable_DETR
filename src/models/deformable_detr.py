from transformers import DeformableDetrConfig, DeformableDetrForObjectDetection


def build_model(num_classes=20, pretrained=True):
    """
    Build Deformable DETR with ResNet-50 backbone using HuggingFace Transformers.
    Args:
        num_classes (int): Number of foreground classes (default: 20 for VOC).
        pretrained (bool): Whether to load pretrained weights from 'SenseTime/deformable-detr'.
    Returns:
        DeformableDetrForObjectDetection: Deformable DETR model instance.
    """
    checkpoint = "SenseTime/deformable-detr"

    if pretrained:
        model = DeformableDetrForObjectDetection.from_pretrained(
            checkpoint,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
    else:
        config = DeformableDetrConfig.from_pretrained(checkpoint, num_labels=num_classes)
        model = DeformableDetrForObjectDetection(config)

    return model
