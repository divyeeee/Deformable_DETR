from transformers import DetrConfig, DetrForObjectDetection


def build_model(num_classes=20, pretrained=True):
    """
    Build DETR with ResNet-50 backbone using HuggingFace Transformers.
    Args:
        num_classes (int): Number of foreground classes (default: 20 for VOC).
        pretrained (bool): Whether to load pretrained weights from 'facebook/detr-resnet-50'.
    Returns:
        DetrForObjectDetection: DETR model instance.
    """
    checkpoint = "facebook/detr-resnet-50"

    if pretrained:
        model = DetrForObjectDetection.from_pretrained(
            checkpoint,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
    else:
        config = DetrConfig.from_pretrained(checkpoint, num_labels=num_classes)
        model = DetrForObjectDetection(config)

    return model
