import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(num_classes=20, pretrained=True):
    """
    Build Faster R-CNN with ResNet-50 FPN backbone.
    Args:
        num_classes (int): Number of foreground classes (default: 20 for VOC).
        pretrained (bool): Whether to load pretrained weights.
    Returns:
        torch.nn.Module: Faster R-CNN model instance.
    """
    # torchvision Faster R-CNN requires num_classes + 1 (for background)
    num_classes_with_bg = num_classes + 1

    if pretrained:
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_with_bg)
    else:
        model = fasterrcnn_resnet50_fpn_v2(weights=None, num_classes=num_classes_with_bg)

    return model
