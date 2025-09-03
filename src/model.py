import torch.nn as nn
import torchvision.models as models

def build_model(backbone: str = "resnet50", num_classes: int = 2, pretrained: bool = True, dropout: float = 0.2):
    if backbone == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_feats = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, num_classes)
        )
    elif backbone == "resnet34":
        net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        in_feats = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, num_classes)
        )
    elif backbone == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_feats = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, num_classes)
        )
    elif backbone == "efficientnet_b0":
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        in_feats = net.classifier[-1].in_features
        net.classifier[-1] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, num_classes)
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return net