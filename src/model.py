import torch
import torch.nn as nn
import timm


class ODIRModel(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super().__init__()

        # EfficientNet-B0 (lighter than B4, good for CPU training)
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )
        embed_dim = self.backbone.num_features  # 1280

        # Single multi-label classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)