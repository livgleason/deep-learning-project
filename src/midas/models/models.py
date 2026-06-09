import torch
import torch.nn as nn
from torchvision import models

class DetectionModel(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = backbone.fc.in_features

        self.attention = nn.Sequential(nn.Linear(self.feature_dim, 128), nn.Tanh(), nn.Linear(128, 1))

        self.classifier = nn.Sequential(nn.Linear(self.feature_dim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, images):
        feats = []
        for img in images:
            f = self.encoder(img.unsqueeze(0))
            feats.append(f.view(-1))

        feats = torch.stack(feats)

        attn_logits = self.attention(feats)
        attn_weights = torch.softmax(attn_logits.squeeze(-1) / 0.3, dim=0)
        attn_weights = attn_weights.unsqueeze(-1)

        pooled = (feats * attn_weights).sum(dim=0)

        x = pooled.unsqueeze(0)
        out = self.classifier(x)

        return out.squeeze() * 3.0
