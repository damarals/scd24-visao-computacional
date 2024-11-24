import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights

class PersonClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)