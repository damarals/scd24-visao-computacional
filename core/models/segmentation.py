import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

class PersonSegmenter(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
        self.model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
    
    def forward(self, x):
        return self.model(x)['out']