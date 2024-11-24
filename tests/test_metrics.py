import pytest
import torch
import numpy as np
from core.metrics import ClassificationMetrics, DetectionMetrics, SegmentationMetrics

class MockClassificationModel:
    def __init__(self, predictions):
        self.predictions = predictions
        self.eval_called = False
    
    def eval(self):
        self.eval_called = True
    
    def __call__(self, x):
        return torch.tensor(self.predictions)

class MockDetectionModel:
    def __init__(self, predictions):
        self.predictions = predictions
        self.eval_called = False
    
    def eval(self):
        self.eval_called = True
    
    def __call__(self, x):
        return self.predictions

class MockSegmentationModel:
    def __init__(self, predictions):
        self.predictions = predictions
        self.eval_called = False
    
    def eval(self):
        self.eval_called = True
    
    def __call__(self, x):
        return self.predictions

class MockDataLoader:
    def __init__(self, data):
        self.data = data
    
    def __iter__(self):
        return iter(self.data)

def test_detection_metrics():
    # Criar mock data com batch_size = 1
    boxes = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    target = {
        'boxes': boxes,
        'labels': torch.ones(1, dtype=torch.int64)
    }
    
    # Criar uma única imagem de batch_size=1
    image = torch.randn(1, 3, 416, 416)
    dataloader = MockDataLoader([(image, [target])])
    
    # Mock predictions
    prediction = [{
        'boxes': boxes,
        'scores': torch.tensor([0.9], dtype=torch.float32)
    }]
    model = MockDetectionModel(prediction)
    
    metrics = DetectionMetrics.calculate(model, dataloader, 'cpu')
    
    assert metrics['AP'] > 0
    assert model.eval_called

def test_segmentation_metrics():
    # Criar mock data
    image = torch.randn(3, 512, 512)
    mask = torch.ones(512, 512, dtype=torch.long)  # Usar long para labels
    dataloader = MockDataLoader([(image.unsqueeze(0), mask.unsqueeze(0))])
    
    # Criar predição que corresponde exatamente à máscara
    logits = torch.zeros(1, 2, 512, 512)  # 2 classes: background e foreground
    logits[:, 1, :, :] = 10.0  # Alta confiança na classe 1 (foreground)
    model = MockSegmentationModel(logits)
    
    metrics = SegmentationMetrics.calculate(model, dataloader, 'cpu')
    
    assert metrics['mean_iou'] > 0.9  # Permitir pequena tolerância
    assert metrics['mean_dice'] > 0.9
    assert metrics['mean_precision'] > 0.9
    assert metrics['mean_recall'] > 0.9
    assert model.eval_called

# Corrigir também o método calculate_single em SegmentationMetrics
@staticmethod
def calculate_single(pred_mask, true_mask):
    """Calcula métricas para uma única máscara de segmentação."""
    pred = pred_mask.cpu().numpy().astype(bool)
    true = true_mask.cpu().numpy().astype(bool)
    
    intersection = np.logical_and(true, pred).sum()
    union = np.logical_or(true, pred).sum()
    
    iou = intersection / union if union > 0 else 0
    dice = 2 * intersection / (pred.sum() + true.sum()) if (pred.sum() + true.sum()) > 0 else 0
    
    tp = intersection
    fp = pred.sum() - tp
    fn = true.sum() - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {
        'iou': float(iou),
        'dice': float(dice),
        'precision': float(precision),
        'recall': float(recall)
    }

def test_segmentation_single_metrics():
    # Testar cálculo de métricas para uma única máscara
    pred_mask = torch.ones(10, 10, dtype=torch.long)
    true_mask = torch.ones(10, 10, dtype=torch.long)
    metrics = SegmentationMetrics.calculate_single(pred_mask, true_mask)
    
    assert metrics['iou'] > 0.9
    assert metrics['dice'] > 0.9
    assert metrics['precision'] > 0.9
    assert metrics['recall'] > 0.9