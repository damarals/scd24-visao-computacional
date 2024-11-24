import torch
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryConfusionMatrix, BinaryROC, BinaryAUROC, BinarySpecificity

class ClassificationMetrics:
    @staticmethod
    def calculate(model, data_loader, device):
        """Calcula métricas para classificação binária usando TorchMetrics."""
        model.eval()
        
        # Inicializar métricas
        accuracy = BinaryAccuracy().to(device)
        precision = BinaryPrecision().to(device)
        recall = BinaryRecall().to(device)
        f1_score = BinaryF1Score().to(device)
        specificity = BinarySpecificity().to(device)
        confusion_matrix = BinaryConfusionMatrix().to(device)
        roc = BinaryROC().to(device)
        auroc = BinaryAUROC().to(device)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                # Converter labels para long (inteiros)
                labels = labels.long().to(device)
                
                outputs = model(images)
                outputs = outputs.squeeze(1)
                preds = (outputs > 0.5).float()
                
                # Atualizar métricas
                accuracy.update(preds, labels)
                precision.update(preds, labels)
                recall.update(preds, labels)
                f1_score.update(preds, labels)
                specificity.update(preds, labels)
                confusion_matrix.update(preds, labels)
                # Para ROC e AUROC, usar as probabilidades brutas
                roc.update(outputs.float(), labels)
                auroc.update(outputs.float(), labels)
                
                all_preds.extend(preds.cpu())
                all_labels.extend(labels.cpu())
        
        # Computar métricas finais
        metrics = {
            'accuracy': accuracy.compute().item(),
            'precision': precision.compute().item(),
            'recall': recall.compute().item(),
            'f1': f1_score.compute().item(),
            'specificity': specificity.compute().item(),
            'confusion_matrix': confusion_matrix.compute().cpu().numpy(),
            'auroc': auroc.compute().item()
        }
        
        # Computar pontos da curva ROC
        fpr, tpr, thresholds = roc.compute()
        metrics['roc'] = {
            'fpr': fpr.cpu().numpy(),
            'tpr': tpr.cpu().numpy(),
            'thresholds': thresholds.cpu().numpy()
        }
        
        return metrics

class DetectionMetrics:
    @staticmethod
    def calculate(model, data_loader, device, iou_threshold=0.5):
        """
        TODO: Implementar métricas de detecção usando TorchMetrics.
        Pode usar: torchmetrics.detection.mean_ap.MeanAveragePrecision
        """
        pass

class SegmentationMetrics:
    @staticmethod
    def calculate(model, data_loader, device):
        """
        TODO: Implementar métricas de segmentação usando TorchMetrics.
        Pode usar: torchmetrics.JaccardIndex, torchmetrics.Dice, etc.
        """
        pass