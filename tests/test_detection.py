import torch

from core.dataset import PersonDetectionDataset
from core.models.detection import PersonDetector

def test_person_detector_output_format():
    model = PersonDetector()
    model.eval()
    batch_size = 2
    x = torch.randn(batch_size, 3, 416, 416)
    outputs = model(x)
    
    assert isinstance(outputs, list)
    assert len(outputs) == batch_size
    
    for output in outputs:
        assert isinstance(output, dict)
        assert "boxes" in output
        assert "scores" in output
        assert "labels" in output
        assert output["boxes"].shape[1] == 4

def test_detection_dataset_getitem(mock_coco_data):
    dataset = PersonDetectionDataset(mock_coco_data)
    img, target = dataset[0]
    
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 416, 416)
    assert isinstance(target, dict)
    assert "boxes" in target
    assert "labels" in target
    assert target["boxes"].shape[1] == 4