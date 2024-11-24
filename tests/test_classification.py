import torch

from core.dataset import PersonClassificationDataset
from core.models.classification import PersonClassifier

def test_person_classifier_output_shape():
    model = PersonClassifier()
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    output = model(x)
    assert output.shape == (batch_size, 1)
    assert torch.all((output >= 0) & (output <= 1))

def test_classifier_dataset_getitem(mock_coco_data):
    dataset = PersonClassificationDataset(mock_coco_data)
    img, label = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert isinstance(label, torch.Tensor)
    assert label.shape == ()
    assert label.item() in [0.0, 1.0]