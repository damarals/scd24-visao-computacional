import torch

from core.dataset import PersonSegmentationDataset
from core.models.segmentation import PersonSegmenter

def test_person_segmenter_output():
    model = PersonSegmenter()
    model.eval() 
    batch_size = 2
    x = torch.randn(batch_size, 3, 512, 512)
    with torch.no_grad():
        output = model(x)
        output = torch.nn.functional.softmax(output, dim=1)
    
    assert output.shape == (batch_size, 2, 512, 512)
    assert torch.all((output >= 0) & (output <= 1))

def test_segmentation_dataset_getitem(mock_coco_data):
    dataset = PersonSegmentationDataset(mock_coco_data)
    img, mask = dataset[0]
    
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 512, 512)
    assert isinstance(mask, torch.Tensor) 
    assert mask.shape == (512, 512)
    assert torch.all((mask == 0) | (mask == 1))