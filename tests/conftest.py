"""tests/conftest.py"""
import pytest
import numpy as np

@pytest.fixture
def mock_coco_data():
    class MockCOCO:
        def loadImgs(self, ids):
            return [{"file_name": "test.jpg", "height": 416, "width": 416}]
            
        def getAnnIds(self, imgIds, catIds):
            return [1]
            
        def loadAnns(self, ids):
            return [{
                "bbox": [150, 100, 100, 250],
                "segmentation": [[150, 100, 250, 100, 250, 350, 150, 350]],
                "category_id": 1
            }]
            
        def annToMask(self, ann):
            mask = np.zeros((416, 416), dtype=np.uint8)
            mask[100:350, 150:250] = 1  # Ajuste para coincidir com a bbox
            return mask
            
    return {
        "coco": MockCOCO(),
        "image_ids": [1],
        "image_dir": "tests"
    }