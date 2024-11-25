import os
import numpy as np
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.coco as fouc
from fiftyone import ViewField as F

def setup_coco_person(output_dir="data/coco", samples_per_class=1000):
    """
    Download and setup person detection dataset using FiftyOne.
    
    Args:
        output_dir (str): Directory to store the dataset
        samples_per_class (int): Number of samples per class
    """
    print("\nInitializing COCO dataset setup...")
    
    # Create directories
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

    print("\nDownloading and loading COCO dataset...")
    
    # Load COCO-2017 dataset
    dataset = foz.load_zoo_dataset(
        "sama-coco",
        splits=["train", "validation"],
        label_types=["segmentations"],
        max_samples=samples_per_class * 4  # Load extra samples to ensure we have enough after filtering
    )

    print("\nProcessing dataset...")
    
    # Pre-process dataset to have only the labels we want
    processed_dataset = fo.Dataset()

    for sample in dataset:
        new_sample = sample.copy()
        
        # Check if image has person in segmentations
        has_person = False
        if (hasattr(sample, 'ground_truth') and sample.ground_truth is not None):
            has_person = any(det.label == "person" for det in sample.ground_truth.detections)
        
        # Process segmentations field
        if has_person:
            # Keep only person segmentations
            person_segmentations = [
                det for det in sample.ground_truth.detections 
                if det.label == "person"
            ]
            new_sample['segmentations'] = fo.Detections(detections=person_segmentations)
        else:
            # Create a single "no-person" detection for the image
            h, w = sample.metadata.height, sample.metadata.width
            empty_mask = np.zeros((h, w), dtype=bool)
            
            no_person_det = fo.Detection(
                label="no-person",
                bounding_box=[0, 0, 1, 1],
                mask=empty_mask,
                iscrowd=0
            )
            new_sample['segmentations'] = fo.Detections(detections=[no_person_det])
        
        processed_dataset.add_sample(new_sample)

    print(f"\nTotal processed samples: {len(processed_dataset)}")

    # Create views for images with and without people
    people_view = processed_dataset.match_labels(
        filter=F("label") == "person",
        fields="segmentations"
    )

    no_people_view = processed_dataset.match_labels(
        filter=F("label") == "no-person",
        fields="segmentations"
    )

    print(f"Samples with person: {len(people_view)}")
    print(f"Samples without person: {len(no_people_view)}")

    def create_balanced_split(people_view, no_people_view, n_samples):
        n_each = n_samples // 2
        people_samples = people_view.take(n_each, seed=51)
        no_people_samples = no_people_view.take(n_each, seed=51)
        split_dataset = fo.Dataset()
        split_dataset.add_samples(people_samples)
        split_dataset.add_samples(no_people_samples)
        return split_dataset

    # Create balanced splits
    print("\nCreating balanced splits...")
    train_dataset = create_balanced_split(
        people_view.match_tags("train"),
        no_people_view.match_tags("train"),
        samples_per_class * 2  # Multiply by 2 to account for both classes
    )

    val_dataset = create_balanced_split(
        people_view.match_tags("validation"),
        no_people_view.match_tags("validation"),
        samples_per_class  # Use half as many samples for validation
    )

    # Configure categories
    categories = [
        {"id": 1, "name": "person", "supercategory": "person"},
        {"id": 2, "name": "no-person", "supercategory": "object"}
    ]

    # Create COCO exporters
    print("\nExporting datasets...")
    train_exporter = fouc.COCODetectionDatasetExporter(
        export_dir=output_dir,
        data_path="train",
        labels_path="annotations/instances_train.json",
        categories=categories,
        export_media="move"
    )

    val_exporter = fouc.COCODetectionDatasetExporter(
        export_dir=output_dir, 
        data_path="val",
        labels_path="annotations/instances_val.json",
        categories=categories,
        export_media="move"
    )

    # Export datasets
    train_dataset.export(dataset_exporter=train_exporter, label_field="segmentations")
    val_dataset.export(dataset_exporter=val_exporter, label_field="segmentations")

    print("\nDataset exported successfully!")
    print(f"Total training images: {len(train_dataset)}")
    print(f"Total validation images: {len(val_dataset)}")
    
    print("\nDataset statistics:")
    train_labels = train_dataset.count_values("segmentations.detections.label")
    val_labels = val_dataset.count_values("segmentations.detections.label")
    print("Training label distribution:", train_labels)
    print("Validation label distribution:", val_labels)