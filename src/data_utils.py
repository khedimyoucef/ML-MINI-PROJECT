"""
Data Loading and Preprocessing Utilities

This module provides utilities for loading the grocery dataset,
creating labeled/unlabeled splits for semi-supervised learning,
and preprocessing images.
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Class names for the 20 grocery categories
CLASS_NAMES = [
    'bacon', 'banana', 'bread', 'broccoli', 'butter',
    'carrots', 'cheese', 'chicken', 'cucumber', 'eggs',
    'fish', 'lettuce', 'milk', 'onions', 'peppers',
    'potatoes', 'sausages', 'spinach', 'tomato', 'yogurt'
]

# Create mappings
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}


def get_image_transform(image_size: int = 224) -> transforms.Compose:
    """
    Get the image preprocessing transform.
    
    Args:
        image_size: Target size for images
        
    Returns:
        Composed transform for image preprocessing
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_training_transform(image_size: int = 224) -> transforms.Compose:
    """
    Get the training image preprocessing transform with augmentation.
    
    Args:
        image_size: Target size for images
        
    Returns:
        Composed transform for image preprocessing
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


class GroceryDataset(Dataset):
    """
    PyTorch Dataset for the grocery classification dataset.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        max_samples_per_class: Optional[int] = None
    ):
        """
        Initialize the grocery dataset.
        
        Args:
            data_dir: Path to the dataset root (e.g., DS2GROCERIES)
            split: One of 'train', 'test', or 'val'
            transform: Optional transform to apply to images
            max_samples_per_class: Limit samples per class (for faster testing)
        """
        self.data_dir = Path(data_dir) / split
        self.split = split
        if transform:
            self.transform = transform
        else:
            self.transform = get_training_transform() if split == 'train' else get_image_transform()
        self.max_samples_per_class = max_samples_per_class
        
        self.samples: List[Tuple[str, int]] = []
        self._load_samples()
    
    def _load_samples(self):
        """Load all image paths and their labels."""
        for class_name in CLASS_NAMES:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
            
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            
            if self.max_samples_per_class:
                images = images[:self.max_samples_per_class]
            
            for img_path in images:
                self.samples.append((str(img_path), CLASS_TO_IDX[class_name]))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load and convert image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_paths(self) -> List[str]:
        """Get all image paths."""
        return [path for path, _ in self.samples]
    
    def get_labels(self) -> np.ndarray:
        """Get all labels as numpy array."""
        return np.array([label for _, label in self.samples])


def create_semi_supervised_split(
    labels: np.ndarray,
    labeled_ratio: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a semi-supervised split by masking some labels.
    
    For semi-supervised learning, we pretend some labels are unknown
    by replacing them with -1.
    
    Args:
        labels: Original labels array
        labeled_ratio: Fraction of labels to keep (0.0 to 1.0)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (semi_supervised_labels, labeled_indices)
        - semi_supervised_labels: Labels with some replaced by -1
        - labeled_indices: Boolean mask of which samples are labeled
    """
    np.random.seed(random_state)
    n_samples = len(labels)
    n_labeled = int(n_samples * labeled_ratio)
    
    # Ensure at least one sample per class is labeled
    labeled_mask = np.zeros(n_samples, dtype=bool)
    
    # First, select at least one sample per class
    for class_idx in range(len(CLASS_NAMES)):
        class_indices = np.where(labels == class_idx)[0]
        if len(class_indices) > 0:
            selected = np.random.choice(class_indices, size=1)
            labeled_mask[selected] = True
    
    # Then fill up to the desired ratio
    unlabeled_indices = np.where(~labeled_mask)[0]
    n_additional = max(0, n_labeled - labeled_mask.sum())
    
    if n_additional > 0 and len(unlabeled_indices) > 0:
        additional = np.random.choice(
            unlabeled_indices,
            size=min(n_additional, len(unlabeled_indices)),
            replace=False
        )
        labeled_mask[additional] = True
    
    # Create semi-supervised labels (-1 for unlabeled)
    ssl_labels = labels.copy()
    ssl_labels[~labeled_mask] = -1
    
    return ssl_labels, labeled_mask


def load_image_for_prediction(
    image_path: str,
    transform: Optional[transforms.Compose] = None
) -> torch.Tensor:
    """
    Load a single image for prediction.
    
    Args:
        image_path: Path to the image file
        transform: Optional transform (uses default if None)
        
    Returns:
        Preprocessed image tensor with batch dimension
    """
    transform = transform or get_image_transform()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)  # Add batch dimension


def get_dataset_stats(data_dir: str) -> Dict:
    """
    Get statistics about the dataset.
    
    Args:
        data_dir: Path to dataset root
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'splits': {},
        'total_images': 0
    }
    
    data_path = Path(data_dir)
    
    for split in ['train', 'test', 'val']:
        split_path = data_path / split
        if not split_path.exists():
            continue
        
        split_stats = {'classes': {}, 'total': 0}
        
        for class_name in CLASS_NAMES:
            class_dir = split_path / class_name
            if class_dir.exists():
                n_images = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))
                split_stats['classes'][class_name] = n_images
                split_stats['total'] += n_images
        
        stats['splits'][split] = split_stats
        stats['total_images'] += split_stats['total']
    
    return stats


if __name__ == "__main__":
    # Test the data utilities
    import sys
    
    # Get dataset path from command line or use default
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "DS2GROCERIES"
    
    print("Testing Data Utilities")
    print("=" * 50)
    
    # Get dataset stats
    stats = get_dataset_stats(dataset_path)
    print(f"\nDataset Statistics:")
    print(f"Total images: {stats['total_images']}")
    
    for split, split_stats in stats['splits'].items():
        print(f"\n{split.upper()} split: {split_stats['total']} images")
        for class_name, count in list(split_stats['classes'].items())[:5]:
            print(f"  - {class_name}: {count}")
        print("  ...")
    
    # Test dataset loading (with limit for speed)
    print("\n\nTesting Dataset Loading (limited samples)...")
    dataset = GroceryDataset(dataset_path, split='train', max_samples_per_class=5)
    print(f"Loaded {len(dataset)} samples")
    
    # Test semi-supervised split
    labels = dataset.get_labels()
    ssl_labels, labeled_mask = create_semi_supervised_split(labels, labeled_ratio=0.1)
    print(f"\nSemi-supervised split (10% labeled):")
    print(f"  - Labeled samples: {labeled_mask.sum()}")
    print(f"  - Unlabeled samples: {(~labeled_mask).sum()}")
