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
# We need to convert string class names (like 'apple') to numbers (like 0) because
# neural networks only understand numbers.
# enumerate(CLASS_NAMES) gives us pairs like (0, 'bacon'), (1, 'banana'), etc.
# The dictionary comprehension {name: idx for ...} creates a lookup table.
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# We also need the reverse mapping to convert the model's numeric predictions
# back into human-readable names.
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}


def get_image_transform(image_size: int = 224) -> transforms.Compose:
    """
    Get the image preprocessing transform.
    
    Args:
        image_size: Target size for images
        
    Returns:
        Composed transform for image preprocessing
    """
    # This function defines the standard preprocessing pipeline for images.
    # It prepares raw images to be fed into our neural network.
    return transforms.Compose([
        # 1. Resize the image to a fixed size (e.g., 224x224 pixels).
        # Neural networks require all input images to have the same dimensions.
        transforms.Resize((image_size, image_size)),
        
        # 2. Convert the PIL Image or NumPy array to a PyTorch Tensor.
        # This also scales pixel values from [0, 255] to [0.0, 1.0]
        # and changes the channel order to (Channels, Height, Width).
        transforms.ToTensor(),
        
        # 3. Normalize the color channels.
        # We subtract the mean and divide by the standard deviation for each channel (R, G, B).
        # These specific values (mean=[0.485, ...], std=[0.229, ...]) are the standard 
        # statistics from the ImageNet dataset. Since we are using a model pretrained 
        # on ImageNet, we must normalize our data in the exact same way.
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_training_transform(image_size: int = 224) -> transforms.Compose:
    """
    Get the training image preprocessing transform with moderate augmentation.
    
    Args:
        image_size: Target size for images
        
    Returns:
        Composed transform for image preprocessing
    """
    # Moderate augmentation - strong enough to help, not so strong it hurts
    return transforms.Compose([
        # 1. Random Resized Crop with CONSERVATIVE scale:
        # Only crop up to 20% of the image to avoid cutting out the food item
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        
        # 2. Random Horizontal Flip:
        transforms.RandomHorizontalFlip(),
        
        # 3. Slight rotation (food items can be at slight angles):
        transforms.RandomRotation(10),
        
        # 4. Color Jitter (moderate - colors are important for food!):
        # Lower values than before, especially no hue shift
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        
        # NOTE: Removed RandomGrayscale - colors are critical for food recognition!
        # NOTE: Removed GaussianBlur - texture is important for grocery items!
        
        # 5. Convert to Tensor
        transforms.ToTensor(),
        
        # 6. Normalize with ImageNet statistics
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
        # We use different transforms for training and testing.
        # For training, we add augmentation (random crops, flips, color jitter)
        # to make the model more robust and prevent overfitting.
        # For testing, we just resize and normalize to keep it standard.
        if transform:
            self.transform = transform
        else:
            self.transform = get_training_transform() if split == 'train' else get_image_transform()
        self.max_samples_per_class = max_samples_per_class
        
        self.samples: List[Tuple[str, int]] = []
        # _load_samples is a "private" helper method (indicated by the underscore prefix).
        # It populates the self.samples list with all image paths and labels.
        self._load_samples()
    
    def _load_samples(self):
        """Load all image paths and their labels."""
        # Loop through each class name (e.g., 'apple', 'banana')
        for class_name in CLASS_NAMES:
            # Construct the path to the class directory
            # e.g., DS2GROCERIES/train/apple
            class_dir = self.data_dir / class_name
            
            # Check if the directory exists to avoid errors
            if not class_dir.exists():
                continue
            
            # Find all images in the directory
            # glob("*.jpg") finds all files ending with .jpg
            # We combine lists of .jpg and .png files using the + operator
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            
            # If max_samples_per_class is set (not None), we slice the list
            # [:n] takes the first n elements. This is useful for quick testing.
            if self.max_samples_per_class:
                images = images[:self.max_samples_per_class]
            
            # Add each image to our samples list
            for img_path in images:
                # We store a tuple: (path_as_string, numeric_label)
                # str(img_path) converts the Path object to a string
                # CLASS_TO_IDX[class_name] looks up the number for this class
                self.samples.append((str(img_path), CLASS_TO_IDX[class_name]))
    
    def __len__(self) -> int:
        # This magic method allows us to use len(dataset)
        # It must return the total number of samples.
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # This magic method allows us to access items using dataset[idx]
        # It retrieves the image and label at the given index.
        # Unpack the tuple stored in self.samples
        img_path, label = self.samples[idx]
        
        # Load the image from disk using PIL (Python Imaging Library)
        # .convert('RGB') ensures the image has 3 color channels (Red, Green, Blue)
        # even if it was grayscale or had an alpha channel.
        image = Image.open(img_path).convert('RGB')
        #this ensures all the data is rgb images
        # Apply the transformations (resize, normalize, etc.)
        # This converts the PIL image into a PyTorch Tensor ready for the model.
        if self.transform:
            image = self.transform(image)
        
        # Return the processed image and its label
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
    # 1. Set the random seed for reproducibility.
    # This ensures that we get the exact same split every time we run the code.
    np.random.seed(random_state)
    n_samples = len(labels)
    n_labeled = int(n_samples * labeled_ratio)
    
    # Ensure at least one sample per class is labeled
    labeled_mask = np.zeros(n_samples, dtype=bool)
    
    # First, select at least one sample per class
    # This is important! If a class has NO labeled samples, the algorithm
    # might never learn to recognize it. We want to give it a fighting chance.
    for class_idx in range(len(CLASS_NAMES)):
        class_indices = np.where(labels == class_idx)[0]
        if len(class_indices) > 0:
            # Randomly select one index from this class
            selected = np.random.choice(class_indices, size=1)
            labeled_mask[selected] = True
    
    # Then fill up to the desired ratio
    # We find all indices that are not yet labeled
    unlabeled_indices = np.where(~labeled_mask)[0]
    
    # Calculate how many more samples we need to reach the target ratio
    n_additional = max(0, n_labeled - labeled_mask.sum())
    
    if n_additional > 0 and len(unlabeled_indices) > 0:
        # Randomly select additional samples from the unlabeled pool
        additional = np.random.choice(
            unlabeled_indices,
            size=min(n_additional, len(unlabeled_indices)),
            replace=False
        )
        labeled_mask[additional] = True
    
    # Create semi-supervised labels
    # We start with a copy of the true labels
    ssl_labels = labels.copy()
    
    # Then we "hide" the labels for all samples that are NOT in our labeled mask
    # We use -1 to represent an "unlabeled" or "unknown" class.
    # The semi-supervised algorithms will try to predict the true labels for these -1 entries.
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
    #this operation wraps the single image into a batch of size 1
    #for example : if we have an image of shape (3, 224, 224) it will be converted to (1, 3, 224, 224) 
    #[32,3,224,224] for a batch of 32 RGB images of size 224x224

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
