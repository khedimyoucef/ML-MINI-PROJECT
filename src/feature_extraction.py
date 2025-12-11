"""
Feature Extraction Module

This module uses a pretrained ResNet18 model to extract feature vectors
from grocery images. These features are then used by semi-supervised
learning algorithms for classification.
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from .data_utils import GroceryDataset, get_image_transform, load_image_for_prediction


class FeatureExtractor:
    """
    Feature extractor using pretrained ResNet18.
    
    Extracts 512-dimensional feature vectors from images using
    a pretrained ResNet18 model with the classification layer removed.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the feature extractor.
        
        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load pretrained ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Remove the final classification layer
        # This gives us the 512-dimensional feature vector
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        # Set to evaluation mode
        self.model.eval()
        self.model.to(self.device)
        
        # Feature dimension
        self.feature_dim = 512
        
        # Image transform
        self.transform = get_image_transform()
    
    @torch.no_grad()
    def extract_single(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Extract features from a single image tensor.
        
        Args:
            image_tensor: Preprocessed image tensor (with batch dim)
            
        Returns:
            Feature vector as numpy array (512,)
        """
        image_tensor = image_tensor.to(self.device)
        features = self.model(image_tensor)
        features = features.squeeze()  # Remove spatial dimensions
        return features.cpu().numpy()
    
    @torch.no_grad()
    def extract_from_path(self, image_path: str) -> np.ndarray:
        """
        Extract features from an image file.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Feature vector as numpy array (512,)
        """
        image_tensor = load_image_for_prediction(image_path, self.transform)
        return self.extract_single(image_tensor)
    
    @torch.no_grad()
    def extract_batch(
        self,
        dataset: GroceryDataset,
        batch_size: int = 32,
        show_progress: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from an entire dataset.
        
        Args:
            dataset: GroceryDataset instance
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Tuple of (features, labels) numpy arrays
        """
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0  # Avoid multiprocessing issues
        )
        
        all_features = []
        all_labels = []
        
        iterator = tqdm(dataloader, desc="Extracting features") if show_progress else dataloader
        
        for images, labels in iterator:
            images = images.to(self.device)
            features = self.model(images)
            features = features.squeeze(-1).squeeze(-1)  # Remove spatial dims
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
        
        return np.vstack(all_features), np.concatenate(all_labels)
    
    def extract_and_cache(
        self,
        dataset: GroceryDataset,
        cache_path: str,
        batch_size: int = 32,
        force_recompute: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features with caching support.
        
        Args:
            dataset: GroceryDataset instance
            cache_path: Path to save/load cached features
            batch_size: Batch size for processing
            force_recompute: Force recomputation even if cache exists
            
        Returns:
            Tuple of (features, labels) numpy arrays
        """
        cache_file = Path(cache_path)
        
        if cache_file.exists() and not force_recompute:
            print(f"Loading cached features from {cache_path}")
            data = np.load(cache_path)
            return data['features'], data['labels']
        
        print(f"Extracting features (this may take a while)...")
        features, labels = self.extract_batch(dataset, batch_size)
        
        # Save to cache
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, features=features, labels=labels)
        print(f"Cached features to {cache_path}")
        
        return features, labels


def extract_dataset_features(
    data_dir: str,
    output_dir: str = "features",
    splits: List[str] = ['train', 'test', 'val'],
    batch_size: int = 32,
    max_samples_per_class: Optional[int] = None,
    device: Optional[str] = None
) -> dict:
    """
    Extract features for entire dataset and cache results.
    
    Args:
        data_dir: Path to dataset root
        output_dir: Directory to save cached features
        splits: List of splits to process
        batch_size: Batch size for feature extraction
        max_samples_per_class: Limit samples per class (for testing)
        device: Device to use
        
    Returns:
        Dictionary with features and labels for each split
    """
    extractor = FeatureExtractor(device=device)
    results = {}
    
    for split in splits:
        print(f"\nProcessing {split} split...")
        
        dataset = GroceryDataset(
            data_dir,
            split=split,
            max_samples_per_class=max_samples_per_class
        )
        
        if len(dataset) == 0:
            print(f"  No samples found for {split} split, skipping.")
            continue
        
        cache_path = os.path.join(output_dir, f"{split}_features.npz")
        features, labels = extractor.extract_and_cache(
            dataset,
            cache_path,
            batch_size=batch_size
        )
        
        results[split] = {
            'features': features,
            'labels': labels
        }
        
        print(f"  Extracted {len(features)} feature vectors of dim {features.shape[1]}")
    
    return results


if __name__ == "__main__":
    import sys
    
    # Get dataset path from command line or use default
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "DS2GROCERIES"
    
    print("Testing Feature Extraction")
    print("=" * 50)
    
    # Test with limited samples
    print("\nExtracting features (limited samples for testing)...")
    results = extract_dataset_features(
        dataset_path,
        output_dir="features",
        splits=['train'],
        batch_size=16,
        max_samples_per_class=10  # Limit for faster testing
    )
    
    if 'train' in results:
        features = results['train']['features']
        labels = results['train']['labels']
        print(f"\nFeature shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")
