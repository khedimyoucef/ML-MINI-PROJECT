"""
Feature Extraction Module

This module uses a pretrained ResNet50` model to extract feature vectors
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
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

from .data_utils import GroceryDataset, get_image_transform, load_image_for_prediction


class FeatureExtractor:
    """
    Feature extractor using pretrained ResNet50.
    
    Extracts 2048-dimensional feature vectors from images using
    a pretrained ResNet50 model. Can be fine-tuned on labeled data.
    """
    #the resnet50 model is designed to extract 2048 features from an image regardless of its size 
    #for example if we crop the images to 224x224x3 (for rgb images) we get 150528 total features and we only extract 2048 
    #learnt features 
    #it applies many convolutional layers and pooling operations that transform and compress the information into 2048 learned features.
    def __init__(self, device: Optional[str] = None, model_path: Optional[str] = None):
        """
        Initialize the feature extractor.
        
        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            model_path: Optional path to load fine-tuned weights from
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load pretrained ResNet50
        # We use a ResNet50 model that has been pretrained on the ImageNet dataset.
        # ImageNet contains 1.2 million images across 1000 categories.
        # By using this, we leverage the "knowledge" the model has already learned
        # about detecting edges, textures, and shapes.
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Save feature dim before removing last layer
        # The last fully connected layer of ResNet50 outputs 1000 class probabilities.
        # The input to that layer is a vector of size 2048.
        # We want that 2048-dimensional vector as our "feature".
        self.feature_dim = self.model.fc.in_features  # 2048 for ResNet50
        
        # Remove the final classification layer
        # We strip off the last layer (fc) so that the model outputs the raw features
        # instead of class predictions.
        # list(self.model.children())[:-1] gets all layers except the last one.
        # nn.Sequential wraps them back into a single module.
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        # Load fine-tuned weights if provided
        if model_path and os.path.exists(model_path):
            print(f"Loading feature extractor weights from {model_path}")
            self.load(model_path)
        
        # Set to evaluation mode by default
        # This is crucial! It tells PyTorch to behave in "inference" mode.
        # Layers like Dropout and BatchNorm behave differently during training vs evaluation.
        self.model.eval()
        self.model.to(self.device)
        
        # Image transform
        self.transform = get_image_transform()
    
    def save(self, path: str):
        """Save model weights to path."""
        torch.save(self.model.state_dict(), path)
        print(f"Feature extractor saved to {path}")
        
    def load(self, path: str):
        """Load model weights from path."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
    
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
        
        # Pass the image through the model to get features
        features = self.model(image_tensor)
        
        # The output of ResNet50 (without the last layer) is (Batch, 2048, 1, 1).
        # We use .squeeze() to remove the dimensions of size 1.
        # Resulting shape: (2048,) for a single image.
        features = features.squeeze()  # Remove spatial dimensions
        
        # Move back to CPU and convert to NumPy for compatibility with scikit-learn
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
        batch_size: int = 16,
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
        # Create a DataLoader to handle batching
        # DataLoader is a PyTorch utility that helps us iterate over the dataset in batches.
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # No need to shuffle for feature extraction
            num_workers=0  # Use 0 workers (main process) to avoid multiprocessing complexity on Windows
        )
        
        all_features = []
        all_labels = []
        
        # tqdm creates a progress bar so we can see how fast we are processing
        iterator = tqdm(dataloader, desc="Extracting features") if show_progress else dataloader
        
        # Iterate through the dataset in batches
        for images, labels in iterator:
            # Move images to the GPU (if available)
            images = images.to(self.device)
            
            # Forward pass: compute features
            features = self.model(images)
            
            # Remove the extra dimensions (1x1 spatial dims)
            # .squeeze(-1) removes the last dimension if it has size 1
            features = features.squeeze(-1).squeeze(-1)
            
            # Move features back to CPU and convert to NumPy array
            # We need to do this because we can't store GPU tensors in a standard list
            all_features.append(features.cpu().numpy())
            
            # Also store the labels
            all_labels.append(labels.numpy())
        
        # Combine all batches into single large arrays
        # np.vstack stacks arrays vertically (row by row)
        # np.concatenate joins arrays end-to-end
        return np.vstack(all_features), np.concatenate(all_labels)
    
    def extract_and_cache(
        self,
        dataset: GroceryDataset,
        cache_path: str,
        batch_size: int = 16,
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
        
        # Check if we already have cached features on disk
        # This saves a lot of time by avoiding re-running the heavy model
        if cache_file.exists() and not force_recompute:
            print(f"Loading cached features from {cache_path}")
            # Load the .npz file (NumPy Zip)
            data = np.load(cache_path)
            return data['features'], data['labels']
        
        print(f"Extracting features (this may take a while)...")
        # Call the batch extraction method we defined above
        features, labels = self.extract_batch(dataset, batch_size)
        
        # Save the results to disk for next time
        # .mkdir(parents=True, exist_ok=True) creates the directory if it doesn't exist
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        # np.savez compresses the arrays into a single file
        np.savez(cache_path, features=features, labels=labels)
        print(f"Cached features to {cache_path}")
        
        return features, labels

    
    def fine_tune(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_classes: int = 20,
        epochs: int = 5,
        learning_rate: float = 1e-4
    ):
        """
        Fine-tune the model on labeled data.
        
        Args:
            train_loader: DataLoader for labeled training data
            val_loader: Optional DataLoader for validation
            num_classes: Number of classes for the temporary classification head
            epochs: Number of epochs to train
            learning_rate: Learning rate for optimizer
        """
        print(f"\nFine-tuning feature extractor for {epochs} epochs...")
        
        # Restore classification head for training
        # We need to recreate the full model structure temporarily
        # The current self.model is just the feature extractor (Sequential)
        
        # Create a temporary classification head
        # This linear layer maps the 2048 features to our 20 grocery classes.
        classifier = nn.Linear(self.feature_dim, num_classes).to(self.device)
        
        # Optimizer
        # We use Adam, a standard optimizer.
        # We optimize parameters of BOTH the backbone (self.model) and the new classifier.
        # This allows the backbone to adapt its features to our specific grocery images.
        
        optimizer = optim.Adam([
            {'params': self.model.parameters()},
            {'params': classifier.parameters()}
        ], lr=learning_rate)
        
        # CrossEntropyLoss is the standard loss function for multi-class classification.
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            classifier.train()
            
            running_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                # 1. Extract features using the backbone (ResNet50)
                features = self.model(images)
                # 2. Flatten features to (Batch, 2048)
                features = features.flatten(1)
                # 3. Pass features through the classifier to get class scores
                outputs = classifier(features)
                
                # Calculate loss (how wrong was the model?)
                loss = criterion(outputs, labels)
                
                # Backward pass (Backpropagation)
                # 1. Calculate gradients (how much each parameter contributed to the error)
                loss.backward()
                # 2. Update parameters using the optimizer
                optimizer.step()
                
                # Update statistics for progress bar
                running_loss += loss.item() * images.size(0)
                
                # Get predictions (index of the highest score)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update the progress bar description
                pbar.set_postfix({
                    'loss': running_loss / total,
                    'acc': correct / total
                })
            
            # Calculate average loss and accuracy for the epoch
            train_loss = running_loss / total
            train_acc = correct / total
            
            # Validation
            val_info = ""
            if val_loader:
                self.model.eval()
                classifier.eval()
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        features = self.model(images).flatten(1)
                        outputs = classifier(features)
                        _, predicted = outputs.max(1)
                        val_total += labels.size(0)
                        val_correct += predicted.eq(labels).sum().item()
                
                val_acc = val_correct / val_total
                val_info = f" | Val Acc: {val_acc:.4f}"
            
            print(f"Epoch {epoch+1}: Loss: {train_loss:.4f} | Acc: {train_acc:.4f}{val_info}")
        
        # Cleanup
        del classifier
        del optimizer
        self.model.eval()  # Return to eval mode for feature extraction
        print("Fine-tuning complete.")


def extract_dataset_features(
    data_dir: str,
    output_dir: str = "features",
    splits: List[str] = ['train', 'test', 'val'],
    batch_size: int = 16,
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
