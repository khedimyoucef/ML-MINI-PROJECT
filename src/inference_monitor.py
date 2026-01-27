"""
Inference Monitoring Module with Weights & Biases

This module simulates production model monitoring by tracking:
- Prediction distributions over time
- Model confidence scores
- Performance metrics on batches
- Potential data drift detection
"""

import os
import sys
import argparse
import time
import random
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pickle
import wandb
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_utils import (
    GroceryDataset,
    CLASS_NAMES,
    get_image_transform,
    load_image_for_prediction
)
from src.feature_extraction import FeatureExtractor
from src.semi_supervised import SemiSupervisedClassifier
from src.wandb_config import (
    init_wandb,
    finish_wandb,
    log_metrics,
    log_inference_prediction,
    create_summary_table,
    WANDB_PROJECT
)


class InferenceMonitor:
    """
    Monitor for tracking model inference in production-like settings.
    
    Logs predictions, confidence scores, and performance metrics to W&B.
    """
    
    def __init__(
        self,
        model_path: str,
        extractor_path: str = None,
        run_name: str = None,
        device: str = None
    ):
        """
        Initialize the inference monitor.
        
        Args:
            model_path: Path to the trained classifier (.pkl)
            extractor_path: Path to the feature extractor (.pth)
            run_name: Optional W&B run name
            device: Device for inference
        """
        self.model_path = model_path
        self.device = device
        
        # Load feature extractor
        print("Loading feature extractor...")
        self.extractor = FeatureExtractor(
            device=device,
            model_path=extractor_path
        )
        
        # Load classifier
        print("Loading classifier...")
        with open(model_path, 'rb') as f:
            self.classifier = pickle.load(f)
        
        # Track statistics
        self.prediction_counts = {name: 0 for name in CLASS_NAMES}
        self.confidence_history = []
        self.correct_predictions = 0
        self.total_predictions = 0
        self.inference_times = []
        
        # Initialize W&B
        self.run = init_wandb(
            run_name=run_name or "inference-monitoring",
            tags=["inference", "monitoring", "production"],
            job_type="inference"
        )
        
        print(f"✓ Inference Monitor initialized")
        print(f"  Model: {model_path}")
        print(f"  W&B Run: {wandb.run.name if wandb.run else 'N/A'}")
    
    def predict(
        self,
        image_path: str,
        true_label: int = None,
        log_to_wandb: bool = True
    ) -> Tuple[int, float, Dict[str, float]]:
        """
        Make a prediction and log to W&B.
        
        Args:
            image_path: Path to the image
            true_label: Optional true label for accuracy tracking
            log_to_wandb: Whether to log to W&B
            
        Returns:
            Tuple of (predicted_class_idx, confidence, probabilities)
        """
        start_time = time.time()
        
        # Extract features
        features = self.extractor.extract_from_path(image_path)
        features = features.reshape(1, -1)
        
        # Predict
        predicted_idx = self.classifier.predict(features)[0]
        probabilities = self.classifier.predict_proba(features)[0]
        confidence = probabilities[predicted_idx]
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Update statistics
        predicted_class = CLASS_NAMES[predicted_idx]
        self.prediction_counts[predicted_class] += 1
        self.confidence_history.append(confidence)
        self.inference_times.append(inference_time)
        self.total_predictions += 1
        
        if true_label is not None:
            if predicted_idx == true_label:
                self.correct_predictions += 1
        
        # Log to W&B
        if log_to_wandb:
            true_class = CLASS_NAMES[true_label] if true_label is not None else None
            log_inference_prediction(
                image_path=image_path,
                predicted_class=predicted_class,
                true_class=true_class,
                confidence=confidence,
                probabilities={CLASS_NAMES[i]: float(p) for i, p in enumerate(probabilities)},
                inference_time_ms=inference_time
            )
        
        # Log aggregated metrics periodically
        if self.total_predictions % 10 == 0:
            self._log_aggregate_metrics()
        
        return predicted_idx, confidence, {CLASS_NAMES[i]: float(p) for i, p in enumerate(probabilities)}
    
    def _log_aggregate_metrics(self):
        """Log aggregated monitoring metrics."""
        if not self.confidence_history:
            return
        
        metrics = {
            "monitor/total_predictions": self.total_predictions,
            "monitor/avg_confidence": np.mean(self.confidence_history),
            "monitor/min_confidence": np.min(self.confidence_history),
            "monitor/max_confidence": np.max(self.confidence_history),
            "monitor/avg_inference_time_ms": np.mean(self.inference_times),
        }
        
        if self.total_predictions > 0:
            metrics["monitor/accuracy"] = self.correct_predictions / self.total_predictions
        
        log_metrics(metrics)
    
    def run_batch_monitoring(
        self,
        test_dir: str,
        n_samples: int = 100,
        random_sample: bool = True
    ):
        """
        Run monitoring on a batch of test images.
        
        Args:
            test_dir: Directory with test images
            n_samples: Number of samples to process
            random_sample: Whether to randomly sample
        """
        print(f"\n{'='*60}")
        print("Running Batch Monitoring")
        print(f"{'='*60}")
        
        # Load test dataset
        test_dir = Path(test_dir)
        
        all_samples = []
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_path = test_dir / class_name
            if class_path.exists():
                for img_path in class_path.glob("*.jpg"):
                    all_samples.append((str(img_path), class_idx))
                for img_path in class_path.glob("*.png"):
                    all_samples.append((str(img_path), class_idx))
        
        if random_sample:
            samples = random.sample(all_samples, min(n_samples, len(all_samples)))
        else:
            samples = all_samples[:n_samples]
        
        print(f"Processing {len(samples)} samples...")
        
        batch_start = time.time()
        
        for i, (img_path, true_label) in enumerate(samples):
            predicted_idx, confidence, probs = self.predict(
                img_path, 
                true_label=true_label
            )
            
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(samples)} images")
        
        batch_time = time.time() - batch_start
        
        # Log final summary
        print(f"\n{'='*60}")
        print("Batch Monitoring Summary")
        print(f"{'='*60}")
        print(f"Total predictions: {self.total_predictions}")
        print(f"Accuracy: {self.correct_predictions / self.total_predictions:.2%}")
        print(f"Average confidence: {np.mean(self.confidence_history):.2%}")
        print(f"Average inference time: {np.mean(self.inference_times):.1f}ms")
        print(f"Total batch time: {batch_time:.1f}s")
        
        # Log summary metrics
        log_metrics({
            "monitor/batch_size": len(samples),
            "monitor/batch_accuracy": self.correct_predictions / self.total_predictions,
            "monitor/batch_time_seconds": batch_time,
            "monitor/throughput_samples_per_sec": len(samples) / batch_time,
        })
        
        # Create prediction distribution table
        dist_data = [[name, count] for name, count in self.prediction_counts.items()]
        dist_table = wandb.Table(data=dist_data, columns=["class", "count"])
        wandb.log({
            "prediction_distribution": wandb.plot.bar(
                dist_table, "class", "count",
                title="Prediction Distribution"
            )
        })
        
        # Log confidence histogram
        wandb.log({
            "confidence_histogram": wandb.Histogram(self.confidence_history)
        })
    
    def detect_drift(self, baseline_distribution: Dict[str, float] = None):
        """
        Detect potential data drift by comparing prediction distributions.
        
        Args:
            baseline_distribution: Expected prediction distribution
        """
        if not baseline_distribution:
            # Use uniform distribution as baseline
            baseline_distribution = {name: 1/len(CLASS_NAMES) for name in CLASS_NAMES}
        
        # Calculate current distribution
        total = sum(self.prediction_counts.values())
        if total == 0:
            return
        
        current_distribution = {
            name: count / total 
            for name, count in self.prediction_counts.items()
        }
        
        # Calculate KL divergence (simplified drift metric)
        drift_score = 0
        for name in CLASS_NAMES:
            p = current_distribution.get(name, 0)
            q = baseline_distribution.get(name, 1/len(CLASS_NAMES))
            if p > 0 and q > 0:
                drift_score += p * np.log(p / q)
        
        log_metrics({
            "monitor/drift_score": drift_score,
            "monitor/drift_detected": int(drift_score > 0.5)
        })
        
        if drift_score > 0.5:
            print(f"⚠️  Potential data drift detected! Score: {drift_score:.3f}")
        else:
            print(f"✓ No significant drift. Score: {drift_score:.3f}")
    
    def finish(self):
        """Finish monitoring and close W&B run."""
        self._log_aggregate_metrics()
        finish_wandb()
        print("\n✓ Monitoring session completed")


def simulate_production_monitoring(
    model_path: str,
    test_dir: str,
    extractor_path: str = None,
    n_samples: int = 100,
    device: str = None
):
    """
    Simulate production monitoring session.
    
    Args:
        model_path: Path to trained model
        test_dir: Path to test data directory
        extractor_path: Path to feature extractor
        n_samples: Number of samples to monitor
        device: Device for inference
    """
    monitor = InferenceMonitor(
        model_path=model_path,
        extractor_path=extractor_path,
        device=device
    )
    
    try:
        # Run batch monitoring
        monitor.run_batch_monitoring(
            test_dir=test_dir,
            n_samples=n_samples
        )
        
        # Check for drift
        monitor.detect_drift()
        
    finally:
        monitor.finish()


def create_monitoring_dashboard(project: str = None):
    """
    Print instructions for creating a W&B monitoring dashboard.
    
    Args:
        project: W&B project name
    """
    project = project or WANDB_PROJECT
    
    print(f"""
{'='*60}
Creating a Monitoring Dashboard in W&B
{'='*60}

1. Go to your W&B project: https://wandb.ai/{project}

2. Click "Create dashboard" -> "Blank dashboard"

3. Add the following panels:

   📊 Metrics Over Time:
   - Add a "Line plot" for: monitor/accuracy, monitor/avg_confidence
   
   📈 Prediction Distribution:
   - Add a "Bar chart" for: prediction_distribution
   
   ⏱️ Performance Metrics:
   - Add a "Line plot" for: monitor/avg_inference_time_ms
   - Add a "Scalar" for: monitor/throughput_samples_per_sec
   
   🎯 Drift Detection:
   - Add a "Line plot" for: monitor/drift_score
   - Add a "Scalar" for: monitor/drift_detected
   
   📸 Sample Predictions:
   - Add a "Media panel" for: inference/image

4. Save and share your dashboard!

{'='*60}
""")


def main():
    parser = argparse.ArgumentParser(
        description="Production inference monitoring with W&B"
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/label_spreading_model.pkl',
        help='Path to trained model'
    )
    parser.add_argument(
        '--extractor-path',
        type=str,
        default='models/feature_extractor.pth',
        help='Path to feature extractor'
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        default=str(Path(__file__).parent.parent / 'DS2GROCERIES' / 'test'),
        help='Path to test data directory'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=100,
        help='Number of samples to monitor'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device for inference'
    )
    parser.add_argument(
        '--dashboard-help',
        action='store_true',
        help='Print dashboard creation instructions'
    )
    
    args = parser.parse_args()
    
    if args.dashboard_help:
        create_monitoring_dashboard()
    else:
        simulate_production_monitoring(
            model_path=args.model_path,
            test_dir=args.test_dir,
            extractor_path=args.extractor_path,
            n_samples=args.n_samples,
            device=args.device
        )


if __name__ == "__main__":
    main()
