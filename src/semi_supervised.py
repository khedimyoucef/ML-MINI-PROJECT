"""
Semi-Supervised Learning Algorithms

This module implements three semi-supervised learning algorithms:
1. Label Propagation - Graph-based label spreading using k-NN
2. Label Spreading - Normalized graph Laplacian variant
3. Self-Training Classifier - Iterative pseudo-labeling approach
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score
)
import joblib
from pathlib import Path


class SemiSupervisedClassifier:
    """
    Unified interface for semi-supervised learning algorithms.
    
    Supports:
    - label_propagation: Graph-based method using k-NN similarity
    - label_spreading: Uses normalized graph Laplacian
    - self_training: Iterative pseudo-labeling with base classifier
    """
    
    ALGORITHMS = ['label_propagation', 'label_spreading', 'self_training']
    
    def __init__(
        self,
        algorithm: str = 'label_propagation',
        normalize_features: bool = True,
        **kwargs
    ):
        """
        Initialize the semi-supervised classifier.
        
        Args:
            algorithm: One of 'label_propagation', 'label_spreading', 'self_training'
            normalize_features: Whether to standardize features
            **kwargs: Additional arguments for the specific algorithm
        """
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}. Choose from {self.ALGORITHMS}")
        
        self.algorithm = algorithm
        self.normalize_features = normalize_features
        self.scaler = StandardScaler() if normalize_features else None
        self.model = None
        self.kwargs = kwargs
        
        self._create_model()
    
    def _create_model(self):
        """Create the underlying model based on algorithm choice."""
        if self.algorithm == 'label_propagation':
            # Label Propagation:
            # This algorithm builds a graph where nodes are data points.
            # Edges represent similarity between points (using k-Nearest Neighbors).
            # Labels "flow" from labeled nodes to unlabeled nodes based on these edges.
            # It's like spreading ink on a network: if a node is connected to many "Red" nodes, it becomes Red.
            self.model = LabelPropagation(
                kernel=self.kwargs.get('kernel', 'knn'),
                n_neighbors=self.kwargs.get('n_neighbors', 7),
                max_iter=self.kwargs.get('max_iter', 1000),
                n_jobs=self.kwargs.get('n_jobs', -1)
            )
        
        elif self.algorithm == 'label_spreading':
            # Label Spreading:
            # Similar to Label Propagation, but uses a "Normalized Graph Laplacian".
            # This makes it more robust to noise and irregular graph structures.
            # The 'alpha' parameter controls how much the initial labels can change.
            # It essentially "smooths" the labels over the graph.
            self.model = LabelSpreading(
                kernel=self.kwargs.get('kernel', 'knn'),
                n_neighbors=self.kwargs.get('n_neighbors', 7),
                alpha=self.kwargs.get('alpha', 0.2),
                max_iter=self.kwargs.get('max_iter', 1000),
                n_jobs=self.kwargs.get('n_jobs', -1)
            )
        
        elif self.algorithm == 'self_training':
            # Self-Training:
            # This is a wrapper around a standard supervised classifier (like Random Forest).
            # 1. Train the classifier on the small labeled set.
            # 2. Use it to predict labels for the unlabeled set.
            # 3. Take the most confident predictions and add them to the labeled set as "pseudo-labels".
            # 4. Repeat step 1 with the larger labeled set.
            base_classifier = self.kwargs.get('base_classifier', None)
            if base_classifier is None:
                # Default to Random Forest for better scalability
                base_classifier = RandomForestClassifier(
                    n_estimators=100,
                    n_jobs=-1,
                    random_state=42
                )
            
            self.model = SelfTrainingClassifier(
                estimator=base_classifier,
                threshold=self.kwargs.get('threshold', 0.75),
                max_iter=self.kwargs.get('max_iter', 10),
                verbose=self.kwargs.get('verbose', False)
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SemiSupervisedClassifier':
        """
        Fit the semi-supervised classifier.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels with -1 for unlabeled samples
            
        Returns:
            self
        """
        # Normalize features if requested
        # Standardization (mean=0, std=1) is crucial for many ML algorithms,
        # especially those based on distance (like k-NN in Label Propagation).
        # It ensures that all features contribute equally to the distance calculation.
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
        
        # Fit the model
        # For semi-supervised models, X contains BOTH labeled and unlabeled data.
        # y contains labels for labeled data, and -1 for unlabeled data.
        self.model.fit(X, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for new samples.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predicted labels
        """
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for new samples.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Class probability matrix (n_samples, n_classes)
        """
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # Return one-hot for models without probabilities
            predictions = self.model.predict(X)
            n_classes = len(np.unique(predictions))
            proba = np.zeros((len(X), n_classes))
            proba[np.arange(len(X)), predictions] = 1.0
            return proba
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score on test data.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def get_classification_report(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_names: Optional[list] = None
    ) -> str:
        """
        Get detailed classification report.
        
        Args:
            X: Feature matrix
            y: True labels
            class_names: List of class names
            
        Returns:
            Classification report string
        """
        predictions = self.predict(X)
        return classification_report(y, predictions, target_names=class_names)
    
    def get_confusion_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get confusion matrix.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Confusion matrix
        """
        predictions = self.predict(X)
        return confusion_matrix(y, predictions)
    
    def save(self, path: str):
        """
        Save the trained model.
        
        Args:
            path: Path to save the model
        """
        save_dict = {
            'algorithm': self.algorithm,
            'model': self.model,
            'scaler': self.scaler,
            'kwargs': self.kwargs
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(save_dict, path)
    
    @classmethod
    def load(cls, path: str) -> 'SemiSupervisedClassifier':
        """
        Load a trained model.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded SemiSupervisedClassifier
        """
        save_dict = joblib.load(path)
        
        instance = cls.__new__(cls)
        instance.algorithm = save_dict['algorithm']
        instance.model = save_dict['model']
        instance.scaler = save_dict['scaler']
        instance.kwargs = save_dict['kwargs']
        instance.normalize_features = instance.scaler is not None
        
        return instance


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    algorithm: str,
    class_names: Optional[list] = None,
    **kwargs
) -> Tuple[SemiSupervisedClassifier, Dict[str, Any]]:
    """
    Train and evaluate a semi-supervised classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels (with -1 for unlabeled)
        X_test: Test features
        y_test: Test labels
        algorithm: Algorithm name
        class_names: Class names for reporting
        **kwargs: Algorithm-specific parameters
        
    Returns:
        Tuple of (trained_classifier, results_dict)
    """
    print(f"\nTraining {algorithm}...")
    
    # Count labeled/unlabeled
    n_labeled = np.sum(y_train != -1)
    n_unlabeled = np.sum(y_train == -1)
    print(f"  Labeled: {n_labeled}, Unlabeled: {n_unlabeled}")
    
    # Create and train classifier
    clf = SemiSupervisedClassifier(algorithm=algorithm, **kwargs)
    clf.fit(X_train, y_train)
    
    # Evaluate
    # Calculate comprehensive metrics
    # We use 'weighted' average to account for class imbalance, which is common in real-world datasets
    train_acc = clf.score(X_train[y_train != -1], y_train[y_train != -1])
    test_acc = clf.score(X_test, y_test)
    
    # Get predictions for detailed metrics
    y_pred = clf.predict(X_test)
    
    # Calculate F1 Score
    # F1 Score is the harmonic mean of precision and recall.
    # It provides a balance between the two.
    # We use 'weighted' average to account for class imbalance (some classes have more samples than others).
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Calculate Recall
    # Recall (Sensitivity) measures the proportion of actual positives that were correctly identified.
    # "Out of all the actual apples, how many did we correctly label as apples?"
    recall = recall_score(y_test, y_pred, average='weighted')
    
    # Calculate Precision
    # Precision measures the proportion of positive identifications that were actually correct.
    # "Out of all the items we labeled as apples, how many were actually apples?"
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Generate a detailed classification report
    # This string contains precision, recall, f1-score for EACH class individually.
    report = classification_report(y_test, y_pred, target_names=class_names)
    
    # Generate a confusion matrix
    # This matrix shows where the model is making mistakes.
    # Rows represent actual classes, columns represent predicted classes.
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"  Train accuracy (labeled only): {train_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # Return the trained model and a dictionary containing all the results
    results = {
        'algorithm': algorithm,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'f1_score': f1,
        'recall_score': recall,
        'precision_score': precision,
        'n_labeled': n_labeled,
        'n_unlabeled': n_unlabeled,
        'classification_report': report,
        'confusion_matrix': conf_matrix
    }
    
    return clf, results


def compare_algorithms(
    X_train: np.ndarray,
    y_train_ssl: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[list] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compare all semi-supervised algorithms.
    
    Args:
        X_train: Training features
        y_train_ssl: Training labels with -1 for unlabeled
        X_test: Test features
        y_test: Test labels
        class_names: Class names for reporting
        
    Returns:
        Dictionary of results for each algorithm
    """
    results = {}
    
    for algorithm in SemiSupervisedClassifier.ALGORITHMS:
        try:
            clf, result = train_and_evaluate(
                X_train, y_train_ssl, X_test, y_test,
                algorithm=algorithm,
                class_names=class_names
            )
            results[algorithm] = result
            results[algorithm]['classifier'] = clf
        except Exception as e:
            print(f"  Error training {algorithm}: {e}")
            results[algorithm] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing Semi-Supervised Classifiers")
    print("=" * 50)
    
    # Create synthetic data
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=5,
        random_state=42
    )
    
    # Split into train/test
    train_size = 800
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create semi-supervised labels (90% unlabeled)
    y_train_ssl = y_train.copy()
    np.random.seed(42)
    unlabeled_mask = np.random.rand(len(y_train_ssl)) > 0.1
    y_train_ssl[unlabeled_mask] = -1
    
    print(f"\nDataset:")
    print(f"  Train: {len(X_train)} samples ({(~unlabeled_mask).sum()} labeled)")
    print(f"  Test: {len(X_test)} samples")
    
    # Compare algorithms
    results = compare_algorithms(X_train, y_train_ssl, X_test, y_test)
    
    # Summary
    print("\n" + "=" * 50)
    print("Summary of Results:")
    print("-" * 50)
    for algo, result in results.items():
        if 'error' not in result:
            print(f"{algo}: {result['test_accuracy']:.4f}")
        else:
            print(f"{algo}: ERROR - {result['error']}")
