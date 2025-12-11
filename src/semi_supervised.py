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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
            # Label Propagation uses k-NN graph
            self.model = LabelPropagation(
                kernel=self.kwargs.get('kernel', 'knn'),
                n_neighbors=self.kwargs.get('n_neighbors', 7),
                max_iter=self.kwargs.get('max_iter', 1000),
                n_jobs=self.kwargs.get('n_jobs', -1)
            )
        
        elif self.algorithm == 'label_spreading':
            # Label Spreading with normalized Laplacian
            self.model = LabelSpreading(
                kernel=self.kwargs.get('kernel', 'knn'),
                n_neighbors=self.kwargs.get('n_neighbors', 7),
                alpha=self.kwargs.get('alpha', 0.2),
                max_iter=self.kwargs.get('max_iter', 1000),
                n_jobs=self.kwargs.get('n_jobs', -1)
            )
        
        elif self.algorithm == 'self_training':
            # Self-Training with a base classifier
            base_classifier = self.kwargs.get('base_classifier', None)
            if base_classifier is None:
                # Default to SVM with probability estimates
                base_classifier = SVC(
                    kernel='rbf',
                    probability=True,
                    random_state=42
                )
            
            self.model = SelfTrainingClassifier(
                base_estimator=base_classifier,
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
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
        
        # Fit the model
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
    train_acc = clf.score(X_train[y_train != -1], y_train[y_train != -1])
    test_acc = clf.score(X_test, y_test)
    
    print(f"  Train accuracy (labeled only): {train_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")
    
    # Get detailed report
    report = clf.get_classification_report(X_test, y_test, class_names)
    conf_matrix = clf.get_confusion_matrix(X_test, y_test)
    
    results = {
        'algorithm': algorithm,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
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
