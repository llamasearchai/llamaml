"""
Tree-based models for classification and regression tasks.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
from collections import Counter

from llamaml.core import Model, ModelConfig


class Node:
    """Node in a decision tree."""
    
    def __init__(self, feature_idx: Optional[int] = None, threshold: Optional[float] = None, 
                 left: Optional['Node'] = None, right: Optional['Node'] = None, 
                 value: Optional[np.ndarray] = None, is_leaf: bool = False):
        """
        Initialize a decision tree node.
        
        Args:
            feature_idx: Index of the feature to split on
            threshold: Threshold value for the split
            left: Left child node
            right: Right child node
            value: Value for leaf nodes (class probabilities or regression value)
            is_leaf: Whether the node is a leaf
        """
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.is_leaf = is_leaf


class DecisionTree(Model):
    """Decision tree implementation for classification and regression."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the decision tree model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.root = None
        self.task = config.params.get("task", "classification")  # "classification" or "regression"
        self.max_depth = config.params.get("max_depth", 10)
        self.min_samples_split = config.params.get("min_samples_split", 2)
        self.min_samples_leaf = config.params.get("min_samples_leaf", 1)
        self.criterion = config.params.get("criterion", "gini" if self.task == "classification" else "mse")
        
        if self.task not in ["classification", "regression"]:
            raise ValueError(f"Task must be 'classification' or 'regression', got {self.task}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        """Fit the decision tree to the data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            
        Returns:
            self: The fitted model
        """
        self.n_features = X.shape[1]
        
        if self.task == "classification":
            self.classes = np.unique(y)
            self.n_classes = len(self.classes)
        
        self.root = self._grow_tree(X, y)
        self.is_fitted = True
        self.feature_names = getattr(X, 'columns', None)
        self.target_names = getattr(y, 'name', None)
        
        return self
    
    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Recursively grow the decision tree.
        
        Args:
            X: Feature matrix
            y: Target values
            depth: Current depth of the tree
            
        Returns:
            Root node of the subtree
        """
        n_samples, n_features = X.shape
        
        # Check if we should stop splitting
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_samples < 2 * self.min_samples_leaf or
            np.all(y == y[0])):
            return self._create_leaf_node(y)
        
        # Find the best split
        best_feature, best_threshold = self._find_best_split(X, y)
        
        # If no good split found, create leaf node
        if best_feature is None:
            return self._create_leaf_node(y)
        
        # Split data
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices
        
        # Check if any child has too few samples
        if np.sum(left_indices) < self.min_samples_leaf or np.sum(right_indices) < self.min_samples_leaf:
            return self._create_leaf_node(y)
        
        # Recursively grow left and right trees
        left = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        
        return Node(feature_idx=best_feature, threshold=best_threshold, left=left, right=right)
    
    def _create_leaf_node(self, y: np.ndarray) -> Node:
        """Create a leaf node based on the target values.
        
        Args:
            y: Target values
            
        Returns:
            Leaf node
        """
        if self.task == "classification":
            # Store class probabilities
            values = np.zeros(self.n_classes)
            for i, cls in enumerate(self.classes):
                values[i] = np.sum(y == cls) / len(y)
        else:
            # Store mean value for regression
            values = np.mean(y)
        
        return Node(value=values, is_leaf=True)
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        """Find the best feature and threshold for splitting.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Tuple of (best_feature_idx, best_threshold)
        """
        n_samples, n_features = X.shape
        
        # Calculate current impurity
        if self.task == "classification":
            current_impurity = self._gini_impurity(y) if self.criterion == "gini" else self._entropy(y)
        else:
            current_impurity = self._mse(y)
        
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        
        # Try each feature
        for feature_idx in range(n_features):
            # Get unique values as potential thresholds
            thresholds = np.unique(X[:, feature_idx])
            
            # Skip if only one unique value
            if len(thresholds) <= 1:
                continue
            
            # Use midpoints between unique values as thresholds
            thresholds = (thresholds[:-1] + thresholds[1:]) / 2
            
            # Evaluate each threshold
            for threshold in thresholds:
                # Split data
                left_indices = X[:, feature_idx] <= threshold
                right_indices = ~left_indices
                
                # Skip if split is too imbalanced
                if np.sum(left_indices) < self.min_samples_leaf or np.sum(right_indices) < self.min_samples_leaf:
                    continue
                
                # Calculate impurity for left and right children
                left_impurity = self._calculate_impurity(y[left_indices])
                right_impurity = self._calculate_impurity(y[right_indices])
                
                # Calculate information gain
                n_left, n_right = np.sum(left_indices), np.sum(right_indices)
                gain = current_impurity - (n_left / n_samples * left_impurity + 
                                          n_right / n_samples * right_impurity)
                
                # Update best split if this one is better
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _calculate_impurity(self, y: np.ndarray) -> float:
        """Calculate impurity based on the criterion.
        
        Args:
            y: Target values
            
        Returns:
            Impurity value
        """
        if self.task == "classification":
            if self.criterion == "gini":
                return self._gini_impurity(y)
            else:  # entropy
                return self._entropy(y)
        else:  # regression
            return self._mse(y)
    
    def _gini_impurity(self, y: np.ndarray) -> float:
        """Calculate Gini impurity.
        
        Args:
            y: Target values
            
        Returns:
            Gini impurity
        """
        m = len(y)
        if m == 0:
            return 0.0
        
        class_counts = [np.sum(y == c) for c in self.classes]
        return 1.0 - sum((count / m) ** 2 for count in class_counts)
    
    def _entropy(self, y: np.ndarray) -> float:
        """Calculate entropy.
        
        Args:
            y: Target values
            
        Returns:
            Entropy
        """
        m = len(y)
        if m == 0:
            return 0.0
        
        class_counts = [np.sum(y == c) for c in self.classes]
        proportions = [count / m for count in class_counts if count > 0]
        return -sum(p * np.log2(p) for p in proportions)
    
    def _mse(self, y: np.ndarray) -> float:
        """Calculate mean squared error.
        
        Args:
            y: Target values
            
        Returns:
            Mean squared error
        """
        if len(y) == 0:
            return 0.0
        
        return np.var(y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the fitted model.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predicted values of shape (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
        
        return np.array([self._predict_sample(x) for x in X])
    
    def _predict_sample(self, x: np.ndarray) -> Union[float, int]:
        """Predict for a single sample.
        
        Args:
            x: A single sample
            
        Returns:
            Predicted value
        """
        node = self.root
        
        while not node.is_leaf:
            if x[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        
        if self.task == "classification":
            # Return class with highest probability
            return self.classes[np.argmax(node.value)]
        else:
            # Return predicted value for regression
            return node.value
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for classification.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predicted probabilities of shape (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
        
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification")
        
        return np.array([self._predict_proba_sample(x) for x in X])
    
    def _predict_proba_sample(self, x: np.ndarray) -> np.ndarray:
        """Predict probabilities for a single sample.
        
        Args:
            x: A single sample
            
        Returns:
            Predicted probabilities
        """
        node = self.root
        
        while not node.is_leaf:
            if x[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        
        return node.value
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on test data.
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X)
        
        if self.task == "classification":
            # Calculate accuracy
            accuracy = np.mean(y_pred == y)
            
            # Calculate precision, recall, F1 for each class
            metrics = {"accuracy": accuracy}
            
            if self.n_classes == 2:  # Binary classification
                # Assuming class 1 is the positive class
                positive_class = self.classes[1]
                true_pos = np.sum((y_pred == positive_class) & (y == positive_class))
                false_pos = np.sum((y_pred == positive_class) & (y != positive_class))
                false_neg = np.sum((y_pred != positive_class) & (y == positive_class))
                
                precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
                recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics.update({
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                })
            
            return metrics
        else:
            # Regression metrics
            mse = np.mean((y - y_pred) ** 2)
            mae = np.mean(np.abs(y - y_pred))
            
            # Calculate R^2
            y_mean = np.mean(y)
            ss_total = np.sum((y - y_mean) ** 2)
            ss_residual = np.sum((y - y_pred) ** 2)
            r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
            
            return {
                "mse": mse,
                "mae": mae,
                "r2": r2
            }
    
    def get_params(self) -> Dict[str, Any]:
        """Get the model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return {
            "task": self.task,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "criterion": self.criterion
        }
    
    def set_params(self, params: Dict[str, Any]) -> "DecisionTree":
        """Set the model parameters.
        
        Args:
            params: Dictionary of model parameters
            
        Returns:
            self: The updated model
        """
        for param, value in params.items():
            if param in ["task", "max_depth", "min_samples_split", "min_samples_leaf", "criterion"]:
                setattr(self, param, value)
        
        return self


class RandomForest(Model):
    """Random Forest implementation for classification and regression."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the random forest model.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.task = config.params.get("task", "classification")  # "classification" or "regression"
        self.n_estimators = config.params.get("n_estimators", 100)
        self.max_depth = config.params.get("max_depth", 10)
        self.min_samples_split = config.params.get("min_samples_split", 2)
        self.min_samples_leaf = config.params.get("min_samples_leaf", 1)
        self.criterion = config.params.get("criterion", "gini" if self.task == "classification" else "mse")
        self.max_features = config.params.get("max_features", "sqrt")
        self.bootstrap = config.params.get("bootstrap", True)
        self.trees = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForest":
        """Fit the random forest to the data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            
        Returns:
            self: The fitted model
        """
        n_samples, n_features = X.shape
        
        # Determine max_features
        if self.max_features == "sqrt":
            max_features = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            max_features = int(np.log2(n_features))
        elif isinstance(self.max_features, float) and 0.0 < self.max_features <= 1.0:
            max_features = int(self.max_features * n_features)
        elif isinstance(self.max_features, int) and 1 <= self.max_features <= n_features:
            max_features = self.max_features
        else:
            max_features = n_features
        
        # Store for prediction
        self.max_features_value = max_features
        
        if self.task == "classification":
            self.classes = np.unique(y)
            self.n_classes = len(self.classes)
        
        # Train each tree
        self.trees = []
        for _ in range(self.n_estimators):
            # Configure tree
            tree_config = ModelConfig(
                name=f"{self.config.name}_tree",
                params={
                    "task": self.task,
                    "max_depth": self.max_depth,
                    "min_samples_split": self.min_samples_split,
                    "min_samples_leaf": self.min_samples_leaf,
                    "criterion": self.criterion
                }
            )
            tree = DecisionTree(tree_config)
            
            # Bootstrap sampling
            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_bootstrap, y_bootstrap = X[indices], y[indices]
            else:
                X_bootstrap, y_bootstrap = X, y
            
            # Feature sampling
            feature_indices = np.random.choice(n_features, max_features, replace=False)
            
            # Train tree with selected features
            tree.fit(X_bootstrap[:, feature_indices], y_bootstrap)
            
            # Store feature indices with the tree
            tree.feature_indices = feature_indices
            
            self.trees.append(tree)
        
        self.is_fitted = True
        self.feature_names = getattr(X, 'columns', None)
        self.target_names = getattr(y, 'name', None)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the fitted model.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predicted values of shape (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
        
        # Get predictions from each tree
        predictions = []
        for tree in self.trees:
            # Select features used by this tree
            X_tree = X[:, tree.feature_indices]
            predictions.append(tree.predict(X_tree))
        
        # Combine predictions
        if self.task == "classification":
            # Majority voting
            final_predictions = []
            for i in range(len(X)):
                votes = [pred[i] for pred in predictions]
                final_predictions.append(Counter(votes).most_common(1)[0][0])
            return np.array(final_predictions)
        else:
            # Average for regression
            return np.mean(predictions, axis=0)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for classification.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predicted probabilities of shape (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")
        
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification")
        
        # Get probability predictions from each tree
        all_proba = []
        for tree in self.trees:
            # Select features used by this tree
            X_tree = X[:, tree.feature_indices]
            all_proba.append(tree.predict_proba(X_tree))
        
        # Average probabilities
        return np.mean(all_proba, axis=0)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on test data.
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X)
        
        if self.task == "classification":
            # Calculate accuracy
            accuracy = np.mean(y_pred == y)
            
            # Calculate precision, recall, F1 for each class
            metrics = {"accuracy": accuracy}
            
            if self.n_classes == 2:  # Binary classification
                # Assuming class 1 is the positive class
                positive_class = self.classes[1]
                true_pos = np.sum((y_pred == positive_class) & (y == positive_class))
                false_pos = np.sum((y_pred == positive_class) & (y != positive_class))
                false_neg = np.sum((y_pred != positive_class) & (y == positive_class))
                
                precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
                recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics.update({
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                })
            
            return metrics
        else:
            # Regression metrics
            mse = np.mean((y - y_pred) ** 2)
            mae = np.mean(np.abs(y - y_pred))
            
            # Calculate R^2
            y_mean = np.mean(y)
            ss_total = np.sum((y - y_mean) ** 2)
            ss_residual = np.sum((y - y_pred) ** 2)
            r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
            
            return {
                "mse": mse,
                "mae": mae,
                "r2": r2
            }
    
    def get_params(self) -> Dict[str, Any]:
        """Get the model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return {
            "task": self.task,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "criterion": self.criterion,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap
        }
    
    def set_params(self, params: Dict[str, Any]) -> "RandomForest":
        """Set the model parameters.
        
        Args:
            params: Dictionary of model parameters
            
        Returns:
            self: The updated model
        """
        for param, value in params.items():
            if param in ["task", "n_estimators", "max_depth", "min_samples_split", 
                        "min_samples_leaf", "criterion", "max_features", "bootstrap"]:
                setattr(self, param, value)
        
        return self 