"""
Linear models for regression and classification tasks.
"""

from typing import Any, Dict, Optional

import numpy as np

from llamaml.core import Model, ModelConfig


class LinearRegression(Model):
    """Linear regression model implementation."""

    def __init__(self, config: ModelConfig):
        """Initialize the linear regression model.

        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.weights = None
        self.bias = None
        self.fit_intercept = config.params.get("fit_intercept", True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """Fit the linear regression model using normal equation.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)

        Returns:
            self: The fitted model
        """
        n_samples, n_features = X.shape
        X_b = X.copy()

        if self.fit_intercept:
            X_b = np.c_[np.ones((n_samples, 1)), X]
            n_features += 1

        # Normal equation: theta = (X^T X)^(-1) X^T y
        try:
            inverse = np.linalg.inv(X_b.T.dot(X_b))
            theta = inverse.dot(X_b.T).dot(y)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse for ill-conditioned matrices
            theta = np.linalg.pinv(X_b).dot(y)

        if self.fit_intercept:
            self.bias = theta[0]
            self.weights = theta[1:]
        else:
            self.bias = 0
            self.weights = theta

        self.is_fitted = True
        self.feature_names = getattr(X, "columns", None)
        self.target_names = getattr(y, "name", None)

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

        return X.dot(self.weights) + self.bias

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on test data.

        Args:
            X: Feature matrix
            y: True target values

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X)

        # Calculate MSE
        mse = np.mean((y - y_pred) ** 2)

        # Calculate R^2
        y_mean = np.mean(y)
        ss_total = np.sum((y - y_mean) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0

        # Calculate MAE
        mae = np.mean(np.abs(y - y_pred))

        return {"mse": mse, "r2": r2, "mae": mae}

    def get_params(self) -> Dict[str, Any]:
        """Get the model parameters.

        Returns:
            Dictionary of model parameters
        """
        return {
            "weights": self.weights.tolist() if self.weights is not None else None,
            "bias": float(self.bias) if self.bias is not None else None,
            "fit_intercept": self.fit_intercept,
        }

    def set_params(self, params: Dict[str, Any]) -> "LinearRegression":
        """Set the model parameters.

        Args:
            params: Dictionary of model parameters

        Returns:
            self: The updated model
        """
        if "weights" in params and params["weights"] is not None:
            self.weights = np.array(params["weights"])

        if "bias" in params and params["bias"] is not None:
            self.bias = params["bias"]

        if "fit_intercept" in params:
            self.fit_intercept = params["fit_intercept"]

        self.is_fitted = self.weights is not None
        return self


class LogisticRegression(Model):
    """Logistic regression model for binary classification."""

    def __init__(self, config: ModelConfig):
        """Initialize the logistic regression model.

        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.weights = None
        self.bias = None
        self.fit_intercept = config.params.get("fit_intercept", True)
        self.learning_rate = config.learning_rate
        self.max_iter = config.params.get("max_iter", 1000)
        self.tol = config.params.get("tol", 1e-4)
        self.C = config.params.get("C", 1.0)  # Regularization strength

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Apply sigmoid function.

        Args:
            z: Input values

        Returns:
            Sigmoid of input values
        """
        # Clip to avoid overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """Fit the logistic regression model using gradient descent.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target values of shape (n_samples,) with values 0 or 1

        Returns:
            self: The fitted model
        """
        n_samples, n_features = X.shape

        # Initialize parameters
        if self.fit_intercept:
            self.weights = np.zeros(n_features)
            self.bias = 0
        else:
            self.weights = np.zeros(n_features)
            self.bias = 0

        # Gradient descent
        for i in range(self.max_iter):
            # Linear combination
            linear_model = X.dot(self.weights) + self.bias

            # Predictions
            y_pred = self._sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * X.T.dot(y_pred - y) + (1 / self.C) * self.weights
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            if self.fit_intercept:
                self.bias -= self.learning_rate * db

            # Check convergence
            if i > 0 and np.linalg.norm(dw) < self.tol:
                break

        self.is_fitted = True
        self.feature_names = getattr(X, "columns", None)
        self.target_names = getattr(y, "name", None)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Predicted probabilities of shape (n_samples, 2)
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' first.")

        proba_1 = self._sigmoid(X.dot(self.weights) + self.bias)
        proba_0 = 1 - proba_1

        return np.column_stack((proba_0, proba_1))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Predicted class labels of shape (n_samples,)
        """
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on test data.

        Args:
            X: Feature matrix
            y: True class labels

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X)

        # Calculate accuracy
        accuracy = np.mean(y_pred == y)

        # Calculate precision, recall, F1
        true_pos = np.sum((y_pred == 1) & (y == 1))
        false_pos = np.sum((y_pred == 1) & (y == 0))
        false_neg = np.sum((y_pred == 0) & (y == 1))

        precision = (
            true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        )
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Calculate log loss
        y_prob = self.predict_proba(X)[:, 1]
        eps = 1e-15
        y_prob = np.clip(y_prob, eps, 1 - eps)
        log_loss = -np.mean(y * np.log(y_prob) + (1 - y) * np.log(1 - y_prob))

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "log_loss": log_loss,
        }

    def get_params(self) -> Dict[str, Any]:
        """Get the model parameters.

        Returns:
            Dictionary of model parameters
        """
        return {
            "weights": self.weights.tolist() if self.weights is not None else None,
            "bias": float(self.bias) if self.bias is not None else None,
            "fit_intercept": self.fit_intercept,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "C": self.C,
        }

    def set_params(self, params: Dict[str, Any]) -> "LogisticRegression":
        """Set the model parameters.

        Args:
            params: Dictionary of model parameters

        Returns:
            self: The updated model
        """
        if "weights" in params and params["weights"] is not None:
            self.weights = np.array(params["weights"])

        if "bias" in params and params["bias"] is not None:
            self.bias = params["bias"]

        for param in ["fit_intercept", "learning_rate", "max_iter", "tol", "C"]:
            if param in params:
                setattr(self, param, params[param])

        self.is_fitted = self.weights is not None
        return self
