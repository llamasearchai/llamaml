#!/usr/bin/env python3
"""
Linear Regression Example with LlamaML

This example demonstrates how to use LlamaML's LinearRegression model
for a simple regression task on the Boston Housing dataset.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add the parent directory to the path to import llamaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llamaml.core import ModelConfig
from llamaml.models import LinearRegression


def main():
    """Run a linear regression example using LlamaML."""
    print("LlamaML Linear Regression Example")
    print("=================================")

    # Load the Boston Housing dataset
    print("\nLoading Boston Housing dataset...")
    boston = load_boston()
    X, y = boston.data, boston.target
    feature_names = boston.feature_names

    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Feature names: {feature_names}")

    # Split the data into training and testing sets
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create model configuration
    print("\nConfiguring the model...")
    config = ModelConfig(
        name="boston_housing_linear_model",
        learning_rate=0.01,
        params={"fit_intercept": True},
    )

    # Initialize and train the model
    print("\nTraining the model...")
    model = LinearRegression(config)
    model.fit(X_train_scaled, y_train)

    # Print model parameters
    print("\nModel parameters:")
    weights = model.weights
    bias = model.bias

    print(f"Bias: {bias:.4f}")
    print("Weights:")
    for name, weight in zip(feature_names, weights):
        print(f"  {name}: {weight:.4f}")

    # Make predictions
    print("\nMaking predictions...")
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # Evaluate the model
    print("\nEvaluating the model...")
    train_metrics = model.evaluate(X_train_scaled, y_train)
    test_metrics = model.evaluate(X_test_scaled, y_test)

    print("\nTraining set metrics:")
    print(f"  MSE: {train_metrics['mse']:.4f}")
    print(f"  MAE: {train_metrics['mae']:.4f}")
    print(f"  R²: {train_metrics['r2']:.4f}")

    print("\nTest set metrics:")
    print(f"  MSE: {test_metrics['mse']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    print(f"  R²: {test_metrics['r2']:.4f}")

    # Visualize results
    print("\nVisualizing results...")
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_pred_train, alpha=0.5)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "r--")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Training Set")

    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Test Set")

    plt.tight_layout()
    plt.savefig("linear_regression_results.png")
    print("Plot saved as 'linear_regression_results.png'")

    # Save model
    model_path = "boston_housing_model.pkl"
    model.save(model_path)
    print(f"\nModel saved to {model_path}")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
