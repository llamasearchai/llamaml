#!/usr/bin/env python3
"""
Classification Example with LlamaML

This example demonstrates how to use LlamaML's LogisticRegression and RandomForest
models for classification on the Iris dataset.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add the parent directory to the path to import llamaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llamaml.core import ModelConfig
from llamaml.models import LogisticRegression, RandomForest


def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    """
    Plot a confusion matrix for the classification results.

    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        classes: List of class names
        model_name: Name of the model to include in plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    print(
        f"Confusion matrix saved as 'confusion_matrix_{model_name.lower().replace(' ', '_')}.png'"
    )


def main():
    """Run a classification example using LlamaML."""
    print("LlamaML Classification Example")
    print("==============================")

    # Load the Iris dataset
    print("\nLoading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Feature names: {feature_names}")
    print(f"Classes: {target_names}")

    # Split the data into training and testing sets
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==================== LogisticRegression ====================
    print("\n" + "=" * 50)
    print("Logistic Regression Model")
    print("=" * 50)

    # Create model configuration for LogisticRegression
    print("\nConfiguring the Logistic Regression model...")
    lr_config = ModelConfig(
        name="iris_logistic_regression",
        learning_rate=0.01,
        batch_size=32,
        epochs=200,
        params={
            "fit_intercept": True,
            "max_iter": 1000,
            "tol": 1e-4,
            "C": 1.0,  # Regularization strength
        },
    )

    # Initialize and train the model
    print("\nTraining the Logistic Regression model...")
    lr_model = LogisticRegression(lr_config)
    lr_model.fit(X_train_scaled, y_train)

    # Make predictions
    print("\nMaking predictions with Logistic Regression...")
    y_pred_lr = lr_model.predict(X_test_scaled)

    # Evaluate the model
    print("\nEvaluating the Logistic Regression model...")
    lr_metrics = lr_model.evaluate(X_test_scaled, y_test)

    print("\nLogistic Regression metrics:")
    for metric, value in lr_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred_lr, target_names, "Logistic Regression")

    # ==================== RandomForest ====================
    print("\n" + "=" * 50)
    print("Random Forest Model")
    print("=" * 50)

    # Create model configuration for RandomForest
    print("\nConfiguring the Random Forest model...")
    rf_config = ModelConfig(
        name="iris_random_forest",
        params={
            "task": "classification",
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "criterion": "gini",
            "max_features": "sqrt",
            "bootstrap": True,
        },
    )

    # Initialize and train the model
    print("\nTraining the Random Forest model...")
    rf_model = RandomForest(rf_config)
    rf_model.fit(X_train_scaled, y_train)

    # Make predictions
    print("\nMaking predictions with Random Forest...")
    y_pred_rf = rf_model.predict(X_test_scaled)

    # Evaluate the model
    print("\nEvaluating the Random Forest model...")
    rf_metrics = rf_model.evaluate(X_test_scaled, y_test)

    print("\nRandom Forest metrics:")
    for metric, value in rf_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred_rf, target_names, "Random Forest")

    # ==================== Compare models ====================
    print("\n" + "=" * 50)
    print("Model Comparison")
    print("=" * 50)

    # Compare model accuracy
    print("\nAccuracy comparison:")
    print(f"  Logistic Regression: {lr_metrics['accuracy']:.4f}")
    print(f"  Random Forest: {rf_metrics['accuracy']:.4f}")

    # Plot feature importance for Random Forest
    # Note: This is a simple approximation of feature importance
    print("\nVisualizing feature importance from Random Forest...")

    # Create a basic feature importance visualization
    plt.figure(figsize=(10, 6))

    # Count how many times each feature is used in the Random Forest
    feature_counts = np.zeros(X.shape[1])
    for tree in rf_model.trees:
        # This is a simple approximation since our implementation
        # doesn't explicitly track feature importances
        for idx in tree.feature_indices:
            feature_counts[idx] += 1

    # Normalize to get relative importance
    feature_importance = feature_counts / feature_counts.sum()

    # Plot feature importance
    plt.bar(feature_names, feature_importance)
    plt.xlabel("Features")
    plt.ylabel("Relative Importance")
    plt.title("Feature Importance (Random Forest)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("Feature importance plot saved as 'feature_importance.png'")

    # Save models
    lr_model_path = "iris_logistic_regression.pkl"
    rf_model_path = "iris_random_forest.pkl"

    lr_model.save(lr_model_path)
    rf_model.save(rf_model_path)

    print(f"\nModels saved to {lr_model_path} and {rf_model_path}")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
