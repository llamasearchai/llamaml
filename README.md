# LlamaML

A Python framework for machine learning experimentation, model development, and deployment that simplifies the workflow from data processing to model evaluation.

## Features

- **Data Processing**: Tools for data loading, transformation, and preprocessing
- **Model Development**: Easy-to-use abstractions for creating and training ML models
- **Experiment Management**: Track experiments and model performance
- **Evaluation**: Comprehensive metrics and evaluation tools
- **Visualization**: Visualize model results and data distributions
- **Integration**: Seamless integration with popular ML libraries and frameworks
- **CLI Interface**: Command-line tools for common ML tasks

## Installation

```bash
# Basic installation
pip install llamaml

# With extras
pip install llamaml[viz]     # Visualization extras
pip install llamaml[dev]     # Development tools
pip install llamaml[torch]   # PyTorch integration
pip install llamaml[tf]      # TensorFlow integration
pip install llamaml[xgboost] # XGBoost integration

# Full installation
pip install llamaml[all]
```

## Quick Start

### Creating and Training a Model

```python
import numpy as np
from llamaml.core import ModelConfig
from llamaml.models import LinearRegression
from llamaml.data import DataLoader
from llamaml.metrics import RegressionMetrics
from llamaml.viz import plot_regression_results

# Load data
data_loader = DataLoader("path/to/data.csv")
X_train, X_test, y_train, y_test = data_loader.train_test_split(test_size=0.2)

# Create model config
config = ModelConfig(
    name="my_linear_model",
    learning_rate=0.01,
    params={"fit_intercept": True}
)

# Initialize and train model
model = LinearRegression(config)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
metrics = RegressionMetrics()
results = metrics.compute_all(y_test, y_pred)
print(results)

# Visualize results
plot_regression_results(y_test, y_pred)

# Save model
model.save("models/my_linear_model.pkl")
```

### Using the Command Line Interface

```bash
# Train a model
llamaml train --config config.yaml --data data.csv --output models/

# Evaluate a model
llamaml evaluate --model models/my_model.pkl --test-data test_data.csv

# Generate predictions
llamaml predict --model models/my_model.pkl --data new_data.csv --output predictions.csv

# Run an experiment
llamaml experiment --config experiment.yaml
```

## Core Components

### Data Module

- **DataLoader**: Load and preprocess data from various sources
- **Transformers**: Apply transformations to features and targets
- **Splitters**: Split data for training, validation, and testing

### Models

- **Base Models**: Abstract base classes for different model types
- **Linear Models**: Linear and logistic regression
- **Tree Models**: Decision trees and ensemble methods
- **Neural Networks**: Simple feed-forward and convolutional networks
- **Custom Models**: Easily create custom models

### Metrics

- **Classification**: Accuracy, precision, recall, F1, ROC AUC, etc.
- **Regression**: MSE, MAE, R², etc.
- **Clustering**: Silhouette score, Davies-Bouldin index, etc.

### Visualization

- **Data Visualization**: Histograms, scatter plots, correlation matrices
- **Model Visualization**: Learning curves, feature importance, decision boundaries
- **Evaluation Visualization**: Confusion matrices, ROC curves, residual plots

## Advanced Usage

### Custom Models

```python
from llamaml.core import Model, ModelConfig
import numpy as np

class MyCustomModel(Model):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # Initialize your model here
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Model":
        # Implement training logic
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Implement prediction logic
        return predictions
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        # Implement evaluation logic
        return {"metric1": value1, "metric2": value2}
```

### Experiment Configuration

Example `experiment.yaml`:

```yaml
name: classification_experiment
models:
  - name: random_forest
    type: RandomForest
    params:
      n_estimators: 100
      max_depth: 10
  - name: logistic_regression
    type: LogisticRegression
    params:
      C: 1.0
data:
  path: data/dataset.csv
  target_column: target
  feature_columns: [feat1, feat2, feat3]
  preprocessing:
    - type: StandardScaler
      columns: [feat1, feat2]
    - type: OneHotEncoder
      columns: [feat3]
evaluation:
  metrics: [accuracy, precision, recall, f1]
  cross_validation: 5
```

## Documentation

For more detailed documentation, visit our [documentation site](https://llamasearch.ai

## Examples

Check out the `examples/` directory for more usage examples:

- Basic regression and classification
- Time series analysis
- Feature engineering
- Hyperparameter tuning
- Model ensembling
- And more!

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute.

## License

LlamaML is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 
# Updated in commit 1 - 2025-04-04 17:29:54

# Updated in commit 9 - 2025-04-04 17:29:54

# Updated in commit 17 - 2025-04-04 17:29:55

# Updated in commit 25 - 2025-04-04 17:29:55

# Updated in commit 1 - 2025-04-05 14:35:13

# Updated in commit 9 - 2025-04-05 14:35:14

# Updated in commit 17 - 2025-04-05 14:35:14

# Updated in commit 25 - 2025-04-05 14:35:14

# Updated in commit 1 - 2025-04-05 15:21:44

# Updated in commit 9 - 2025-04-05 15:21:44

# Updated in commit 17 - 2025-04-05 15:21:44

# Updated in commit 25 - 2025-04-05 15:21:44

# Updated in commit 1 - 2025-04-05 15:56:02

# Updated in commit 9 - 2025-04-05 15:56:03

# Updated in commit 17 - 2025-04-05 15:56:03

# Updated in commit 25 - 2025-04-05 15:56:03

# Updated in commit 1 - 2025-04-05 17:01:25

# Updated in commit 9 - 2025-04-05 17:01:25

# Updated in commit 17 - 2025-04-05 17:01:25

# Updated in commit 25 - 2025-04-05 17:01:25

# Updated in commit 1 - 2025-04-05 17:33:23

# Updated in commit 9 - 2025-04-05 17:33:23

# Updated in commit 17 - 2025-04-05 17:33:23

# Updated in commit 25 - 2025-04-05 17:33:23

# Updated in commit 1 - 2025-04-05 18:20:02

# Updated in commit 9 - 2025-04-05 18:20:02

# Updated in commit 17 - 2025-04-05 18:20:03

# Updated in commit 25 - 2025-04-05 18:20:03

# Updated in commit 1 - 2025-04-05 18:41:01

# Updated in commit 9 - 2025-04-05 18:41:02

# Updated in commit 17 - 2025-04-05 18:41:02
