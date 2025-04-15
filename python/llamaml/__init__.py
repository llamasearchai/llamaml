"""
LlamaML: Machine Learning Library for the LlamaSearch ecosystem.

A comprehensive package for building, training, and deploying machine learning
models with a focus on NLP and LLM applications.
"""

__version__ = "0.1.0"

from llamaml.core.evaluator import EvaluationConfig, Evaluator

# Core components
from llamaml.core.model import Model, ModelConfig
from llamaml.core.optimizer import Optimizer, OptimizerConfig
from llamaml.core.pipeline import Pipeline, PipelineStage
from llamaml.core.scheduler import Scheduler, SchedulerConfig
from llamaml.core.trainer import Trainer, TrainingConfig
from llamaml.data.augmentation import Augmenter

# Data handling
from llamaml.data.dataset import DataLoader, Dataset
from llamaml.data.preprocessing import Preprocessor
from llamaml.data.splitting import DataSplitter

# Metrics
from llamaml.metrics.classification import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from llamaml.metrics.regression import mean_absolute_error, mean_squared_error, r2_score

# Model implementations
from llamaml.models.classification import ClassificationModel
from llamaml.models.clustering import ClusteringModel
from llamaml.models.embedding import EmbeddingModel
from llamaml.models.ensemble import EnsembleModel
from llamaml.models.regression import RegressionModel
from llamaml.models.transformer import TransformerModel

# Utilities
from llamaml.utils.logging import get_logger, setup_logging
from llamaml.utils.serialization import load_model, save_model
from llamaml.utils.visualization import plot_confusion_matrix, plot_training_history

__all__ = [
    # Core
    "Model",
    "ModelConfig",
    "Trainer",
    "TrainingConfig",
    "Evaluator",
    "EvaluationConfig",
    "Optimizer",
    "OptimizerConfig",
    "Scheduler",
    "SchedulerConfig",
    "Pipeline",
    "PipelineStage",
    # Models
    "ClassificationModel",
    "RegressionModel",
    "ClusteringModel",
    "TransformerModel",
    "EmbeddingModel",
    "EnsembleModel",
    # Data
    "Dataset",
    "DataLoader",
    "Preprocessor",
    "Augmenter",
    "DataSplitter",
    # Metrics
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    # Utils
    "get_logger",
    "setup_logging",
    "plot_training_history",
    "plot_confusion_matrix",
    "save_model",
    "load_model",
]
