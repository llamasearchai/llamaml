"""
LlamaML: Machine Learning Library for the LlamaSearch ecosystem.

A comprehensive package for building, training, and deploying machine learning
models with a focus on NLP and LLM applications.
"""

__version__ = "0.1.0"

# Core components
from llamaml.core.model import Model, ModelConfig
from llamaml.core.trainer import Trainer, TrainingConfig
from llamaml.core.evaluator import Evaluator, EvaluationConfig
from llamaml.core.optimizer import Optimizer, OptimizerConfig
from llamaml.core.scheduler import Scheduler, SchedulerConfig
from llamaml.core.pipeline import Pipeline, PipelineStage

# Model implementations
from llamaml.models.classification import ClassificationModel
from llamaml.models.regression import RegressionModel
from llamaml.models.clustering import ClusteringModel
from llamaml.models.transformer import TransformerModel
from llamaml.models.embedding import EmbeddingModel
from llamaml.models.ensemble import EnsembleModel

# Data handling
from llamaml.data.dataset import Dataset, DataLoader
from llamaml.data.preprocessing import Preprocessor
from llamaml.data.augmentation import Augmenter
from llamaml.data.splitting import DataSplitter

# Metrics
from llamaml.metrics.classification import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)
from llamaml.metrics.regression import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score
)

# Utilities
from llamaml.utils.logging import get_logger, setup_logging
from llamaml.utils.visualization import plot_training_history, plot_confusion_matrix
from llamaml.utils.serialization import save_model, load_model

__all__ = [
    # Core
    "Model", "ModelConfig", "Trainer", "TrainingConfig",
    "Evaluator", "EvaluationConfig", "Optimizer", "OptimizerConfig",
    "Scheduler", "SchedulerConfig", "Pipeline", "PipelineStage",
    
    # Models
    "ClassificationModel", "RegressionModel", "ClusteringModel",
    "TransformerModel", "EmbeddingModel", "EnsembleModel",
    
    # Data
    "Dataset", "DataLoader", "Preprocessor", "Augmenter", "DataSplitter",
    
    # Metrics
    "accuracy_score", "precision_score", "recall_score", "f1_score",
    "mean_squared_error", "mean_absolute_error", "r2_score",
    
    # Utils
    "get_logger", "setup_logging", "plot_training_history",
    "plot_confusion_matrix", "save_model", "load_model"
] 