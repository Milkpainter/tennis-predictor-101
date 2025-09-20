"""Base Models Package for Tennis Predictor 101.

Provides all base machine learning models:
- XGBoostModel: Gradient boosting with advanced optimization
- RandomForestModel: Ensemble trees with feature importance
- NeuralNetworkModel: Deep learning with PyTorch
- SVMModel: Support vector machines with kernel optimization
- LogisticRegressionModel: Linear model with feature selection

All models inherit from BaseModel and support:
- Hyperparameter optimization
- Cross-validation
- Model persistence
- Performance metrics
"""

from .base_model import BaseModel
from .xgboost_model import XGBoostModel
from .random_forest_model import RandomForestModel
from .neural_network_model import NeuralNetworkModel
from .svm_model import SVMModel
from .logistic_regression_model import LogisticRegressionModel

__all__ = [
    'BaseModel',
    'XGBoostModel', 
    'RandomForestModel',
    'NeuralNetworkModel',
    'SVMModel',
    'LogisticRegressionModel'
]


def get_all_models(hyperparameter_optimization: bool = True) -> list:
    """Get instances of all base models.
    
    Args:
        hyperparameter_optimization: Enable hyperparameter tuning
        
    Returns:
        List of initialized base model instances
    """
    
    models = [
        XGBoostModel(hyperparameter_optimization=hyperparameter_optimization),
        RandomForestModel(hyperparameter_tuning=hyperparameter_optimization),
        NeuralNetworkModel(use_scaling=True),
        SVMModel(hyperparameter_tuning=hyperparameter_optimization),
        LogisticRegressionModel(hyperparameter_tuning=hyperparameter_optimization)
    ]
    
    return models


def get_model_by_name(model_name: str, **kwargs):
    """Get model instance by name.
    
    Args:
        model_name: Name of the model (case-insensitive)
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Model instance
    """
    
    model_map = {
        'xgboost': XGBoostModel,
        'randomforest': RandomForestModel,
        'neuralnetwork': NeuralNetworkModel,
        'svm': SVMModel,
        'logisticregression': LogisticRegressionModel
    }
    
    model_class = model_map.get(model_name.lower().replace('_', '').replace(' ', ''))
    
    if model_class is None:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(model_map.keys())}")
    
    return model_class(**kwargs)