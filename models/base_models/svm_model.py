"""Support Vector Machine Model for Tennis Prediction.

Implements optimized SVM classifier with:
- Multiple kernel options (RBF, polynomial, linear)
- Hyperparameter optimization
- Feature scaling integration
- Probability calibration
- Grid search optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import joblib
import logging
from datetime import datetime

from .base_model import BaseModel
from config import get_config


class SVMModel(BaseModel):
    """Support Vector Machine Model for Tennis Prediction."""
    
    def __init__(self, hyperparameter_tuning: bool = True, feature_scaling: bool = True):
        super().__init__()
        
        self.model_name = "SVM"
        self.config = get_config()
        self.logger = logging.getLogger(f"model.{self.model_name}")
        
        # Configuration
        self.hyperparameter_tuning = hyperparameter_tuning
        self.feature_scaling = feature_scaling
        
        # Model components
        self.model = None
        self.scaler = StandardScaler() if feature_scaling else None
        self.best_params = None
        
        # Hyperparameter search space
        self.param_grid = {
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'degree': [2, 3, 4, 5],  # For polynomial kernel
            'class_weight': ['balanced', None]
        }
        
        # Default parameters
        self.default_params = {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'probability': True,
            'class_weight': 'balanced',
            'random_state': 42
        }
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'SVMModel':
        """Train the SVM model."""
        
        self.logger.info(f"Training SVM on {len(X)} samples")
        start_time = datetime.now()
        
        # Feature scaling
        if self.feature_scaling:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.values
        
        # Hyperparameter optimization
        if self.hyperparameter_tuning:
            self.best_params = self._optimize_hyperparameters(X_scaled, y)
            model_params = {**self.default_params, **self.best_params}
        else:
            model_params = self.default_params
        
        # Train model
        self.model = SVC(**model_params)
        self.model.fit(X_scaled, y)
        
        # Training metrics
        train_predictions = self.model.predict(X_scaled)
        train_accuracy = accuracy_score(y, train_predictions)
        training_time = (datetime.now() - start_time).total_seconds()
        
        self.training_metrics = {
            'accuracy': train_accuracy,
            'training_time_seconds': training_time,
            'kernel': model_params['kernel'],
            'C': model_params['C'],
            'gamma': model_params['gamma'],
            'n_support_vectors': self.model.n_support_.sum() if hasattr(self.model, 'n_support_') else None
        }
        
        self.is_trained = True
        self.logger.info(f"SVM training completed - Accuracy: {train_accuracy:.3f}")
        
        return self
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: pd.Series) -> Dict[str, Any]:
        """Optimize SVM hyperparameters."""
        
        base_model = SVC(probability=True, random_state=42)
        
        # Use randomized search for efficiency
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=self.param_grid,
            n_iter=20,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )
        
        search.fit(X, y)
        return search.best_params_
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if self.feature_scaling:
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
        else:
            return self.model.predict(X.values)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if self.feature_scaling:
            X_scaled = self.scaler.transform(X)
            return self.model.predict_proba(X_scaled)
        else:
            return self.model.predict_proba(X.values)