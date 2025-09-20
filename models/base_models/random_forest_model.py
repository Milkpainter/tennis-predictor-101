"""Random Forest Model for Tennis Prediction.

Implements optimized Random Forest classifier with:
- Advanced hyperparameter tuning
- Feature importance analysis
- Out-of-bag scoring
- Bootstrap sampling optimization
- Ensemble diversity metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging
from datetime import datetime

from .base_model import BaseModel
from config import get_config


class RandomForestModel(BaseModel):
    """Advanced Random Forest Model for Tennis Prediction."""
    
    def __init__(self, hyperparameter_tuning: bool = True, n_jobs: int = -1):
        super().__init__()
        
        self.model_name = "RandomForest"
        self.config = get_config()
        self.logger = logging.getLogger(f"model.{self.model_name}")
        
        # Model configuration
        self.hyperparameter_tuning = hyperparameter_tuning
        self.n_jobs = n_jobs
        
        # Model components
        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.oob_score = None
        
        # Hyperparameter search space
        self.param_grid = {
            'n_estimators': [300, 500, 700, 1000],
            'max_depth': [8, 10, 12, 15, None],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 6],
            'max_features': ['sqrt', 'log2', 0.3, 0.5],
            'bootstrap': [True, False],
            'class_weight': ['balanced', 'balanced_subsample', None]
        }
        
        # Default parameters
        self.default_params = {
            'n_estimators': 500,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'oob_score': True,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': self.n_jobs
        }
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RandomForestModel':
        """Train the Random Forest model."""
        
        self.logger.info(f"Training RandomForest on {len(X)} samples")
        start_time = datetime.now()
        
        # Hyperparameter optimization
        if self.hyperparameter_tuning:
            self.best_params = self._optimize_hyperparameters(X, y)
            model_params = {**self.default_params, **self.best_params}
        else:
            model_params = self.default_params
        
        # Train model
        self.model = RandomForestClassifier(**model_params)
        self.model.fit(X, y)
        
        # Extract information
        self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))
        self.oob_score = getattr(self.model, 'oob_score_', None)
        
        # Training metrics
        train_accuracy = accuracy_score(y, self.model.predict(X))
        training_time = (datetime.now() - start_time).total_seconds()
        
        self.training_metrics = {
            'accuracy': train_accuracy,
            'oob_score': self.oob_score,
            'training_time_seconds': training_time
        }
        
        self.is_trained = True
        self.logger.info(f"RandomForest training completed - Accuracy: {train_accuracy:.3f}")
        
        return self
    
    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize hyperparameters."""
        
        base_model = RandomForestClassifier(random_state=42, n_jobs=self.n_jobs)
        
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=self.param_grid,
            n_iter=30,
            cv=5,
            scoring='accuracy',
            n_jobs=self.n_jobs,
            random_state=42
        )
        
        search.fit(X, y)
        return search.best_params_
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.model.predict_proba(X)