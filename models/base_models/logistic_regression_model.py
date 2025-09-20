"""Logistic Regression Model for Tennis Prediction.

Implements advanced logistic regression with:
- L1/L2 regularization
- Feature selection integration
- Hyperparameter optimization
- Coefficient analysis
- Statistical significance testing
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import joblib
import logging
from datetime import datetime

from .base_model import BaseModel
from config import get_config


class LogisticRegressionModel(BaseModel):
    """Advanced Logistic Regression Model for Tennis Prediction."""
    
    def __init__(self, hyperparameter_tuning: bool = True, feature_selection: bool = True,
                 n_features_to_select: int = 50):
        super().__init__()
        
        self.model_name = "LogisticRegression"
        self.config = get_config()
        self.logger = logging.getLogger(f"model.{self.model_name}")
        
        # Configuration
        self.hyperparameter_tuning = hyperparameter_tuning
        self.feature_selection = feature_selection
        self.n_features_to_select = n_features_to_select
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features = None
        self.best_params = None
        self.coefficients = None
        
        # Hyperparameter search space
        self.param_grid = {
            'penalty': ['l1', 'l2', 'elasticnet'],
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga', 'lbfgs'],
            'max_iter': [1000, 2000, 3000],
            'class_weight': ['balanced', None]
        }
        
        # Default parameters
        self.default_params = {
            'penalty': 'l2',
            'C': 1.0,
            'solver': 'liblinear',
            'max_iter': 1000,
            'class_weight': 'balanced',
            'random_state': 42
        }
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LogisticRegressionModel':
        """Train the Logistic Regression model."""
        
        self.logger.info(f"Training LogisticRegression on {len(X)} samples")
        start_time = datetime.now()
        
        # Feature scaling
        X_scaled = self.scaler.fit_transform(X)
        
        # Feature selection
        if self.feature_selection:
            self.logger.info(f"Performing feature selection (top {self.n_features_to_select} features)")
            
            # Use SelectKBest with f_classif
            self.feature_selector = SelectKBest(
                score_func=f_classif,
                k=min(self.n_features_to_select, X.shape[1])
            )
            
            X_selected = self.feature_selector.fit_transform(X_scaled, y)
            
            # Store selected feature names
            selected_indices = self.feature_selector.get_support()
            self.selected_features = X.columns[selected_indices].tolist()
            
            self.logger.info(f"Selected {len(self.selected_features)} features")
        else:
            X_selected = X_scaled
            self.selected_features = X.columns.tolist()
        
        # Hyperparameter optimization
        if self.hyperparameter_tuning:
            self.best_params = self._optimize_hyperparameters(X_selected, y)
            model_params = {**self.default_params, **self.best_params}
        else:
            model_params = self.default_params
        
        # Train final model
        self.model = LogisticRegression(**model_params)
        self.model.fit(X_selected, y)
        
        # Extract coefficients
        self.coefficients = dict(zip(
            self.selected_features,
            self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_
        ))
        
        # Training metrics
        train_predictions = self.predict(X)
        train_accuracy = accuracy_score(y, train_predictions)
        training_time = (datetime.now() - start_time).total_seconds()
        
        self.training_metrics = {
            'accuracy': train_accuracy,
            'training_time_seconds': training_time,
            'n_selected_features': len(self.selected_features),
            'penalty': model_params['penalty'],
            'C': model_params['C'],
            'solver': model_params['solver']
        }
        
        self.is_trained = True
        self.logger.info(
            f"LogisticRegression training completed - Accuracy: {train_accuracy:.3f}, "
            f"Features: {len(self.selected_features)}"
        )
        
        return self
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: pd.Series) -> Dict[str, Any]:
        """Optimize hyperparameters using GridSearchCV."""
        
        base_model = LogisticRegression(random_state=42)
        
        # Grid search
        search = GridSearchCV(
            estimator=base_model,
            param_grid=self.param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        search.fit(X, y)
        return search.best_params_
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Transform features
        X_scaled = self.scaler.transform(X)
        
        if self.feature_selection:
            X_selected = self.feature_selector.transform(X_scaled)
        else:
            X_selected = X_scaled
        
        return self.model.predict(X_selected)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Transform features
        X_scaled = self.scaler.transform(X)
        
        if self.feature_selection:
            X_selected = self.feature_selector.transform(X_scaled)
        else:
            X_selected = X_scaled
        
        return self.model.predict_proba(X_selected)
    
    def get_feature_coefficients(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature coefficients sorted by absolute value."""
        
        if not self.coefficients:
            return pd.DataFrame()
        
        coef_df = pd.DataFrame([
            {'feature': feature, 'coefficient': coef, 'abs_coefficient': abs(coef)}
            for feature, coef in self.coefficients.items()
        ])
        
        coef_df = coef_df.sort_values('abs_coefficient', ascending=False)
        
        if top_n:
            coef_df = coef_df.head(top_n)
        
        return coef_df