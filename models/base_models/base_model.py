"""Base model class for all machine learning models."""

import logging
import pickle
import joblib
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, brier_score_loss, roc_auc_score
)
from sklearn.model_selection import cross_val_score
from datetime import datetime
import os

from config import get_config


class BaseModel(ABC):
    """Base class for all machine learning models.
    
    Provides common functionality for training, prediction,
    evaluation, and model persistence.
    """
    
    def __init__(self, name: str, **kwargs):
        """Initialize base model.
        
        Args:
            name: Model name
            **kwargs: Model-specific parameters
        """
        self.name = name
        self.config = get_config()
        self.logger = logging.getLogger(f"models.{name}")
        
        # Model state
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.feature_importance = None
        self.training_metrics = {}
        self.validation_metrics = {}
        
        # Training metadata
        self.training_data_shape = None
        self.training_timestamp = None
        self.model_version = "1.0.0"
        
        # Model parameters
        self.model_params = kwargs
        
        # Performance tracking
        self.prediction_count = 0
        self.last_prediction_time = None
    
    @abstractmethod
    def _create_model(self) -> Any:
        """Create the underlying ML model.
        
        Returns:
            Initialized model object
        """
        pass
    
    @abstractmethod
    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Fit the model to training data.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Fitted model object
        """
        pass
    
    @abstractmethod
    def _predict_proba_model(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities from the model.
        
        Args:
            X: Features to predict
            
        Returns:
            Prediction probabilities
        """
        pass
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
             X_val: Optional[pd.DataFrame] = None,
             y_val: Optional[pd.Series] = None) -> Dict[str, float]:
        """Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Training metrics
        """
        self.logger.info(f"Training {self.name} model")
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Create and fit model
        if self.model is None:
            self.model = self._create_model()
        
        # Fit the model
        start_time = datetime.now()
        self.model = self._fit_model(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Update state
        self.is_trained = True
        self.training_data_shape = X_train.shape
        self.training_timestamp = datetime.now()
        
        # Calculate training metrics
        train_pred = self.predict(X_train)
        train_proba = self.predict_proba(X_train)
        
        self.training_metrics = self._calculate_metrics(
            y_train, train_pred, train_proba, 'training'
        )
        self.training_metrics['training_time'] = training_time
        
        # Calculate validation metrics if provided
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            val_proba = self.predict_proba(X_val)
            
            self.validation_metrics = self._calculate_metrics(
                y_val, val_pred, val_proba, 'validation'
            )
        
        # Extract feature importance if available
        self._extract_feature_importance()
        
        self.logger.info(
            f"Training completed - Accuracy: {self.training_metrics.get('accuracy', 0):.3f}, "
            f"Training time: {training_time:.2f}s"
        )
        
        return self.training_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Binary predictions (0 or 1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure feature consistency
        X = self._ensure_feature_consistency(X)
        
        # Get probabilities and convert to binary predictions
        probabilities = self.predict_proba(X)
        predictions = (probabilities[:, 1] > 0.5).astype(int)
        
        # Update prediction tracking
        self.prediction_count += len(predictions)
        self.last_prediction_time = datetime.now()
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities.
        
        Args:
            X: Features to predict
            
        Returns:
            Prediction probabilities [prob_class_0, prob_class_1]
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure feature consistency
        X = self._ensure_feature_consistency(X)
        
        # Get probabilities from model
        probabilities = self._predict_proba_model(X)
        
        # Ensure proper shape (n_samples, 2)
        if probabilities.ndim == 1:
            # Convert to 2D array
            prob_positive = probabilities
            prob_negative = 1 - prob_positive
            probabilities = np.column_stack([prob_negative, prob_positive])
        
        return probabilities
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Evaluation metrics
        """
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        
        return self._calculate_metrics(y_test, predictions, probabilities, 'test')
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series,
                      cv: int = 5, scoring: str = 'accuracy') -> Dict[str, float]:
        """Perform cross-validation.
        
        Args:
            X: Features
            y: Targets
            cv: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Cross-validation results
        """
        if not self.is_trained:
            # Create model for cross-validation
            temp_model = self._create_model()
        else:
            temp_model = self.model
        
        scores = cross_val_score(temp_model, X, y, cv=cv, scoring=scoring)
        
        return {
            f'cv_{scoring}_mean': scores.mean(),
            f'cv_{scoring}_std': scores.std(),
            f'cv_{scoring}_scores': scores.tolist()
        }
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance if available.
        
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            return None
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
    
    def save_model(self, filepath: str):
        """Save model to file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare model data
        model_data = {
            'model': self.model,
            'name': self.name,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'training_data_shape': self.training_data_shape,
            'training_timestamp': self.training_timestamp,
            'model_version': self.model_version,
            'model_params': self.model_params
        }
        
        # Save using joblib for sklearn models or pickle for others
        try:
            joblib.dump(model_data, filepath)
        except:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file.
        
        Args:
            filepath: Path to load the model from
        """
        try:
            # Try joblib first
            model_data = joblib.load(filepath)
        except:
            # Fall back to pickle
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
        
        # Restore model state
        self.model = model_data['model']
        self.name = model_data.get('name', self.name)
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data.get('feature_importance')
        self.training_metrics = model_data.get('training_metrics', {})
        self.validation_metrics = model_data.get('validation_metrics', {})
        self.training_data_shape = model_data.get('training_data_shape')
        self.training_timestamp = model_data.get('training_timestamp')
        self.model_version = model_data.get('model_version', '1.0.0')
        self.model_params = model_data.get('model_params', {})
        
        self.is_trained = True
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information.
        
        Returns:
            Dictionary with model information
        """
        return {
            'name': self.name,
            'is_trained': self.is_trained,
            'model_version': self.model_version,
            'training_timestamp': self.training_timestamp,
            'training_data_shape': self.training_data_shape,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'prediction_count': self.prediction_count,
            'last_prediction_time': self.last_prediction_time,
            'model_params': self.model_params
        }
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray,
                          y_proba: np.ndarray, phase: str) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}
        
        try:
            # Basic classification metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Probability-based metrics
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                metrics['log_loss'] = log_loss(y_true, y_proba)
                metrics['brier_score'] = brier_score_loss(y_true, y_proba[:, 1])
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            
            # Additional metrics
            metrics['sample_count'] = len(y_true)
            metrics['positive_rate'] = y_true.mean()
            metrics['prediction_positive_rate'] = y_pred.mean()
            
        except Exception as e:
            self.logger.warning(f"Error calculating metrics for {phase}: {e}")
        
        return metrics
    
    def _extract_feature_importance(self):
        """Extract feature importance from the model if available."""
        try:
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                # For linear models, use absolute coefficients
                coef = self.model.coef_
                if coef.ndim > 1:
                    coef = coef[0]
                self.feature_importance = np.abs(coef)
            else:
                self.feature_importance = None
        except Exception as e:
            self.logger.warning(f"Could not extract feature importance: {e}")
            self.feature_importance = None
    
    def _ensure_feature_consistency(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure feature consistency with training data."""
        if self.feature_names is None:
            return X
        
        # Check for missing features
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            self.logger.warning(f"Missing features: {missing_features}")
            # Add missing features with default value (0)
            for feature in missing_features:
                X[feature] = 0
        
        # Remove extra features
        extra_features = set(X.columns) - set(self.feature_names)
        if extra_features:
            self.logger.warning(f"Extra features removed: {extra_features}")
            X = X.drop(columns=list(extra_features))
        
        # Ensure correct order
        X = X[self.feature_names]
        
        return X