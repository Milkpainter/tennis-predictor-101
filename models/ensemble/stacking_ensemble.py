"""Stacking ensemble implementation for Tennis Predictor 101.

Implements multi-level stacking with meta-learning based on
research showing superior performance in competitive ML.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
import logging

from .base_ensemble import BaseEnsemble
from ..base_models import BaseModel
from config import get_config


class StackingEnsemble(BaseEnsemble):
    """Advanced stacking ensemble for tennis prediction.
    
    Features:
    - Multi-level stacking architecture
    - Cross-validation for base model predictions
    - Multiple meta-learner options
    - Feature augmentation with base model outputs
    - Dynamic model weighting
    - Out-of-fold prediction generation
    """
    
    def __init__(self, base_models: List[BaseModel], 
                 meta_learner: str = 'logistic_regression',
                 cv_folds: int = 5, use_probabilities: bool = True,
                 use_original_features: bool = True, **kwargs):
        """Initialize stacking ensemble.
        
        Args:
            base_models: List of base models
            meta_learner: Meta-learner type ('logistic_regression', 'random_forest', 'xgboost')
            cv_folds: Number of cross-validation folds
            use_probabilities: Use probabilities instead of predictions
            use_original_features: Include original features in meta-learner
            **kwargs: Additional parameters
        """
        super().__init__("StackingEnsemble", base_models, **kwargs)
        
        self.config = get_config()
        self.logger = logging.getLogger("models.stacking_ensemble")
        
        # Stacking configuration
        self.meta_learner_type = meta_learner
        self.cv_folds = cv_folds
        self.use_probabilities = use_probabilities
        self.use_original_features = use_original_features
        
        # Model components
        self.meta_learner = None
        self.cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Training artifacts
        self.base_predictions = None
        self.meta_features = None
        self.base_model_scores = {}
        self.stacking_scores = {}
        
        # Feature names
        self.meta_feature_names = []
        self.original_feature_names = []
    
    def _create_meta_learner(self) -> Any:
        """Create meta-learner model."""
        if self.meta_learner_type == 'logistic_regression':
            return LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced'
            )
        elif self.meta_learner_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                class_weight='balanced'
            )
        elif self.meta_learner_type == 'xgboost':
            import xgboost as xgb
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown meta-learner type: {self.meta_learner_type}")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
             X_val: Optional[pd.DataFrame] = None,
             y_val: Optional[pd.Series] = None) -> Dict[str, float]:
        """Train stacking ensemble.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Training metrics
        """
        self.logger.info("Training stacking ensemble")
        
        # Store original feature names
        self.original_feature_names = list(X_train.columns)
        
        # Step 1: Train base models and generate out-of-fold predictions
        self.logger.info("Generating out-of-fold predictions from base models")
        meta_features_train = self._generate_meta_features(X_train, y_train, training=True)
        
        # Step 2: Train meta-learner
        self.logger.info("Training meta-learner")
        self.meta_learner = self._create_meta_learner()
        self.meta_learner.fit(meta_features_train, y_train)
        
        # Update state
        self.is_trained = True
        self.training_data_shape = X_train.shape
        self.training_timestamp = pd.Timestamp.now()
        
        # Calculate training metrics
        train_pred = self.predict(X_train)
        train_proba = self.predict_proba(X_train)
        
        self.training_metrics = self._calculate_metrics(
            y_train, train_pred, train_proba, 'training'
        )
        
        # Calculate validation metrics if provided
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            val_proba = self.predict_proba(X_val)
            
            self.validation_metrics = self._calculate_metrics(
                y_val, val_pred, val_proba, 'validation'
            )
        
        # Calculate individual base model performance
        self._evaluate_base_models(X_train, y_train)
        
        self.logger.info(
            f"Stacking ensemble training completed - "
            f"Accuracy: {self.training_metrics.get('accuracy', 0):.3f}"
        )
        
        return self.training_metrics
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities from stacking ensemble.
        
        Args:
            X: Features to predict
            
        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        # Generate meta-features
        meta_features = self._generate_meta_features(X, training=False)
        
        # Get predictions from meta-learner
        return self.meta_learner.predict_proba(meta_features)
    
    def _generate_meta_features(self, X: pd.DataFrame, y: pd.Series = None,
                               training: bool = False) -> pd.DataFrame:
        """Generate meta-features from base model predictions.
        
        Args:
            X: Input features
            y: Target values (required for training)
            training: Whether this is called during training
            
        Returns:
            DataFrame with meta-features
        """
        meta_features_list = []
        
        if training:
            # Use cross-validation for training to avoid overfitting
            for i, model in enumerate(self.base_models):
                self.logger.info(f"Generating CV predictions for {model.name}")
                
                if self.use_probabilities:
                    # Get probability predictions using cross-validation
                    cv_probs = cross_val_predict(
                        model.model if model.is_trained else model._create_model(),
                        X, y, cv=self.cv_splitter, method='predict_proba'
                    )
                    
                    # Use probability of positive class
                    meta_features_list.append(cv_probs[:, 1])
                    self.meta_feature_names.append(f"{model.name}_proba")
                else:
                    # Get binary predictions using cross-validation
                    cv_preds = cross_val_predict(
                        model.model if model.is_trained else model._create_model(),
                        X, y, cv=self.cv_splitter
                    )
                    
                    meta_features_list.append(cv_preds)
                    self.meta_feature_names.append(f"{model.name}_pred")
                
                # Train the actual model on full training data
                if not model.is_trained:
                    model.train(X, y)
        else:
            # Use trained models for prediction
            for model in self.base_models:
                if self.use_probabilities:
                    proba = model.predict_proba(X)
                    meta_features_list.append(proba[:, 1])
                else:
                    pred = model.predict(X)
                    meta_features_list.append(pred)
        
        # Combine meta-features
        meta_features = np.column_stack(meta_features_list)
        meta_df = pd.DataFrame(meta_features, columns=self.meta_feature_names, index=X.index)
        
        # Add original features if requested
        if self.use_original_features:
            # Concatenate original features with meta-features
            meta_df = pd.concat([X, meta_df], axis=1)
        
        return meta_df
    
    def _evaluate_base_models(self, X: pd.DataFrame, y: pd.Series):
        """Evaluate individual base model performance."""
        self.logger.info("Evaluating base model performance")
        
        for model in self.base_models:
            try:
                pred = model.predict(X)
                proba = model.predict_proba(X)
                
                accuracy = accuracy_score(y, pred)
                logloss = log_loss(y, proba)
                
                self.base_model_scores[model.name] = {
                    'accuracy': accuracy,
                    'log_loss': logloss
                }
                
                self.logger.info(f"{model.name}: Accuracy={accuracy:.3f}, LogLoss={logloss:.3f}")
                
            except Exception as e:
                self.logger.warning(f"Error evaluating {model.name}: {e}")
    
    def get_base_model_weights(self) -> Dict[str, float]:
        """Get effective weights of base models in the ensemble.
        
        Returns:
            Dictionary with model weights
        """
        if not self.is_trained or self.meta_learner is None:
            return {}
        
        weights = {}
        
        try:
            # For linear meta-learners, coefficients represent weights
            if hasattr(self.meta_learner, 'coef_'):
                coefs = self.meta_learner.coef_[0] if self.meta_learner.coef_.ndim > 1 else self.meta_learner.coef_
                
                # Map coefficients to base models
                for i, model in enumerate(self.base_models):
                    if i < len(coefs):
                        weights[model.name] = abs(coefs[i])
            
            # For tree-based meta-learners, use feature importance
            elif hasattr(self.meta_learner, 'feature_importances_'):
                importances = self.meta_learner.feature_importances_
                
                for i, model in enumerate(self.base_models):
                    if i < len(importances):
                        weights[model.name] = importances[i]
            
            # Normalize weights
            if weights:
                total_weight = sum(weights.values())
                if total_weight > 0:
                    weights = {k: v/total_weight for k, v in weights.items()}
                    
        except Exception as e:
            self.logger.warning(f"Error calculating model weights: {e}")
        
        return weights
    
    def get_meta_learner_info(self) -> Dict[str, Any]:
        """Get information about the meta-learner.
        
        Returns:
            Dictionary with meta-learner information
        """
        if not self.is_trained:
            return {}
        
        info = {
            'meta_learner_type': self.meta_learner_type,
            'cv_folds': self.cv_folds,
            'use_probabilities': self.use_probabilities,
            'use_original_features': self.use_original_features,
            'meta_feature_count': len(self.meta_feature_names),
            'meta_feature_names': self.meta_feature_names,
            'base_model_weights': self.get_base_model_weights(),
            'base_model_scores': self.base_model_scores
        }
        
        # Add meta-learner specific information
        if hasattr(self.meta_learner, 'coef_'):
            info['meta_learner_coefficients'] = self.meta_learner.coef_.tolist()
        
        if hasattr(self.meta_learner, 'feature_importances_'):
            info['meta_learner_feature_importance'] = self.meta_learner.feature_importances_.tolist()
        
        return info
    
    def analyze_model_contributions(self, X_sample: pd.DataFrame) -> pd.DataFrame:
        """Analyze individual model contributions to predictions.
        
        Args:
            X_sample: Sample of features to analyze
            
        Returns:
            DataFrame with model contributions
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained first")
        
        contributions = []
        
        # Get base model predictions
        for model in self.base_models:
            if self.use_probabilities:
                pred = model.predict_proba(X_sample)[:, 1]
            else:
                pred = model.predict(X_sample).astype(float)
            
            contributions.append({
                'model': model.name,
                'predictions': pred.tolist(),
                'mean_prediction': pred.mean(),
                'std_prediction': pred.std()
            })
        
        # Get ensemble prediction
        ensemble_proba = self.predict_proba(X_sample)[:, 1]
        
        contributions.append({
            'model': 'Ensemble',
            'predictions': ensemble_proba.tolist(),
            'mean_prediction': ensemble_proba.mean(),
            'std_prediction': ensemble_proba.std()
        })
        
        return pd.DataFrame(contributions)
    
    def get_diversity_metrics(self) -> Dict[str, float]:
        """Calculate ensemble diversity metrics.
        
        Returns:
            Dictionary with diversity metrics
        """
        if len(self.base_models) < 2:
            return {}
        
        # This would require a validation set to calculate properly
        # For now, return placeholder
        return {
            'model_count': len(self.base_models),
            'meta_learner_type': self.meta_learner_type,
            'uses_original_features': self.use_original_features
        }