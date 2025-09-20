"""Support Vector Machine model for tennis prediction.

Implements SVM with RBF kernel, probability calibration,
and hyperparameter optimization for tennis match prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from .base_model import BaseModel
from config import get_config


class SVMModel(BaseModel):
    """Support Vector Machine model for tennis prediction.
    
    Features:
    - RBF kernel for non-linear patterns
    - Probability calibration (Platt scaling)
    - Feature scaling integration
    - Hyperparameter optimization
    - Support vector analysis
    - Kernel parameter tuning
    """
    
    def __init__(self, **kwargs):
        # Get default parameters from config
        config = get_config()
        default_params = config.get('models.base_models.svm', {})
        
        # Merge with provided parameters
        params = {**default_params, **kwargs}
        
        super().__init__("SVM", **params)
        
        # SVM specific parameters
        self.kernel = params.get('kernel', 'rbf')
        self.C = params.get('C', 1.0)
        self.gamma = params.get('gamma', 'scale')
        self.degree = params.get('degree', 3)  # For poly kernel
        self.coef0 = params.get('coef0', 0.0)   # For poly/sigmoid kernels
        self.probability = params.get('probability', True)
        self.class_weight = params.get('class_weight', 'balanced')
        self.random_state = params.get('random_state', 42)
        
        # Advanced parameters
        self.calibration_method = params.get('calibration_method', 'sigmoid')  # or 'isotonic'
        self.hyperparameter_tuning = params.get('hyperparameter_tuning', True)
        self.feature_scaling = params.get('feature_scaling', True)
        
        # Components
        self.scaler = StandardScaler() if self.feature_scaling else None
        self.calibrated_model = None
        
        # Training artifacts
        self.support_vectors_count = None
        self.support_vectors_ratio = None
        self.optimization_results = {}
    
    def _create_model(self) -> SVC:
        """Create SVM classifier."""
        return SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            probability=self.probability,
            class_weight=self.class_weight,
            random_state=self.random_state
        )
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Fit SVM with scaling and calibration."""
        
        # Prepare data
        if self.feature_scaling:
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        else:
            X_scaled = X
        
        # Hyperparameter optimization
        if self.hyperparameter_tuning:
            optimized_params = self._optimize_hyperparameters(X_scaled, y)
            
            # Update model parameters
            for param, value in optimized_params.items():
                setattr(self, param, value)
            
            # Recreate model with optimized parameters
            self.model = self._create_model()
        
        # Fit the model
        self.model.fit(X_scaled, y)
        
        # Store support vector information
        self.support_vectors_count = self.model.n_support_
        self.support_vectors_ratio = self.model.n_support_.sum() / len(X)
        
        self.logger.info(
            f"SVM trained - Support vectors: {self.support_vectors_count.sum()}/{len(X)} "
            f"({self.support_vectors_ratio:.1%})"
        )
        
        # Probability calibration if needed
        if self.probability and not self.model.probability:
            self.logger.info("Applying probability calibration")
            self.calibrated_model = CalibratedClassifierCV(
                self.model, method=self.calibration_method, cv=3
            )
            self.calibrated_model.fit(X_scaled, y)
        else:
            self.calibrated_model = self.model
        
        return self.calibrated_model
    
    def _predict_proba_model(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities with scaling."""
        if self.feature_scaling:
            X_scaled = self.scaler.transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        else:
            X_scaled = X
        
        if self.calibrated_model:
            return self.calibrated_model.predict_proba(X_scaled)
        else:
            return self.model.predict_proba(X_scaled)
    
    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize SVM hyperparameters using GridSearch."""
        self.logger.info("Starting SVM hyperparameter optimization")
        
        # Define parameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        
        # Create base model
        base_model = SVC(
            random_state=self.random_state,
            class_weight=self.class_weight
        )
        
        # Perform grid search
        search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        search.fit(X, y)
        
        # Store optimization results
        self.optimization_results = {
            'best_score': search.best_score_,
            'best_params': search.best_params_,
            'cv_results': search.cv_results_
        }
        
        self.logger.info(
            f"SVM optimization completed. Best score: {search.best_score_:.3f}"
        )
        self.logger.info(f"Best parameters: {search.best_params_}")
        
        return search.best_params_
    
    def analyze_support_vectors(self) -> Dict[str, Any]:
        """Analyze support vector characteristics."""
        if not self.is_trained or not hasattr(self.model, 'support_vectors_'):
            return {}
        
        sv = self.model.support_vectors_
        sv_indices = self.model.support_
        
        analysis = {
            'total_support_vectors': len(sv),
            'support_vectors_by_class': self.support_vectors_count.tolist(),
            'support_vectors_ratio': self.support_vectors_ratio,
            'support_vector_stats': {
                'mean': sv.mean(axis=0).tolist() if len(sv) > 0 else [],
                'std': sv.std(axis=0).tolist() if len(sv) > 0 else [],
                'shape': sv.shape
            }
        }
        
        return analysis
    
    def get_decision_function_analysis(self, X_sample: pd.DataFrame) -> Dict[str, Any]:
        """Analyze decision function values."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Scale features if needed
        if self.feature_scaling:
            X_scaled = self.scaler.transform(X_sample)
            X_scaled = pd.DataFrame(X_scaled, columns=X_sample.columns, index=X_sample.index)
        else:
            X_scaled = X_sample
        
        # Get decision function values
        decision_values = self.model.decision_function(X_scaled)
        
        analysis = {
            'decision_values': decision_values.tolist(),
            'mean_decision_value': float(np.mean(decision_values)),
            'std_decision_value': float(np.std(decision_values)),
            'min_decision_value': float(np.min(decision_values)),
            'max_decision_value': float(np.max(decision_values))
        }
        
        # Convert to probabilities for comparison
        probabilities = self.predict_proba(X_sample)[:, 1]
        analysis['probabilities'] = probabilities.tolist()
        
        return analysis
    
    def get_svm_info(self) -> Dict[str, Any]:
        """Get SVM-specific model information."""
        info = self.get_model_info()
        
        info.update({
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'support_vectors_analysis': self.analyze_support_vectors(),
            'optimization_results': self.optimization_results,
            'calibration_method': self.calibration_method,
            'feature_scaling_used': self.feature_scaling
        })
        
        return info