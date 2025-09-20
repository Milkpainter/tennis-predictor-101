"""XGBoost model implementation with BSA optimization.

Implements XGBoost with Bird Swarm Algorithm optimization
based on research showing superior performance for tennis prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import xgboost as xgb
from sklearn.model_selection import train_test_split

from .base_model import BaseModel
from config import get_config


class XGBoostModel(BaseModel):
    """XGBoost model for tennis match prediction.
    
    Features:
    - Bird Swarm Algorithm (BSA) optimization
    - Early stopping and regularization
    - Feature importance analysis
    - GPU acceleration support
    - Hyperparameter optimization
    """
    
    def __init__(self, **kwargs):
        # Get default parameters from config
        config = get_config()
        default_params = config.get('models.base_models.xgboost', {})
        
        # Merge with provided parameters
        params = {**default_params, **kwargs}
        
        super().__init__("XGBoost", **params)
        
        # XGBoost specific parameters
        self.n_estimators = params.get('n_estimators', 500)
        self.max_depth = params.get('max_depth', 6)
        self.learning_rate = params.get('learning_rate', 0.1)
        self.subsample = params.get('subsample', 0.8)
        self.colsample_bytree = params.get('colsample_bytree', 0.8)
        self.reg_alpha = params.get('reg_alpha', 0.1)
        self.reg_lambda = params.get('reg_lambda', 1.0)
        self.random_state = params.get('random_state', 42)
        
        # Advanced parameters
        self.early_stopping_rounds = params.get('early_stopping_rounds', 50)
        self.use_gpu = params.get('use_gpu', False)
        self.bsa_optimization = params.get('bsa_optimization', True)
        
        # BSA optimization parameters
        self.bsa_population_size = params.get('bsa_population_size', 10)
        self.bsa_generations = params.get('bsa_generations', 20)
        self.bsa_c1 = params.get('bsa_c1', 2.0)
        self.bsa_c2 = params.get('bsa_c2', 2.0)
        
        # Training state
        self.validation_scores = []
        self.best_iteration = None
    
    def _create_model(self) -> xgb.XGBClassifier:
        """Create XGBoost classifier."""
        params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        
        # GPU acceleration
        if self.use_gpu:
            params['tree_method'] = 'gpu_hist'
            params['gpu_id'] = 0
        
        return xgb.XGBClassifier(**params)
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
        """Fit XGBoost model with optional BSA optimization."""
        if self.bsa_optimization:
            # Use BSA optimization for hyperparameters
            optimized_params = self._bsa_optimize(X, y)
            
            # Update model with optimized parameters
            for param, value in optimized_params.items():
                setattr(self.model, param, value)
        
        # Split data for early stopping
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Fit with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=False
        )
        
        self.best_iteration = self.model.best_iteration
        self.validation_scores = self.model.evals_result()['validation_0']['logloss']
        
        return self.model
    
    def _predict_proba_model(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities from XGBoost."""
        return self.model.predict_proba(X)
    
    def _bsa_optimize(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize hyperparameters using Bird Swarm Algorithm.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Dictionary of optimized parameters
        """
        self.logger.info("Starting BSA hyperparameter optimization")
        
        # Define parameter search space
        param_bounds = {
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.3),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
            'reg_alpha': (0.0, 1.0),
            'reg_lambda': (0.0, 2.0)
        }
        
        # Initialize bird population
        population = self._initialize_population(param_bounds)
        
        best_params = None
        best_score = float('inf')
        
        for generation in range(self.bsa_generations):
            # Evaluate population
            scores = []
            for bird in population:
                score = self._evaluate_params(bird, X, y)
                scores.append(score)
                
                if score < best_score:
                    best_score = score
                    best_params = bird.copy()
            
            # Update population using BSA rules
            population = self._update_population(population, scores, param_bounds)
            
            if generation % 5 == 0:
                self.logger.info(f"BSA Generation {generation}: Best score = {best_score:.4f}")
        
        self.logger.info(f"BSA optimization completed. Best score: {best_score:.4f}")
        return best_params
    
    def _initialize_population(self, param_bounds: Dict[str, tuple]) -> list:
        """Initialize BSA population."""
        population = []
        
        for _ in range(self.bsa_population_size):
            bird = {}
            for param, (min_val, max_val) in param_bounds.items():
                if param == 'max_depth':
                    bird[param] = np.random.randint(min_val, max_val + 1)
                else:
                    bird[param] = np.random.uniform(min_val, max_val)
            population.append(bird)
        
        return population
    
    def _evaluate_params(self, params: Dict[str, Any], X: pd.DataFrame, y: pd.Series) -> float:
        """Evaluate parameter set using cross-validation."""
        try:
            # Create temporary model with these parameters
            temp_model = xgb.XGBClassifier(
                n_estimators=100,  # Reduced for speed
                **params,
                random_state=self.random_state,
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False
            )
            
            # Simple train-validation split for speed
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.3, random_state=self.random_state
            )
            
            temp_model.fit(X_train, y_train)
            y_pred = temp_model.predict_proba(X_val)
            
            # Calculate log loss
            from sklearn.metrics import log_loss
            score = log_loss(y_val, y_pred)
            
            return score
            
        except Exception as e:
            self.logger.warning(f"Error evaluating params {params}: {e}")
            return float('inf')
    
    def _update_population(self, population: list, scores: list, param_bounds: Dict) -> list:
        """Update population using BSA algorithm."""
        new_population = []
        
        # Find best bird
        best_idx = np.argmin(scores)
        best_bird = population[best_idx]
        
        for i, bird in enumerate(population):
            new_bird = bird.copy()
            
            # BSA update rules
            for param in bird.keys():
                min_val, max_val = param_bounds[param]
                
                # Random factors
                r1, r2 = np.random.rand(), np.random.rand()
                
                # Attraction to best bird
                if r1 < 0.5:
                    # Move towards best bird
                    new_val = bird[param] + self.bsa_c1 * r2 * (best_bird[param] - bird[param])
                else:
                    # Random movement
                    new_val = bird[param] + self.bsa_c2 * (np.random.rand() - 0.5) * (max_val - min_val)
                
                # Ensure bounds
                new_val = np.clip(new_val, min_val, max_val)
                
                # Handle integer parameters
                if param == 'max_depth':
                    new_val = int(round(new_val))
                
                new_bird[param] = new_val
            
            new_population.append(new_bird)
        
        return new_population
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get XGBoost-specific training information."""
        info = self.get_model_info()
        
        info.update({
            'best_iteration': self.best_iteration,
            'total_boosting_rounds': len(self.validation_scores),
            'final_validation_score': self.validation_scores[-1] if self.validation_scores else None,
            'early_stopped': self.best_iteration is not None and self.best_iteration < self.n_estimators,
            'bsa_optimization_used': self.bsa_optimization
        })
        
        return info
    
    def plot_training_progress(self) -> None:
        """Plot training progress (validation scores)."""
        if not self.validation_scores:
            self.logger.warning("No validation scores available for plotting")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.validation_scores, label='Validation Loss')
            
            if self.best_iteration:
                plt.axvline(x=self.best_iteration, color='red', linestyle='--', 
                           label=f'Best Iteration ({self.best_iteration})')
            
            plt.xlabel('Boosting Round')
            plt.ylabel('Log Loss')
            plt.title(f'{self.name} Training Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
        except ImportError:
            self.logger.warning("Matplotlib not available for plotting")
    
    def get_feature_importance_detailed(self) -> pd.DataFrame:
        """Get detailed feature importance with multiple importance types."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importance_types = ['weight', 'gain', 'cover']
        importance_data = []
        
        for imp_type in importance_types:
            try:
                importance = self.model.get_booster().get_score(importance_type=imp_type)
                
                for feature, score in importance.items():
                    importance_data.append({
                        'feature': feature,
                        'importance_type': imp_type,
                        'importance': score
                    })
            except Exception as e:
                self.logger.warning(f"Could not get {imp_type} importance: {e}")
        
        if not importance_data:
            return self.get_feature_importance()
        
        return pd.DataFrame(importance_data)