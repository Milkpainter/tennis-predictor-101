"""Random Forest model for tennis prediction.

Implements Random Forest with feature importance analysis,
out-of-bag scoring, and hyperparameter optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from .base_model import BaseModel
from config import get_config


class RandomForestModel(BaseModel):
    """Random Forest model for tennis match prediction.
    
    Features:
    - Feature importance analysis
    - Out-of-bag scoring
    - Hyperparameter optimization
    - Class balancing
    - Bootstrap aggregation
    """
    
    def __init__(self, **kwargs):
        # Get default parameters from config
        config = get_config()
        default_params = config.get('models.base_models.random_forest', {})
        
        # Merge with provided parameters
        params = {**default_params, **kwargs}
        
        super().__init__("RandomForest", **params)
        
        # Random Forest specific parameters
        self.n_estimators = params.get('n_estimators', 500)
        self.max_depth = params.get('max_depth', 10)
        self.min_samples_split = params.get('min_samples_split', 5)
        self.min_samples_leaf = params.get('min_samples_leaf', 2)
        self.max_features = params.get('max_features', 'sqrt')
        self.bootstrap = params.get('bootstrap', True)
        self.oob_score = params.get('oob_score', True)
        self.class_weight = params.get('class_weight', 'balanced')
        self.random_state = params.get('random_state', 42)
        self.n_jobs = params.get('n_jobs', -1)
        
        # Advanced parameters
        self.hyperparameter_tuning = params.get('hyperparameter_tuning', True)
        self.feature_selection = params.get('feature_selection', True)
        
        # Performance tracking
        self.oob_score_value = None
        self.feature_importance_ranking = None
    
    def _create_model(self) -> RandomForestClassifier:
        """Create Random Forest classifier."""
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
        """Fit Random Forest with optional hyperparameter tuning."""
        
        if self.hyperparameter_tuning:
            # Hyperparameter optimization
            optimized_params = self._optimize_hyperparameters(X, y)
            
            # Update model with optimized parameters
            for param, value in optimized_params.items():
                setattr(self, param, value)
            
            # Recreate model with optimized parameters
            self.model = self._create_model()
        
        # Fit the model
        self.model.fit(X, y)
        
        # Store OOB score if available
        if self.oob_score and hasattr(self.model, 'oob_score_'):
            self.oob_score_value = self.model.oob_score_
            self.logger.info(f"Out-of-bag score: {self.oob_score_value:.3f}")
        
        # Analyze feature importance
        self._analyze_feature_importance()
        
        return self.model
    
    def _predict_proba_model(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities from Random Forest."""
        return self.model.predict_proba(X)
    
    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize hyperparameters using RandomizedSearch."""
        self.logger.info("Starting Random Forest hyperparameter optimization")
        
        # Define parameter distribution
        param_dist = {
            'n_estimators': [100, 200, 300, 500, 800, 1000],
            'max_depth': [None, 5, 10, 15, 20, 25],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 4, 8, 10],
            'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7],
            'bootstrap': [True, False]
        }
        
        # Create base model for search
        base_model = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            oob_score=True,
            class_weight=self.class_weight
        )
        
        # Perform randomized search
        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_dist,
            n_iter=50,  # Number of combinations to try
            cv=5,
            scoring='accuracy',
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=0
        )
        
        search.fit(X, y)
        
        self.logger.info(
            f"Hyperparameter optimization completed. Best score: {search.best_score_:.3f}"
        )
        
        return search.best_params_
    
    def _analyze_feature_importance(self):
        """Analyze and rank feature importance."""
        if not hasattr(self.model, 'feature_importances_'):
            return
        
        # Get feature importance
        importance_scores = self.model.feature_importances_
        
        # Create ranking
        if self.feature_names:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance_scores
            }).sort_values('importance', ascending=False)
            
            self.feature_importance_ranking = importance_df
            
            # Log top features
            self.logger.info("Top 10 most important features:")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                self.logger.info(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    def get_tree_info(self) -> Dict[str, Any]:
        """Get information about the forest structure."""
        if not self.is_trained:
            return {}
        
        # Analyze tree structure
        tree_depths = []
        tree_leaves = []
        
        for tree in self.model.estimators_:
            tree_depths.append(tree.tree_.max_depth)
            tree_leaves.append(tree.tree_.n_leaves)
        
        return {
            'n_estimators': self.model.n_estimators,
            'avg_tree_depth': np.mean(tree_depths),
            'max_tree_depth': np.max(tree_depths),
            'min_tree_depth': np.min(tree_depths),
            'avg_tree_leaves': np.mean(tree_leaves),
            'total_leaves': np.sum(tree_leaves),
            'oob_score': self.oob_score_value
        }
    
    def get_prediction_path(self, X_sample: pd.DataFrame, tree_idx: int = 0) -> List[Dict]:
        """Get decision path for a specific prediction."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if tree_idx >= len(self.model.estimators_):
            raise ValueError(f"Tree index {tree_idx} out of range")
        
        tree = self.model.estimators_[tree_idx]
        
        # Get decision path
        leaf_id = tree.apply(X_sample)
        feature_names = X_sample.columns.tolist()
        
        # Extract path information
        path_info = []
        
        try:
            decision_path = tree.decision_path(X_sample)
            
            for sample_id in range(X_sample.shape[0]):
                path = []
                node_ids = decision_path[sample_id].indices
                
                for node_id in node_ids[:-1]:  # Exclude leaf
                    feature_idx = tree.tree_.feature[node_id]
                    threshold = tree.tree_.threshold[node_id]
                    feature_name = feature_names[feature_idx]
                    
                    path.append({
                        'node': int(node_id),
                        'feature': feature_name,
                        'threshold': float(threshold),
                        'value': float(X_sample.iloc[sample_id, feature_idx])
                    })
                
                path_info.append(path)
        
        except Exception as e:
            self.logger.warning(f"Could not extract decision path: {e}")
        
        return path_info
    
    def analyze_model_stability(self, X_test: pd.DataFrame, n_runs: int = 10) -> Dict[str, float]:
        """Analyze model stability across multiple runs."""
        predictions = []
        
        for run in range(n_runs):
            # Create new model with different random state
            temp_model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=run,
                n_jobs=self.n_jobs
            )
            
            # Fit and predict
            temp_model.fit(self.training_data[0], self.training_data[1])  # Assumes training data stored
            pred = temp_model.predict_proba(X_test)[:, 1]
            predictions.append(pred)
        
        # Calculate stability metrics
        predictions_array = np.array(predictions)
        
        return {
            'mean_prediction_std': np.mean(np.std(predictions_array, axis=0)),
            'max_prediction_std': np.max(np.std(predictions_array, axis=0)),
            'prediction_correlation': np.corrcoef(predictions).mean(),
            'stability_score': 1 - np.mean(np.std(predictions_array, axis=0))  # Higher = more stable
        }