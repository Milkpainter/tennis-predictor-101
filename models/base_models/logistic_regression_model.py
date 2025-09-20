"""Logistic Regression model for tennis prediction.

Implements logistic regression with L1/L2 regularization,
feature selection, and comprehensive coefficient analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from .base_model import BaseModel
from config import get_config


class LogisticRegressionModel(BaseModel):
    """Logistic Regression model for tennis prediction.
    
    Features:
    - L1/L2/Elastic Net regularization
    - Feature selection integration
    - Coefficient analysis and interpretation
    - Multicollinearity detection
    - Statistical significance testing
    - Hyperparameter optimization
    """
    
    def __init__(self, **kwargs):
        # Get default parameters from config
        config = get_config()
        default_params = config.get('models.base_models.logistic_regression', {})
        
        # Merge with provided parameters
        params = {**default_params, **kwargs}
        
        super().__init__("LogisticRegression", **params)
        
        # Logistic regression specific parameters
        self.penalty = params.get('penalty', 'l2')
        self.C = params.get('C', 1.0)
        self.l1_ratio = params.get('l1_ratio', None)  # For elastic net
        self.solver = params.get('solver', 'liblinear')
        self.max_iter = params.get('max_iter', 1000)
        self.class_weight = params.get('class_weight', 'balanced')
        self.random_state = params.get('random_state', 42)
        
        # Advanced parameters
        self.feature_selection = params.get('feature_selection', True)
        self.feature_selection_method = params.get('feature_selection_method', 'rfe')
        self.n_features_to_select = params.get('n_features_to_select', 50)
        self.hyperparameter_tuning = params.get('hyperparameter_tuning', True)
        self.scaling = params.get('scaling', True)
        
        # Components
        self.scaler = StandardScaler() if self.scaling else None
        self.feature_selector = None
        
        # Analysis artifacts
        self.coefficient_analysis = {}
        self.feature_importance_stats = {}
        self.selected_features = []
        self.multicollinearity_analysis = {}
    
    def _create_model(self) -> LogisticRegression:
        """Create Logistic Regression classifier."""
        # Adjust solver based on penalty
        solver = self.solver
        if self.penalty == 'elasticnet' and solver != 'saga':
            solver = 'saga'
        elif self.penalty == 'l1' and solver not in ['liblinear', 'saga']:
            solver = 'liblinear'
        
        params = {
            'penalty': self.penalty,
            'C': self.C,
            'solver': solver,
            'max_iter': self.max_iter,
            'class_weight': self.class_weight,
            'random_state': self.random_state
        }
        
        # Add l1_ratio for elastic net
        if self.penalty == 'elasticnet':
            params['l1_ratio'] = self.l1_ratio or 0.5
        
        return LogisticRegression(**params)
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Fit logistic regression with preprocessing."""
        
        # Feature scaling
        if self.scaling:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        # Feature selection
        if self.feature_selection:
            X_selected = self._perform_feature_selection(X_scaled, y)
        else:
            X_selected = X_scaled
        
        # Hyperparameter optimization
        if self.hyperparameter_tuning:
            optimized_params = self._optimize_hyperparameters(X_selected, y)
            
            # Update model parameters
            for param, value in optimized_params.items():
                setattr(self, param, value)
            
            # Recreate model with optimized parameters
            self.model = self._create_model()
        
        # Fit the model
        self.model.fit(X_selected, y)
        
        # Perform coefficient analysis
        self._analyze_coefficients(X_selected)
        
        # Check for multicollinearity
        self._check_multicollinearity(X_selected)
        
        return self.model
    
    def _predict_proba_model(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities with preprocessing."""
        # Apply same preprocessing as training
        X_processed = self._preprocess_features(X)
        return self.model.predict_proba(X_processed)
    
    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing pipeline to features."""
        X_processed = X.copy()
        
        # Scaling
        if self.scaling and self.scaler:
            X_processed = pd.DataFrame(
                self.scaler.transform(X_processed),
                columns=X_processed.columns,
                index=X_processed.index
            )
        
        # Feature selection
        if self.feature_selector:
            X_processed = pd.DataFrame(
                self.feature_selector.transform(X_processed),
                columns=self.selected_features,
                index=X_processed.index
            )
        
        return X_processed
    
    def _perform_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Perform feature selection."""
        self.logger.info(f"Performing feature selection using {self.feature_selection_method}")
        
        if self.feature_selection_method == 'rfe':
            # Recursive Feature Elimination
            base_estimator = LogisticRegression(
                random_state=self.random_state,
                max_iter=500,
                class_weight=self.class_weight
            )
            
            self.feature_selector = RFE(
                base_estimator,
                n_features_to_select=min(self.n_features_to_select, X.shape[1]),
                step=1
            )
            
        elif self.feature_selection_method == 'univariate':
            # Univariate feature selection
            self.feature_selector = SelectKBest(
                score_func=f_classif,
                k=min(self.n_features_to_select, X.shape[1])
            )
        
        else:
            raise ValueError(f"Unknown feature selection method: {self.feature_selection_method}")
        
        # Fit and transform
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Store selected feature names
        if hasattr(self.feature_selector, 'get_support'):
            selected_mask = self.feature_selector.get_support()
            self.selected_features = X.columns[selected_mask].tolist()
        else:
            self.selected_features = X.columns.tolist()[:self.n_features_to_select]
        
        self.logger.info(f"Selected {len(self.selected_features)} features")
        
        return pd.DataFrame(
            X_selected,
            columns=self.selected_features,
            index=X.index
        )
    
    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize hyperparameters using GridSearch."""
        self.logger.info("Starting logistic regression hyperparameter optimization")
        
        # Define parameter grid based on current penalty
        if self.penalty == 'elasticnet':
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            }
        else:
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100, 1000],
                'penalty': ['l1', 'l2'] if self.solver in ['liblinear', 'saga'] else ['l2']
            }
        
        # Create base model
        base_model = LogisticRegression(
            solver=self.solver,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state
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
        
        self.logger.info(
            f"Hyperparameter optimization completed. Best score: {search.best_score_:.3f}"
        )
        
        return search.best_params_
    
    def _analyze_coefficients(self, X: pd.DataFrame):
        """Analyze model coefficients for interpretation."""
        if not hasattr(self.model, 'coef_'):
            return
        
        coefficients = self.model.coef_[0] if self.model.coef_.ndim > 1 else self.model.coef_
        intercept = self.model.intercept_[0] if hasattr(self.model, 'intercept_') else 0
        
        # Create coefficient analysis
        coef_data = []
        feature_names = self.selected_features if self.selected_features else X.columns.tolist()
        
        for i, (feature, coef) in enumerate(zip(feature_names, coefficients)):
            coef_data.append({
                'feature': feature,
                'coefficient': float(coef),
                'abs_coefficient': float(abs(coef)),
                'odds_ratio': float(np.exp(coef)),  # Odds ratio
                'rank': i + 1
            })
        
        # Sort by absolute coefficient value
        coef_data.sort(key=lambda x: x['abs_coefficient'], reverse=True)
        
        # Update ranks
        for i, data in enumerate(coef_data):
            data['rank'] = i + 1
        
        self.coefficient_analysis = {
            'intercept': float(intercept),
            'coefficients': coef_data,
            'top_positive_features': [c for c in coef_data if c['coefficient'] > 0][:10],
            'top_negative_features': [c for c in coef_data if c['coefficient'] < 0][:10],
            'coefficient_stats': {
                'mean_abs_coef': float(np.mean([abs(c['coefficient']) for c in coef_data])),
                'max_abs_coef': float(max([abs(c['coefficient']) for c in coef_data])),
                'n_positive_coef': len([c for c in coef_data if c['coefficient'] > 0]),
                'n_negative_coef': len([c for c in coef_data if c['coefficient'] < 0])
            }
        }
        
        # Log most important features
        self.logger.info("Top 5 most important features (by abs coefficient):")
        for i, coef_info in enumerate(coef_data[:5]):
            self.logger.info(
                f"  {i+1}. {coef_info['feature']}: {coef_info['coefficient']:.4f} "
                f"(OR: {coef_info['odds_ratio']:.3f})"
            )
    
    def _check_multicollinearity(self, X: pd.DataFrame):
        """Check for multicollinearity among features."""
        try:
            # Calculate correlation matrix
            correlation_matrix = X.corr()
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr = correlation_matrix.iloc[i, j]
                    if abs(corr) > 0.8:  # High correlation threshold
                        high_corr_pairs.append({
                            'feature1': correlation_matrix.columns[i],
                            'feature2': correlation_matrix.columns[j],
                            'correlation': float(corr)
                        })
            
            # Calculate VIF (simplified)
            from sklearn.linear_model import LinearRegression
            vif_scores = []
            
            for i, feature in enumerate(X.columns):
                try:
                    # Regress feature on all other features
                    y_temp = X.iloc[:, i]
                    X_temp = X.drop(X.columns[i], axis=1)
                    
                    lr = LinearRegression()
                    lr.fit(X_temp, y_temp)
                    r_squared = lr.score(X_temp, y_temp)
                    
                    # Calculate VIF
                    vif = 1 / (1 - r_squared) if r_squared < 0.99 else float('inf')
                    vif_scores.append({'feature': feature, 'vif': vif})
                    
                except:
                    vif_scores.append({'feature': feature, 'vif': float('inf')})
            
            self.multicollinearity_analysis = {
                'high_correlation_pairs': high_corr_pairs,
                'vif_scores': sorted(vif_scores, key=lambda x: x['vif'], reverse=True),
                'multicollinearity_detected': len(high_corr_pairs) > 0 or any(v['vif'] > 10 for v in vif_scores)
            }
            
            if self.multicollinearity_analysis['multicollinearity_detected']:
                self.logger.warning(f"Multicollinearity detected: {len(high_corr_pairs)} high correlations")
            
        except Exception as e:
            self.logger.warning(f"Could not perform multicollinearity analysis: {e}")
    
    def get_feature_significance(self) -> pd.DataFrame:
        """Get statistical significance of features (simplified)."""
        if not self.coefficient_analysis:
            return pd.DataFrame()
        
        # Create significance analysis (simplified without p-values)
        significance_data = []
        
        for coef_info in self.coefficient_analysis['coefficients']:
            # Simple heuristic for significance based on coefficient magnitude
            abs_coef = abs(coef_info['coefficient'])
            
            if abs_coef > 1.0:
                significance = 'high'
            elif abs_coef > 0.5:
                significance = 'medium'
            elif abs_coef > 0.1:
                significance = 'low'
            else:
                significance = 'very_low'
            
            significance_data.append({
                'feature': coef_info['feature'],
                'coefficient': coef_info['coefficient'],
                'odds_ratio': coef_info['odds_ratio'],
                'significance': significance,
                'impact_direction': 'positive' if coef_info['coefficient'] > 0 else 'negative'
            })
        
        return pd.DataFrame(significance_data)
    
    def interpret_prediction(self, X_sample: pd.DataFrame) -> Dict[str, Any]:
        """Interpret prediction by analyzing coefficient contributions."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Preprocess sample
        X_processed = self._preprocess_features(X_sample)
        
        if len(X_processed) != 1:
            raise ValueError("Can only interpret single prediction")
        
        # Get prediction and probability
        prediction = self.predict(X_sample)[0]
        probability = self.predict_proba(X_sample)[0, 1]
        
        # Calculate feature contributions
        coefficients = self.model.coef_[0] if self.model.coef_.ndim > 1 else self.model.coef_
        intercept = self.model.intercept_[0] if hasattr(self.model, 'intercept_') else 0
        
        contributions = []
        feature_names = self.selected_features if self.selected_features else X_processed.columns.tolist()
        
        for i, (feature, coef) in enumerate(zip(feature_names, coefficients)):
            feature_value = float(X_processed.iloc[0, i])
            contribution = coef * feature_value
            
            contributions.append({
                'feature': feature,
                'value': feature_value,
                'coefficient': float(coef),
                'contribution': float(contribution),
                'abs_contribution': float(abs(contribution))
            })
        
        # Sort by absolute contribution
        contributions.sort(key=lambda x: x['abs_contribution'], reverse=True)
        
        # Calculate total logit
        total_logit = intercept + sum(c['contribution'] for c in contributions)
        
        return {
            'prediction': int(prediction),
            'probability': float(probability),
            'total_logit': float(total_logit),
            'intercept': float(intercept),
            'feature_contributions': contributions[:10],  # Top 10
            'top_positive_contributors': [c for c in contributions if c['contribution'] > 0][:5],
            'top_negative_contributors': [c for c in contributions if c['contribution'] < 0][:5]
        }
    
    def get_odds_ratios_analysis(self) -> pd.DataFrame:
        """Get comprehensive odds ratios analysis."""
        if not self.coefficient_analysis:
            return pd.DataFrame()
        
        odds_data = []
        
        for coef_info in self.coefficient_analysis['coefficients']:
            odds_ratio = coef_info['odds_ratio']
            
            # Interpret odds ratio
            if odds_ratio > 2.0:
                interpretation = f"Very high positive effect (>{odds_ratio:.1f}x odds)"
            elif odds_ratio > 1.5:
                interpretation = f"High positive effect ({odds_ratio:.1f}x odds)"
            elif odds_ratio > 1.1:
                interpretation = f"Moderate positive effect ({odds_ratio:.1f}x odds)"
            elif odds_ratio > 0.9:
                interpretation = f"Neutral effect ({odds_ratio:.1f}x odds)"
            elif odds_ratio > 0.67:
                interpretation = f"Moderate negative effect ({odds_ratio:.1f}x odds)"
            elif odds_ratio > 0.5:
                interpretation = f"High negative effect ({odds_ratio:.1f}x odds)"
            else:
                interpretation = f"Very high negative effect (<{odds_ratio:.1f}x odds)"
            
            odds_data.append({
                'feature': coef_info['feature'],
                'coefficient': coef_info['coefficient'],
                'odds_ratio': odds_ratio,
                'interpretation': interpretation,
                'effect_size': 'large' if abs(odds_ratio - 1) > 0.5 else 'medium' if abs(odds_ratio - 1) > 0.2 else 'small'
            })
        
        return pd.DataFrame(odds_data)
    
    def get_logistic_info(self) -> Dict[str, Any]:
        """Get logistic regression specific information."""
        info = self.get_model_info()
        
        info.update({
            'penalty': self.penalty,
            'C': self.C,
            'solver': self.solver,
            'n_selected_features': len(self.selected_features),
            'selected_features': self.selected_features,
            'coefficient_analysis': self.coefficient_analysis,
            'multicollinearity_analysis': self.multicollinearity_analysis,
            'feature_selection_method': self.feature_selection_method if self.feature_selection else None
        })
        
        return info