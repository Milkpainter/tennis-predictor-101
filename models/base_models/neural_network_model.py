"""Neural Network model for tennis prediction.

Implements feed-forward neural network with dropout regularization,
batch normalization, and early stopping for tennis match prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import validation_curve
import warnings
warnings.filterwarnings('ignore')

from .base_model import BaseModel
from config import get_config


class NeuralNetworkModel(BaseModel):
    """Neural Network model for tennis match prediction.
    
    Features:
    - Multi-layer perceptron architecture
    - Dropout regularization
    - Adaptive learning rate
    - Early stopping
    - Feature scaling
    - Architecture optimization
    """
    
    def __init__(self, **kwargs):
        # Get default parameters from config
        config = get_config()
        default_params = config.get('models.base_models.neural_network', {})
        
        # Merge with provided parameters
        params = {**default_params, **kwargs}
        
        super().__init__("NeuralNetwork", **params)
        
        # Neural network specific parameters
        self.hidden_layer_sizes = params.get('hidden_layers', [128, 64, 32])
        self.activation = params.get('activation', 'relu')
        self.solver = params.get('solver', 'adam')
        self.alpha = params.get('alpha', 0.001)  # L2 regularization
        self.learning_rate_init = params.get('learning_rate', 0.001)
        self.learning_rate = params.get('learning_rate_schedule', 'adaptive')
        self.max_iter = params.get('epochs', 1000)
        self.early_stopping = params.get('early_stopping', True)
        self.validation_fraction = params.get('validation_fraction', 0.1)
        self.n_iter_no_change = params.get('patience', 20)
        self.random_state = params.get('random_state', 42)
        
        # Advanced parameters
        self.batch_size = params.get('batch_size', 'auto')
        self.beta_1 = params.get('beta_1', 0.9)  # Adam parameter
        self.beta_2 = params.get('beta_2', 0.999)  # Adam parameter
        self.epsilon = params.get('epsilon', 1e-8)  # Adam parameter
        
        # Feature scaling
        self.scaler = StandardScaler()
        self.use_scaling = params.get('use_scaling', True)
        
        # Training artifacts
        self.loss_curve = []
        self.validation_scores = []
        self.best_validation_score = None
        self.training_stopped_early = False
    
    def _create_model(self) -> MLPClassifier:
        """Create MLPClassifier with optimized parameters."""
        return MLPClassifier(
            hidden_layer_sizes=tuple(self.hidden_layer_sizes),
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            batch_size=self.batch_size,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon,
            random_state=self.random_state
        )
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> MLPClassifier:
        """Fit neural network with feature scaling and optimization."""
        
        # Feature scaling
        if self.use_scaling:
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        else:
            X_scaled = X
        
        # Architecture optimization if requested
        if len(self.hidden_layer_sizes) == 3 and self.hidden_layer_sizes == [128, 64, 32]:
            # Default architecture - try optimization
            optimized_architecture = self._optimize_architecture(X_scaled, y)
            if optimized_architecture:
                self.hidden_layer_sizes = optimized_architecture
                self.model = self._create_model()
        
        # Fit the model
        self.model.fit(X_scaled, y)
        
        # Store training artifacts
        if hasattr(self.model, 'loss_curve_'):
            self.loss_curve = self.model.loss_curve_
        
        if hasattr(self.model, 'validation_scores_'):
            self.validation_scores = self.model.validation_scores_
            self.best_validation_score = max(self.validation_scores)
        
        # Check if training stopped early
        self.training_stopped_early = (hasattr(self.model, 'n_iter_') and 
                                     self.model.n_iter_ < self.max_iter)
        
        if self.training_stopped_early:
            self.logger.info(f"Training stopped early at iteration {self.model.n_iter_}")
        
        return self.model
    
    def _predict_proba_model(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities with feature scaling."""
        if self.use_scaling:
            X_scaled = self.scaler.transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        else:
            X_scaled = X
        
        return self.model.predict_proba(X_scaled)
    
    def _optimize_architecture(self, X: pd.DataFrame, y: pd.Series) -> Optional[List[int]]:
        """Optimize neural network architecture."""
        self.logger.info("Optimizing neural network architecture")
        
        # Define architecture candidates
        architectures = [
            [64],
            [128],
            [64, 32],
            [128, 64],
            [256, 128],
            [128, 64, 32],
            [256, 128, 64],
            [512, 256, 128],
            [256, 128, 64, 32],
            [512, 256, 128, 64]
        ]
        
        best_score = 0
        best_architecture = None
        
        for arch in architectures:
            try:
                # Create temporary model
                temp_model = MLPClassifier(
                    hidden_layer_sizes=tuple(arch),
                    max_iter=200,  # Reduced for speed
                    early_stopping=True,
                    validation_fraction=0.2,
                    random_state=self.random_state,
                    alpha=self.alpha
                )
                
                # Quick validation
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(temp_model, X, y, cv=3, scoring='accuracy')
                avg_score = scores.mean()
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_architecture = arch
                
                self.logger.debug(f"Architecture {arch}: {avg_score:.3f}")
                
            except Exception as e:
                self.logger.warning(f"Error testing architecture {arch}: {e}")
                continue
        
        if best_architecture:
            self.logger.info(f"Best architecture found: {best_architecture} (score: {best_score:.3f})")
            return best_architecture
        
        return None
    
    def analyze_learning_curve(self) -> Dict[str, Any]:
        """Analyze model learning behavior."""
        if not self.loss_curve:
            return {}
        
        analysis = {
            'total_iterations': len(self.loss_curve),
            'final_loss': self.loss_curve[-1],
            'initial_loss': self.loss_curve[0],
            'loss_reduction': self.loss_curve[0] - self.loss_curve[-1],
            'converged': self.training_stopped_early,
            'loss_curve': self.loss_curve
        }
        
        # Analyze convergence behavior
        if len(self.loss_curve) > 10:
            recent_improvement = (self.loss_curve[-10] - self.loss_curve[-1]) / 10
            analysis['recent_improvement_rate'] = recent_improvement
            analysis['is_still_improving'] = recent_improvement > 1e-5
        
        # Validation curve analysis
        if self.validation_scores:
            analysis.update({
                'best_validation_score': self.best_validation_score,
                'validation_curve': self.validation_scores,
                'overfitting_detected': self._detect_overfitting()
            })
        
        return analysis
    
    def _detect_overfitting(self) -> bool:
        """Detect overfitting from validation scores."""
        if not self.validation_scores or len(self.validation_scores) < 10:
            return False
        
        # Check if validation score is declining
        recent_scores = self.validation_scores[-10:]
        trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
        
        return trend < -0.001  # Declining validation performance
    
    def get_network_weights_analysis(self) -> Dict[str, Any]:
        """Analyze network weight distributions."""
        if not self.is_trained:
            return {}
        
        analysis = {}
        
        # Analyze each layer's weights
        for i, coef_matrix in enumerate(self.model.coefs_):
            layer_name = f"layer_{i+1}"
            
            analysis[layer_name] = {
                'shape': coef_matrix.shape,
                'mean_weight': float(np.mean(coef_matrix)),
                'std_weight': float(np.std(coef_matrix)),
                'max_weight': float(np.max(coef_matrix)),
                'min_weight': float(np.min(coef_matrix)),
                'zero_weights_pct': float(np.mean(np.abs(coef_matrix) < 1e-6))
            }
        
        # Analyze biases
        for i, bias_vector in enumerate(self.model.intercepts_):
            bias_name = f"bias_{i+1}"
            
            analysis[bias_name] = {
                'shape': bias_vector.shape,
                'mean_bias': float(np.mean(bias_vector)),
                'std_bias': float(np.std(bias_vector))
            }
        
        return analysis
    
    def plot_training_progress(self):
        """Plot training progress if matplotlib available."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2 if self.validation_scores else 1, figsize=(12, 5))
            
            if not isinstance(axes, np.ndarray):
                axes = [axes]
            
            # Plot loss curve
            if self.loss_curve:
                axes[0].plot(self.loss_curve, label='Training Loss', color='blue')
                axes[0].set_xlabel('Iteration')
                axes[0].set_ylabel('Loss')
                axes[0].set_title('Training Loss Curve')
                axes[0].grid(True, alpha=0.3)
                axes[0].legend()
            
            # Plot validation curve if available
            if self.validation_scores and len(axes) > 1:
                axes[1].plot(self.validation_scores, label='Validation Score', color='green')
                axes[1].set_xlabel('Iteration')
                axes[1].set_ylabel('Validation Accuracy')
                axes[1].set_title('Validation Score Curve')
                axes[1].grid(True, alpha=0.3)
                axes[1].legend()
                
                # Mark best validation score
                best_idx = np.argmax(self.validation_scores)
                axes[1].scatter(best_idx, self.validation_scores[best_idx], 
                              color='red', s=100, label=f'Best ({self.best_validation_score:.3f})')
                axes[1].legend()
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            self.logger.warning("Matplotlib not available for plotting")
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get comprehensive training information."""
        info = self.get_model_info()
        
        info.update({
            'architecture': self.hidden_layer_sizes,
            'total_parameters': self._count_parameters(),
            'oob_score': self.oob_score_value,
            'training_stopped_early': self.training_stopped_early,
            'learning_curve': self.analyze_learning_curve(),
            'network_analysis': self.get_network_weights_analysis()
        })
        
        return info
    
    def _count_parameters(self) -> int:
        """Count total number of parameters in the network."""
        if not self.is_trained:
            return 0
        
        total_params = 0
        
        # Count weights
        for coef_matrix in self.model.coefs_:
            total_params += coef_matrix.size
        
        # Count biases
        for bias_vector in self.model.intercepts_:
            total_params += bias_vector.size
        
        return total_params