"""Configuration Management for Tennis Predictor 101.

Centralized configuration system supporting:
- Environment-specific settings
- Model parameters
- API configurations
- Database connections
- External service integrations
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "tennis_predictor"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    log_level: str = "info"
    cors_origins: list = None
    rate_limit_per_minute: int = 100
    max_batch_size: int = 100


@dataclass
class ModelConfig:
    """Machine learning model configuration."""
    # Base models
    xgboost: Dict[str, Any] = None
    random_forest: Dict[str, Any] = None
    neural_network: Dict[str, Any] = None
    svm: Dict[str, Any] = None
    logistic_regression: Dict[str, Any] = None
    
    # Ensemble
    ensemble: Dict[str, Any] = None
    
    # Training
    training: Dict[str, Any] = None


@dataclass
class ELOConfig:
    """ELO rating system configuration."""
    initial_rating: int = 1500
    k_factor: int = 32
    surface_weights: Dict[str, float] = None
    tournament_multipliers: Dict[str, float] = None
    time_decay: float = 0.95


@dataclass
class MomentumConfig:
    """Momentum analysis configuration."""
    indicators_count: int = 42
    serving_indicators: int = 14
    return_indicators: int = 14
    rally_indicators: int = 14
    momentum_weights: Dict[str, Dict[str, float]] = None
    thresholds: Dict[str, float] = None


@dataclass
class BettingConfig:
    """Betting analysis configuration."""
    kelly_fraction_cap: float = 0.25
    min_edge_threshold: float = 0.02
    min_confidence_threshold: float = 0.6
    bankroll_size: float = 1000.0
    risk_tolerance: str = "moderate"


@dataclass
class DataConfig:
    """Data collection and processing configuration."""
    # Data sources
    jeff_sackmann_enabled: bool = True
    odds_api_enabled: bool = False
    real_time_data_enabled: bool = False
    
    # API keys
    odds_api_key: str = ""
    
    # Data storage
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    models_path: str = "models/saved"
    
    # Processing
    min_matches_per_player: int = 10
    max_match_age_days: int = 1095  # 3 years


class Config:
    """Main configuration class."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or self._find_config_file()
        self._config_data = self._load_config()
        
        # Initialize configuration sections
        self.database = DatabaseConfig(**self._config_data.get('database', {}))
        self.api = APIConfig(**self._config_data.get('api', {}))
        self.models = self._init_model_config()
        self.elo = self._init_elo_config()
        self.momentum = self._init_momentum_config()
        self.betting = BettingConfig(**self._config_data.get('betting', {}))
        self.data = DataConfig(**self._config_data.get('data', {}))
        
        # Environment
        self.environment = os.getenv('ENVIRONMENT', 'development')
        self.debug = self.environment == 'development'
    
    def _find_config_file(self) -> str:
        """Find configuration file."""
        
        # Check environment variable first
        if 'CONFIG_FILE' in os.environ:
            return os.environ['CONFIG_FILE']
        
        # Look for config files in order of preference
        config_files = [
            'config.yaml',
            'config.yml',
            'config/config.yaml',
            'config/config.yml',
            'tennis_predictor.yaml'
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                return config_file
        
        # Return default config file path
        return 'config.yaml'
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        
        config_path = Path(self.config_file)
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Warning: Could not load config file {config_path}: {e}")
                config_data = {}
        else:
            print(f"Warning: Config file {config_path} not found, using defaults")
            config_data = {}
        
        # Override with environment variables
        config_data = self._apply_env_overrides(config_data)
        
        return config_data
    
    def _apply_env_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        
        # Database overrides
        if 'DATABASE_URL' in os.environ:
            # Parse database URL (simplified)
            db_url = os.environ['DATABASE_URL']
            config_data.setdefault('database', {})['url'] = db_url
        
        # API overrides
        if 'PORT' in os.environ:
            config_data.setdefault('api', {})['port'] = int(os.environ['PORT'])
        
        if 'HOST' in os.environ:
            config_data.setdefault('api', {})['host'] = os.environ['HOST']
        
        # Betting overrides
        if 'ODDS_API_KEY' in os.environ:
            config_data.setdefault('data', {})['odds_api_key'] = os.environ['ODDS_API_KEY']
        
        return config_data
    
    def _init_model_config(self) -> ModelConfig:
        """Initialize model configuration with defaults."""
        
        models_config = self._config_data.get('models', {})
        
        # Default model configurations
        default_xgboost = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42
        }
        
        default_random_forest = {
            'n_estimators': 500,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42
        }
        
        default_neural_network = {
            'hidden_layers': [128, 64, 32],
            'activation': 'relu',
            'learning_rate': 0.001,
            'epochs': 1000,
            'early_stopping': True,
            'dropout_rate': 0.3
        }
        
        default_svm = {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'probability': True
        }
        
        default_logistic_regression = {
            'penalty': 'l2',
            'C': 1.0,
            'solver': 'liblinear',
            'max_iter': 1000
        }
        
        default_ensemble = {
            'meta_learner': 'logistic_regression',
            'cv_folds': 5,
            'use_probabilities': True,
            'calibration_method': 'sigmoid',
            'dynamic_weighting': True
        }
        
        default_training = {
            'test_size': 0.2,
            'validation_size': 0.1,
            'cv_folds': 5,
            'scoring': 'accuracy',
            'hyperparameter_optimization': True
        }
        
        return ModelConfig(
            xgboost=models_config.get('xgboost', default_xgboost),
            random_forest=models_config.get('random_forest', default_random_forest),
            neural_network=models_config.get('neural_network', default_neural_network),
            svm=models_config.get('svm', default_svm),
            logistic_regression=models_config.get('logistic_regression', default_logistic_regression),
            ensemble=models_config.get('ensemble', default_ensemble),
            training=models_config.get('training', default_training)
        )
    
    def _init_elo_config(self) -> ELOConfig:
        """Initialize ELO configuration with defaults."""
        
        elo_config = self._config_data.get('elo', {})
        
        default_surface_weights = {
            'clay': {'elo': 0.024, 'surface': 0.976},
            'hard': {'elo': 0.379, 'surface': 0.621},
            'grass': {'elo': 0.870, 'surface': 0.130}
        }
        
        default_tournament_multipliers = {
            'grand_slam': 1.5,
            'masters_1000': 1.3,
            'atp_500': 1.15,
            'atp_250': 1.0,
            'challenger': 0.8
        }
        
        return ELOConfig(
            initial_rating=elo_config.get('initial_rating', 1500),
            k_factor=elo_config.get('k_factor', 32),
            surface_weights=elo_config.get('surface_weights', default_surface_weights),
            tournament_multipliers=elo_config.get('tournament_multipliers', default_tournament_multipliers),
            time_decay=elo_config.get('time_decay', 0.95)
        )
    
    def _init_momentum_config(self) -> MomentumConfig:
        """Initialize momentum configuration with defaults."""
        
        momentum_config = self._config_data.get('momentum', {})
        
        default_weights = {
            'serving': {
                'break_points_saved': 3.0,
                'service_hold_rate': 2.5,
                'service_games_streak': 2.0,
                'first_serve_trend': 1.8,
                'pressure_point_serving': 1.8,
                'ace_rate_momentum': 1.5
            },
            'return': {
                'break_point_conversion': 3.0,
                'return_points_trend': 2.0,
                'pressure_return_performance': 1.8,
                'return_games_streak': 1.8,
                'break_attempt_frequency': 1.5
            },
            'rally': {
                'rally_win_percentage': 3.0,
                'pressure_rally_performance': 2.5,
                'court_position_dominance': 2.0,
                'groundstroke_winner_rate': 1.8,
                'unforced_error_control': 1.8
            }
        }
        
        default_thresholds = {
            'very_high': 0.85,
            'high': 0.70,
            'medium_high': 0.60,
            'medium': 0.50,
            'medium_low': 0.40,
            'low': 0.30,
            'very_low': 0.15
        }
        
        return MomentumConfig(
            momentum_weights=momentum_config.get('weights', default_weights),
            thresholds=momentum_config.get('thresholds', default_thresholds)
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key."""
        
        keys = key.split('.')
        value = self._config_data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        
        return {
            'database': self.database.__dict__,
            'api': self.api.__dict__,
            'models': {
                'xgboost': self.models.xgboost,
                'random_forest': self.models.random_forest,
                'neural_network': self.models.neural_network,
                'svm': self.models.svm,
                'logistic_regression': self.models.logistic_regression,
                'ensemble': self.models.ensemble,
                'training': self.models.training
            },
            'elo': self.elo.__dict__,
            'momentum': self.momentum.__dict__,
            'betting': self.betting.__dict__,
            'data': self.data.__dict__,
            'environment': self.environment,
            'debug': self.debug
        }


# Global configuration instance
_config_instance = None


@lru_cache(maxsize=1)
def get_config(config_file: Optional[str] = None) -> Config:
    """Get singleton configuration instance."""
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config(config_file)
    
    return _config_instance


def reload_config(config_file: Optional[str] = None) -> Config:
    """Reload configuration (useful for testing)."""
    global _config_instance
    
    # Clear cache
    get_config.cache_clear()
    _config_instance = None
    
    return get_config(config_file)