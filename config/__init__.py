"""Configuration management for Tennis Predictor 101."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for Tennis Predictor 101."""
    
    def __init__(self, config_path: str = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file. Defaults to config/config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._apply_environment_overrides()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides."""
        # API Keys from environment
        if 'ODDS_API_KEY' in os.environ:
            self._config.setdefault('api_keys', {})['odds_api'] = os.environ['ODDS_API_KEY']
        
        if 'WEATHER_API_KEY' in os.environ:
            self._config.setdefault('api_keys', {})['weather_api'] = os.environ['WEATHER_API_KEY']
        
        # Database URLs
        if 'DATABASE_URL' in os.environ:
            self._config['database']['primary']['url'] = os.environ['DATABASE_URL']
        
        if 'REDIS_URL' in os.environ:
            self._config['api']['caching']['redis_url'] = os.environ['REDIS_URL']
        
        # Environment-specific settings
        env = os.environ.get('TENNIS_ENV', 'development')
        self._config['system']['environment'] = env
        
        if env == 'production':
            self._config['system']['log_level'] = 'WARNING'
            self._config['api']['prediction_service']['workers'] = 8
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value
            default: Default value if key not found
        
        Returns:
            Configuration value
        
        Example:
            config.get('models.xgboost.n_estimators')
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value):
        """Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value
            value: Value to set
        """
        keys = key_path.split('.')
        config = self._config
        
        for key in keys[:-1]:
            if key not in config or not isinstance(config[key], dict):
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def save(self, path: str = None):
        """Save configuration to file.
        
        Args:
            path: Path to save configuration. Defaults to original config path
        """
        if path is None:
            path = self.config_path
        
        with open(path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
    
    @property
    def data_sources(self) -> Dict[str, Any]:
        """Get data sources configuration."""
        return self._config.get('data_sources', {})
    
    @property
    def models(self) -> Dict[str, Any]:
        """Get models configuration."""
        return self._config.get('models', {})
    
    @property
    def feature_engineering(self) -> Dict[str, Any]:
        """Get feature engineering configuration."""
        return self._config.get('feature_engineering', {})
    
    @property
    def validation(self) -> Dict[str, Any]:
        """Get validation configuration."""
        return self._config.get('validation', {})
    
    @property
    def api(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self._config.get('api', {})
    
    @property
    def monitoring(self) -> Dict[str, Any]:
        """Get monitoring configuration."""
        return self._config.get('monitoring', {})
    
    @property
    def market_analysis(self) -> Dict[str, Any]:
        """Get market analysis configuration."""
        return self._config.get('market_analysis', {})


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get global configuration instance."""
    return config


def reload_config():
    """Reload configuration from file."""
    global config
    config = Config()