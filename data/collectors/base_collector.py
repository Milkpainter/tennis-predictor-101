"""Base collector class for all data sources."""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from datetime import datetime, timedelta

from config import get_config


class BaseCollector(ABC):
    """Base class for all data collectors.
    
    Provides common functionality for HTTP requests, rate limiting,
    error handling, and data validation.
    """
    
    def __init__(self, name: str, rate_limit: float = 1.0):
        """Initialize base collector.
        
        Args:
            name: Name of the collector
            rate_limit: Minimum seconds between requests
        """
        self.name = name
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.config = get_config()
        self.logger = logging.getLogger(f"collector.{name}")
        
        # Setup HTTP session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default timeout
        self.timeout = self.config.get('optimization.timeout', 30)
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit:
            sleep_time = self.rate_limit - time_since_last
            self.logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, **kwargs) -> requests.Response:
        """Make HTTP request with rate limiting and error handling.
        
        Args:
            url: URL to request
            **kwargs: Additional arguments for requests
            
        Returns:
            HTTP response
            
        Raises:
            requests.RequestException: If request fails
        """
        self._rate_limit()
        
        # Set default timeout if not provided
        kwargs.setdefault('timeout', self.timeout)
        
        try:
            self.logger.debug(f"Making request to {url}")
            response = self.session.get(url, **kwargs)
            response.raise_for_status()
            return response
            
        except requests.RequestException as e:
            self.logger.error(f"Request failed for {url}: {e}")
            raise
    
    def _validate_data(self, data: pd.DataFrame, required_columns: List[str]) -> bool:
        """Validate collected data.
        
        Args:
            data: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            True if data is valid
            
        Raises:
            ValueError: If data validation fails
        """
        if data.empty:
            raise ValueError("Collected data is empty")
        
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for reasonable data size
        if len(data) == 0:
            raise ValueError("No data rows collected")
        
        self.logger.info(f"Data validation passed: {len(data)} rows, {len(data.columns)} columns")
        return True
    
    @abstractmethod
    def collect(self, **kwargs) -> pd.DataFrame:
        """Collect data from the source.
        
        Args:
            **kwargs: Source-specific parameters
            
        Returns:
            DataFrame with collected data
        """
        pass
    
    @abstractmethod
    def get_latest(self) -> pd.DataFrame:
        """Get the latest available data.
        
        Returns:
            DataFrame with latest data
        """
        pass
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on data source.
        
        Returns:
            Dictionary with health status
        """
        try:
            # Attempt to collect small sample
            sample = self.get_latest()
            return {
                'status': 'healthy',
                'last_updated': datetime.now().isoformat(),
                'sample_size': len(sample) if isinstance(sample, pd.DataFrame) else 0,
                'error': None
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'last_updated': datetime.now().isoformat(),
                'sample_size': 0,
                'error': str(e)
            }
    
    def get_data_freshness(self) -> timedelta:
        """Get age of the latest data.
        
        Returns:
            Time since last data update
        """
        try:
            latest_data = self.get_latest()
            if 'date' in latest_data.columns:
                latest_date = pd.to_datetime(latest_data['date']).max()
                return datetime.now() - latest_date.to_pydatetime()
            else:
                return timedelta(days=999)  # Unknown freshness
        except:
            return timedelta(days=999)  # Error getting freshness