"""Cross-validation strategies for tennis prediction models.

Implements research-validated cross-validation approaches including
tournament-based, time-series, and surface-specific validation.
"""

import numpy as np
import pandas as pd
from typing import Generator, Tuple, List, Dict, Any
from sklearn.model_selection import BaseCrossValidator
from datetime import datetime, timedelta
import logging

from config import get_config


class TournamentBasedCV(BaseCrossValidator):
    """Tournament-based cross-validation.
    
    Ensures no data leakage between tournaments by using
    leave-one-tournament-out or leave-multiple-tournaments-out validation.
    """
    
    def __init__(self, n_splits: int = 5, tournament_col: str = 'tournament_id',
                 min_tournaments_per_fold: int = 1):
        """Initialize tournament-based CV.
        
        Args:
            n_splits: Number of CV splits
            tournament_col: Column containing tournament identifiers
            min_tournaments_per_fold: Minimum tournaments per validation fold
        """
        self.n_splits = n_splits
        self.tournament_col = tournament_col
        self.min_tournaments_per_fold = min_tournaments_per_fold
        self.logger = logging.getLogger("validation.tournament_cv")
    
    def split(self, X: pd.DataFrame, y: pd.Series = None, groups: pd.Series = None) -> Generator:
        """Generate train/validation splits.
        
        Args:
            X: Features DataFrame
            y: Target Series (unused)
            groups: Tournament groups (unused, uses tournament_col from X)
            
        Yields:
            Tuple of (train_indices, validation_indices)
        """
        if self.tournament_col not in X.columns:
            raise ValueError(f"Tournament column '{self.tournament_col}' not found in data")
        
        # Get unique tournaments
        tournaments = X[self.tournament_col].unique()
        n_tournaments = len(tournaments)
        
        if n_tournaments < self.n_splits:
            raise ValueError(f"Not enough tournaments ({n_tournaments}) for {self.n_splits} splits")
        
        # Calculate tournaments per fold
        tournaments_per_fold = max(self.min_tournaments_per_fold, 
                                 n_tournaments // self.n_splits)
        
        self.logger.info(f"Using {tournaments_per_fold} tournaments per validation fold")
        
        # Shuffle tournaments for random splits
        np.random.shuffle(tournaments)
        
        for i in range(self.n_splits):
            # Select validation tournaments
            start_idx = i * tournaments_per_fold
            end_idx = min(start_idx + tournaments_per_fold, n_tournaments)
            
            if start_idx >= n_tournaments:
                break
                
            val_tournaments = tournaments[start_idx:end_idx]
            train_tournaments = tournaments[~np.isin(tournaments, val_tournaments)]
            
            # Get indices
            val_mask = X[self.tournament_col].isin(val_tournaments)
            train_mask = X[self.tournament_col].isin(train_tournaments)
            
            train_indices = X.index[train_mask].tolist()
            val_indices = X.index[val_mask].tolist()
            
            if len(train_indices) == 0 or len(val_indices) == 0:
                continue
            
            self.logger.debug(
                f"Fold {i+1}: {len(train_tournaments)} train tournaments, "
                f"{len(val_tournaments)} val tournaments, "
                f"{len(train_indices)} train samples, {len(val_indices)} val samples"
            )
            
            yield train_indices, val_indices
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Get number of splits."""
        return self.n_splits


class TimeSeriesCV(BaseCrossValidator):
    """Time-series cross-validation for tennis data.
    
    Implements expanding window validation that respects temporal order
    and prevents future data leakage.
    """
    
    def __init__(self, n_splits: int = 5, date_col: str = 'date',
                 min_train_size: int = 1000, gap_days: int = 0):
        """Initialize time-series CV.
        
        Args:
            n_splits: Number of CV splits
            date_col: Column containing date information
            min_train_size: Minimum training samples
            gap_days: Gap between train and validation (to prevent leakage)
        """
        self.n_splits = n_splits
        self.date_col = date_col
        self.min_train_size = min_train_size
        self.gap_days = gap_days
        self.logger = logging.getLogger("validation.timeseries_cv")
    
    def split(self, X: pd.DataFrame, y: pd.Series = None, groups: pd.Series = None) -> Generator:
        """Generate time-series train/validation splits.
        
        Args:
            X: Features DataFrame with date column
            y: Target Series (unused)
            groups: Groups (unused)
            
        Yields:
            Tuple of (train_indices, validation_indices)
        """
        if self.date_col not in X.columns:
            raise ValueError(f"Date column '{self.date_col}' not found in data")
        
        # Sort by date
        X_sorted = X.sort_values(self.date_col).copy()
        
        # Convert date column if needed
        if not pd.api.types.is_datetime64_any_dtype(X_sorted[self.date_col]):
            X_sorted[self.date_col] = pd.to_datetime(X_sorted[self.date_col])
        
        total_samples = len(X_sorted)
        
        if total_samples < self.min_train_size + 100:  # Need some validation samples
            raise ValueError(f"Not enough samples ({total_samples}) for time-series CV")
        
        # Calculate split points
        for i in range(self.n_splits):
            # Expanding window: training set grows with each split
            train_end_ratio = 0.5 + (i + 1) * 0.1  # Start at 50%, grow by 10% each split
            train_end_idx = int(total_samples * train_end_ratio)
            
            if train_end_idx < self.min_train_size:
                continue
                
            if train_end_idx >= total_samples - 50:  # Need at least 50 validation samples
                break
            
            # Apply gap if specified
            if self.gap_days > 0:
                train_end_date = X_sorted.iloc[train_end_idx][self.date_col]
                gap_end_date = train_end_date + timedelta(days=self.gap_days)
                
                # Find validation start after gap
                val_start_mask = X_sorted[self.date_col] > gap_end_date
                val_start_indices = X_sorted.index[val_start_mask]
                
                if len(val_start_indices) == 0:
                    continue
                    
                val_start_idx = val_start_indices[0]
                val_start_pos = X_sorted.index.get_loc(val_start_idx)
            else:
                val_start_pos = train_end_idx
            
            # Define validation end (use remaining data or fixed window)
            val_end_pos = min(val_start_pos + 500, total_samples)  # Max 500 validation samples
            
            if val_end_pos <= val_start_pos + 50:  # Need at least 50 validation samples
                continue
            
            # Get indices
            train_indices = X_sorted.iloc[:train_end_idx].index.tolist()
            val_indices = X_sorted.iloc[val_start_pos:val_end_pos].index.tolist()
            
            self.logger.debug(
                f"Fold {i+1}: Train samples: {len(train_indices)}, "
                f"Val samples: {len(val_indices)}, "
                f"Train end: {X_sorted.iloc[train_end_idx-1][self.date_col].date()}, "
                f"Val start: {X_sorted.iloc[val_start_pos][self.date_col].date()}"
            )
            
            yield train_indices, val_indices
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Get number of splits."""
        return self.n_splits


class SurfaceSpecificCV(BaseCrossValidator):
    """Surface-specific cross-validation.
    
    Validates models separately for each court surface (clay, hard, grass)
    to account for surface-specific performance patterns.
    """
    
    def __init__(self, n_splits: int = 5, surface_col: str = 'surface',
                 target_surface: str = None):
        """Initialize surface-specific CV.
        
        Args:
            n_splits: Number of CV splits
            surface_col: Column containing surface information
            target_surface: Specific surface to validate on (None for all)
        """
        self.n_splits = n_splits
        self.surface_col = surface_col
        self.target_surface = target_surface
        self.logger = logging.getLogger("validation.surface_cv")
    
    def split(self, X: pd.DataFrame, y: pd.Series = None, groups: pd.Series = None) -> Generator:
        """Generate surface-specific train/validation splits.
        
        Args:
            X: Features DataFrame with surface column
            y: Target Series
            groups: Groups (unused)
            
        Yields:
            Tuple of (train_indices, validation_indices)
        """
        if self.surface_col not in X.columns:
            raise ValueError(f"Surface column '{self.surface_col}' not found in data")
        
        surfaces = [self.target_surface] if self.target_surface else X[self.surface_col].unique()
        
        for surface in surfaces:
            if pd.isna(surface):
                continue
                
            self.logger.info(f"Generating CV splits for {surface} surface")
            
            # Filter data for this surface
            surface_mask = X[self.surface_col] == surface
            surface_indices = X.index[surface_mask].tolist()
            
            if len(surface_indices) < 100:  # Need minimum samples
                self.logger.warning(f"Not enough {surface} samples ({len(surface_indices)})")
                continue
            
            # Use stratified splits within surface
            from sklearn.model_selection import StratifiedKFold
            
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            
            surface_X = X.loc[surface_indices]
            surface_y = y.loc[surface_indices] if y is not None else None
            
            if surface_y is None:
                # Use regular KFold if no target
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
                splits = kf.split(surface_X)
            else:
                splits = skf.split(surface_X, surface_y)
            
            for fold, (train_idx, val_idx) in enumerate(splits):
                # Convert to original indices
                train_indices = [surface_indices[i] for i in train_idx]
                val_indices = [surface_indices[i] for i in val_idx]
                
                self.logger.debug(
                    f"{surface} Fold {fold+1}: {len(train_indices)} train, {len(val_indices)} val"
                )
                
                yield train_indices, val_indices
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Get number of splits."""
        if self.target_surface:
            return self.n_splits
        else:
            # Multiple surfaces, each with n_splits
            if X is not None and self.surface_col in X.columns:
                n_surfaces = X[self.surface_col].nunique()
                return self.n_splits * n_surfaces
            return self.n_splits


def select_cv_strategy(strategy: str, **kwargs) -> BaseCrossValidator:
    """Factory function to select cross-validation strategy.
    
    Args:
        strategy: CV strategy name
        **kwargs: Strategy-specific parameters
        
    Returns:
        Cross-validation object
    """
    if strategy == 'tournament_based':
        return TournamentBasedCV(**kwargs)
    elif strategy == 'time_series':
        return TimeSeriesCV(**kwargs)
    elif strategy == 'surface_specific':
        return SurfaceSpecificCV(**kwargs)
    elif strategy == 'stratified':
        from sklearn.model_selection import StratifiedKFold
        return StratifiedKFold(**kwargs)
    else:
        raise ValueError(f"Unknown CV strategy: {strategy}")