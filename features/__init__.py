"""Tennis prediction feature engineering package.

Provides comprehensive feature engineering capabilities:
- ELO rating system with surface-specific adjustments
- Advanced momentum analysis (42 research-validated indicators)
- Surface-specific features and playing style analysis
- Environmental impact analysis (weather, altitude, court conditions)
"""

from .elo_rating import ELORatingSystem
from .momentum.momentum_analyzer import AdvancedMomentumAnalyzer, MomentumIndicators
from .surface import SurfaceSpecificFeatures, SurfaceAnalysis, PlayingStyle
from .environmental import EnvironmentalFeatures, EnvironmentalConditions, EnvironmentalImpact

# Alias for backward compatibility
MomentumAnalyzer = AdvancedMomentumAnalyzer

__all__ = [
    'ELORatingSystem',
    'AdvancedMomentumAnalyzer', 
    'MomentumAnalyzer',  # Backward compatibility
    'MomentumIndicators',
    'SurfaceSpecificFeatures',
    'SurfaceAnalysis', 
    'PlayingStyle',
    'EnvironmentalFeatures',
    'EnvironmentalConditions',
    'EnvironmentalImpact'
]