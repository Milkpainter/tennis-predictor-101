"""Environmental Feature Engineering.

Implements weather, altitude, and court condition impacts on tennis performance.
Based on research showing 10°C = 2-3 mph ball speed change.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum

from config import get_config


class CourtType(Enum):
    OUTDOOR = "outdoor"
    INDOOR = "indoor"
    COVERED = "covered"  # Retractable roof


@dataclass
class EnvironmentalConditions:
    """Environmental condition data."""
    temperature: float  # Celsius
    humidity: float    # Percentage
    wind_speed: float  # km/h
    altitude: float    # meters
    court_type: CourtType
    air_pressure: Optional[float] = None  # hPa
    uv_index: Optional[float] = None
    

@dataclass
class EnvironmentalImpact:
    """Environmental impact analysis results."""
    ball_speed_factor: float
    bounce_height_factor: float
    player_fatigue_factor: float
    serve_advantage_adjustment: float
    rally_length_adjustment: float
    environmental_features: Dict[str, float]


class EnvironmentalFeatures:
    """Environmental impact analysis for tennis matches.
    
    Based on sports science research:
    - Temperature: 10°C change = 2-3 mph ball speed change
    - Humidity: >70% significantly affects performance
    - Altitude: Air density changes affect ball flight
    - Wind: Service game disruption patterns
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger("environmental_features")
        
        # Research-based environmental impact parameters
        self.temperature_coefficients = {
            'ball_speed_per_celsius': 0.02,      # 2% per degree C
            'bounce_height_per_celsius': 0.015,   # 1.5% per degree C
            'fatigue_threshold_celsius': 30.0,    # Fatigue increases above 30°C
            'optimal_temperature': 22.0          # Optimal playing temperature
        }
        
        self.humidity_thresholds = {
            'comfort_max': 60.0,      # Below 60% = comfortable
            'performance_decline': 70.0,  # Above 70% = performance decline
            'severe_impact': 80.0     # Above 80% = severe impact
        }
        
        self.altitude_impacts = {
            'air_density_factor': 0.000012,  # Air density change per meter
            'ball_flight_factor': 0.000008,  # Ball flight change per meter
            'significant_altitude': 1000.0   # Above 1000m = significant impact
        }
        
        self.wind_thresholds = {
            'calm': 10.0,           # <10 km/h = calm
            'moderate': 25.0,       # 10-25 km/h = moderate
            'strong': 40.0,         # 25-40 km/h = strong
            'severe': 60.0          # >40 km/h = severe
        }
    
    def analyze_environmental_impact(self, 
                                   conditions: EnvironmentalConditions,
                                   player1_adaptability: Dict[str, float] = None,
                                   player2_adaptability: Dict[str, float] = None) -> EnvironmentalImpact:
        """Analyze comprehensive environmental impact on match.
        
        Args:
            conditions: Current environmental conditions
            player1_adaptability: Player 1's adaptation to conditions
            player2_adaptability: Player 2's adaptation to conditions
            
        Returns:
            EnvironmentalImpact with all calculated factors
        """
        
        # Calculate individual impact factors
        ball_speed_factor = self._calculate_ball_speed_impact(conditions)
        bounce_height_factor = self._calculate_bounce_height_impact(conditions)
        fatigue_factor = self._calculate_fatigue_impact(conditions)
        serve_advantage_adj = self._calculate_serve_advantage_adjustment(conditions)
        rally_length_adj = self._calculate_rally_length_adjustment(conditions)
        
        # Generate features for ML models
        env_features = self._generate_environmental_features(conditions)
        
        # Add player-specific adaptability if available
        if player1_adaptability and player2_adaptability:
            env_features.update(
                self._calculate_player_adaptability_features(
                    conditions, player1_adaptability, player2_adaptability
                )
            )
        
        return EnvironmentalImpact(
            ball_speed_factor=ball_speed_factor,
            bounce_height_factor=bounce_height_factor,
            player_fatigue_factor=fatigue_factor,
            serve_advantage_adjustment=serve_advantage_adj,
            rally_length_adjustment=rally_length_adj,
            environmental_features=env_features
        )
    
    def _calculate_ball_speed_impact(self, conditions: EnvironmentalConditions) -> float:
        """Calculate ball speed impact based on environmental conditions.
        
        Research: 10°C change = 2-3 mph (3-5 km/h) ball speed change
        """
        
        # Temperature impact (primary factor)
        temp_diff = conditions.temperature - self.temperature_coefficients['optimal_temperature']
        temp_factor = 1.0 + (temp_diff * self.temperature_coefficients['ball_speed_per_celsius'])
        
        # Humidity impact (air density)
        humidity_factor = 1.0
        if conditions.humidity > self.humidity_thresholds['comfort_max']:
            humidity_excess = conditions.humidity - self.humidity_thresholds['comfort_max']
            humidity_factor = 1.0 - (humidity_excess * 0.001)  # Slight reduction
        
        # Altitude impact (air density)
        altitude_factor = 1.0
        if conditions.altitude > 0:
            # Higher altitude = less air resistance = faster ball
            altitude_factor = 1.0 + (conditions.altitude * self.altitude_impacts['air_density_factor'])
            altitude_factor = min(1.15, altitude_factor)  # Cap at 15% increase
        
        # Indoor vs outdoor
        court_factor = 1.0
        if conditions.court_type == CourtType.INDOOR:
            court_factor = 0.98  # Slightly slower indoors (less air circulation)
        
        # Combined impact
        total_factor = temp_factor * humidity_factor * altitude_factor * court_factor
        
        # Reasonable bounds
        return max(0.85, min(1.15, total_factor))
    
    def _calculate_bounce_height_impact(self, conditions: EnvironmentalConditions) -> float:
        """Calculate bounce height impact."""
        
        # Temperature impact on ball pressure and court surface
        temp_diff = conditions.temperature - self.temperature_coefficients['optimal_temperature']
        temp_factor = 1.0 + (temp_diff * self.temperature_coefficients['bounce_height_per_celsius'])
        
        # Humidity impact on court surface
        humidity_factor = 1.0
        if conditions.humidity > self.humidity_thresholds['performance_decline']:
            # High humidity can affect court surface, especially clay
            humidity_factor = 1.05  # Slightly higher bounce on humid courts
        
        # Altitude impact
        altitude_factor = 1.0
        if conditions.altitude > self.altitude_impacts['significant_altitude']:
            # Higher altitude = less air resistance = slightly higher bounce
            altitude_excess = conditions.altitude - self.altitude_impacts['significant_altitude']
            altitude_factor = 1.0 + (altitude_excess * 0.00001)
        
        total_factor = temp_factor * humidity_factor * altitude_factor
        
        return max(0.9, min(1.1, total_factor))
    
    def _calculate_fatigue_impact(self, conditions: EnvironmentalConditions) -> float:
        """Calculate player fatigue impact from environmental conditions."""
        
        fatigue_factor = 1.0  # 1.0 = no extra fatigue
        
        # Temperature-based fatigue
        if conditions.temperature > self.temperature_coefficients['fatigue_threshold_celsius']:
            temp_excess = conditions.temperature - self.temperature_coefficients['fatigue_threshold_celsius']
            fatigue_factor += temp_excess * 0.02  # 2% fatigue per degree above 30°C
        elif conditions.temperature < 10.0:  # Cold weather fatigue
            cold_factor = (10.0 - conditions.temperature) * 0.01
            fatigue_factor += cold_factor
        
        # Humidity-based fatigue
        if conditions.humidity > self.humidity_thresholds['performance_decline']:
            humidity_excess = conditions.humidity - self.humidity_thresholds['performance_decline']
            fatigue_factor += humidity_excess * 0.005  # 0.5% fatigue per % humidity above 70%
        
        # Wind fatigue (mental and physical)
        if conditions.wind_speed > self.wind_thresholds['moderate']:
            wind_excess = conditions.wind_speed - self.wind_thresholds['moderate']
            fatigue_factor += wind_excess * 0.002  # Mental fatigue from wind
        
        # Indoor advantage (no weather stress)
        if conditions.court_type == CourtType.INDOOR:
            fatigue_factor *= 0.95  # 5% less fatigue indoors
        
        # Cap maximum fatigue impact
        return min(1.5, fatigue_factor)
    
    def _calculate_serve_advantage_adjustment(self, conditions: EnvironmentalConditions) -> float:
        """Calculate how conditions affect serve advantage."""
        
        serve_adjustment = 1.0  # 1.0 = no change
        
        # Wind impact (most significant for serving)
        if conditions.wind_speed > self.wind_thresholds['calm']:
            if conditions.wind_speed < self.wind_thresholds['moderate']:
                serve_adjustment = 0.95  # Slight disadvantage
            elif conditions.wind_speed < self.wind_thresholds['strong']:
                serve_adjustment = 0.90  # Moderate disadvantage
            else:
                serve_adjustment = 0.85  # Strong disadvantage
        
        # Temperature impact on ball speed (affects serve)
        if conditions.temperature > 25.0:  # Hot conditions
            serve_adjustment *= 1.02  # Faster ball = slight serve advantage
        elif conditions.temperature < 15.0:  # Cold conditions
            serve_adjustment *= 0.98  # Slower ball = slight disadvantage
        
        # Altitude impact
        if conditions.altitude > self.altitude_impacts['significant_altitude']:
            serve_adjustment *= 1.03  # Less air resistance = serve advantage
        
        # Indoor conditions (no wind)
        if conditions.court_type == CourtType.INDOOR:
            serve_adjustment *= 1.02  # Consistent conditions favor serving
        
        return max(0.8, min(1.2, serve_adjustment))
    
    def _calculate_rally_length_adjustment(self, conditions: EnvironmentalConditions) -> float:
        """Calculate how conditions affect average rally length."""
        
        rally_adjustment = 1.0  # 1.0 = no change
        
        # Wind impact (longer rallies in windy conditions)
        if conditions.wind_speed > self.wind_thresholds['moderate']:
            wind_factor = min(0.2, (conditions.wind_speed - self.wind_thresholds['moderate']) * 0.005)
            rally_adjustment += wind_factor
        
        # Temperature impact
        if conditions.temperature > self.temperature_coefficients['fatigue_threshold_celsius']:
            # Hot conditions = shorter rallies due to fatigue
            rally_adjustment -= 0.1
        elif conditions.temperature < 10.0:
            # Cold conditions = longer rallies (more cautious play)
            rally_adjustment += 0.05
        
        # Humidity impact
        if conditions.humidity > self.humidity_thresholds['severe_impact']:
            rally_adjustment -= 0.08  # High humidity = shorter rallies
        
        # Altitude impact (faster ball = shorter rallies)
        if conditions.altitude > self.altitude_impacts['significant_altitude']:
            rally_adjustment -= 0.05
        
        return max(0.8, min(1.3, rally_adjustment))
    
    def _generate_environmental_features(self, conditions: EnvironmentalConditions) -> Dict[str, float]:
        """Generate environmental features for ML models."""
        
        features = {
            # Raw environmental values (normalized)
            'temperature': conditions.temperature / 35.0,  # Normalize to 0-1 range
            'temperature_squared': (conditions.temperature / 35.0) ** 2,
            'humidity': conditions.humidity / 100.0,
            'wind_speed': min(1.0, conditions.wind_speed / 50.0),  # Cap at 50 km/h
            'altitude_km': conditions.altitude / 1000.0,
            
            # Categorical court type
            'court_indoor': 1.0 if conditions.court_type == CourtType.INDOOR else 0.0,
            'court_outdoor': 1.0 if conditions.court_type == CourtType.OUTDOOR else 0.0,
            'court_covered': 1.0 if conditions.court_type == CourtType.COVERED else 0.0,
            
            # Temperature categories
            'temp_very_hot': 1.0 if conditions.temperature > 32.0 else 0.0,
            'temp_hot': 1.0 if 27.0 < conditions.temperature <= 32.0 else 0.0,
            'temp_warm': 1.0 if 22.0 < conditions.temperature <= 27.0 else 0.0,
            'temp_cool': 1.0 if 15.0 < conditions.temperature <= 22.0 else 0.0,
            'temp_cold': 1.0 if conditions.temperature <= 15.0 else 0.0,
            
            # Humidity categories
            'humidity_high': 1.0 if conditions.humidity > self.humidity_thresholds['severe_impact'] else 0.0,
            'humidity_moderate': 1.0 if self.humidity_thresholds['performance_decline'] < conditions.humidity <= self.humidity_thresholds['severe_impact'] else 0.0,
            'humidity_comfortable': 1.0 if conditions.humidity <= self.humidity_thresholds['comfort_max'] else 0.0,
            
            # Wind categories
            'wind_calm': 1.0 if conditions.wind_speed <= self.wind_thresholds['calm'] else 0.0,
            'wind_moderate': 1.0 if self.wind_thresholds['calm'] < conditions.wind_speed <= self.wind_thresholds['moderate'] else 0.0,
            'wind_strong': 1.0 if self.wind_thresholds['moderate'] < conditions.wind_speed <= self.wind_thresholds['strong'] else 0.0,
            'wind_severe': 1.0 if conditions.wind_speed > self.wind_thresholds['strong'] else 0.0,
            
            # Altitude categories
            'altitude_sea_level': 1.0 if conditions.altitude < 300.0 else 0.0,
            'altitude_elevated': 1.0 if 300.0 <= conditions.altitude < 1000.0 else 0.0,
            'altitude_high': 1.0 if conditions.altitude >= 1000.0 else 0.0,
            
            # Combined stress factors
            'heat_humidity_stress': self._calculate_heat_humidity_index(conditions.temperature, conditions.humidity),
            'wind_temperature_factor': (conditions.wind_speed / 50.0) * (abs(conditions.temperature - 22.0) / 20.0),
            
            # Calculated impact factors
            'ball_speed_environmental_factor': self._calculate_ball_speed_impact(conditions),
            'bounce_height_environmental_factor': self._calculate_bounce_height_impact(conditions),
            'fatigue_environmental_factor': self._calculate_fatigue_impact(conditions),
            'serve_advantage_environmental': self._calculate_serve_advantage_adjustment(conditions),
            'rally_length_environmental': self._calculate_rally_length_adjustment(conditions)
        }
        
        # Add air pressure features if available
        if conditions.air_pressure is not None:
            features.update({
                'air_pressure': conditions.air_pressure / 1013.25,  # Normalize to sea level
                'low_pressure': 1.0 if conditions.air_pressure < 1000.0 else 0.0,
                'high_pressure': 1.0 if conditions.air_pressure > 1025.0 else 0.0
            })
        
        return features
    
    def _calculate_heat_humidity_index(self, temperature: float, humidity: float) -> float:
        """Calculate heat-humidity stress index."""
        
        # Simplified heat index calculation
        if temperature < 25.0:
            return 0.0
        
        temp_factor = (temperature - 25.0) / 15.0  # 0-1 scale for temps 25-40°C
        humidity_factor = max(0.0, (humidity - 40.0) / 60.0)  # 0-1 scale for humidity 40-100%
        
        # Combined heat-humidity stress
        combined_stress = (temp_factor + humidity_factor) / 2.0
        
        return min(1.0, combined_stress)
    
    def _calculate_player_adaptability_features(self, 
                                              conditions: EnvironmentalConditions,
                                              player1_adapt: Dict[str, float],
                                              player2_adapt: Dict[str, float]) -> Dict[str, float]:
        """Calculate player-specific environmental adaptability features."""
        
        adaptability_features = {}
        
        # Heat adaptability
        if conditions.temperature > 28.0:
            heat_adapt_diff = player1_adapt.get('heat_tolerance', 0.5) - player2_adapt.get('heat_tolerance', 0.5)
            adaptability_features['heat_adaptability_advantage'] = heat_adapt_diff
        
        # Cold adaptability
        if conditions.temperature < 15.0:
            cold_adapt_diff = player1_adapt.get('cold_tolerance', 0.5) - player2_adapt.get('cold_tolerance', 0.5)
            adaptability_features['cold_adaptability_advantage'] = cold_adapt_diff
        
        # Wind adaptability
        if conditions.wind_speed > self.wind_thresholds['moderate']:
            wind_adapt_diff = player1_adapt.get('wind_tolerance', 0.5) - player2_adapt.get('wind_tolerance', 0.5)
            adaptability_features['wind_adaptability_advantage'] = wind_adapt_diff
        
        # Humidity adaptability
        if conditions.humidity > self.humidity_thresholds['performance_decline']:
            humid_adapt_diff = player1_adapt.get('humidity_tolerance', 0.5) - player2_adapt.get('humidity_tolerance', 0.5)
            adaptability_features['humidity_adaptability_advantage'] = humid_adapt_diff
        
        # Indoor/outdoor preference
        indoor_pref_diff = (player1_adapt.get('indoor_preference', 0.5) - 
                           player2_adapt.get('indoor_preference', 0.5))
        
        if conditions.court_type == CourtType.INDOOR:
            adaptability_features['court_type_advantage'] = indoor_pref_diff
        else:
            adaptability_features['court_type_advantage'] = -indoor_pref_diff
        
        return adaptability_features
    
    def get_environmental_summary(self, conditions: EnvironmentalConditions) -> str:
        """Get human-readable environmental conditions summary."""
        
        summary_parts = []
        
        # Temperature
        if conditions.temperature > 32.0:
            summary_parts.append("Very hot conditions")
        elif conditions.temperature > 27.0:
            summary_parts.append("Hot conditions")
        elif conditions.temperature < 15.0:
            summary_parts.append("Cold conditions")
        
        # Humidity
        if conditions.humidity > self.humidity_thresholds['severe_impact']:
            summary_parts.append("very high humidity")
        elif conditions.humidity > self.humidity_thresholds['performance_decline']:
            summary_parts.append("high humidity")
        
        # Wind
        if conditions.wind_speed > self.wind_thresholds['strong']:
            summary_parts.append("strong wind")
        elif conditions.wind_speed > self.wind_thresholds['moderate']:
            summary_parts.append("moderate wind")
        
        # Altitude
        if conditions.altitude > 1500.0:
            summary_parts.append("high altitude")
        elif conditions.altitude > 1000.0:
            summary_parts.append("elevated altitude")
        
        # Court type
        if conditions.court_type == CourtType.INDOOR:
            summary_parts.append("indoor court")
        
        if not summary_parts:
            return "Favorable playing conditions"
        
        return ", ".join(summary_parts).capitalize() + " expected to impact play"