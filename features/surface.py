"""Surface-Specific Feature Engineering.

Implements advanced surface analysis including player style matching,
historical surface performance, and court-specific adjustments.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

from config import get_config


class SurfaceType(Enum):
    CLAY = "Clay"
    HARD = "Hard"
    GRASS = "Grass"
    CARPET = "Carpet"


class PlayingStyle(Enum):
    AGGRESSIVE_BASELINER = "aggressive_baseliner"
    DEFENSIVE_BASELINER = "defensive_baseliner"
    ALL_COURT = "all_court"
    SERVE_AND_VOLLEY = "serve_and_volley"
    BIG_SERVER = "big_server"
    COUNTER_PUNCHER = "counter_puncher"


@dataclass
class SurfaceAnalysis:
    """Surface-specific analysis results."""
    surface_advantage: Optional[str]
    player1_surface_rating: float
    player2_surface_rating: float
    style_matchup_advantage: Optional[str]
    historical_performance: Dict[str, Any]
    surface_specific_features: Dict[str, float]


class SurfaceSpecificFeatures:
    """Advanced surface-specific feature engineering.
    
    Features:
    - Historical surface performance tracking
    - Playing style classification and matchups
    - Surface transition analysis
    - Court-specific adjustments
    - Weather and condition impacts
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger("surface_features")
        
        # Surface characteristics
        self.surface_characteristics = {
            SurfaceType.CLAY: {
                'speed_index': 0.3,      # Slower surface
                'bounce_height': 0.8,    # High bounce
                'slide_factor': 0.9,     # High sliding
                'rally_length_multiplier': 1.4,  # Longer rallies
                'serve_advantage': 0.6,  # Reduced serve advantage
                'defensive_bonus': 1.3   # Defensive play bonus
            },
            SurfaceType.HARD: {
                'speed_index': 0.6,      # Medium-fast surface
                'bounce_height': 0.6,    # Medium bounce
                'slide_factor': 0.2,     # Limited sliding
                'rally_length_multiplier': 1.0,  # Neutral rallies
                'serve_advantage': 0.8,  # Balanced serve advantage
                'defensive_bonus': 1.0   # Neutral defensive bonus
            },
            SurfaceType.GRASS: {
                'speed_index': 0.9,      # Fastest surface
                'bounce_height': 0.3,    # Low bounce
                'slide_factor': 0.1,     # Minimal sliding
                'rally_length_multiplier': 0.7,  # Shorter rallies
                'serve_advantage': 1.2,  # High serve advantage
                'defensive_bonus': 0.7   # Reduced defensive effectiveness
            }
        }
        
        # Style advantages by surface
        self.style_surface_advantages = {
            SurfaceType.CLAY: {
                PlayingStyle.DEFENSIVE_BASELINER: 1.3,
                PlayingStyle.COUNTER_PUNCHER: 1.2,
                PlayingStyle.AGGRESSIVE_BASELINER: 1.0,
                PlayingStyle.ALL_COURT: 0.9,
                PlayingStyle.BIG_SERVER: 0.7,
                PlayingStyle.SERVE_AND_VOLLEY: 0.6
            },
            SurfaceType.HARD: {
                PlayingStyle.ALL_COURT: 1.1,
                PlayingStyle.AGGRESSIVE_BASELINER: 1.0,
                PlayingStyle.BIG_SERVER: 1.0,
                PlayingStyle.DEFENSIVE_BASELINER: 1.0,
                PlayingStyle.COUNTER_PUNCHER: 0.9,
                PlayingStyle.SERVE_AND_VOLLEY: 0.8
            },
            SurfaceType.GRASS: {
                PlayingStyle.SERVE_AND_VOLLEY: 1.4,
                PlayingStyle.BIG_SERVER: 1.3,
                PlayingStyle.ALL_COURT: 1.1,
                PlayingStyle.AGGRESSIVE_BASELINER: 0.9,
                PlayingStyle.COUNTER_PUNCHER: 0.7,
                PlayingStyle.DEFENSIVE_BASELINER: 0.6
            }
        }
    
    def analyze_surface_matchup(self, 
                              player1_id: str,
                              player2_id: str,
                              surface: str,
                              historical_data: pd.DataFrame) -> SurfaceAnalysis:
        """Comprehensive surface matchup analysis."""
        
        surface_type = SurfaceType(surface)
        
        # Calculate surface-specific ratings
        p1_surface_rating = self._calculate_surface_rating(player1_id, surface_type, historical_data)
        p2_surface_rating = self._calculate_surface_rating(player2_id, surface_type, historical_data)
        
        # Determine surface advantage
        surface_advantage = None
        if abs(p1_surface_rating - p2_surface_rating) > 0.1:
            surface_advantage = 'player1' if p1_surface_rating > p2_surface_rating else 'player2'
        
        # Analyze playing styles
        p1_style = self._classify_playing_style(player1_id, historical_data)
        p2_style = self._classify_playing_style(player2_id, historical_data)
        
        # Style matchup analysis
        style_advantage = self._analyze_style_matchup(p1_style, p2_style, surface_type)
        
        # Historical performance
        historical_perf = self._analyze_historical_surface_performance(
            player1_id, player2_id, surface_type, historical_data
        )
        
        # Generate surface-specific features
        surface_features = self._generate_surface_features(
            player1_id, player2_id, surface_type, historical_data
        )
        
        return SurfaceAnalysis(
            surface_advantage=surface_advantage,
            player1_surface_rating=p1_surface_rating,
            player2_surface_rating=p2_surface_rating,
            style_matchup_advantage=style_advantage,
            historical_performance=historical_perf,
            surface_specific_features=surface_features
        )
    
    def _calculate_surface_rating(self, player_id: str, surface: SurfaceType, 
                                historical_data: pd.DataFrame) -> float:
        """Calculate player's surface-specific rating."""
        
        # Filter for player's matches on this surface
        player_matches = historical_data[
            ((historical_data['winner_name'] == player_id) | 
             (historical_data['loser_name'] == player_id)) &
            (historical_data['surface'] == surface.value)
        ].copy()
        
        if len(player_matches) == 0:
            return 0.5  # Neutral rating if no data
        
        # Calculate win rate
        wins = len(player_matches[player_matches['winner_name'] == player_id])
        total_matches = len(player_matches)
        win_rate = wins / total_matches
        
        # Adjust for opponent quality (simplified)
        opponent_quality_adj = 0.0
        for _, match in player_matches.iterrows():
            opponent = match['loser_name'] if match['winner_name'] == player_id else match['winner_name']
            # Simple opponent ranking adjustment (would use ELO in real implementation)
            opp_ranking = match.get('loser_rank', 50) if match['winner_name'] == player_id else match.get('winner_rank', 50)
            if opp_ranking and opp_ranking <= 20:
                opponent_quality_adj += 0.1
            elif opp_ranking and opp_ranking <= 50:
                opponent_quality_adj += 0.05
        
        opponent_quality_adj = min(0.2, opponent_quality_adj / total_matches)
        
        # Time decay for recent performance
        recent_weight = 0.0
        if 'tourney_date' in player_matches.columns:
            recent_matches = player_matches[player_matches['tourney_date'] >= 
                                          datetime.now() - timedelta(days=365*2)]
            if len(recent_matches) > 0:
                recent_wins = len(recent_matches[recent_matches['winner_name'] == player_id])
                recent_win_rate = recent_wins / len(recent_matches)
                recent_weight = 0.3 * (recent_win_rate - win_rate)
        
        # Surface characteristics bonus
        surface_chars = self.surface_characteristics[surface]
        player_style = self._classify_playing_style(player_id, historical_data)
        style_bonus = self.style_surface_advantages[surface].get(player_style, 1.0) - 1.0
        style_bonus *= 0.1  # Scale down the bonus
        
        # Final rating
        surface_rating = min(0.95, max(0.05,
            win_rate + opponent_quality_adj + recent_weight + style_bonus
        ))
        
        return surface_rating
    
    def _classify_playing_style(self, player_id: str, historical_data: pd.DataFrame) -> PlayingStyle:
        """Classify player's playing style based on historical data."""
        
        player_matches = historical_data[
            (historical_data['winner_name'] == player_id) | 
            (historical_data['loser_name'] == player_id)
        ]
        
        if len(player_matches) == 0:
            return PlayingStyle.ALL_COURT  # Default style
        
        # Analyze match characteristics to determine style
        style_indicators = {
            'avg_match_duration': 0,
            'grass_win_rate': 0,
            'clay_win_rate': 0,
            'hard_win_rate': 0,
            'break_point_conversion': 0,
            'service_games_won_pct': 0
        }
        
        # Calculate style indicators (simplified)
        for surface in ['Grass', 'Clay', 'Hard']:
            surface_matches = player_matches[player_matches['surface'] == surface]
            if len(surface_matches) > 0:
                wins = len(surface_matches[surface_matches['winner_name'] == player_id])
                win_rate = wins / len(surface_matches)
                style_indicators[f'{surface.lower()}_win_rate'] = win_rate
        
        # Classification logic
        if style_indicators['grass_win_rate'] > 0.7 and style_indicators['clay_win_rate'] < 0.5:
            return PlayingStyle.SERVE_AND_VOLLEY
        elif style_indicators['grass_win_rate'] > 0.65:
            return PlayingStyle.BIG_SERVER
        elif style_indicators['clay_win_rate'] > 0.7:
            if style_indicators['hard_win_rate'] > 0.6:
                return PlayingStyle.DEFENSIVE_BASELINER
            else:
                return PlayingStyle.COUNTER_PUNCHER
        elif all(wr > 0.55 for wr in [style_indicators['grass_win_rate'], 
                                     style_indicators['clay_win_rate'],
                                     style_indicators['hard_win_rate']]):
            return PlayingStyle.ALL_COURT
        else:
            return PlayingStyle.AGGRESSIVE_BASELINER
    
    def _analyze_style_matchup(self, style1: PlayingStyle, style2: PlayingStyle, 
                             surface: SurfaceType) -> Optional[str]:
        """Analyze playing style matchup advantages."""
        
        # Style matchup matrix (simplified)
        matchup_advantages = {
            (PlayingStyle.AGGRESSIVE_BASELINER, PlayingStyle.DEFENSIVE_BASELINER): {
                SurfaceType.HARD: 'player1',
                SurfaceType.GRASS: 'player1',
                SurfaceType.CLAY: 'player2'
            },
            (PlayingStyle.BIG_SERVER, PlayingStyle.COUNTER_PUNCHER): {
                SurfaceType.GRASS: 'player1',
                SurfaceType.HARD: 'player1',
                SurfaceType.CLAY: 'player2'
            },
            (PlayingStyle.SERVE_AND_VOLLEY, PlayingStyle.DEFENSIVE_BASELINER): {
                SurfaceType.GRASS: 'player1',
                SurfaceType.HARD: 'neutral',
                SurfaceType.CLAY: 'player2'
            }
        }
        
        # Check direct matchup
        if (style1, style2) in matchup_advantages:
            return matchup_advantages[(style1, style2)].get(surface, 'neutral')
        elif (style2, style1) in matchup_advantages:
            result = matchup_advantages[(style2, style1)].get(surface, 'neutral')
            if result == 'player1':
                return 'player2'
            elif result == 'player2':
                return 'player1'
            else:
                return result
        
        # Surface-based style advantages
        style1_advantage = self.style_surface_advantages[surface].get(style1, 1.0)
        style2_advantage = self.style_surface_advantages[surface].get(style2, 1.0)
        
        if abs(style1_advantage - style2_advantage) > 0.2:
            return 'player1' if style1_advantage > style2_advantage else 'player2'
        
        return None
    
    def _analyze_historical_surface_performance(self, player1_id: str, player2_id: str,
                                              surface: SurfaceType, 
                                              historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze historical performance on specific surface."""
        
        analysis = {
            'player1_stats': {},
            'player2_stats': {},
            'head_to_head': {}
        }
        
        # Analyze each player's performance
        for player_key, player_id in [('player1_stats', player1_id), ('player2_stats', player2_id)]:
            player_surface_matches = historical_data[
                ((historical_data['winner_name'] == player_id) | 
                 (historical_data['loser_name'] == player_id)) &
                (historical_data['surface'] == surface.value)
            ]
            
            if len(player_surface_matches) > 0:
                wins = len(player_surface_matches[player_surface_matches['winner_name'] == player_id])
                total = len(player_surface_matches)
                
                # Recent performance (last 2 years)
                recent_matches = player_surface_matches[
                    player_surface_matches['tourney_date'] >= datetime.now() - timedelta(days=730)
                ] if 'tourney_date' in player_surface_matches.columns else pd.DataFrame()
                
                recent_wins = 0
                recent_total = 0
                if len(recent_matches) > 0:
                    recent_wins = len(recent_matches[recent_matches['winner_name'] == player_id])
                    recent_total = len(recent_matches)
                
                analysis[player_key] = {
                    'total_matches': total,
                    'wins': wins,
                    'win_rate': wins / total,
                    'recent_matches': recent_total,
                    'recent_wins': recent_wins,
                    'recent_win_rate': recent_wins / recent_total if recent_total > 0 else 0
                }
        
        # Head-to-head analysis on surface
        h2h_matches = historical_data[
            ((historical_data['winner_name'] == player1_id) & (historical_data['loser_name'] == player2_id)) |
            ((historical_data['winner_name'] == player2_id) & (historical_data['loser_name'] == player1_id))
        ]
        
        h2h_surface = h2h_matches[h2h_matches['surface'] == surface.value]
        
        if len(h2h_surface) > 0:
            p1_wins = len(h2h_surface[h2h_surface['winner_name'] == player1_id])
            total_h2h = len(h2h_surface)
            
            analysis['head_to_head'] = {
                'total_matches': total_h2h,
                'player1_wins': p1_wins,
                'player2_wins': total_h2h - p1_wins,
                'player1_win_rate': p1_wins / total_h2h
            }
        
        return analysis
    
    def _generate_surface_features(self, player1_id: str, player2_id: str,
                                 surface: SurfaceType, 
                                 historical_data: pd.DataFrame) -> Dict[str, float]:
        """Generate surface-specific features for ML models."""
        
        surface_chars = self.surface_characteristics[surface]
        
        features = {
            # Basic surface encoding
            f'surface_{surface.value.lower()}': 1.0,
            
            # Surface characteristics
            'surface_speed_index': surface_chars['speed_index'],
            'surface_bounce_height': surface_chars['bounce_height'],
            'surface_slide_factor': surface_chars['slide_factor'],
            'rally_length_multiplier': surface_chars['rally_length_multiplier'],
            'serve_advantage_factor': surface_chars['serve_advantage'],
            'defensive_bonus_factor': surface_chars['defensive_bonus'],
        }
        
        # Add other surface encodings as 0
        for other_surface in SurfaceType:
            if other_surface != surface:
                features[f'surface_{other_surface.value.lower()}'] = 0.0
        
        # Player-specific surface features
        p1_surface_rating = self._calculate_surface_rating(player1_id, surface, historical_data)
        p2_surface_rating = self._calculate_surface_rating(player2_id, surface, historical_data)
        
        features.update({
            'player1_surface_rating': p1_surface_rating,
            'player2_surface_rating': p2_surface_rating,
            'surface_rating_diff': p1_surface_rating - p2_surface_rating,
            'surface_rating_advantage': 1.0 if p1_surface_rating > p2_surface_rating else 0.0
        })
        
        # Style-based features
        p1_style = self._classify_playing_style(player1_id, historical_data)
        p2_style = self._classify_playing_style(player2_id, historical_data)
        
        p1_style_advantage = self.style_surface_advantages[surface].get(p1_style, 1.0)
        p2_style_advantage = self.style_surface_advantages[surface].get(p2_style, 1.0)
        
        features.update({
            'player1_style_surface_advantage': p1_style_advantage,
            'player2_style_surface_advantage': p2_style_advantage,
            'style_advantage_diff': p1_style_advantage - p2_style_advantage
        })
        
        # Surface transition features
        p1_transition_penalty = self._calculate_surface_transition_penalty(player1_id, surface, historical_data)
        p2_transition_penalty = self._calculate_surface_transition_penalty(player2_id, surface, historical_data)
        
        features.update({
            'player1_surface_transition_penalty': p1_transition_penalty,
            'player2_surface_transition_penalty': p2_transition_penalty
        })
        
        return features
    
    def _calculate_surface_transition_penalty(self, player_id: str, current_surface: SurfaceType,
                                            historical_data: pd.DataFrame) -> float:
        """Calculate penalty for switching surfaces."""
        
        player_matches = historical_data[
            ((historical_data['winner_name'] == player_id) | 
             (historical_data['loser_name'] == player_id))
        ].sort_values('tourney_date') if 'tourney_date' in historical_data.columns else pd.DataFrame()
        
        if len(player_matches) < 2:
            return 0.0
        
        # Find most recent match
        recent_match = player_matches.iloc[-1]
        last_surface = recent_match['surface']
        
        # If same surface, no penalty
        if last_surface == current_surface.value:
            return 0.0
        
        # Calculate penalty based on surface difference
        surface_transition_penalties = {
            ('Clay', 'Grass'): 0.15,    # Biggest transition
            ('Grass', 'Clay'): 0.15,    # Biggest transition
            ('Clay', 'Hard'): 0.08,     # Moderate transition
            ('Hard', 'Clay'): 0.06,     # Moderate transition
            ('Hard', 'Grass'): 0.10,    # Moderate transition
            ('Grass', 'Hard'): 0.12,    # Moderate transition
        }
        
        penalty = surface_transition_penalties.get((last_surface, current_surface.value), 0.05)
        
        # Time decay - penalty decreases over time
        if 'tourney_date' in historical_data.columns:
            days_since = (datetime.now() - pd.to_datetime(recent_match['tourney_date'])).days
            time_decay = max(0.1, 1.0 - days_since / 30.0)  # Penalty decays over 30 days
            penalty *= time_decay
        
        return penalty