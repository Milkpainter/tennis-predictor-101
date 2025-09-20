"""ELO Rating System for Tennis Players.

Implements a sophisticated ELO rating system with surface-specific
adjustments, time decay, and match importance weighting.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging

from config import get_config


class ELORatingSystem:
    """Advanced ELO rating system for tennis players.
    
    Features:
    - Surface-specific ratings (clay, hard, grass)
    - Time-based rating decay
    - Match importance weighting (Grand Slams, Masters, etc.)
    - Head-to-head adjustments
    - Uncertainty quantification
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger("features.elo")
        
        # ELO parameters from config
        self.initial_rating = self.config.get('feature_engineering.elo.initial_rating', 1500)
        self.k_factor = self.config.get('feature_engineering.elo.k_factor', 32)
        self.decay_factor = self.config.get('feature_engineering.elo.decay_factor', 0.95)
        
        # Surface adjustments
        self.surface_adjustments = self.config.get(
            'feature_engineering.elo.surface_adjustments',
            {'clay': 1.1, 'hard': 1.0, 'grass': 0.9}
        )
        
        # Player ratings storage
        self.overall_ratings = {}  # player_id -> rating
        self.surface_ratings = {}  # (player_id, surface) -> rating
        self.rating_history = {}   # player_id -> [(date, rating)]
        self.match_count = {}      # player_id -> number of matches
        self.last_update = {}      # player_id -> last match date
        
        # Match importance weights
        self.tournament_weights = {
            'Grand Slam': 1.5,
            'ATP Finals': 1.4,
            'Masters 1000': 1.3,
            'Olympics': 1.3,
            'ATP 500': 1.1,
            'ATP 250': 1.0,
            'Challenger': 0.8,
            'Futures': 0.6
        }
    
    def initialize_player(self, player_id: str, surface: str = None) -> float:
        """Initialize a new player with default rating.
        
        Args:
            player_id: Unique player identifier
            surface: Specific surface ('clay', 'hard', 'grass')
            
        Returns:
            Initial rating
        """
        if player_id not in self.overall_ratings:
            self.overall_ratings[player_id] = self.initial_rating
            self.rating_history[player_id] = [(datetime.now(), self.initial_rating)]
            self.match_count[player_id] = 0
            self.last_update[player_id] = datetime.now()
        
        if surface and (player_id, surface) not in self.surface_ratings:
            # Initialize surface rating slightly adjusted from overall
            surface_mult = self.surface_adjustments.get(surface, 1.0)
            surface_rating = self.initial_rating * surface_mult
            self.surface_ratings[(player_id, surface)] = surface_rating
        
        return self.overall_ratings[player_id]
    
    def get_rating(self, player_id: str, surface: str = None, 
                  as_of_date: datetime = None) -> float:
        """Get player's current or historical rating.
        
        Args:
            player_id: Player identifier
            surface: Specific surface rating
            as_of_date: Get rating as of specific date
            
        Returns:
            Player rating
        """
        if as_of_date:
            return self._get_historical_rating(player_id, as_of_date, surface)
        
        if surface and (player_id, surface) in self.surface_ratings:
            return self.surface_ratings[(player_id, surface)]
        
        return self.overall_ratings.get(player_id, self.initial_rating)
    
    def update_ratings(self, winner_id: str, loser_id: str, 
                      match_date: datetime, surface: str,
                      tournament_category: str = 'ATP 250',
                      score: str = None) -> Tuple[float, float]:
        """Update ratings after a match.
        
        Args:
            winner_id: Winner player ID
            loser_id: Loser player ID
            match_date: Date of the match
            surface: Court surface
            tournament_category: Tournament importance
            score: Match score for additional weighting
            
        Returns:
            Tuple of (new_winner_rating, new_loser_rating)
        """
        # Initialize players if needed
        self.initialize_player(winner_id, surface)
        self.initialize_player(loser_id, surface)
        
        # Apply time decay if needed
        self._apply_time_decay(winner_id, match_date)
        self._apply_time_decay(loser_id, match_date)
        
        # Get current ratings
        winner_rating = self.get_rating(winner_id, surface)
        loser_rating = self.get_rating(loser_id, surface)
        
        # Calculate expected scores
        winner_expected = self._expected_score(winner_rating, loser_rating)
        loser_expected = 1 - winner_expected
        
        # Calculate K-factor adjustments
        k_winner = self._calculate_k_factor(winner_id, tournament_category, score)
        k_loser = self._calculate_k_factor(loser_id, tournament_category, score)
        
        # Update ratings
        new_winner_rating = winner_rating + k_winner * (1 - winner_expected)
        new_loser_rating = loser_rating + k_loser * (0 - loser_expected)
        
        # Store updated ratings
        self._store_rating_update(winner_id, new_winner_rating, match_date, surface)
        self._store_rating_update(loser_id, new_loser_rating, match_date, surface)
        
        self.logger.debug(
            f"Rating update: {winner_id}: {winner_rating:.1f} -> {new_winner_rating:.1f}, "
            f"{loser_id}: {loser_rating:.1f} -> {new_loser_rating:.1f}"
        )
        
        return new_winner_rating, new_loser_rating
    
    def calculate_match_probability(self, player1_id: str, player2_id: str,
                                  surface: str) -> float:
        """Calculate probability of player1 beating player2.
        
        Args:
            player1_id: First player ID
            player2_id: Second player ID  
            surface: Court surface
            
        Returns:
            Probability of player1 winning (0-1)
        """
        rating1 = self.get_rating(player1_id, surface)
        rating2 = self.get_rating(player2_id, surface)
        
        return self._expected_score(rating1, rating2)
    
    def get_rating_difference(self, player1_id: str, player2_id: str,
                            surface: str) -> float:
        """Get rating difference between two players.
        
        Args:
            player1_id: First player ID
            player2_id: Second player ID
            surface: Court surface
            
        Returns:
            Rating difference (player1 - player2)
        """
        rating1 = self.get_rating(player1_id, surface)
        rating2 = self.get_rating(player2_id, surface)
        
        return rating1 - rating2
    
    def get_player_statistics(self, player_id: str) -> Dict:
        """Get comprehensive player statistics.
        
        Args:
            player_id: Player identifier
            
        Returns:
            Dictionary with player stats
        """
        if player_id not in self.overall_ratings:
            return {}
        
        # Calculate rating volatility (standard deviation of recent ratings)
        recent_ratings = [r for d, r in self.rating_history.get(player_id, [])[-10:]]
        volatility = np.std(recent_ratings) if len(recent_ratings) > 1 else 0
        
        # Rating trend (last 5 matches)
        trend_ratings = recent_ratings[-5:] if len(recent_ratings) >= 5 else recent_ratings
        trend = np.polyfit(range(len(trend_ratings)), trend_ratings, 1)[0] if len(trend_ratings) > 1 else 0
        
        # Peak rating
        all_ratings = [r for d, r in self.rating_history.get(player_id, [])]
        peak_rating = max(all_ratings) if all_ratings else self.initial_rating
        
        return {
            'current_rating': self.overall_ratings[player_id],
            'peak_rating': peak_rating,
            'rating_volatility': volatility,
            'rating_trend': trend,
            'match_count': self.match_count.get(player_id, 0),
            'last_update': self.last_update.get(player_id),
            'surface_ratings': {
                surface: rating for (pid, surface), rating in self.surface_ratings.items() 
                if pid == player_id
            }
        }
    
    def _expected_score(self, rating1: float, rating2: float) -> float:
        """Calculate expected score using ELO formula."""
        return 1 / (1 + 10 ** ((rating2 - rating1) / 400))
    
    def _calculate_k_factor(self, player_id: str, tournament_category: str, 
                          score: str = None) -> float:
        """Calculate adaptive K-factor based on various factors."""
        base_k = self.k_factor
        
        # Tournament importance adjustment
        tournament_mult = self.tournament_weights.get(tournament_category, 1.0)
        
        # Experience adjustment (lower K for experienced players)
        match_count = self.match_count.get(player_id, 0)
        if match_count > 100:
            experience_mult = 0.8
        elif match_count > 50:
            experience_mult = 0.9
        else:
            experience_mult = 1.0
        
        # Score-based adjustment (closer matches get higher K)
        score_mult = 1.0
        if score:
            score_mult = self._calculate_score_multiplier(score)
        
        return base_k * tournament_mult * experience_mult * score_mult
    
    def _calculate_score_multiplier(self, score: str) -> float:
        """Calculate score-based K-factor multiplier."""
        try:
            # Parse sets and games to determine match closeness
            sets = score.split()
            total_games_diff = 0
            set_count = 0
            
            for set_score in sets:
                if '-' in set_score:
                    games = set_score.split('-')
                    if len(games) == 2:
                        diff = abs(int(games[0]) - int(games[1]))
                        total_games_diff += diff
                        set_count += 1
            
            if set_count > 0:
                avg_games_diff = total_games_diff / set_count
                # Closer matches (lower avg diff) get higher multiplier
                return max(0.8, 1.2 - (avg_games_diff / 10))
        except:
            pass
        
        return 1.0
    
    def _apply_time_decay(self, player_id: str, current_date: datetime):
        """Apply time-based rating decay."""
        if player_id not in self.last_update:
            return
        
        last_match = self.last_update[player_id]
        months_inactive = (current_date - last_match).days / 30.0
        
        if months_inactive > 1:
            decay = self.decay_factor ** months_inactive
            
            # Apply decay to overall rating
            current_rating = self.overall_ratings[player_id]
            decayed_rating = self.initial_rating + (current_rating - self.initial_rating) * decay
            self.overall_ratings[player_id] = decayed_rating
            
            # Apply decay to surface ratings
            for (pid, surface), rating in list(self.surface_ratings.items()):
                if pid == player_id:
                    decayed_surface_rating = self.initial_rating + (rating - self.initial_rating) * decay
                    self.surface_ratings[(pid, surface)] = decayed_surface_rating
    
    def _store_rating_update(self, player_id: str, new_rating: float,
                           match_date: datetime, surface: str):
        """Store rating update in history."""
        # Update overall rating
        self.overall_ratings[player_id] = new_rating
        
        # Update surface-specific rating
        if surface:
            self.surface_ratings[(player_id, surface)] = new_rating
        
        # Update history
        if player_id not in self.rating_history:
            self.rating_history[player_id] = []
        self.rating_history[player_id].append((match_date, new_rating))
        
        # Update match count and last update
        self.match_count[player_id] = self.match_count.get(player_id, 0) + 1
        self.last_update[player_id] = match_date
    
    def _get_historical_rating(self, player_id: str, as_of_date: datetime,
                             surface: str = None) -> float:
        """Get player rating as of specific date."""
        if player_id not in self.rating_history:
            return self.initial_rating
        
        history = self.rating_history[player_id]
        
        # Find rating closest to but not after the specified date
        valid_ratings = [(date, rating) for date, rating in history if date <= as_of_date]
        
        if not valid_ratings:
            return self.initial_rating
        
        # Return the most recent valid rating
        return valid_ratings[-1][1]
    
    def export_ratings(self) -> pd.DataFrame:
        """Export all current ratings to DataFrame."""
        data = []
        
        for player_id, rating in self.overall_ratings.items():
            player_data = {
                'player_id': player_id,
                'overall_rating': rating,
                'match_count': self.match_count.get(player_id, 0),
                'last_update': self.last_update.get(player_id)
            }
            
            # Add surface ratings
            for surface in ['clay', 'hard', 'grass']:
                surface_rating = self.surface_ratings.get((player_id, surface))
                player_data[f'{surface}_rating'] = surface_rating
            
            data.append(player_data)
        
        return pd.DataFrame(data)
    
    def load_ratings(self, ratings_df: pd.DataFrame):
        """Load ratings from DataFrame."""
        for _, row in ratings_df.iterrows():
            player_id = row['player_id']
            
            self.overall_ratings[player_id] = row['overall_rating']
            self.match_count[player_id] = row.get('match_count', 0)
            
            if 'last_update' in row and pd.notna(row['last_update']):
                self.last_update[player_id] = pd.to_datetime(row['last_update'])
            
            # Load surface ratings
            for surface in ['clay', 'hard', 'grass']:
                surface_col = f'{surface}_rating'
                if surface_col in row and pd.notna(row[surface_col]):
                    self.surface_ratings[(player_id, surface)] = row[surface_col]