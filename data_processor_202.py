#!/usr/bin/env python3
"""
Data Processor 202 - Advanced Tennis Data Processing Module

Processes and analyzes tennis match data to extract predictive features:
- Score analysis and momentum indicators
- Player performance statistics
- Surface-specific metrics
- Head-to-head analysis
- Tournament progression patterns
- Break point and serve statistics

Compatible with Tennis Predictor 202

Author: Advanced Tennis Analytics Research
Version: 2.0.2
Date: September 21, 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from collections import defaultdict, Counter
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class TennisDataProcessor202:
    """
    Advanced tennis data processing system for extracting predictive features
    from match results and player statistics.
    """
    
    def __init__(self):
        self.setup_logging()
        self.player_stats = defaultdict(lambda: defaultdict(float))
        self.elo_ratings = {}
        self.h2h_records = defaultdict(lambda: defaultdict(int))
        self.surface_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.tournament_progression = defaultdict(list)
        
    def setup_logging(self):
        """Setup logging for the data processor"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def analyze_score(self, score_str: str) -> Dict[str, Any]:
        """Analyze match score to extract detailed gameplay dynamics"""
        if pd.isna(score_str) or score_str == '':
            return self._empty_score_analysis()
        
        # Handle special cases
        if any(term in score_str for term in ['RET', 'Walkover', 'W/O', 'DEF']):
            return {'match_type': 'incomplete', 'sets_played': 0, 'retirement': True}
        
        # Clean score string
        score_clean = str(score_str).replace('"', '').strip()
        sets = [s.strip() for s in score_clean.split(',') if s.strip()]
        
        analysis = {
            'sets_played': len(sets),
            'total_games': 0,
            'winner_games': 0,
            'loser_games': 0,
            'tiebreaks': 0,
            'close_sets': 0,
            'bagel_sets': 0,
            'breadstick_sets': 0,
            'dominant_sets': 0,
            'comeback_indicators': 0,
            'match_type': 'complete',
            'retirement': False,
            'set_scores': [],
            'game_differentials': [],
            'momentum_shifts': 0
        }
        
        previous_winner_advantage = 0
        
        for i, set_score in enumerate(sets):
            if not set_score:
                continue
                
            set_analysis = self._analyze_set_score(set_score)
            if set_analysis:
                analysis['total_games'] += set_analysis['total_games']
                analysis['winner_games'] += set_analysis['winner_games']
                analysis['loser_games'] += set_analysis['loser_games']
                analysis['tiebreaks'] += set_analysis['tiebreaks']
                analysis['close_sets'] += set_analysis['close_sets']
                analysis['bagel_sets'] += set_analysis['bagel_sets']
                analysis['breadstick_sets'] += set_analysis['breadstick_sets']
                analysis['dominant_sets'] += set_analysis['dominant_sets']
                
                analysis['set_scores'].append(set_analysis)
                
                game_diff = set_analysis['winner_games'] - set_analysis['loser_games']
                analysis['game_differentials'].append(game_diff)
                
                # Detect momentum shifts between sets
                current_advantage = 1 if game_diff > 0 else -1 if game_diff < 0 else 0
                if i > 0 and current_advantage != previous_winner_advantage and previous_winner_advantage != 0:
                    analysis['momentum_shifts'] += 1
                previous_winner_advantage = current_advantage
        
        # Calculate additional metrics
        if analysis['sets_played'] > 0:
            analysis['avg_games_per_set'] = analysis['total_games'] / analysis['sets_played']
            analysis['competitiveness_index'] = analysis['close_sets'] / analysis['sets_played']
            analysis['dominance_index'] = (analysis['bagel_sets'] + analysis['breadstick_sets']) / analysis['sets_played']
            analysis['tiebreak_frequency'] = analysis['tiebreaks'] / analysis['sets_played']
        else:
            analysis['avg_games_per_set'] = 0
            analysis['competitiveness_index'] = 0
            analysis['dominance_index'] = 0
            analysis['tiebreak_frequency'] = 0
            
        return analysis
        
    def _analyze_set_score(self, set_score: str) -> Dict[str, Any]:
        """Analyze individual set score"""
        # Handle tiebreak notation like 7-6(4) or 6-7(3)
        tiebreak_match = re.search(r'(\d+)-(\d+)\((\d+)\)', set_score)
        if tiebreak_match:
            winner_games = int(tiebreak_match.group(1))
            loser_games = int(tiebreak_match.group(2))
            tiebreak_score = int(tiebreak_match.group(3))
            tiebreaks = 1
        else:
            # Regular set like 6-4, 6-3, etc.
            games_match = re.search(r'(\d+)-(\d+)', set_score)
            if games_match:
                winner_games = int(games_match.group(1))
                loser_games = int(games_match.group(2))
                tiebreaks = 0
            else:
                return None
        
        total_games = winner_games + loser_games
        game_diff = abs(winner_games - loser_games)
        
        # Classify set type
        close_sets = 1 if game_diff <= 2 and tiebreaks == 0 else tiebreaks
        bagel_sets = 1 if (winner_games == 6 and loser_games == 0) or (winner_games == 0 and loser_games == 6) else 0
        breadstick_sets = 1 if (winner_games == 6 and loser_games == 1) or (winner_games == 1 and loser_games == 6) else 0
        dominant_sets = 1 if game_diff >= 4 and tiebreaks == 0 else 0
        
        return {
            'winner_games': winner_games,
            'loser_games': loser_games,
            'total_games': total_games,
            'tiebreaks': tiebreaks,
            'close_sets': close_sets,
            'bagel_sets': bagel_sets,
            'breadstick_sets': breadstick_sets,
            'dominant_sets': dominant_sets,
            'game_differential': game_diff
        }
        
    def _empty_score_analysis(self) -> Dict[str, Any]:
        """Return empty score analysis for missing data"""
        return {
            'match_type': 'unknown',
            'sets_played': 0,
            'total_games': 0,
            'winner_games': 0,
            'loser_games': 0,
            'tiebreaks': 0,
            'close_sets': 0,
            'bagel_sets': 0,
            'breadstick_sets': 0,
            'dominant_sets': 0,
            'comeback_indicators': 0,
            'retirement': False,
            'avg_games_per_set': 0,
            'competitiveness_index': 0,
            'dominance_index': 0,
            'tiebreak_frequency': 0,
            'momentum_shifts': 0
        }
        
    def process_match_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw match data to extract all predictive features"""
        self.logger.info(f"Processing {len(df)} matches for feature extraction...")
        
        # Initialize processed DataFrame with original columns
        processed_df = df.copy()
        
        # Add score analysis features
        score_features = []
        for _, match in df.iterrows():
            score_analysis = self.analyze_score(match.get('Score', ''))
            score_features.append(score_analysis)
            
        # Convert score features to DataFrame and merge
        score_df = pd.DataFrame(score_features)
        for col in score_df.columns:
            processed_df[f'score_{col}'] = score_df[col]
            
        # Process player statistics
        self._extract_player_statistics(processed_df)
        
        # Add Elo ratings
        self._calculate_elo_ratings(processed_df)
        
        # Add head-to-head features
        self._extract_h2h_features(processed_df)
        
        # Add surface-specific features
        self._extract_surface_features(processed_df)
        
        # Add tournament progression features
        self._extract_tournament_features(processed_df)
        
        # Add momentum and form features
        self._extract_momentum_features(processed_df)
        
        # Add psychological pressure indicators
        self._extract_pressure_indicators(processed_df)
        
        self.logger.info(f"Processing complete. Added {len(processed_df.columns) - len(df.columns)} new features.")
        return processed_df
        
    def _extract_player_statistics(self, df: pd.DataFrame):
        """Extract comprehensive player statistics"""
        # Process matches chronologically
        df['Date'] = pd.to_datetime(df['Date'])
        df_sorted = df.sort_values('Date')
        
        # Track cumulative statistics for each player
        for idx, match in df_sorted.iterrows():
            winner = match['Winner']
            loser = match['Loser']
            surface = match.get('Surface', 'Hard')
            
            # Update winner stats
            self._update_player_stats(winner, True, match)
            
            # Update loser stats  
            self._update_player_stats(loser, False, match)
            
    def _update_player_stats(self, player: str, won: bool, match: Dict):
        """Update individual player statistics"""
        stats = self.player_stats[player]
        
        # Basic match statistics
        stats['total_matches'] += 1
        if won:
            stats['wins'] += 1
        stats['win_rate'] = stats['wins'] / stats['total_matches']
        
        # Surface-specific stats
        surface = match.get('Surface', 'Hard')
        stats[f'{surface.lower()}_matches'] += 1
        if won:
            stats[f'{surface.lower()}_wins'] += 1
        stats[f'{surface.lower()}_win_rate'] = (stats[f'{surface.lower()}_wins'] / 
                                               max(stats[f'{surface.lower()}_matches'], 1))
        
        # Tournament category stats
        category = match.get('Category', 'Unknown')
        stats[f'{category.lower().replace(" ", "_")}_matches'] += 1
        if won:
            stats[f'{category.lower().replace(" ", "_")}_wins'] += 1
            
        # Score-based performance metrics
        score_analysis = self.analyze_score(match.get('Score', ''))
        if score_analysis['match_type'] == 'complete':
            stats['total_sets'] += score_analysis['sets_played']
            stats['total_games'] += (score_analysis['winner_games'] if won else score_analysis['loser_games'])
            stats['tiebreaks_played'] += score_analysis['tiebreaks']
            stats['close_sets_played'] += score_analysis['close_sets']
            
            if won:
                stats['tiebreaks_won'] += score_analysis['tiebreaks']
                stats['close_sets_won'] += score_analysis['close_sets']
                
        # Recent form (last 10 matches)
        if 'recent_results' not in stats:
            stats['recent_results'] = []
        stats['recent_results'].append(1 if won else 0)
        if len(stats['recent_results']) > 10:
            stats['recent_results'] = stats['recent_results'][-10:]
        stats['recent_form'] = np.mean(stats['recent_results'])
        
    def _calculate_elo_ratings(self, df: pd.DataFrame):
        """Calculate Elo ratings for all players"""
        self.logger.info("Calculating Elo ratings...")
        
        df_sorted = df.sort_values('Date')
        
        # Initialize Elo ratings
        elo_history = defaultdict(list)
        
        for idx, match in df_sorted.iterrows():
            winner = match['Winner']
            loser = match['Loser']
            surface = match.get('Surface', 'Hard')
            
            # Initialize ratings if not present
            if winner not in self.elo_ratings:
                self.elo_ratings[winner] = {'Hard': 1500, 'Clay': 1500, 'Grass': 1500, 'overall': 1500}
            if loser not in self.elo_ratings:
                self.elo_ratings[loser] = {'Hard': 1500, 'Clay': 1500, 'Grass': 1500, 'overall': 1500}
                
            # Get current ratings
            winner_rating = self.elo_ratings[winner][surface]
            loser_rating = self.elo_ratings[loser][surface]
            
            # Calculate expected scores
            winner_expected = 1 / (1 + 10**((loser_rating - winner_rating) / 400))
            loser_expected = 1 - winner_expected
            
            # K-factor (higher for fewer matches played)
            winner_matches = self.player_stats[winner]['total_matches']
            loser_matches = self.player_stats[loser]['total_matches']
            
            winner_k = 32 if winner_matches < 10 else 24 if winner_matches < 30 else 16
            loser_k = 32 if loser_matches < 10 else 24 if loser_matches < 30 else 16
            
            # Update ratings
            new_winner_rating = winner_rating + winner_k * (1 - winner_expected)
            new_loser_rating = loser_rating + loser_k * (0 - loser_expected)
            
            # Store updated ratings
            self.elo_ratings[winner][surface] = new_winner_rating
            self.elo_ratings[loser][surface] = new_loser_rating
            
            # Update overall ratings (weighted average)
            self.elo_ratings[winner]['overall'] = np.mean([
                self.elo_ratings[winner]['Hard'],
                self.elo_ratings[winner]['Clay'],
                self.elo_ratings[winner]['Grass']
            ])
            self.elo_ratings[loser]['overall'] = np.mean([
                self.elo_ratings[loser]['Hard'],
                self.elo_ratings[loser]['Clay'], 
                self.elo_ratings[loser]['Grass']
            ])
            
            # Store rating at time of match
            elo_history[winner].append((match['Date'], new_winner_rating))
            elo_history[loser].append((match['Date'], new_loser_rating))
            
        # Add Elo features to dataframe
        df['winner_elo'] = [self.elo_ratings.get(winner, {}).get(surface, 1500) 
                           for winner, surface in zip(df['Winner'], df['Surface'])]
        df['loser_elo'] = [self.elo_ratings.get(loser, {}).get(surface, 1500) 
                          for loser, surface in zip(df['Loser'], df['Surface'])]
        df['elo_difference'] = df['winner_elo'] - df['loser_elo']
        
    def _extract_h2h_features(self, df: pd.DataFrame):
        """Extract head-to-head features"""
        self.logger.info("Extracting head-to-head features...")
        
        h2h_wins = []
        h2h_total = []
        h2h_surface = []
        
        df_sorted = df.sort_values('Date')
        
        for idx, match in df_sorted.iterrows():
            winner = match['Winner']
            loser = match['Loser']
            surface = match.get('Surface', 'Hard')
            
            # Create consistent matchup key
            matchup = tuple(sorted([winner, loser]))
            
            # Get historical H2H record
            h2h_record = self.h2h_records[matchup]
            total_matches = h2h_record['total']
            
            if total_matches > 0:
                winner_h2h_wins = h2h_record.get(winner, 0)
                h2h_rate = winner_h2h_wins / total_matches
            else:
                h2h_rate = 0.5  # No prior history
                
            h2h_wins.append(h2h_rate)
            h2h_total.append(total_matches)
            h2h_surface.append(h2h_record.get(f'{surface.lower()}_total', 0))
            
            # Update H2H record
            h2h_record['total'] += 1
            h2h_record[winner] = h2h_record.get(winner, 0) + 1
            h2h_record[f'{surface.lower()}_total'] += 1
            h2h_record[f'{surface.lower()}_{winner}'] = h2h_record.get(f'{surface.lower()}_{winner}', 0) + 1
            
        df['h2h_win_rate'] = h2h_wins
        df['h2h_total_matches'] = h2h_total
        df['h2h_surface_matches'] = h2h_surface
        
    def _extract_surface_features(self, df: pd.DataFrame):
        """Extract surface-specific performance features"""
        self.logger.info("Extracting surface-specific features...")
        
        winner_surface_wr = []
        loser_surface_wr = []
        surface_advantage = []
        
        for _, match in df.iterrows():
            winner = match['Winner']
            loser = match['Loser']
            surface = match.get('Surface', 'Hard')
            
            # Get surface-specific win rates
            winner_stats = self.player_stats[winner]
            loser_stats = self.player_stats[loser]
            
            winner_surface_rate = winner_stats.get(f'{surface.lower()}_win_rate', 0.5)
            loser_surface_rate = loser_stats.get(f'{surface.lower()}_win_rate', 0.5)
            
            winner_surface_wr.append(winner_surface_rate)
            loser_surface_wr.append(loser_surface_rate)
            surface_advantage.append(winner_surface_rate - loser_surface_rate)
            
        df['winner_surface_win_rate'] = winner_surface_wr
        df['loser_surface_win_rate'] = loser_surface_wr
        df['surface_advantage'] = surface_advantage
        
    def _extract_tournament_features(self, df: pd.DataFrame):
        """Extract tournament progression and category features"""
        category_mapping = {
            'Grand Slam': 4,
            'Masters 1000': 3,
            'ATP 500': 2.5,
            'WTA 1000': 3,
            'WTA 500': 2.5,
            'ATP 250': 2,
            'WTA 250': 2,
            'ATP Challenger': 1.5,
            'Team Event': 1,
            'Exhibition': 0.5
        }
        
        df['tournament_level'] = df['Category'].map(category_mapping).fillna(2.0)
        
        # Round progression features
        round_mapping = {
            'Final': 7,
            'Semifinal': 6,
            'Quarterfinal': 5,
            'Round of 16': 4,
            'Round of 32': 3,
            'Round of 64': 2,
            'Round of 128': 1,
            'Singles': 1
        }
        
        df['round_level'] = df['Round'].map(round_mapping).fillna(3.0)
        
    def _extract_momentum_features(self, df: pd.DataFrame):
        """Extract momentum and recent form features"""
        self.logger.info("Extracting momentum features...")
        
        winner_form = []
        loser_form = []
        winner_streak = []
        loser_streak = []
        
        for _, match in df.iterrows():
            winner = match['Winner']
            loser = match['Loser']
            
            winner_stats = self.player_stats[winner]
            loser_stats = self.player_stats[loser]
            
            # Recent form (last 10 matches)
            winner_form.append(winner_stats.get('recent_form', 0.5))
            loser_form.append(loser_stats.get('recent_form', 0.5))
            
            # Current winning streak
            winner_recent = winner_stats.get('recent_results', [])
            loser_recent = loser_stats.get('recent_results', [])
            
            winner_streak_count = self._calculate_current_streak(winner_recent)
            loser_streak_count = self._calculate_current_streak(loser_recent)
            
            winner_streak.append(winner_streak_count)
            loser_streak.append(loser_streak_count)
            
        df['winner_recent_form'] = winner_form
        df['loser_recent_form'] = loser_form
        df['winner_streak'] = winner_streak
        df['loser_streak'] = loser_streak
        df['form_advantage'] = np.array(winner_form) - np.array(loser_form)
        
    def _calculate_current_streak(self, results: List[int]) -> int:
        """Calculate current winning/losing streak"""
        if not results:
            return 0
            
        streak = 0
        current_result = results[-1]
        
        for result in reversed(results):
            if result == current_result:
                streak += 1
            else:
                break
                
        return streak if current_result == 1 else -streak
        
    def _extract_pressure_indicators(self, df: pd.DataFrame):
        """Extract psychological pressure and clutch performance indicators"""
        self.logger.info("Extracting pressure indicators...")
        
        # Add clutch performance metrics based on match context
        clutch_scores = []
        pressure_levels = []
        
        for _, match in df.iterrows():
            # Calculate pressure level based on tournament and round
            tournament_level = match.get('tournament_level', 2.0)
            round_level = match.get('round_level', 3.0)
            
            pressure_level = (tournament_level * round_level) / 28  # Normalize to [0,1]
            pressure_levels.append(pressure_level)
            
            # Clutch score based on score analysis
            score_analysis = self.analyze_score(match.get('Score', ''))
            
            # Higher clutch score for tiebreaks, close sets, comebacks
            clutch_score = (
                score_analysis.get('tiebreak_frequency', 0) * 0.4 +
                score_analysis.get('competitiveness_index', 0) * 0.3 +
                score_analysis.get('momentum_shifts', 0) * 0.1 * 0.2 +
                (1 - score_analysis.get('dominance_index', 0)) * 0.1
            )
            
            clutch_scores.append(clutch_score)
            
        df['pressure_level'] = pressure_levels
        df['clutch_score'] = clutch_scores
        
    def get_player_features(self, player: str, date: datetime = None) -> Dict[str, float]:
        """Get comprehensive player features at a specific date"""
        if player not in self.player_stats:
            return {}
            
        stats = dict(self.player_stats[player])
        
        # Add Elo ratings
        if player in self.elo_ratings:
            for surface in ['Hard', 'Clay', 'Grass', 'overall']:
                stats[f'elo_{surface.lower()}'] = self.elo_ratings[player][surface]
                
        return stats
        
    def export_processed_data(self, df: pd.DataFrame, filename: str):
        """Export processed data to CSV"""
        try:
            df.to_csv(filename, index=False)
            self.logger.info(f"Processed data exported to {filename}")
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            
    def generate_feature_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary of extracted features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        summary = {
            'total_matches': len(df),
            'total_features': len(df.columns),
            'numeric_features': len(numeric_cols),
            'unique_players': len(set(df['Winner'].tolist() + df['Loser'].tolist())),
            'date_range': f"{df['Date'].min()} to {df['Date'].max()}",
            'surfaces': df['Surface'].value_counts().to_dict(),
            'tournaments': df['Category'].value_counts().to_dict(),
            'feature_correlations': df[numeric_cols].corr().abs().mean().to_dict()
        }
        
        return summary

def main():
    """Main function for testing the data processor"""
    processor = TennisDataProcessor202()
    
    # Load sample data
    try:
        df = pd.read_csv('tennis_matches_500_ultimate.csv')
        print(f"Loaded {len(df)} matches")
        
        # Process the data
        processed_df = processor.process_match_data(df)
        
        # Generate summary
        summary = processor.generate_feature_summary(processed_df)
        
        print("\n" + "="*60)
        print("TENNIS DATA PROCESSOR 202 - PROCESSING SUMMARY")
        print("="*60)
        print(f"Total matches processed: {summary['total_matches']}")
        print(f"Total features extracted: {summary['total_features']}")
        print(f"Numeric features: {summary['numeric_features']}")
        print(f"Unique players: {summary['unique_players']}")
        print(f"Date range: {summary['date_range']}")
        print("-"*60)
        print("Surface distribution:")
        for surface, count in summary['surfaces'].items():
            print(f"  {surface}: {count}")
        print("-"*60)
        print("Tournament categories:")
        for category, count in summary['tournaments'].items():
            print(f"  {category}: {count}")
        print("="*60)
        
        # Export processed data
        processor.export_processed_data(processed_df, 'tennis_matches_processed_202.csv')
        
    except FileNotFoundError:
        print("Sample data file not found. Please ensure 'tennis_matches_500_ultimate.csv' is available.")
    except Exception as e:
        print(f"Error processing data: {e}")

if __name__ == "__main__":
    main()
