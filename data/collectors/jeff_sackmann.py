"""Jeff Sackmann tennis data collector.

Collects historical ATP and WTA match data from Jeff Sackmann's
comprehensive tennis databases on GitHub.
"""

import pandas as pd
import io
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from .base_collector import BaseCollector


class JeffSackmannCollector(BaseCollector):
    """Collector for Jeff Sackmann's tennis data.
    
    This is the industry standard for historical tennis data,
    covering ATP (1968+) and WTA (1968+) matches with comprehensive
    statistics and player information.
    """
    
    def __init__(self):
        super().__init__("jeff_sackmann", rate_limit=1.0)
        
        # Base URLs from config
        self.atp_base = self.config.get('data_sources.jeff_sackmann.atp_url')
        self.wta_base = self.config.get('data_sources.jeff_sackmann.wta_url')
        
        # File mappings
        self.atp_files = {
            'matches': 'atp_matches_{year}.csv',
            'rankings': 'atp_rankings_{date}.csv', 
            'players': 'atp_players.csv'
        }
        
        self.wta_files = {
            'matches': 'wta_matches_{year}.csv',
            'rankings': 'wta_rankings_{date}.csv',
            'players': 'wta_players.csv'
        }
    
    def collect_matches(self, tour: str = 'atp', years: List[int] = None) -> pd.DataFrame:
        """Collect match data for specified years.
        
        Args:
            tour: 'atp' or 'wta'
            years: List of years to collect. Defaults to last 5 years
            
        Returns:
            DataFrame with match data
        """
        if years is None:
            current_year = datetime.now().year
            years = list(range(current_year - 4, current_year + 1))
        
        tour = tour.lower()
        if tour not in ['atp', 'wta']:
            raise ValueError("Tour must be 'atp' or 'wta'")
        
        base_url = self.atp_base if tour == 'atp' else self.wta_base
        file_template = self.atp_files['matches'] if tour == 'atp' else self.wta_files['matches']
        
        all_matches = []
        
        for year in years:
            try:
                filename = file_template.format(year=year)
                url = f"{base_url}{filename}"
                
                self.logger.info(f"Collecting {tour.upper()} matches for {year}")
                response = self._make_request(url)
                
                # Read CSV from response content
                matches_df = pd.read_csv(io.StringIO(response.text))
                matches_df['year'] = year
                matches_df['tour'] = tour.upper()
                
                all_matches.append(matches_df)
                
            except Exception as e:
                self.logger.warning(f"Failed to collect {tour} matches for {year}: {e}")
                continue
        
        if not all_matches:
            raise ValueError(f"No match data collected for {tour} tours")
        
        combined_matches = pd.concat(all_matches, ignore_index=True)
        
        # Validate required columns
        required_columns = [
            'tourney_date', 'winner_name', 'loser_name', 
            'score', 'surface', 'draw_size'
        ]
        self._validate_data(combined_matches, required_columns)
        
        # Data preprocessing
        combined_matches = self._preprocess_matches(combined_matches)
        
        self.logger.info(f"Collected {len(combined_matches)} {tour.upper()} matches")
        return combined_matches
    
    def collect_rankings(self, tour: str = 'atp', date: str = None) -> pd.DataFrame:
        """Collect ranking data for specific date.
        
        Args:
            tour: 'atp' or 'wta'
            date: Date in YYYYMMDD format. Defaults to most recent Monday
            
        Returns:
            DataFrame with ranking data
        """
        if date is None:
            # Get most recent Monday
            today = datetime.now()
            days_since_monday = today.weekday()
            last_monday = today - timedelta(days=days_since_monday)
            date = last_monday.strftime('%Y%m%d')
        
        tour = tour.lower()
        base_url = self.atp_base if tour == 'atp' else self.wta_base
        file_template = self.atp_files['rankings'] if tour == 'atp' else self.wta_files['rankings']
        
        filename = file_template.format(date=date)
        url = f"{base_url}{filename}"
        
        self.logger.info(f"Collecting {tour.upper()} rankings for {date}")
        response = self._make_request(url)
        
        rankings_df = pd.read_csv(io.StringIO(response.text))
        rankings_df['tour'] = tour.upper()
        
        # Validate required columns
        required_columns = ['ranking_date', 'rank', 'player', 'points']
        self._validate_data(rankings_df, required_columns)
        
        return rankings_df
    
    def collect_players(self, tour: str = 'atp') -> pd.DataFrame:
        """Collect player biographical data.
        
        Args:
            tour: 'atp' or 'wta'
            
        Returns:
            DataFrame with player data
        """
        tour = tour.lower()
        base_url = self.atp_base if tour == 'atp' else self.wta_base
        filename = self.atp_files['players'] if tour == 'atp' else self.wta_files['players']
        
        url = f"{base_url}{filename}"
        
        self.logger.info(f"Collecting {tour.upper()} player data")
        response = self._make_request(url)
        
        players_df = pd.read_csv(io.StringIO(response.text))
        players_df['tour'] = tour.upper()
        
        # Validate required columns
        required_columns = ['player_id', 'name_first', 'name_last', 'hand', 'dob', 'country']
        self._validate_data(players_df, required_columns)
        
        return players_df
    
    def collect(self, tour: str = 'atp', data_type: str = 'matches', **kwargs) -> pd.DataFrame:
        """Main collection method.
        
        Args:
            tour: 'atp' or 'wta'
            data_type: 'matches', 'rankings', or 'players'
            **kwargs: Additional parameters for specific data types
            
        Returns:
            DataFrame with requested data
        """
        if data_type == 'matches':
            return self.collect_matches(tour, **kwargs)
        elif data_type == 'rankings':
            return self.collect_rankings(tour, **kwargs)
        elif data_type == 'players':
            return self.collect_players(tour, **kwargs)
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
    
    def get_latest(self) -> pd.DataFrame:
        """Get latest match data for both tours.
        
        Returns:
            DataFrame with recent matches
        """
        current_year = datetime.now().year
        
        try:
            atp_matches = self.collect_matches('atp', [current_year])
        except:
            atp_matches = pd.DataFrame()
        
        try:
            wta_matches = self.collect_matches('wta', [current_year])
        except:
            wta_matches = pd.DataFrame()
        
        if atp_matches.empty and wta_matches.empty:
            raise ValueError("No recent match data available")
        
        return pd.concat([atp_matches, wta_matches], ignore_index=True)
    
    def _preprocess_matches(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess match data for consistency.
        
        Args:
            matches_df: Raw match data
            
        Returns:
            Preprocessed match data
        """
        # Convert date to datetime
        matches_df['tourney_date'] = pd.to_datetime(matches_df['tourney_date'], format='%Y%m%d')
        
        # Standardize surface names
        surface_mapping = {
            'Hard': 'Hard',
            'Clay': 'Clay', 
            'Grass': 'Grass',
            'Carpet': 'Hard'  # Treat carpet as hard court
        }
        matches_df['surface'] = matches_df['surface'].map(surface_mapping).fillna('Hard')
        
        # Parse score for additional features
        matches_df['sets_won_winner'] = matches_df['score'].apply(self._parse_sets_won)
        matches_df['match_duration_estimated'] = matches_df['score'].apply(self._estimate_duration)
        
        # Add derived features
        matches_df['is_upset'] = matches_df.apply(self._identify_upset, axis=1)
        matches_df['round_numeric'] = matches_df['round'].map(self._round_to_numeric)
        
        return matches_df
    
    def _parse_sets_won(self, score: str) -> int:
        """Parse number of sets won by winner."""
        if pd.isna(score):
            return 0
        
        try:
            # Count sets (separated by spaces)
            sets = score.split()
            return len([s for s in sets if '-' in s])
        except:
            return 0
    
    def _estimate_duration(self, score: str) -> int:
        """Estimate match duration in minutes based on score."""
        if pd.isna(score):
            return 90  # Default estimate
        
        try:
            sets_count = self._parse_sets_won(score)
            # Rough estimation: 45 minutes per set
            return max(45 * sets_count, 60)
        except:
            return 90
    
    def _identify_upset(self, row) -> bool:
        """Identify potential upsets based on entry type."""
        # Simple heuristic: qualifier beating non-qualifier
        winner_entry = str(row.get('winner_entry', '')).lower()
        loser_entry = str(row.get('loser_entry', '')).lower()
        
        return 'q' in winner_entry and 'q' not in loser_entry
    
    def _round_to_numeric(self, round_str: str) -> int:
        """Convert round string to numeric value for ordering."""
        round_mapping = {
            'F': 7,    # Final
            'SF': 6,   # Semifinal
            'QF': 5,   # Quarterfinal
            'R16': 4,  # Round of 16
            'R32': 3,  # Round of 32
            'R64': 2,  # Round of 64
            'R128': 1, # Round of 128
            'RR': 0    # Round Robin
        }
        
        return round_mapping.get(str(round_str), 0)