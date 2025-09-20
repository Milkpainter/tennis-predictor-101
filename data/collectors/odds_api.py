"""The Odds API collector for real-time tennis betting odds.

Integrates with The Odds API to collect live and pre-match
betting odds from multiple bookmakers.
"""

import pandas as pd
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from .base_collector import BaseCollector


class OddsAPICollector(BaseCollector):
    """Collector for The Odds API tennis betting odds.
    
    Provides access to real-time odds from 20+ bookmakers
    for tennis matches worldwide.
    """
    
    def __init__(self):
        super().__init__("odds_api", rate_limit=6.0)  # API limit: 10 requests/min
        
        # API configuration
        self.api_key = os.getenv('ODDS_API_KEY')
        if not self.api_key:
            raise ValueError("ODDS_API_KEY environment variable not set")
        
        self.base_url = self.config.get('data_sources.odds_api.base_url')
        self.endpoints = self.config.get('data_sources.odds_api.endpoints')
        
        # Tennis sport key
        self.sport_key = 'tennis'
        
        # Default parameters
        self.default_regions = ['us', 'uk', 'eu', 'au']
        self.default_markets = ['h2h', 'spreads', 'totals']
        self.default_bookmakers = [
            'pinnacle', 'bet365', 'betfair', 'william_hill',
            'unibet', '888sport', 'betway', 'marathon'
        ]
    
    def collect_live_odds(self, markets: List[str] = None, 
                         bookmakers: List[str] = None) -> pd.DataFrame:
        """Collect live tennis odds.
        
        Args:
            markets: List of markets to collect (h2h, spreads, totals)
            bookmakers: List of bookmaker keys
            
        Returns:
            DataFrame with live odds
        """
        if markets is None:
            markets = self.default_markets
        if bookmakers is None:
            bookmakers = self.default_bookmakers
        
        odds_data = []
        
        for market in markets:
            try:
                market_odds = self._fetch_odds(market, live=True, bookmakers=bookmakers)
                odds_data.extend(market_odds)
            except Exception as e:
                self.logger.warning(f"Failed to collect live odds for {market}: {e}")
                continue
        
        if not odds_data:
            return pd.DataFrame()
        
        odds_df = pd.DataFrame(odds_data)
        return self._process_odds_data(odds_df)
    
    def collect_upcoming_odds(self, days_ahead: int = 7,
                            markets: List[str] = None) -> pd.DataFrame:
        """Collect odds for upcoming matches.
        
        Args:
            days_ahead: Number of days ahead to collect
            markets: List of markets to collect
            
        Returns:
            DataFrame with upcoming match odds
        """
        if markets is None:
            markets = self.default_markets
        
        odds_data = []
        
        for market in markets:
            try:
                market_odds = self._fetch_odds(market, live=False)
                # Filter for matches within specified days
                filtered_odds = self._filter_by_date(market_odds, days_ahead)
                odds_data.extend(filtered_odds)
            except Exception as e:
                self.logger.warning(f"Failed to collect upcoming odds for {market}: {e}")
                continue
        
        if not odds_data:
            return pd.DataFrame()
        
        odds_df = pd.DataFrame(odds_data)
        return self._process_odds_data(odds_df)
    
    def collect_odds_history(self, match_id: str) -> pd.DataFrame:
        """Collect historical odds movement for a specific match.
        
        Args:
            match_id: Unique match identifier
            
        Returns:
            DataFrame with odds history
        """
        # Note: Historical odds require premium API access
        # This is a placeholder for the implementation
        
        url = f"{self.base_url}sports/{self.sport_key}/events/{match_id}/odds/history"
        params = {
            'apiKey': self.api_key,
            'regions': ','.join(self.default_regions),
            'markets': 'h2h'
        }
        
        try:
            response = self._make_request(url, params=params)
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            return self._process_historical_odds(data)
            
        except Exception as e:
            self.logger.error(f"Failed to collect odds history for {match_id}: {e}")
            return pd.DataFrame()
    
    def _fetch_odds(self, market: str, live: bool = False, 
                   bookmakers: List[str] = None) -> List[Dict]:
        """Fetch odds for specific market.
        
        Args:
            market: Market type (h2h, spreads, totals)
            live: Whether to fetch live odds
            bookmakers: List of bookmaker keys
            
        Returns:
            List of odds dictionaries
        """
        endpoint = 'live-scores' if live else 'odds'
        url = f"{self.base_url}sports/{self.sport_key}/{endpoint}"
        
        params = {
            'apiKey': self.api_key,
            'regions': ','.join(self.default_regions),
            'markets': market,
            'oddsFormat': 'decimal',
            'dateFormat': 'iso'
        }
        
        if bookmakers:
            params['bookmakers'] = ','.join(bookmakers)
        
        if not live:
            # For upcoming matches, set date range
            params['commenceTimeFrom'] = datetime.now().isoformat()
            params['commenceTimeTo'] = (datetime.now() + timedelta(days=7)).isoformat()
        
        response = self._make_request(url, params=params)
        data = response.json()
        
        odds_list = []
        
        for event in data:
            event_data = {
                'event_id': event['id'],
                'sport_key': event['sport_key'],
                'sport_title': event['sport_title'],
                'commence_time': event['commence_time'],
                'home_team': event['home_team'],
                'away_team': event['away_team'],
                'market': market
            }
            
            # Add live score if available
            if 'scores' in event:
                event_data.update({
                    'home_score': event['scores'][0]['score'] if event['scores'] else None,
                    'away_score': event['scores'][1]['score'] if len(event['scores']) > 1 else None,
                    'last_update': event.get('last_update')
                })
            
            # Process bookmaker odds
            if 'bookmakers' in event:
                for bookmaker in event['bookmakers']:
                    bookmaker_data = event_data.copy()
                    bookmaker_data.update({
                        'bookmaker': bookmaker['key'],
                        'bookmaker_title': bookmaker['title'],
                        'last_update': bookmaker['last_update']
                    })
                    
                    # Add market-specific odds
                    for market_data in bookmaker['markets']:
                        if market_data['key'] == market:
                            for outcome in market_data['outcomes']:
                                outcome_data = bookmaker_data.copy()
                                outcome_data.update({
                                    'outcome_name': outcome['name'],
                                    'outcome_price': outcome['price'],
                                    'point': outcome.get('point'),  # For spreads/totals
                                })
                                odds_list.append(outcome_data)
            
        return odds_list
    
    def _filter_by_date(self, odds_data: List[Dict], days_ahead: int) -> List[Dict]:
        """Filter odds data by date range."""
        cutoff_date = datetime.now() + timedelta(days=days_ahead)
        
        filtered_data = []
        for odds in odds_data:
            commence_time = datetime.fromisoformat(odds['commence_time'].replace('Z', '+00:00'))
            if commence_time <= cutoff_date:
                filtered_data.append(odds)
        
        return filtered_data
    
    def _process_odds_data(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean odds data."""
        if odds_df.empty:
            return odds_df
        
        # Convert datetime columns
        odds_df['commence_time'] = pd.to_datetime(odds_df['commence_time'])
        odds_df['last_update'] = pd.to_datetime(odds_df['last_update'])
        
        # Add derived features
        odds_df['implied_probability'] = 1 / odds_df['outcome_price']
        odds_df['data_collection_time'] = datetime.now()
        
        # Calculate market overround by event and bookmaker
        odds_df['overround'] = odds_df.groupby(['event_id', 'bookmaker', 'market'])['implied_probability'].transform('sum')
        
        # Calculate true probability (removing overround)
        odds_df['true_probability'] = odds_df['implied_probability'] / odds_df['overround']
        
        return odds_df
    
    def _process_historical_odds(self, historical_data: Dict) -> pd.DataFrame:
        """Process historical odds data."""
        # Implementation for historical odds processing
        # This would parse the historical odds API response
        return pd.DataFrame()
    
    def collect(self, data_type: str = 'live', **kwargs) -> pd.DataFrame:
        """Main collection method.
        
        Args:
            data_type: 'live', 'upcoming', or 'history'
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with odds data
        """
        if data_type == 'live':
            return self.collect_live_odds(**kwargs)
        elif data_type == 'upcoming':
            return self.collect_upcoming_odds(**kwargs)
        elif data_type == 'history':
            match_id = kwargs.get('match_id')
            if not match_id:
                raise ValueError("match_id required for historical odds")
            return self.collect_odds_history(match_id)
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
    
    def get_latest(self) -> pd.DataFrame:
        """Get latest live odds."""
        return self.collect_live_odds()
    
    def get_bookmaker_summary(self) -> pd.DataFrame:
        """Get summary of available bookmakers."""
        url = f"{self.base_url}sports/{self.sport_key}/odds"
        params = {
            'apiKey': self.api_key,
            'regions': ','.join(self.default_regions)
        }
        
        try:
            response = self._make_request(url, params=params)
            data = response.json()
            
            bookmakers = set()
            for event in data:
                if 'bookmakers' in event:
                    for bm in event['bookmakers']:
                        bookmakers.add((bm['key'], bm['title']))
            
            return pd.DataFrame(list(bookmakers), columns=['key', 'title'])
            
        except Exception as e:
            self.logger.error(f"Failed to get bookmaker summary: {e}")
            return pd.DataFrame()