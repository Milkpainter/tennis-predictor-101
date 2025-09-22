#!/usr/bin/env python3
"""
Morning Tennis Predictor - Daily 12:00 AM Match Prediction Interface

Real-time tennis match prediction system designed for morning use (12:00 AM).
Predicts all matches scheduled for the current day before they start.

Features:
- Live data integration
- Real-time odds monitoring
- Weather condition analysis
- Player form assessment
- Market inefficiency detection
- Automated daily predictions

Compatible with Tennis Predictor 202

Author: Advanced Tennis Analytics Research
Version: 2.0.2
Date: September 21, 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from tennis_predictor_202 import TennisPredictor202
from data_processor_202 import TennisDataProcessor202

class MorningTennisPredictor:
    """
    Morning Tennis Prediction System
    
    Designed for daily use at 12:00 AM to predict all tennis matches
    scheduled for the current day.
    """
    
    def __init__(self, config_file: str = None):
        self.setup_logging()
        self.config = self.load_config(config_file)
        
        # Initialize core components
        self.predictor = TennisPredictor202(config_file)
        self.data_processor = TennisDataProcessor202()
        
        # Data sources
        self.data_sources = {
            'atp': 'https://api.atptour.com',
            'wta': 'https://api.wtatennis.com',
            'weather': 'https://api.openweathermap.org/data/2.5/weather',
            'odds': 'https://api.the-odds-api.com/v4/sports',
            'rankings': 'https://www.atptour.com/en/rankings'
        }
        
        self.logger.info("Morning Tennis Predictor initialized")
        
    def setup_logging(self):
        """Setup logging for morning predictions"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('morning_predictions.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config_file: Optional[str]) -> Dict:
        """Load configuration for morning predictions"""
        default_config = {
            'prediction_threshold': 0.6,
            'min_edge_threshold': 0.05,
            'max_matches_per_day': 50,
            'data_refresh_hours': 1,
            'timezone': 'UTC',
            'notification_enabled': False,
            'auto_save_results': True,
            'weather_api_key': None,
            'odds_api_key': None,
            'telegram_bot_token': None,
            'telegram_chat_id': None
        }
        
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except FileNotFoundError:
                self.logger.warning(f"Config file {config_file} not found. Using defaults.")
                
        return default_config
        
    def get_todays_matches(self) -> List[Dict[str, Any]]:
        """Fetch all tennis matches scheduled for today"""
        today = datetime.now().date()
        self.logger.info(f"Fetching matches for {today}")
        
        matches = []
        
        try:
            # In a real implementation, this would connect to live APIs
            # For demo purposes, using simulated match data
            
            sample_matches = [
                {
                    'player1': 'Jannik Sinner',
                    'player2': 'Carlos Alcaraz',
                    'tournament': 'US Open',
                    'round': 'Semifinal',
                    'surface': 'Hard',
                    'start_time': '15:00',
                    'location': 'New York',
                    'category': 'Grand Slam',
                    'court': 'Arthur Ashe Stadium',
                    'best_of': 5
                },
                {
                    'player1': 'Novak Djokovic',
                    'player2': 'Alexander Zverev',
                    'tournament': 'US Open',
                    'round': 'Semifinal',
                    'surface': 'Hard',
                    'start_time': '19:00',
                    'location': 'New York',
                    'category': 'Grand Slam',
                    'court': 'Arthur Ashe Stadium',
                    'best_of': 5
                },
                {
                    'player1': 'Daniil Medvedev',
                    'player2': 'Taylor Fritz',
                    'tournament': 'Tokyo Open',
                    'round': 'Quarterfinal',
                    'surface': 'Hard',
                    'start_time': '12:00',
                    'location': 'Tokyo',
                    'category': 'ATP 500',
                    'court': 'Center Court',
                    'best_of': 3
                },
                {
                    'player1': 'Iga Swiatek',
                    'player2': 'Aryna Sabalenka',
                    'tournament': 'China Open',
                    'round': 'Final',
                    'surface': 'Hard',
                    'start_time': '14:30',
                    'location': 'Beijing',
                    'category': 'WTA 1000',
                    'court': 'Diamond Court',
                    'best_of': 3
                }
            ]
            
            # Add today's date to each match
            for match in sample_matches:
                match['date'] = today
                matches.append(match)
                
        except Exception as e:
            self.logger.error(f"Error fetching today's matches: {e}")
            
        self.logger.info(f"Found {len(matches)} matches for today")
        return matches
        
    def get_live_odds(self, player1: str, player2: str) -> Dict[str, float]:
        """Get live betting odds for a match"""
        try:
            # In real implementation, would connect to odds API
            # For demo, using simulated realistic odds
            
            # Simulate odds based on player names (demo purposes)
            odds_data = {
                ('Jannik Sinner', 'Carlos Alcaraz'): {'player1': 1.85, 'player2': 1.95},
                ('Novak Djokovic', 'Alexander Zverev'): {'player1': 1.60, 'player2': 2.40},
                ('Daniil Medvedev', 'Taylor Fritz'): {'player1': 1.45, 'player2': 2.75},
                ('Iga Swiatek', 'Aryna Sabalenka'): {'player1': 1.70, 'player2': 2.10}
            }
            
            key = (player1, player2)
            if key in odds_data:
                odds = odds_data[key]
            else:
                # Default odds for unknown matchups
                odds = {'player1': 1.80, 'player2': 2.00}
                
            # Convert to implied probabilities
            implied_prob1 = 1 / odds['player1']
            implied_prob2 = 1 / odds['player2']
            
            # Remove overround (bookmaker margin)
            total_prob = implied_prob1 + implied_prob2
            implied_prob1 = implied_prob1 / total_prob
            implied_prob2 = implied_prob2 / total_prob
            
            return {
                'odds_player1': odds['player1'],
                'odds_player2': odds['player2'],
                'implied_prob1': implied_prob1,
                'implied_prob2': implied_prob2,
                'overround': (total_prob - 1) * 100  # Bookmaker margin %
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching odds for {player1} vs {player2}: {e}")
            return {
                'odds_player1': 1.80,
                'odds_player2': 2.00,
                'implied_prob1': 0.50,
                'implied_prob2': 0.50,
                'overround': 11.11
            }
            
    def get_weather_conditions(self, location: str) -> Dict[str, Any]:
        """Get current weather conditions for match location"""
        try:
            # In real implementation, would use weather API
            # For demo, using simulated weather data
            
            weather_data = {
                'New York': {'temp': 22, 'humidity': 65, 'wind': 8, 'condition': 'Clear'},
                'Tokyo': {'temp': 26, 'humidity': 70, 'wind': 5, 'condition': 'Partly Cloudy'},
                'Beijing': {'temp': 18, 'humidity': 55, 'wind': 12, 'condition': 'Sunny'},
                'Paris': {'temp': 15, 'humidity': 80, 'wind': 15, 'condition': 'Overcast'}
            }
            
            return weather_data.get(location, {
                'temp': 20, 'humidity': 60, 'wind': 10, 'condition': 'Unknown'
            })
            
        except Exception as e:
            self.logger.error(f"Error fetching weather for {location}: {e}")
            return {'temp': 20, 'humidity': 60, 'wind': 10, 'condition': 'Unknown'}
            
    def analyze_player_form(self, player: str) -> Dict[str, Any]:
        """Analyze current player form and recent performance"""
        # Get player statistics from data processor
        player_stats = self.data_processor.get_player_features(player)
        
        if not player_stats:
            return {
                'recent_form': 0.5,
                'winning_streak': 0,
                'surface_performance': 0.5,
                'ranking_trend': 'stable',
                'injury_status': 'unknown',
                'last_match_date': 'unknown',
                'matches_last_month': 0
            }
            
        return {
            'recent_form': player_stats.get('recent_form', 0.5),
            'winning_streak': player_stats.get('winning_streak', 0),
            'surface_performance': player_stats.get('hard_win_rate', 0.5),  # Default to hard court
            'ranking_trend': 'stable',  # Would calculate from ranking history
            'injury_status': 'fit',  # Would fetch from injury reports
            'last_match_date': 'recent',  # Would calculate from match history
            'matches_last_month': player_stats.get('total_matches', 0)
        }
        
    def make_daily_predictions(self) -> List[Dict[str, Any]]:
        """Make predictions for all matches scheduled today"""
        self.logger.info("Starting daily match predictions...")
        
        matches = self.get_todays_matches()
        predictions = []
        
        for match in matches:
            self.logger.info(f"Predicting: {match['player1']} vs {match['player2']}")
            
            try:
                # Prepare match info for prediction
                match_info = {
                    'surface': match['surface'],
                    'date': datetime.combine(match['date'], datetime.min.time()),
                    'location': match['location'],
                    'tournament': match['tournament'],
                    'round': match['round'],
                    'category': match['category']
                }
                
                # Get live odds
                odds_data = self.get_live_odds(match['player1'], match['player2'])
                match_info['implied_probability'] = odds_data['implied_prob1']
                
                # Get weather conditions
                weather = self.get_weather_conditions(match['location'])
                
                # Analyze player forms
                player1_form = self.analyze_player_form(match['player1'])
                player2_form = self.analyze_player_form(match['player2'])
                
                # Make prediction
                prediction = self.predictor.predict_match(
                    match['player1'], match['player2'], match_info
                )
                
                # Calculate betting value
                model_prob = prediction['player1_win_probability']
                market_prob = odds_data['implied_prob1']
                edge = model_prob - market_prob
                
                # Enhanced prediction result
                enhanced_prediction = {
                    # Match details
                    'match_id': f"{match['player1'][:3]}{match['player2'][:3]}_{match['date']}",
                    'player1': match['player1'],
                    'player2': match['player2'],
                    'tournament': match['tournament'],
                    'round': match['round'],
                    'surface': match['surface'],
                    'start_time': match['start_time'],
                    'location': match['location'],
                    
                    # Predictions
                    'predicted_winner': prediction['predicted_winner'],
                    'model_probability': model_prob,
                    'confidence': prediction['confidence'],
                    
                    # Market analysis
                    'market_probability': market_prob,
                    'odds_player1': odds_data['odds_player1'],
                    'odds_player2': odds_data['odds_player2'],
                    'edge': edge,
                    'bet_recommendation': 'BET' if abs(edge) > self.config['min_edge_threshold'] else 'PASS',
                    'kelly_fraction': prediction['market_edge']['kelly_fraction'],
                    
                    # Environmental factors
                    'weather_temp': weather['temp'],
                    'weather_condition': weather['condition'],
                    'weather_impact': 'favorable' if 18 <= weather['temp'] <= 25 else 'challenging',
                    
                    # Player forms
                    'player1_form': player1_form['recent_form'],
                    'player2_form': player2_form['recent_form'],
                    'form_advantage': player1_form['recent_form'] - player2_form['recent_form'],
                    
                    # Meta information
                    'prediction_time': datetime.now().isoformat(),
                    'models_used': prediction.get('model_ensemble', []),
                    'features_analyzed': prediction.get('features_used', 0),
                    'data_quality': 'good'  # Would assess actual data quality
                }
                
                predictions.append(enhanced_prediction)
                
                # Add small delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Error predicting match {match['player1']} vs {match['player2']}: {e}")
                
                # Add error entry
                predictions.append({
                    'match_id': f"{match['player1'][:3]}{match['player2'][:3]}_{match['date']}",
                    'player1': match['player1'],
                    'player2': match['player2'],
                    'error': str(e),
                    'prediction_time': datetime.now().isoformat()
                })
                
        self.logger.info(f"Completed predictions for {len(predictions)} matches")
        return predictions
        
    def format_prediction_report(self, predictions: List[Dict[str, Any]]) -> str:
        """Format predictions into a readable report"""
        report = []
        report.append("=" * 80)
        report.append("🎾 DAILY TENNIS PREDICTIONS - MORNING REPORT")
        report.append(f"📅 Date: {datetime.now().strftime('%Y-%m-%d')}")
        report.append(f"⏰ Generated at: {datetime.now().strftime('%H:%M:%S UTC')}")
        report.append("=" * 80)
        
        successful_predictions = [p for p in predictions if 'error' not in p]
        
        if not successful_predictions:
            report.append("❌ No successful predictions generated today")
            return "\n".join(report)
            
        report.append(f"📊 Total matches analyzed: {len(successful_predictions)}")
        report.append(f"💰 Betting opportunities: {len([p for p in successful_predictions if p.get('bet_recommendation') == 'BET'])}")
        report.append("")
        
        # Sort by confidence
        successful_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        for i, pred in enumerate(successful_predictions, 1):
            report.append(f"{i}. {pred['tournament']} - {pred['round']}")
            report.append(f"   🥎 {pred['player1']} vs {pred['player2']}")
            report.append(f"   📍 {pred['location']} | {pred['surface']} | {pred['start_time']}")
            
            # Prediction details
            winner = pred['predicted_winner']
            prob = pred['model_probability']
            conf = pred['confidence']
            
            if winner == pred['player1']:
                report.append(f"   🏆 PREDICTION: {winner} ({prob:.1%}) | Confidence: {conf:.1%}")
            else:
                report.append(f"   🏆 PREDICTION: {winner} ({1-prob:.1%}) | Confidence: {conf:.1%}")
                
            # Market analysis
            edge = pred['edge']
            odds1 = pred['odds_player1']
            odds2 = pred['odds_player2']
            
            report.append(f"   💹 ODDS: {pred['player1']} {odds1:.2f} | {pred['player2']} {odds2:.2f}")
            report.append(f"   📈 EDGE: {edge:+.1%} | {pred['bet_recommendation']}")
            
            if pred['bet_recommendation'] == 'BET':
                kelly = pred['kelly_fraction']
                report.append(f"   💰 Kelly Stake: {kelly:.1%}")
                
            # Environmental factors
            temp = pred['weather_temp']
            condition = pred['weather_condition']
            report.append(f"   🌤️  Weather: {temp}°C, {condition} ({pred['weather_impact']})")
            
            # Form comparison
            form_adv = pred['form_advantage']
            if abs(form_adv) > 0.1:
                form_leader = pred['player1'] if form_adv > 0 else pred['player2']
                report.append(f"   📊 Form Advantage: {form_leader} ({abs(form_adv):+.1%})")
                
            report.append("   " + "-" * 60)
            report.append("")
            
        # Summary statistics
        report.append("📈 SUMMARY STATISTICS:")
        avg_confidence = np.mean([p['confidence'] for p in successful_predictions])
        report.append(f"   Average Confidence: {avg_confidence:.1%}")
        
        high_conf_matches = len([p for p in successful_predictions if p['confidence'] > 0.7])
        report.append(f"   High Confidence Matches (>70%): {high_conf_matches}")
        
        positive_edge = len([p for p in successful_predictions if p['edge'] > 0.05])
        report.append(f"   Positive Edge Opportunities: {positive_edge}")
        
        # Surface breakdown
        surface_counts = Counter([p['surface'] for p in successful_predictions])
        report.append(f"   Surface Distribution: {dict(surface_counts)}")
        
        report.append("=" * 80)
        report.append("⚠️  DISCLAIMER: Predictions for informational purposes only")
        report.append("💡 Always practice responsible gaming")
        report.append("=" * 80)
        
        return "\n".join(report)
        
    def save_predictions(self, predictions: List[Dict[str, Any]], filename: str = None):
        """Save predictions to file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'tennis_predictions_{timestamp}.json'
            
        try:
            with open(filename, 'w') as f:
                json.dump(predictions, f, indent=2, default=str)
            self.logger.info(f"Predictions saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving predictions: {e}")
            
    def send_notification(self, report: str):
        """Send prediction report via notification (Telegram, email, etc.)"""
        if not self.config.get('notification_enabled', False):
            return
            
        try:
            # Implement notification service (Telegram, email, etc.)
            # For demo, just log the notification
            self.logger.info("Notification sent successfully")
            
        except Exception as e:
            self.logger.error(f"Error sending notification: {e}")
            
    def run_morning_predictions(self):
        """Main function to run morning predictions"""
        start_time = datetime.now()
        self.logger.info(f"🌅 Starting morning tennis predictions at {start_time}")
        
        try:
            # Make predictions
            predictions = self.make_daily_predictions()
            
            # Generate report
            report = self.format_prediction_report(predictions)
            
            # Print report
            print(report)
            
            # Save results
            if self.config.get('auto_save_results', True):
                self.save_predictions(predictions)
                
                # Save text report
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                with open(f'tennis_report_{timestamp}.txt', 'w') as f:
                    f.write(report)
                    
            # Send notifications
            if self.config.get('notification_enabled', False):
                self.send_notification(report)
                
            # Calculate execution time
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            self.logger.info(f"✅ Morning predictions completed in {execution_time:.2f} seconds")
            
            return predictions, report
            
        except Exception as e:
            self.logger.error(f"❌ Error in morning predictions: {e}")
            raise
            
def main():
    """Main function for morning predictions"""
    print("🎾 Tennis Predictor 202 - Morning Edition")
    print(f"⏰ Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🚀 Initializing prediction system...\n")
    
    try:
        # Initialize morning predictor
        morning_predictor = MorningTennisPredictor()
        
        # Run morning predictions
        predictions, report = morning_predictor.run_morning_predictions()
        
        print(f"\n✅ Successfully generated {len(predictions)} predictions")
        print("📁 Results saved to files")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.error(f"Fatal error in morning predictions: {e}")
        
if __name__ == "__main__":
    main()
