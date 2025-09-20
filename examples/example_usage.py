#!/usr/bin/env python3
"""Example usage of Tennis Predictor 101.

Demonstrates how to use the various components of the tennis prediction system.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import get_config
from data.collectors import JeffSackmannCollector, OddsAPICollector
from features import ELORatingSystem, MomentumAnalyzer
from models.base_models import XGBoostModel, RandomForestModel
from models.ensemble import StackingEnsemble
from validation import TournamentBasedCV, ModelEvaluator


def example_data_collection():
    """Example: Collect tennis data."""
    print("=" * 60)
    print("EXAMPLE: Data Collection")
    print("=" * 60)
    
    # Initialize data collector
    collector = JeffSackmannCollector()
    
    # Collect recent ATP matches
    print("Collecting ATP matches for 2023-2024...")
    atp_matches = collector.collect_matches('atp', [2023, 2024])
    print(f"Collected {len(atp_matches)} ATP matches")
    
    # Show sample data
    print("\nSample matches:")
    print(atp_matches[['tourney_name', 'winner_name', 'loser_name', 'surface', 'score']].head())
    
    # Collect player data
    print("\nCollecting player data...")
    players = collector.collect_players('atp')
    print(f"Collected data for {len(players)} players")
    
    return atp_matches, players


def example_elo_system(match_data):
    """Example: ELO rating system."""
    print("\n" + "=" * 60)
    print("EXAMPLE: ELO Rating System")
    print("=" * 60)
    
    # Initialize ELO system
    elo_system = ELORatingSystem()
    
    # Process matches chronologically
    match_data_sorted = match_data.sort_values('tourney_date').head(1000)  # First 1000 for demo
    
    print("Processing matches and updating ELO ratings...")
    
    for idx, match in match_data_sorted.iterrows():
        if idx % 200 == 0:
            print(f"Processed {idx} matches...")
        
        winner_id = match.get('winner_name', f"player_w_{idx}")
        loser_id = match.get('loser_name', f"player_l_{idx}")
        match_date = pd.to_datetime(match['tourney_date'])
        surface = match.get('surface', 'Hard')
        
        # Update ratings
        new_winner_rating, new_loser_rating = elo_system.update_ratings(
            winner_id, loser_id, match_date, surface, 'ATP 250', match.get('score', '')
        )
    
    # Show top players
    ratings_df = elo_system.export_ratings()
    top_players = ratings_df.nlargest(10, 'overall_rating')
    
    print("\nTop 10 players by ELO rating:")
    for i, (_, player) in enumerate(top_players.iterrows(), 1):
        print(f"{i:2d}. {player['player_id']}: {player['overall_rating']:.0f}")
    
    # Example prediction
    if len(top_players) >= 2:
        player1 = top_players.iloc[0]['player_id']
        player2 = top_players.iloc[1]['player_id']
        
        prob = elo_system.calculate_match_probability(player1, player2, 'Hard')
        print(f"\nExample prediction:")
        print(f"{player1} vs {player2} on Hard court")
        print(f"Probability of {player1} winning: {prob:.1%}")
    
    return elo_system


def example_momentum_analysis():
    """Example: Momentum analysis."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Momentum Analysis")
    print("=" * 60)
    
    # Initialize momentum analyzer
    momentum_analyzer = MomentumAnalyzer()
    
    # Create sample player statistics
    sample_stats = {
        'aces': 12,
        'double_faults': 3,
        'first_serve_pct': 0.68,
        'service_points_won': 45,
        'break_points_saved': 4,
        'break_points_faced': 6
    }
    
    # Create sample recent matches data
    recent_matches = pd.DataFrame({
        'service_games_won': [6, 7, 5, 6, 8],
        'first_serve_pct': [0.72, 0.65, 0.70, 0.68, 0.75],
        'break_points_saved': [3, 2, 1, 4, 2],
        'break_points_faced': [4, 3, 2, 5, 3],
        'return_games_won': [2, 3, 1, 2, 4],
        'unforced_errors': [15, 20, 18, 12, 16]
    })
    
    # Calculate momentum score
    print("Calculating momentum analysis...")
    momentum_results = momentum_analyzer.calculate_momentum_score(sample_stats, recent_matches)
    
    print(f"\nMomentum Analysis Results:")
    print(f"Overall Momentum: {momentum_results['overall_momentum']:.3f}")
    print(f"Classification: {momentum_results['momentum_classification']}")
    print(f"Offense Component: {momentum_results['offense_component']:.3f}")
    print(f"Stability Component: {momentum_results['stability_component']:.3f}")
    print(f"Defense Component: {momentum_results['defense_component']:.3f}")
    
    return momentum_analyzer


def example_model_training(features_df):
    """Example: Model training and ensemble."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Model Training")
    print("=" * 60)
    
    # Prepare sample data
    print("Preparing training data...")
    
    # Create sample feature data if not provided
    if features_df is None or len(features_df) < 100:
        print("Creating sample feature data...")
        np.random.seed(42)
        n_samples = 1000
        
        features_df = pd.DataFrame({
            'elo_diff': np.random.normal(0, 100, n_samples),
            'surface_clay': np.random.binomial(1, 0.3, n_samples),
            'surface_grass': np.random.binomial(1, 0.1, n_samples),
            'surface_hard': np.random.binomial(1, 0.6, n_samples),
            'momentum_score': np.random.beta(2, 2, n_samples),
            'recent_form': np.random.beta(2, 2, n_samples),
            'confidence_index': np.random.beta(2, 2, n_samples),
            'is_final': np.random.binomial(1, 0.05, n_samples),
            'indoor_outdoor': np.random.binomial(1, 0.2, n_samples)
        })
        
        # Create target based on elo_diff and some noise
        prob = 1 / (1 + np.exp(-features_df['elo_diff'] / 100))
        features_df['target'] = np.random.binomial(1, prob)
    
    # Prepare features and target
    feature_columns = [col for col in features_df.columns if col != 'target']
    X = features_df[feature_columns]
    y = features_df['target']
    
    print(f"Training data: {len(X)} samples, {len(feature_columns)} features")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train base models
    print("\nTraining base models...")
    
    models = {
        'xgboost': XGBoostModel(n_estimators=100),  # Reduced for demo
        'random_forest': RandomForestModel(n_estimators=100)
    }
    
    trained_models = []
    model_results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        try:
            metrics = model.train(X_train, y_train, X_test, y_test)
            model_results[name] = metrics
            trained_models.append(model)
            print(f"{name} accuracy: {metrics.get('accuracy', 0):.3f}")
        except Exception as e:
            print(f"Error training {name}: {e}")
    
    if not trained_models:
        print("No models trained successfully")
        return None
    
    # Train ensemble
    print("\nTraining stacking ensemble...")
    ensemble = StackingEnsemble(
        base_models=trained_models,
        meta_learner='logistic_regression',
        cv_folds=3  # Reduced for demo
    )
    
    ensemble_metrics = ensemble.train(X_train, y_train, X_test, y_test)
    print(f"Ensemble accuracy: {ensemble_metrics.get('accuracy', 0):.3f}")
    
    # Show model weights
    weights = ensemble.get_base_model_weights()
    print("\nBase model weights in ensemble:")
    for model_name, weight in weights.items():
        print(f"{model_name}: {weight:.3f}")
    
    return ensemble, model_results


def example_cross_validation(X, y):
    """Example: Cross-validation."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Cross-Validation")
    print("=" * 60)
    
    # Create sample tournament data
    tournaments = [f"Tournament_{i}" for i in range(1, 11)] * (len(X) // 10 + 1)
    tournament_data = tournaments[:len(X)]
    X_with_tournament = X.copy()
    X_with_tournament['tournament_id'] = tournament_data
    
    # Tournament-based cross-validation
    print("Performing tournament-based cross-validation...")
    
    cv = TournamentBasedCV(n_splits=3, tournament_col='tournament_id')
    
    # Simple model for CV demo
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    # Perform CV
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(model, X_with_tournament.drop('tournament_id', axis=1), y, cv=cv, scoring='accuracy')
    
    print(f"CV Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    
    return cv_scores


def example_prediction_api():
    """Example: Using the prediction API."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Prediction API Usage")
    print("=" * 60)
    
    try:
        import requests
        
        # API endpoint (assuming server is running)
        api_url = "http://localhost:8000"
        
        # Test health check
        print("Testing API health check...")
        response = requests.get(f"{api_url}/health", timeout=5)
        
        if response.status_code == 200:
            health = response.json()
            print(f"API Status: {health['status']}")
            print(f"Model Loaded: {health['model_loaded']}")
        else:
            print("API health check failed")
            return
        
        # Example prediction request
        print("\nTesting match prediction...")
        
        prediction_request = {
            "player1": {
                "player_id": "novak_djokovic",
                "name": "Novak Djokovic",
                "ranking": 1
            },
            "player2": {
                "player_id": "carlos_alcaraz",
                "name": "Carlos Alcaraz",
                "ranking": 2
            },
            "surface": "Hard",
            "tournament": "US Open",
            "round_info": "F",
            "best_of": 5
        }
        
        response = requests.post(
            f"{api_url}/predict",
            json=prediction_request,
            timeout=10
        )
        
        if response.status_code == 200:
            prediction = response.json()
            print(f"\nPrediction Results:")
            print(f"Player 1 Win Probability: {prediction['player1_win_probability']:.1%}")
            print(f"Player 2 Win Probability: {prediction['player2_win_probability']:.1%}")
            print(f"Prediction Confidence: {prediction['prediction_confidence']:.1%}")
            print(f"ELO Difference: {prediction['elo_difference']:.0f}")
            print(f"Key Factors: {', '.join(prediction['key_factors'])}")
        else:
            print(f"Prediction request failed: {response.status_code}")
            print(response.text)
    
    except ImportError:
        print("Requests library not available. Install with: pip install requests")
    except requests.exceptions.RequestException as e:
        print(f"API not available (make sure server is running): {e}")
        print("Start the API server with: python api/prediction_server.py")


def main():
    """Run all examples."""
    print("Tennis Predictor 101 - Example Usage")
    print("=====================================")
    
    try:
        # Example 1: Data Collection
        match_data, player_data = example_data_collection()
        
        # Example 2: ELO Rating System
        elo_system = example_elo_system(match_data)
        
        # Example 3: Momentum Analysis
        momentum_analyzer = example_momentum_analysis()
        
        # Example 4: Model Training
        ensemble, model_results = example_model_training(None)
        
        # Example 5: Cross-Validation (using sample data)
        if ensemble is not None:
            # Create sample data for CV
            np.random.seed(42)
            n_samples = 500
            X_sample = pd.DataFrame({
                'elo_diff': np.random.normal(0, 100, n_samples),
                'momentum_score': np.random.beta(2, 2, n_samples),
                'surface_clay': np.random.binomial(1, 0.3, n_samples)
            })
            y_sample = np.random.binomial(1, 0.5, n_samples)
            
            cv_scores = example_cross_validation(X_sample, y_sample)
        
        # Example 6: Prediction API
        example_prediction_api()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run the full training pipeline: python scripts/train_ensemble.py")
        print("2. Start the prediction API: python api/prediction_server.py")
        print("3. Access API documentation at: http://localhost:8000/docs")
        print("4. Customize configuration in: config/config.yaml")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")


if __name__ == "__main__":
    main()