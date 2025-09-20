#!/usr/bin/env python3
"""Example: Predict a tennis match using Ultimate Tennis Predictor 101.

Demonstrates how to use the system to predict match outcomes with
comprehensive analysis including momentum, surface effects, and betting insights.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from prediction_engine.ultimate_predictor import UltimateTennisPredictor
from features.environmental import EnvironmentalConditions, CourtType


def main():
    """Example match prediction."""
    
    parser = argparse.ArgumentParser(description='Predict tennis match outcome')
    parser.add_argument('--player1', required=True, help='First player name')
    parser.add_argument('--player2', required=True, help='Second player name')
    parser.add_argument('--tournament', default='US Open', help='Tournament name')
    parser.add_argument('--surface', choices=['Clay', 'Hard', 'Grass'], 
                       default='Hard', help='Court surface')
    parser.add_argument('--round', default='R32', help='Tournament round')
    parser.add_argument('--model-path', help='Path to trained model')
    parser.add_argument('--temperature', type=float, default=25.0, help='Temperature (Celsius)')
    parser.add_argument('--humidity', type=float, default=60.0, help='Humidity (%)')
    parser.add_argument('--wind-speed', type=float, default=15.0, help='Wind speed (km/h)')
    parser.add_argument('--odds-player1', type=float, help='Decimal odds for player 1')
    parser.add_argument('--odds-player2', type=float, help='Decimal odds for player 2')
    
    args = parser.parse_args()
    
    print("\n🎾 TENNIS PREDICTOR 101 - MATCH PREDICTION 🎾\n")
    print(f"Match: {args.player1} vs {args.player2}")
    print(f"Tournament: {args.tournament}")
    print(f"Surface: {args.surface}")
    print(f"Round: {args.round}")
    print(f"Conditions: {args.temperature}°C, {args.humidity}% humidity, {args.wind_speed} km/h wind")
    print("-" * 60)
    
    try:
        # Initialize predictor
        if args.model_path:
            predictor = UltimateTennisPredictor(model_path=args.model_path)
        else:
            # Try to find latest model
            models_dir = project_root / 'models' / 'saved'
            if models_dir.exists():
                model_files = list(models_dir.glob('stacking_ensemble_*.pkl'))
                if model_files:
                    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                    print(f"Using model: {latest_model}")
                    predictor = UltimateTennisPredictor(model_path=str(latest_model))
                else:
                    print("No trained models found. Please train the system first.")
                    print("Run: python scripts/train_ultimate_system.py")
                    sys.exit(1)
            else:
                print("Models directory not found. Please train the system first.")
                sys.exit(1)
        
        # Setup environmental conditions
        conditions = EnvironmentalConditions(
            temperature=args.temperature,
            humidity=args.humidity,
            wind_speed=args.wind_speed,
            altitude=100.0,  # Default altitude
            court_type=CourtType.OUTDOOR
        )
        
        # Setup betting odds if provided
        betting_odds = None
        if args.odds_player1 and args.odds_player2:
            betting_odds = {
                'player1_decimal_odds': args.odds_player1,
                'player2_decimal_odds': args.odds_player2
            }
            print(f"Betting odds: {args.player1} ({args.odds_player1}), {args.player2} ({args.odds_player2})")
        
        # Make prediction
        print("\nMaking prediction...")
        prediction = predictor.predict_match(
            player1_id=args.player1,
            player2_id=args.player2,
            tournament=args.tournament,
            surface=args.surface,
            round_info=args.round,
            environmental_conditions=conditions,
            betting_odds=betting_odds
        )
        
        # Display results
        print("\n" + "=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        
        # Main prediction
        winner = args.player1 if prediction.player1_win_probability > 0.5 else args.player2
        win_prob = max(prediction.player1_win_probability, prediction.player2_win_probability)
        
        print(f"\n🏆 PREDICTED WINNER: {winner}")
        print(f"Win Probability: {win_prob:.1%}")
        print(f"Confidence: {prediction.confidence:.1%}")
        
        print(f"\n📊 DETAILED PROBABILITIES:")
        print(f"{args.player1}: {prediction.player1_win_probability:.1%}")
        print(f"{args.player2}: {prediction.player2_win_probability:.1%}")
        
        # Prediction breakdown
        print(f"\n🔍 PREDICTION BREAKDOWN:")
        breakdown = prediction.prediction_breakdown
        print(f"ELO Contribution: {breakdown['elo_contribution']:.3f}")
        print(f"Momentum Contribution: {breakdown['momentum_contribution']:.3f}")
        print(f"Surface Contribution: {breakdown['surface_contribution']:.3f}")
        print(f"Environmental Contribution: {breakdown['environmental_contribution']:.3f}")
        
        # Momentum analysis
        print(f"\n⚡ MOMENTUM ANALYSIS:")
        momentum = prediction.momentum_analysis
        print(f"{args.player1} Momentum: {momentum['player1_momentum']:.3f}")
        print(f"{args.player2} Momentum: {momentum['player2_momentum']:.3f}")
        print(f"Momentum Advantage: {momentum['momentum_advantage']:.3f}")
        
        # Environmental impact
        print(f"\n🌤️  ENVIRONMENTAL IMPACT:")
        env = prediction.environmental_impact
        print(f"Ball Speed Factor: {env['ball_speed_factor']:.3f}")
        print(f"Fatigue Factor: {env['fatigue_factor']:.3f}")
        print(f"Serve Advantage: {env['serve_advantage_adjustment']:.3f}")
        print(f"Summary: {env['environmental_summary']}")
        
        # Betting analysis
        if prediction.betting_recommendation:
            print(f"\n💰 BETTING ANALYSIS:")
            betting = prediction.betting_recommendation
            
            if betting['recommended_bet']:
                print(f"✅ BET RECOMMENDED on {winner}")
                print(f"Suggested Stake: {betting['suggested_stake']:.1f}% of bankroll")
                print(f"Expected Value: {betting['expected_value']:.3f}")
                print(f"Kelly Fraction: {betting['kelly_fraction']:.3f}")
            else:
                print(f"❌ NO BET RECOMMENDED")
            
            print(f"Risk Assessment: {betting['risk_assessment']}")
            if betting['market_inefficiency']:
                print(f"Market Inefficiency: {betting['market_inefficiency']}")
        
        # Model explanation
        print(f"\n💭 EXPLANATION:")
        print(prediction.model_explanation)
        
        print(f"\n⏰ Prediction made at: {prediction.prediction_timestamp}")
        
        # System status
        status = predictor.get_system_status()
        print(f"\n📈 SYSTEM STATUS:")
        print(f"Total Predictions: {status['performance_metrics']['total_predictions']}")
        if status['performance_metrics']['total_predictions'] > 0:
            print(f"Accuracy: {status['performance_metrics']['accuracy']:.1%}")
        
        print("\n" + "=" * 60)
        print("PREDICTION COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error making prediction: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()