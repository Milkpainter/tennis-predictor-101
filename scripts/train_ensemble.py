#!/usr/bin/env python3
"""Main training script for Tennis Predictor 101.

Comprehensive training pipeline that integrates all components:
- Data collection and preprocessing
- Feature engineering (ELO, momentum, surface, environmental)
- Model training (base models + ensemble)
- Validation and performance evaluation
- Model persistence and monitoring
"""

import os
import sys
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Suppress warnings
warnings.filterwarnings('ignore')

# Import project modules
from config import get_config
from data.collectors import JeffSackmannCollector, OddsAPICollector
from features import (
    ELORatingSystem, MomentumAnalyzer, 
    SurfaceSpecificFeatures, EnvironmentalFeatures
)
from models.base_models import (
    XGBoostModel, RandomForestModel, NeuralNetworkModel, 
    SVMModel, LogisticRegressionModel
)
from models.ensemble import StackingEnsemble
from validation import TournamentBasedCV, ModelEvaluator


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    return logging.getLogger('train_ensemble')


class TennisPredictor101Trainer:
    """Main trainer class for Tennis Predictor 101."""
    
    def __init__(self, config_path: str = None):
        """Initialize trainer.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config = get_config()
        self.logger = logging.getLogger('trainer')
        
        # Initialize components
        self.data_collectors = {}
        self.feature_engineers = {}
        self.models = {}
        self.ensemble = None
        
        # Training artifacts
        self.training_data = None
        self.validation_data = None
        self.feature_data = None
        self.training_metrics = {}
        
        self.logger.info("Tennis Predictor 101 Trainer initialized")
    
    def collect_data(self, years: List[int] = None) -> pd.DataFrame:
        """Collect and prepare training data.
        
        Args:
            years: Years of data to collect
            
        Returns:
            Combined training dataset
        """
        self.logger.info("Starting data collection")
        
        if years is None:
            current_year = datetime.now().year
            years = list(range(current_year - 4, current_year + 1))
        
        # Initialize data collectors
        self.data_collectors['sackmann'] = JeffSackmannCollector()
        
        try:
            self.data_collectors['odds'] = OddsAPICollector()
        except ValueError as e:
            self.logger.warning(f"Odds API not available: {e}")
        
        # Collect ATP and WTA data
        all_matches = []
        
        for tour in ['atp', 'wta']:
            try:
                self.logger.info(f"Collecting {tour.upper()} matches for years {years}")
                matches = self.data_collectors['sackmann'].collect_matches(tour, years)
                all_matches.append(matches)
                
            except Exception as e:
                self.logger.error(f"Error collecting {tour} data: {e}")
                continue
        
        if not all_matches:
            raise ValueError("No match data collected")
        
        # Combine all data
        combined_data = pd.concat(all_matches, ignore_index=True)
        
        # Basic preprocessing
        combined_data = self._preprocess_match_data(combined_data)
        
        self.training_data = combined_data
        self.logger.info(f"Data collection completed: {len(combined_data)} matches")
        
        return combined_data
    
    def engineer_features(self, match_data: pd.DataFrame) -> pd.DataFrame:
        """Engineer comprehensive features for training.
        
        Args:
            match_data: Raw match data
            
        Returns:
            Feature matrix
        """
        self.logger.info("Starting feature engineering")
        
        # Initialize feature engineers
        self.feature_engineers['elo'] = ELORatingSystem()
        self.feature_engineers['momentum'] = MomentumAnalyzer()
        self.feature_engineers['surface'] = SurfaceSpecificFeatures()
        self.feature_engineers['environmental'] = EnvironmentalFeatures()
        
        feature_list = []
        
        # Process matches chronologically for ELO ratings
        match_data_sorted = match_data.sort_values('tourney_date')
        
        self.logger.info("Calculating ELO ratings and features")
        
        for idx, match in match_data_sorted.iterrows():
            try:
                # Extract match information
                winner_id = match.get('winner_id', match.get('winner_name', f"player_{idx}_w"))
                loser_id = match.get('loser_id', match.get('loser_name', f"player_{idx}_l"))
                match_date = pd.to_datetime(match['tourney_date'])
                surface = match.get('surface', 'Hard')
                
                # Get pre-match ELO ratings
                winner_elo = self.feature_engineers['elo'].get_rating(winner_id, surface)
                loser_elo = self.feature_engineers['elo'].get_rating(loser_id, surface)
                
                # Calculate match probability
                match_prob = self.feature_engineers['elo'].calculate_match_probability(
                    winner_id, loser_id, surface
                )
                
                # Create feature vector for this match
                features = {
                    'match_id': idx,
                    'winner_id': winner_id,
                    'loser_id': loser_id,
                    'date': match_date,
                    'surface': surface,
                    'tournament': match.get('tourney_name', 'Unknown'),
                    
                    # ELO features
                    'winner_elo': winner_elo,
                    'loser_elo': loser_elo,
                    'elo_diff': winner_elo - loser_elo,
                    'match_probability': match_prob,
                    
                    # Match features
                    'draw_size': match.get('draw_size', 64),
                    'round': match.get('round', 'R32'),
                    'best_of': match.get('best_of', 3),
                    
                    # Target (winner = 1, loser = 0)
                    'target': 1  # This match represents winner winning
                }
                
                # Add surface-specific features
                surface_features = self._calculate_surface_features(match, winner_id, loser_id)
                features.update(surface_features)
                
                # Add momentum features (simplified for initial version)
                momentum_features = self._calculate_momentum_features(match, winner_id, loser_id)
                features.update(momentum_features)
                
                # Add environmental features
                env_features = self._calculate_environmental_features(match)
                features.update(env_features)
                
                feature_list.append(features)
                
                # Update ELO ratings after match
                tournament_category = self._categorize_tournament(match.get('tourney_name', ''))
                self.feature_engineers['elo'].update_ratings(
                    winner_id, loser_id, match_date, surface, 
                    tournament_category, match.get('score', '')
                )
                
                # Create symmetric entry for loser
                loser_features = features.copy()
                loser_features.update({
                    'winner_id': loser_id,
                    'loser_id': winner_id,
                    'winner_elo': loser_elo,
                    'loser_elo': winner_elo,
                    'elo_diff': loser_elo - winner_elo,
                    'match_probability': 1 - match_prob,
                    'target': 0  # Loser losing
                })
                
                feature_list.append(loser_features)
                
                if idx % 1000 == 0:
                    self.logger.info(f"Processed {idx} matches")
                    
            except Exception as e:
                self.logger.warning(f"Error processing match {idx}: {e}")
                continue
        
        # Convert to DataFrame
        features_df = pd.DataFrame(feature_list)
        
        # Additional feature engineering
        features_df = self._add_derived_features(features_df)
        
        self.feature_data = features_df
        self.logger.info(f"Feature engineering completed: {len(features_df)} samples, {len(features_df.columns)} features")
        
        return features_df
    
    def train_models(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Train base models and ensemble.
        
        Args:
            features_df: Feature matrix
            
        Returns:
            Training results
        """
        self.logger.info("Starting model training")
        
        # Prepare training data
        feature_columns = [col for col in features_df.columns 
                          if col not in ['match_id', 'winner_id', 'loser_id', 'date', 'target']]
        
        X = features_df[feature_columns]
        y = features_df['target']
        
        self.logger.info(f"Training on {len(X)} samples with {len(feature_columns)} features")
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize base models
        self.models = {
            'xgboost': XGBoostModel(),
            'random_forest': RandomForestModel(), 
            'neural_network': NeuralNetworkModel(),
            'svm': SVMModel(),
            'logistic_regression': LogisticRegressionModel()
        }
        
        # Train base models
        base_model_results = {}
        trained_models = []
        
        for name, model in self.models.items():
            try:
                self.logger.info(f"Training {name}")
                
                # Train model
                metrics = model.train(X_train, y_train, X_test, y_test)
                base_model_results[name] = metrics
                trained_models.append(model)
                
                self.logger.info(f"{name} training completed - Accuracy: {metrics.get('accuracy', 0):.3f}")
                
            except Exception as e:
                self.logger.error(f"Error training {name}: {e}")
                continue
        
        if not trained_models:
            raise ValueError("No models were successfully trained")
        
        # Train stacking ensemble
        self.logger.info("Training stacking ensemble")
        
        self.ensemble = StackingEnsemble(
            base_models=trained_models,
            meta_learner='logistic_regression',
            cv_folds=5,
            use_probabilities=True
        )
        
        ensemble_metrics = self.ensemble.train(X_train, y_train, X_test, y_test)
        
        self.training_metrics = {
            'base_models': base_model_results,
            'ensemble': ensemble_metrics,
            'data_shape': X.shape,
            'training_completed': datetime.now().isoformat()
        }
        
        self.logger.info(f"Ensemble training completed - Accuracy: {ensemble_metrics.get('accuracy', 0):.3f}")
        
        return self.training_metrics
    
    def validate_models(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive model validation.
        
        Args:
            features_df: Feature matrix
            
        Returns:
            Validation results
        """
        self.logger.info("Starting model validation")
        
        # Prepare data
        feature_columns = [col for col in features_df.columns 
                          if col not in ['match_id', 'winner_id', 'loser_id', 'date', 'target']]
        
        X = features_df[feature_columns]
        y = features_df['target']
        
        # Tournament-based cross-validation
        if 'tournament' in features_df.columns:
            cv = TournamentBasedCV(n_splits=5, tournament_col='tournament')
        else:
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        validation_results = {}
        
        # Validate ensemble
        if self.ensemble is not None:
            evaluator = ModelEvaluator()
            
            try:
                ensemble_cv_results = evaluator.cross_validate_model(
                    self.ensemble, X, y, cv=cv
                )
                validation_results['ensemble'] = ensemble_cv_results
                
                self.logger.info(
                    f"Ensemble CV Results - Mean Accuracy: {ensemble_cv_results.get('cv_accuracy_mean', 0):.3f} "
                    f"(±{ensemble_cv_results.get('cv_accuracy_std', 0):.3f})"
                )
                
            except Exception as e:
                self.logger.error(f"Error validating ensemble: {e}")
        
        # Validate base models
        for name, model in self.models.items():
            if model.is_trained:
                try:
                    evaluator = ModelEvaluator()
                    cv_results = evaluator.cross_validate_model(model, X, y, cv=cv)
                    validation_results[name] = cv_results
                    
                except Exception as e:
                    self.logger.warning(f"Error validating {name}: {e}")
        
        self.logger.info("Model validation completed")
        return validation_results
    
    def save_models(self, output_dir: str = "models/saved"):
        """Save trained models and artifacts.
        
        Args:
            output_dir: Directory to save models
        """
        self.logger.info(f"Saving models to {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save ensemble
        if self.ensemble is not None:
            ensemble_path = f"{output_dir}/ensemble_{timestamp}.joblib"
            self.ensemble.save_model(ensemble_path)
            self.logger.info(f"Ensemble saved to {ensemble_path}")
        
        # Save base models
        for name, model in self.models.items():
            if model.is_trained:
                model_path = f"{output_dir}/{name}_{timestamp}.joblib"
                model.save_model(model_path)
                self.logger.info(f"{name} saved to {model_path}")
        
        # Save ELO ratings
        if 'elo' in self.feature_engineers:
            elo_ratings = self.feature_engineers['elo'].export_ratings()
            elo_path = f"{output_dir}/elo_ratings_{timestamp}.csv"
            elo_ratings.to_csv(elo_path, index=False)
            self.logger.info(f"ELO ratings saved to {elo_path}")
        
        # Save training metrics
        import json
        metrics_path = f"{output_dir}/training_metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.training_metrics, f, indent=2, default=str)
        self.logger.info(f"Training metrics saved to {metrics_path}")
    
    def _preprocess_match_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Basic preprocessing of match data."""
        # Remove incomplete matches
        data = data.dropna(subset=['winner_name', 'loser_name', 'tourney_date'])
        
        # Standardize surface names
        surface_mapping = {
            'Hard': 'Hard', 'Clay': 'Clay', 'Grass': 'Grass',
            'Carpet': 'Hard'  # Treat carpet as hard
        }
        data['surface'] = data['surface'].map(surface_mapping).fillna('Hard')
        
        # Convert dates
        data['tourney_date'] = pd.to_datetime(data['tourney_date'])
        
        # Remove very old matches (before 2000)
        data = data[data['tourney_date'] >= '2000-01-01']
        
        return data
    
    def _calculate_surface_features(self, match: pd.Series, 
                                   player1_id: str, player2_id: str) -> Dict[str, float]:
        """Calculate surface-specific features."""
        surface = match.get('surface', 'Hard')
        
        # Basic surface features (placeholder)
        return {
            'surface_clay': 1.0 if surface == 'Clay' else 0.0,
            'surface_grass': 1.0 if surface == 'Grass' else 0.0,
            'surface_hard': 1.0 if surface == 'Hard' else 0.0
        }
    
    def _calculate_momentum_features(self, match: pd.Series,
                                   player1_id: str, player2_id: str) -> Dict[str, float]:
        """Calculate momentum features (simplified)."""
        # Placeholder momentum features
        return {
            'momentum_score': 0.5,  # Neutral momentum
            'recent_form': 0.5,
            'confidence_index': 0.5
        }
    
    def _calculate_environmental_features(self, match: pd.Series) -> Dict[str, float]:
        """Calculate environmental features."""
        # Placeholder environmental features
        return {
            'indoor_outdoor': 0.0,  # Assume outdoor
            'altitude_factor': 0.0,  # Sea level
            'temperature_factor': 0.5  # Neutral temperature
        }
    
    def _categorize_tournament(self, tournament_name: str) -> str:
        """Categorize tournament importance."""
        name_lower = tournament_name.lower()
        
        if any(slam in name_lower for slam in ['wimbledon', 'us open', 'french open', 'australian open']):
            return 'Grand Slam'
        elif 'masters' in name_lower or 'atp finals' in name_lower:
            return 'Masters 1000'
        elif 'olympics' in name_lower:
            return 'Olympics'
        else:
            return 'ATP 250'
    
    def _add_derived_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features."""
        # ELO-based features
        features_df['elo_ratio'] = features_df['winner_elo'] / (features_df['loser_elo'] + 1)
        features_df['elo_sum'] = features_df['winner_elo'] + features_df['loser_elo']
        
        # Match importance
        features_df['is_final'] = (features_df['round'] == 'F').astype(int)
        features_df['is_semifinal'] = (features_df['round'] == 'SF').astype(int)
        
        return features_df


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Tennis Predictor 101')
    parser.add_argument('--years', nargs='+', type=int, 
                       help='Years of data to collect (default: last 5 years)')
    parser.add_argument('--output-dir', default='models/saved',
                       help='Output directory for saved models')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--skip-data-collection', action='store_true',
                       help='Skip data collection (use existing data)')
    parser.add_argument('--skip-validation', action='store_true', 
                       help='Skip cross-validation')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting Tennis Predictor 101 training")
    
    try:
        # Initialize trainer
        trainer = TennisPredictor101Trainer()
        
        # Collect data
        if not args.skip_data_collection:
            training_data = trainer.collect_data(args.years)
        else:
            # Load existing data (implement as needed)
            logger.warning("Data collection skipped - implement data loading")
            return
        
        # Engineer features
        features_df = trainer.engineer_features(training_data)
        
        # Train models
        training_results = trainer.train_models(features_df)
        
        # Validate models
        if not args.skip_validation:
            validation_results = trainer.validate_models(features_df)
        
        # Save models
        trainer.save_models(args.output_dir)
        
        logger.info("Tennis Predictor 101 training completed successfully")
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        if 'ensemble' in training_results:
            ensemble_acc = training_results['ensemble'].get('accuracy', 0)
            print(f"Ensemble Accuracy: {ensemble_acc:.3f}")
        
        for model_name, metrics in training_results.get('base_models', {}).items():
            acc = metrics.get('accuracy', 0)
            print(f"{model_name.title()} Accuracy: {acc:.3f}")
        
        print(f"\nModels saved to: {args.output_dir}")
        print("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()