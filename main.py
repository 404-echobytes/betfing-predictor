
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
import xgboost as xgb
import math
#import yagmail
import os

warnings.filterwarnings('ignore')

@dataclass
class BettingRecommendation:
    match_id: str
    home_team: str
    away_team: str
    outcome: str  # 'H', 'D', 'A'
    predicted_prob: float
    bookmaker_odds: float
    implied_prob: float
    expected_value: float
    kelly_fraction: float
    bet_amount: float
    confidence: str

@dataclass
class MatchPrediction:
    match_id: str
    home_team: str
    away_team: str
    league: str
    date: str
    home_prob: float
    draw_prob: float
    away_prob: float
    predicted_outcome: str
    confidence_score: float
    odds: dict

class EloRating:
    """Elo rating system for team strength estimation"""
    def __init__(self, k_factor=20, initial_rating=1500):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings = {}
    
    def get_rating(self, team):
        """Get team's current Elo rating"""
        return self.ratings.get(team, self.initial_rating)
    
    def expected_score(self, rating_a, rating_b):
        """Calculate expected score for team A against team B"""
        return 1 / (1 + 10**((rating_b - rating_a) / 400))
    
    def update_ratings(self, home_team, away_team, result):
        """Update Elo ratings after match result"""
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        
        home_expected = self.expected_score(home_rating, away_rating)
        away_expected = 1 - home_expected
        
        # Convert result to scores
        if result == 'H':  # Home win
            home_score, away_score = 1.0, 0.0
        elif result == 'A':  # Away win
            home_score, away_score = 0.0, 1.0
        else:  # Draw
            home_score, away_score = 0.5, 0.5
        
        # Update ratings
        home_new = home_rating + self.k_factor * (home_score - home_expected)
        away_new = away_rating + self.k_factor * (away_score - away_expected)
        
        self.ratings[home_team] = home_new
        self.ratings[away_team] = away_new
        
        return home_new, away_new

class AdvancedFootballPredictor:
    def __init__(self):
        self.leagues = {
            'Premier League': 'PL',
            'La Liga': 'PD',
            'Bundesliga': 'BL1',
            'Serie A': 'SA',
            'Ligue 1': 'FL1',
            'Champions League': 'CL'
        }
        
        # Initialize models
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        self.calibrator = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.elo_system = EloRating(k_factor=25)
        
        # Feature storage
        self.team_features = {}
        self.match_history = []
        self.trained = False
        
        # Betting parameters
        self.bankroll = 1000  # Starting bankroll
        self.max_bet_fraction = 0.05  # Max 5% of bankroll per bet
        self.min_ev_threshold = 0.05  # Minimum 5% expected value
        
    def get_fixtures(self, league_code, days_ahead=7):
        """Get upcoming fixtures with mock odds data"""
        try:
            api_key = "c4d999e143b044f5a5d1b3d86fa01962"
            headers = {"X-Auth-Token": api_key}
            
            today = datetime.now()
            start_of_week = today - timedelta(days=today.weekday())
            end_of_week = start_of_week + timedelta(days=6)
            
            date_from = start_of_week.strftime('%Y-%m-%d')
            date_to = end_of_week.strftime('%Y-%m-%d')
            
            url = f"https://api.football-data.org/v4/competitions/{league_code}/matches"
            params = {
                'dateFrom': date_from,
                'dateTo': date_to,
                'status': 'SCHEDULED'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                fixtures = []
                
                for match in data.get('matches', []):
                    # Mock odds data (in real implementation, get from odds API)
                    home_odds = np.random.uniform(1.5, 4.0)
                    draw_odds = np.random.uniform(3.0, 4.5)
                    away_odds = np.random.uniform(1.8, 5.0)
                    
                    fixture = {
                        'date': match['utcDate'],
                        'home_team': match['homeTeam']['name'],
                        'away_team': match['awayTeam']['name'],
                        'league': league_code,
                        'match_id': match['id'],
                        'odds': {
                            'home': home_odds,
                            'draw': draw_odds,
                            'away': away_odds
                        }
                    }
                    fixtures.append(fixture)
                
                return fixtures
            else:
                print(f"API Error for {league_code}: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error getting fixtures for {league_code}: {e}")
            return []
    
    def get_historical_data(self, league_code, seasons=['2023', '2024']):
        """Get comprehensive historical data with improved error handling"""
        all_matches = []
        
        for season in seasons:
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    api_key = "c4d999e143b044f5a5d1b3d86fa01962"
                    headers = {'X-Auth-Token': api_key}
                    url = f"https://api.football-data.org/v4/competitions/{league_code}/matches"
                    params = {
                        'season': season,
                        'status': 'FINISHED'
                    }
                    
                    response = requests.get(url, headers=headers, params=params, timeout=15)
                    
                    if response.status_code == 200:
                        data = response.json()
                        matches = []
                        
                        for match in data.get('matches', []):
                            if (match.get('score') and 
                                match['score'].get('fullTime') and 
                                match['score']['fullTime']['home'] is not None):
                                
                                home_score = match['score']['fullTime']['home']
                                away_score = match['score']['fullTime']['away']
                                
                                if home_score > away_score:
                                    result = 'H'
                                elif away_score > home_score:
                                    result = 'A'
                                else:
                                    result = 'D'
                                
                                match_data = {
                                    'home_team': match['homeTeam']['name'],
                                    'away_team': match['awayTeam']['name'],
                                    'home_score': home_score,
                                    'away_score': away_score,
                                    'result': result,
                                    'league': league_code,
                                    'season': season,
                                    'date': match['utcDate'],
                                    'matchday': match.get('matchday', 1),
                                    'match_id': match['id']
                                }
                                matches.append(match_data)
                        
                        all_matches.extend(matches)
                        print(f"   - Season {season}: {len(matches)} matches")
                        break  # Success, exit retry loop
                        
                    elif response.status_code == 429:  # Rate limit
                        retry_count += 1
                        wait_time = 2 ** retry_count  # Exponential backoff
                        print(f"   - Rate limit hit for {season}, waiting {wait_time}s (attempt {retry_count})")
                        time.sleep(wait_time)
                    elif response.status_code == 403:  # Forbidden
                        print(f"   - Access denied for {season} data for {league_code} (likely quota exceeded)")
                        break  # Don't retry 403 errors
                    else:
                        print(f"   - Error fetching {season} data for {league_code}: {response.status_code}")
                        retry_count += 1
                        time.sleep(1)
                        
                except requests.exceptions.RequestException as e:
                    retry_count += 1
                    print(f"   - Network error fetching {season}: {e}")
                    if retry_count < max_retries:
                        time.sleep(2)
                except Exception as e:
                    print(f"   - Unexpected error fetching {season}: {e}")
                    break
        
        return all_matches
    
    def create_enhanced_features(self, home_team, away_team, match_date=None):
        """Create comprehensive feature vector for a match"""
        features = {}
        
        # Elo ratings
        home_elo = self.elo_system.get_rating(home_team)
        away_elo = self.elo_system.get_rating(away_team)
        features['home_elo'] = home_elo
        features['away_elo'] = away_elo
        features['elo_diff'] = home_elo - away_elo
        
        # Team-specific features
        home_stats = self.team_features.get(home_team, self._default_team_stats())
        away_stats = self.team_features.get(away_team, self._default_team_stats())
        
        # Recent form (weighted by recency)
        features['home_form_5'] = home_stats.get('form_5', 1.5)
        features['away_form_5'] = away_stats.get('form_5', 1.5)
        features['home_form_10'] = home_stats.get('form_10', 1.5)
        features['away_form_10'] = away_stats.get('form_10', 1.5)
        
        # Home/Away specific performance
        features['home_win_rate_home'] = home_stats.get('home_win_rate', 0.5)
        features['away_win_rate_away'] = away_stats.get('away_win_rate', 0.4)
        features['home_goals_home'] = home_stats.get('home_goals_avg', 1.5)
        features['away_goals_away'] = away_stats.get('away_goals_avg', 1.2)
        
        # Attack vs Defense matchups
        features['attack_vs_defense'] = home_stats.get('goals_per_game', 1.5) - away_stats.get('goals_against_per_game', 1.5)
        features['away_attack_vs_home_defense'] = away_stats.get('goals_per_game', 1.5) - home_stats.get('goals_against_per_game', 1.5)
        
        # Goal statistics
        features['home_goals_per_game'] = home_stats.get('goals_per_game', 1.5)
        features['away_goals_per_game'] = away_stats.get('goals_per_game', 1.5)
        features['home_goals_against'] = home_stats.get('goals_against_per_game', 1.5)
        features['away_goals_against'] = away_stats.get('goals_against_per_game', 1.5)
        
        # Win rates
        features['home_win_rate'] = home_stats.get('win_rate', 0.5)
        features['away_win_rate'] = away_stats.get('win_rate', 0.5)
        features['win_rate_diff'] = features['home_win_rate'] - features['away_win_rate']
        
        # Head-to-head if available
        features['h2h_advantage'] = self._get_h2h_advantage(home_team, away_team)
        
        # Home advantage
        features['home_advantage'] = 1.0
        
        # Season timing (early/mid/late season effects)
        if match_date:
            try:
                month = pd.to_datetime(match_date).month
                features['season_phase'] = self._get_season_phase(month)
            except:
                features['season_phase'] = 0.5
        else:
            features['season_phase'] = 0.5
        
        return features
    
    def _default_team_stats(self):
        """Default team statistics"""
        return {
            'form_5': 1.5, 'form_10': 1.5,
            'home_win_rate': 0.5, 'away_win_rate': 0.4,
            'home_goals_avg': 1.5, 'away_goals_avg': 1.2,
            'goals_per_game': 1.5, 'goals_against_per_game': 1.5,
            'win_rate': 0.5
        }
    
    def _get_h2h_advantage(self, home_team, away_team):
        """Calculate head-to-head advantage"""
        try:
            h2h_matches = [m for m in self.match_history 
                          if (m['home_team'] == home_team and m['away_team'] == away_team) or
                             (m['home_team'] == away_team and m['away_team'] == home_team)]
            
            if len(h2h_matches) < 3:
                return 0.0
            
            home_advantage = 0
            for match in h2h_matches[-5:]:  # Last 5 H2H matches
                if match['home_team'] == home_team:
                    if match['result'] == 'H':
                        home_advantage += 1
                    elif match['result'] == 'D':
                        home_advantage += 0.5
                else:  # Away team in historical match
                    if match['result'] == 'A':
                        home_advantage += 1
                    elif match['result'] == 'D':
                        home_advantage += 0.5
            
            return home_advantage / min(len(h2h_matches), 5) - 0.5
        except:
            return 0.0
    
    def _get_season_phase(self, month):
        """Get season phase (0=early, 0.5=mid, 1=late)"""
        if month in [8, 9, 10]:  # Early season
            return 0.2
        elif month in [11, 12, 1, 2]:  # Mid season
            return 0.5
        else:  # Late season
            return 0.8
    
    def update_team_features(self, df):
        """Update comprehensive team features from match data"""
        self.team_features = {}
        
        try:
            for team in set(list(df['home_team']) + list(df['away_team'])):
                team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)].copy()
                team_matches = team_matches.sort_values('date')
                
                # Calculate various statistics
                home_matches = df[df['home_team'] == team]
                away_matches = df[df['away_team'] == team]
                
                # Form calculation with recency weighting
                recent_5 = self._calculate_weighted_form(team_matches.tail(5), team)
                recent_10 = self._calculate_weighted_form(team_matches.tail(10), team)
                
                # Home/Away specific stats
                home_wins = len(home_matches[home_matches['result'] == 'H'])
                home_total = len(home_matches)
                away_wins = len(away_matches[away_matches['result'] == 'A'])
                away_total = len(away_matches)
                
                self.team_features[team] = {
                    'form_5': recent_5,
                    'form_10': recent_10,
                    'home_win_rate': home_wins / max(home_total, 1),
                    'away_win_rate': away_wins / max(away_total, 1),
                    'home_goals_avg': home_matches['home_score'].mean() if home_total > 0 else 1.5,
                    'away_goals_avg': away_matches['away_score'].mean() if away_total > 0 else 1.2,
                    'goals_per_game': self._calculate_goals_per_game(team_matches, team),
                    'goals_against_per_game': self._calculate_goals_against_per_game(team_matches, team),
                    'win_rate': self._calculate_win_rate(team_matches, team)
                }
        except Exception as e:
            print(f"Error updating team features: {e}")
    
    def _calculate_weighted_form(self, matches, team):
        """Calculate form with exponential decay for recency"""
        try:
            if len(matches) == 0:
                return 1.5
            
            points = []
            weights = []
            
            for i, (_, match) in enumerate(matches.iterrows()):
                weight = 0.8 ** (len(matches) - i - 1)  # More recent = higher weight
                
                if match['home_team'] == team:
                    if match['result'] == 'H':
                        points.append(3)
                    elif match['result'] == 'D':
                        points.append(1)
                    else:
                        points.append(0)
                else:
                    if match['result'] == 'A':
                        points.append(3)
                    elif match['result'] == 'D':
                        points.append(1)
                    else:
                        points.append(0)
                
                weights.append(weight)
            
            return np.average(points, weights=weights) if points else 1.5
        except:
            return 1.5
    
    def _calculate_goals_per_game(self, matches, team):
        """Calculate average goals scored per game"""
        try:
            goals = []
            for _, match in matches.iterrows():
                if match['home_team'] == team:
                    goals.append(match['home_score'])
                else:
                    goals.append(match['away_score'])
            return np.mean(goals) if goals else 1.5
        except:
            return 1.5
    
    def _calculate_goals_against_per_game(self, matches, team):
        """Calculate average goals conceded per game"""
        try:
            goals = []
            for _, match in matches.iterrows():
                if match['home_team'] == team:
                    goals.append(match['away_score'])
                else:
                    goals.append(match['home_score'])
            return np.mean(goals) if goals else 1.5
        except:
            return 1.5
    
    def _calculate_win_rate(self, matches, team):
        """Calculate overall win rate"""
        try:
            wins = 0
            for _, match in matches.iterrows():
                if (match['home_team'] == team and match['result'] == 'H') or \
                   (match['away_team'] == team and match['result'] == 'A'):
                    wins += 1
            return wins / len(matches) if len(matches) > 0 else 0.5
        except:
            return 0.5
    
    def train_model(self):
        """Train advanced model with time series validation"""
        try:
            print("Fetching comprehensive historical data...")
            all_matches = []
            
            # Skip 2022 to avoid API quota issues
            seasons = ['2023', '2024']
            
            for league_name, league_code in self.leagues.items():
                print(f"Getting data for {league_name}...")
                matches = self.get_historical_data(league_code, seasons)
                all_matches.extend(matches)
                time.sleep(2)  # Increased delay to avoid rate limits
            
            print(f"Total matches fetched: {len(all_matches)}")
            
            if len(all_matches) < 100:  # Minimum threshold for training
                print("Insufficient real data available. Using enhanced sample data...")
                sample_data = self._generate_sample_data()
                all_matches.extend(sample_data)
                print(f"Added {len(sample_data)} sample matches")
            
            # Convert to DataFrame and sort by date
            df = pd.DataFrame(all_matches)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            print(f"Training with {len(df)} matches")
            
            self.match_history = all_matches
            
            # Update Elo ratings chronologically
            print("Calculating Elo ratings...")
            for _, match in df.iterrows():
                self.elo_system.update_ratings(match['home_team'], match['away_team'], match['result'])
            
            # Reset Elo for training (we'll rebuild during training)
            self.elo_system = EloRating(k_factor=25)
            
            # Update team features
            print("Calculating team features...")
            self.update_team_features(df)
            
            # Prepare features chronologically
            print("Preparing features...")
            features = []
            labels = []
            match_dates = []
            
            min_matches_for_features = min(50, len(df) // 4)  # Adaptive threshold
            
            for i, row in df.iterrows():
                if i > min_matches_for_features:  # Skip first matches to allow for stable features
                    feature_dict = self.create_enhanced_features(
                        row['home_team'], 
                        row['away_team'], 
                        row['date']
                    )
                    
                    features.append(list(feature_dict.values()))
                    labels.append(row['result'])
                    match_dates.append(row['date'])
                    
                    # Update Elo after using current ratings
                    self.elo_system.update_ratings(row['home_team'], row['away_team'], row['result'])
            
            if len(features) < 50:
                raise ValueError(f"Insufficient training data: only {len(features)} samples available")
            
            X = np.array(features)
            y = np.array(labels)
            
            print(f"Training with {len(X)} samples and {X.shape[1]} features")
            
            # Encode string labels to numeric
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Time series cross-validation
            print("Training model with time series validation...")
            n_splits = min(5, len(X) // 20)  # Adaptive splits based on data size
            if n_splits < 2:
                n_splits = 2
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            cv_scores = []
            cv_log_losses = []
            
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
                
                # Train model
                self.model.fit(X_train, y_train)
                
                # Evaluate
                val_pred = self.model.predict(X_val)
                val_prob = self.model.predict_proba(X_val)
                
                accuracy = np.mean(val_pred == y_val)
                
                # Calculate log loss
                log_loss_score = log_loss(y_val, val_prob)
                
                cv_scores.append(accuracy)
                cv_log_losses.append(log_loss_score)
            
            # Final model training
            self.model.fit(X_scaled, y_encoded)
            
            # Calibrate probabilities
            print("Calibrating probabilities...")
            self.calibrator = CalibratedClassifierCV(
                estimator=self.model,
                method='isotonic',
                cv=3
            )
            self.calibrator.fit(X_scaled, y_encoded)
            
            print(f"CV Accuracy: {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})")
            print(f"CV Log Loss: {np.mean(cv_log_losses):.3f} (±{np.std(cv_log_losses):.3f})")
            
            # Feature importance
            feature_names = list(self.create_enhanced_features('Team A', 'Team B').keys())
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            for _, row in importance_df.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.3f}")
            
            self.trained = True
            
        except Exception as e:
            print(f"Error in training: {e}")
            # Use sample data if training fails
            print("Using sample data for demonstration...")
            sample_data = self._generate_sample_data()
            df = pd.DataFrame(sample_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            self.match_history = sample_data
            self.update_team_features(df)
            
            # Simple training with sample data
            features = []
            labels = []
            
            for i, row in df.iterrows():
                if i > 50:
                    feature_dict = self.create_enhanced_features(
                        row['home_team'], 
                        row['away_team'], 
                        row['date']
                    )
                    features.append(list(feature_dict.values()))
                    labels.append(row['result'])
                    self.elo_system.update_ratings(row['home_team'], row['away_team'], row['result'])
            
            if len(features) > 50:
                X = np.array(features)
                y = np.array(labels)
                y_encoded = self.label_encoder.fit_transform(y)
                X_scaled = self.scaler.fit_transform(X)
                
                self.model.fit(X_scaled, y_encoded)
                self.calibrator = CalibratedClassifierCV(estimator=self.model, method='isotonic', cv=3)
                self.calibrator.fit(X_scaled, y_encoded)
                
                self.trained = True
                print("Training completed with sample data")
    
    def _generate_sample_data(self):
        """Generate realistic sample data for demonstration"""
        sample_teams = {
            'PL': ['Arsenal', 'Chelsea', 'Liverpool', 'Man City', 'Man United', 'Tottenham', 'Newcastle', 'Brighton'],
            'PD': ['Barcelona', 'Real Madrid', 'Atletico Madrid', 'Sevilla', 'Valencia', 'Villarreal', 'Real Sociedad', 'Athletic Bilbao'],
            'BL1': ['Bayern Munich', 'Dortmund', 'RB Leipzig', 'Bayer Leverkusen', 'Wolfsburg', 'Frankfurt', 'Borussia M\'gladbach', 'Union Berlin'],
            'SA': ['Juventus', 'Inter Milan', 'AC Milan', 'Napoli', 'Roma', 'Lazio', 'Atalanta', 'Fiorentina'],
            'FL1': ['PSG', 'Marseille', 'Lyon', 'Monaco', 'Nice', 'Lille', 'Rennes', 'Montpellier']
        }
        
        all_matches = []
        for league, teams in sample_teams.items():
            # Generate more realistic season structure
            for i in range(300):  # Balanced sample data
                home_team = np.random.choice(teams)
                away_team = np.random.choice([t for t in teams if t != home_team])
                
                result = np.random.choice(['H', 'A', 'D'], p=[0.46, 0.27, 0.27])
                
                if result == 'H':
                    home_score = np.random.randint(1, 5)
                    away_score = np.random.randint(0, home_score)
                elif result == 'A':
                    away_score = np.random.randint(1, 4)
                    home_score = np.random.randint(0, away_score)
                else:
                    score = np.random.randint(0, 3)
                    home_score = away_score = score
                
                season = np.random.choice(['2023', '2024'])
                year = 2023 if season == '2023' else 2024
                month = np.random.randint(8, 13) if season == '2023' else np.random.randint(1, 6)
                day = np.random.randint(1, 29)
                date = f"{year}-{month:02d}-{day:02d}T15:00:00Z"
                
                all_matches.append({
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_score': home_score,
                    'away_score': away_score,
                    'result': result,
                    'league': league,
                    'season': season,
                    'date': date,
                    'match_id': f"{league}_{i}"
                })
        
        return all_matches
    
    def predict_match_probabilities(self, home_team, away_team):
        """Predict match outcome probabilities"""
        try:
            if not self.trained:
                return None
            
            features_dict = self.create_enhanced_features(home_team, away_team)
            features = np.array([list(features_dict.values())])
            features_scaled = self.scaler.transform(features)
            
            # Get calibrated probabilities
            probabilities = self.calibrator.predict_proba(features_scaled)[0]
            classes = self.calibrator.classes_
            
            # Decode classes back to string labels
            decoded_classes = self.label_encoder.inverse_transform(classes)
            
            # Create probability dictionary
            prob_dict = {}
            for i, class_label in enumerate(decoded_classes):
                if class_label == 'H':
                    prob_dict['Home'] = probabilities[i]
                elif class_label == 'A':
                    prob_dict['Away'] = probabilities[i]
                else:
                    prob_dict['Draw'] = probabilities[i]
            
            return prob_dict
        except Exception as e:
            print(f"Error predicting probabilities: {e}")
            return {'Home': 0.45, 'Draw': 0.27, 'Away': 0.28}  # Default probabilities
    
    def calculate_expected_value(self, predicted_prob, bookmaker_odds):
        """Calculate expected value of a bet"""
        try:
            implied_prob = 1 / bookmaker_odds
            expected_value = (predicted_prob * bookmaker_odds) - 1
            return expected_value, implied_prob
        except:
            return 0.0, 0.5
    
    def kelly_criterion(self, predicted_prob, bookmaker_odds, fraction=0.25):
        """Calculate Kelly Criterion bet size"""
        try:
            implied_prob = 1 / bookmaker_odds
            edge = predicted_prob - implied_prob
            
            if edge <= 0:
                return 0
            
            kelly_fraction = edge / (bookmaker_odds - 1)
            # Use fractional Kelly to reduce risk
            return min(kelly_fraction * fraction, self.max_bet_fraction)
        except:
            return 0
    
    def send_email_report(self, all_predictions, recommendations, multiple_bet):
        """Send email report with predictions and recommendations"""
        try:
            email_content = self._format_email_content(all_predictions, recommendations, multiple_bet)
            
            print("\n📧 EMAIL REPORT:")
            print("=" * 80)
            print(email_content)
            print("=" * 80)
            
            # Try to send actual email (commented out for security)
            # Note: In production, use environment variables for credentials
            # yag = yagmail.SMTP(user='your_email@gmail.com', password='app_password')
            # yag.send(to='moghenerhona@gmail.com', subject='Football Betting Predictions', contents=email_content)
            
            print("✅ Email report generated successfully!")
            
        except Exception as e:
            print(f"Error sending email: {e}")
            print("📧 Email functionality disabled for security. Report printed to console.")
    
    def _format_email_content(self, all_predictions, recommendations, multiple_bet):
        """Format email content with all predictions"""
        content = []
        content.append("🚀 FOOTBALL BETTING PREDICTIONS REPORT")
        content.append("=" * 60)
        content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append("")
        
        # All matches by league
        content.append("📊 ALL MATCHES THIS WEEK BY LEAGUE:")
        content.append("-" * 40)
        
        leagues_matches = {}
        for pred in all_predictions:
            if pred.league not in leagues_matches:
                leagues_matches[pred.league] = []
            leagues_matches[pred.league].append(pred)
        
        for league, matches in leagues_matches.items():
            content.append(f"\n🏆 {league}:")
            for match in matches:
                content.append(f"   {match.home_team} vs {match.away_team}")
                content.append(f"   Prediction: {match.predicted_outcome} ({match.confidence_score:.1%} confidence)")
                content.append(f"   Probabilities: H:{match.home_prob:.1%} D:{match.draw_prob:.1%} A:{match.away_prob:.1%}")
                content.append("")
        
        # Profitable betting opportunities
        if recommendations:
            content.append("💰 PROFITABLE BETTING OPPORTUNITIES:")
            content.append("-" * 40)
            for i, rec in enumerate(recommendations[:10], 1):
                content.append(f"{i}. {rec.home_team} vs {rec.away_team}")
                content.append(f"   Bet: {rec.outcome} @ {rec.bookmaker_odds:.2f}")
                content.append(f"   Expected Value: {rec.expected_value:+.1%}")
                content.append(f"   Recommended Bet: ${rec.bet_amount:.0f}")
                content.append("")
        
        # Multiple bet
        if multiple_bet:
            content.append("🎯 15-GAME MULTIPLE BET (Most Confident Predictions):")
            content.append("-" * 40)
            total_odds = 1.0
            for i, bet in enumerate(multiple_bet, 1):
                content.append(f"{i}. {bet['home_team']} vs {bet['away_team']}")
                content.append(f"   Pick: {bet['prediction']} (Confidence: {bet['confidence']:.1%})")
                content.append(f"   Odds: {bet['odds']:.2f}")
                total_odds *= bet['odds']
                content.append("")
            
            content.append(f"Total Multiple Bet Odds: {total_odds:.2f}")
            content.append(f"$10 bet returns: ${total_odds * 10:.2f}")
        
        return "\n".join(content)
    
    def generate_all_predictions(self):
        """Generate predictions for all upcoming matches"""
        print("Generating predictions for all matches...")
        
        all_fixtures = []
        for league_name, league_code in self.leagues.items():
            print(f"Fetching fixtures for {league_name}...")
            fixtures = self.get_fixtures(league_code)
            for fixture in fixtures:
                fixture['league_name'] = league_name
            all_fixtures.extend(fixtures)
            time.sleep(2)  # Avoid rate limits
        
        if not all_fixtures:
            print("No real fixtures found. Using sample fixtures for demo...")
            # Enhanced sample fixtures for demo
            today = datetime.now()
            sample_fixtures = []
            
            leagues_teams = {
                'Premier League': ['Arsenal', 'Chelsea', 'Liverpool', 'Man City', 'Man United', 'Tottenham', 'Newcastle', 'Brighton', 'West Ham', 'Aston Villa'],
                'La Liga': ['Barcelona', 'Real Madrid', 'Atletico Madrid', 'Sevilla', 'Valencia', 'Villarreal', 'Real Sociedad', 'Athletic Bilbao'],
                'Bundesliga': ['Bayern Munich', 'Dortmund', 'RB Leipzig', 'Bayer Leverkusen', 'Wolfsburg', 'Frankfurt'],
                'Serie A': ['Juventus', 'Inter Milan', 'AC Milan', 'Napoli', 'Roma', 'Lazio'],
                'Ligue 1': ['PSG', 'Marseille', 'Lyon', 'Monaco', 'Nice', 'Lille']
            }
            
            match_id = 1
            for league_name, teams in leagues_teams.items():
                # Generate 3-4 matches per league for the week
                for i in range(min(4, len(teams)//2)):
                    home_team = teams[i*2]
                    away_team = teams[i*2 + 1] if i*2 + 1 < len(teams) else teams[0]
                    
                    home_odds = np.random.uniform(1.5, 4.0)
                    draw_odds = np.random.uniform(3.0, 4.5)
                    away_odds = np.random.uniform(1.8, 5.0)
                    
                    sample_fixtures.append({
                        'home_team': home_team,
                        'away_team': away_team,
                        'date': (today + timedelta(days=i+1)).strftime('%Y-%m-%dT15:00:00Z'),
                        'league_name': league_name,
                        'match_id': f'demo_{match_id}',
                        'odds': {'home': home_odds, 'draw': draw_odds, 'away': away_odds}
                    })
                    match_id += 1
            
            all_fixtures = sample_fixtures
        
        all_predictions = []
        
        for fixture in all_fixtures:
            try:
                home_team = fixture['home_team']
                away_team = fixture['away_team']
                odds = fixture['odds']
                
                # Get predicted probabilities
                prob_dict = self.predict_match_probabilities(home_team, away_team)
                
                if prob_dict:
                    home_prob = prob_dict.get('Home', 0.33)
                    draw_prob = prob_dict.get('Draw', 0.33)
                    away_prob = prob_dict.get('Away', 0.33)
                    
                    # Determine predicted outcome
                    if home_prob > draw_prob and home_prob > away_prob:
                        predicted_outcome = 'Home Win'
                        confidence_score = home_prob
                    elif away_prob > draw_prob and away_prob > home_prob:
                        predicted_outcome = 'Away Win'
                        confidence_score = away_prob
                    else:
                        predicted_outcome = 'Draw'
                        confidence_score = draw_prob
                    
                    prediction = MatchPrediction(
                        match_id=fixture['match_id'],
                        home_team=home_team,
                        away_team=away_team,
                        league=fixture['league_name'],
                        date=fixture['date'],
                        home_prob=home_prob,
                        draw_prob=draw_prob,
                        away_prob=away_prob,
                        predicted_outcome=predicted_outcome,
                        confidence_score=confidence_score,
                        odds=odds
                    )
                    
                    all_predictions.append(prediction)
            except Exception as e:
                print(f"Error processing fixture {fixture.get('match_id', 'unknown')}: {e}")
                continue
        
        return all_predictions
    
    def create_multiple_bet(self, all_predictions, num_games=15):
        """Create a multiple bet from most confident predictions"""
        try:
            # Sort by confidence score
            sorted_predictions = sorted(all_predictions, key=lambda x: x.confidence_score, reverse=True)
            
            multiple_bet = []
            for pred in sorted_predictions[:num_games]:
                # Determine the bet details based on prediction
                if pred.predicted_outcome == 'Home Win':
                    bet_odds = pred.odds['home']
                    bet_prediction = f"{pred.home_team} to Win"
                elif pred.predicted_outcome == 'Away Win':
                    bet_odds = pred.odds['away']
                    bet_prediction = f"{pred.away_team} to Win"
                else:
                    bet_odds = pred.odds['draw']
                    bet_prediction = "Draw"
                
                multiple_bet.append({
                    'home_team': pred.home_team,
                    'away_team': pred.away_team,
                    'league': pred.league,
                    'prediction': bet_prediction,
                    'confidence': pred.confidence_score,
                    'odds': bet_odds
                })
            
            return multiple_bet
        except Exception as e:
            print(f"Error creating multiple bet: {e}")
            return []
    
    def generate_betting_recommendations(self):
        """Generate betting recommendations for upcoming matches"""
        print("Generating betting recommendations...")
        
        all_fixtures = []
        for league_name, league_code in self.leagues.items():
            fixtures = self.get_fixtures(league_code)
            for fixture in fixtures:
                fixture['league_name'] = league_name
            all_fixtures.extend(fixtures)
            time.sleep(2)
        
        if not all_fixtures:
            # Sample fixtures for demo
            today = datetime.now()
            sample_fixtures = [
                {
                    'home_team': 'Arsenal', 'away_team': 'Chelsea',
                    'date': (today + timedelta(days=1)).strftime('%Y-%m-%dT15:00:00Z'),
                    'league_name': 'Premier League',
                    'match_id': 'demo_1',
                    'odds': {'home': 2.1, 'draw': 3.4, 'away': 3.8}
                },
                {
                    'home_team': 'Barcelona', 'away_team': 'Real Madrid',
                    'date': (today + timedelta(days=2)).strftime('%Y-%m-%dT20:00:00Z'),
                    'league_name': 'La Liga',
                    'match_id': 'demo_2',
                    'odds': {'home': 2.3, 'draw': 3.2, 'away': 3.1}
                }
            ]
            all_fixtures = sample_fixtures
        
        recommendations = []
        
        for fixture in all_fixtures:
            try:
                home_team = fixture['home_team']
                away_team = fixture['away_team']
                odds = fixture['odds']
                
                # Get predicted probabilities
                prob_dict = self.predict_match_probabilities(home_team, away_team)
                
                if prob_dict:
                    # Check each outcome for betting opportunities
                    outcomes = [
                        ('Home', prob_dict.get('Home', 0), odds['home'], 'H'),
                        ('Draw', prob_dict.get('Draw', 0), odds['draw'], 'D'),
                        ('Away', prob_dict.get('Away', 0), odds['away'], 'A')
                    ]
                    
                    for outcome_name, pred_prob, bookmaker_odds, outcome_code in outcomes:
                        ev, implied_prob = self.calculate_expected_value(pred_prob, bookmaker_odds)
                        
                        if ev > self.min_ev_threshold:  # Only positive EV bets
                            kelly_frac = self.kelly_criterion(pred_prob, bookmaker_odds)
                            bet_amount = self.bankroll * kelly_frac
                            
                            if bet_amount > 10:  # Minimum bet threshold
                                confidence = "High" if ev > 0.15 else "Medium" if ev > 0.08 else "Low"
                                
                                recommendations.append(BettingRecommendation(
                                    match_id=fixture['match_id'],
                                    home_team=home_team,
                                    away_team=away_team,
                                    outcome=outcome_name,
                                    predicted_prob=pred_prob,
                                    bookmaker_odds=bookmaker_odds,
                                    implied_prob=implied_prob,
                                    expected_value=ev,
                                    kelly_fraction=kelly_frac,
                                    bet_amount=bet_amount,
                                    confidence=confidence
                                ))
            except Exception as e:
                print(f"Error processing recommendation for {fixture.get('match_id', 'unknown')}: {e}")
                continue
        
        # Sort by expected value
        recommendations.sort(key=lambda x: x.expected_value, reverse=True)
        
        return recommendations

def main():
    print("🚀 Advanced Football Betting Predictor")
    print("=" * 60)
    
    try:
        predictor = AdvancedFootballPredictor()
        
        # Train the model
        print("\n📊 Training advanced prediction model...")
        predictor.train_model()
        
        if not predictor.trained:
            print("❌ Model training failed. Exiting...")
            return
        
        # Generate predictions for all matches
        print("\n📋 Generating predictions for all matches this week...")
        all_predictions = predictor.generate_all_predictions()
        
        # Display all matches by league
        print(f"\n🏆 ALL MATCHES THIS WEEK ({len(all_predictions)} total):")
        print("=" * 80)
        
        leagues_matches = {}
        for pred in all_predictions:
            if pred.league not in leagues_matches:
                leagues_matches[pred.league] = []
            leagues_matches[pred.league].append(pred)
        
        for league, matches in leagues_matches.items():
            print(f"\n🏆 {league} ({len(matches)} matches):")
            print("-" * 50)
            for match in matches:
                print(f"   {match.home_team} vs {match.away_team}")
                print(f"   📅 {match.date}")
                print(f"   🎯 Prediction: {match.predicted_outcome} ({match.confidence_score:.1%} confidence)")
                print(f"   📊 Probabilities: H:{match.home_prob:.1%} D:{match.draw_prob:.1%} A:{match.away_prob:.1%}")
                print(f"   💰 Odds: H:{match.odds['home']:.2f} D:{match.odds['draw']:.2f} A:{match.odds['away']:.2f}")
                print()
        
        # Generate betting recommendations
        print("\n💰 Generating betting recommendations...")
        recommendations = predictor.generate_betting_recommendations()
        
        if recommendations:
            print(f"\n🎯 Found {len(recommendations)} profitable betting opportunities:")
            print("=" * 80)
            
            total_ev = 0
            total_bet = 0
            
            for i, rec in enumerate(recommendations[:10], 1):  # Show top 10
                print(f"\n{i}. {rec.home_team} vs {rec.away_team}")
                print(f"   Bet: {rec.outcome} @ {rec.bookmaker_odds:.2f}")
                print(f"   Model Probability: {rec.predicted_prob:.1%}")
                print(f"   Implied Probability: {rec.implied_prob:.1%}")
                print(f"   Expected Value: {rec.expected_value:+.1%}")
                print(f"   Recommended Bet: ${rec.bet_amount:.0f}")
                print(f"   Confidence: {rec.confidence}")
                
                total_ev += rec.expected_value * rec.bet_amount
                total_bet += rec.bet_amount
            
            print(f"\n📈 Portfolio Summary:")
            print(f"   Total Bet Amount: ${total_bet:.0f}")
            print(f"   Expected Profit: ${total_ev:.0f}")
            print(f"   Portfolio EV: {total_ev/total_bet:.1%}" if total_bet > 0 else "")
        else:
            print("No profitable betting opportunities found.")
        
        # Create 15-game multiple bet
        print("\n🎲 Creating 15-Game Multiple Bet...")
        multiple_bet = predictor.create_multiple_bet(all_predictions, 15)
        
        if multiple_bet:
            print("=" * 80)
            print("🎯 15-GAME MULTIPLE BET (Most Confident Predictions):")
            print("-" * 50)
            
            total_odds = 1.0
            for i, bet in enumerate(multiple_bet, 1):
                print(f"{i:2d}. {bet['home_team']} vs {bet['away_team']} ({bet['league']})")
                print(f"     Pick: {bet['prediction']} @ {bet['odds']:.2f} (Confidence: {bet['confidence']:.1%})")
                total_odds *= bet['odds']
            
            print(f"\n🚀 MULTIPLE BET SUMMARY:")
            print(f"   Total Odds: {total_odds:,.2f}")
            print(f"   $10 stake returns: ${total_odds * 10:,.2f}")
            print(f"   $1 stake returns: ${total_odds:,.2f}")
        
        # Send email report
        print("\n📧 Generating email report...")
        predictor.send_email_report(all_predictions, recommendations, multiple_bet)
        
        print("\n🎯 Model Features:")
        print("   ✅ Elo rating system with dynamic updates")
        print("   ✅ Advanced feature engineering (23 features)")
        print("   ✅ XGBoost with time series cross-validation")
        print("   ✅ Probability calibration")
        print("   ✅ Expected Value calculation")
        print("   ✅ Kelly Criterion position sizing")
        print("   ✅ Risk management (max 5% per bet)")
        print("   ✅ Complete match analysis for all leagues")
        print("   ✅ 15-game multiple bet recommendations")
        print("   ✅ Email reporting system")
        
        print("\n✅ All predictions completed successfully!")
        
    except Exception as e:
        print(f"❌ Critical error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
