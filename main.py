"""
Football Betting Predictor with Rate Limiting
Self-contained script - no interfaces, just core prediction functionality
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
import xgboost as xgb
from collections import deque

warnings.filterwarnings('ignore')

class RateLimiter:
    """Rate limiter to control API calls - 9 requests per minute to avoid 429 errors"""
    
    def __init__(self, max_requests_per_minute=9):
        self.max_requests = max_requests_per_minute
        self.interval = 60  # 1 minute in seconds
        self.requests = deque()
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limit"""
        with self.lock:
            current_time = time.time()
            
            # Remove requests older than 1 minute
            while self.requests and self.requests[0] <= current_time - self.interval:
                self.requests.popleft()
            
            # If we've hit the limit, wait
            if len(self.requests) >= self.max_requests:
                wait_time = self.interval - (current_time - self.requests[0])
                if wait_time > 0:
                    print(f"Rate limit: Waiting {wait_time:.1f} seconds before next API call...")
                    time.sleep(wait_time)
                    # Remove the old request after waiting
                    self.requests.popleft()
            
            # Record this request
            self.requests.append(current_time)
    
    def get_status(self):
        """Get current rate limiter status"""
        with self.lock:
            current_time = time.time()
            # Remove old requests
            while self.requests and self.requests[0] <= current_time - self.interval:
                self.requests.popleft()
            
            return {
                'requests_used': len(self.requests),
                'requests_remaining': max(0, self.max_requests - len(self.requests)),
                'next_reset': self.requests[0] + self.interval if self.requests else current_time
            }

@dataclass
class BettingRecommendation:
    match_id: str
    home_team: str
    away_team: str
    outcome: str  # 'H', 'D', 'A', '1X', 'X2', '12', 'O1.5', 'U1.5', 'O2.5', 'U2.5', 'OC8.5', 'UC8.5'
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
    # Double chance probabilities
    double_chance_1x: float = 0.0  # Home win OR Draw
    double_chance_x2: float = 0.0  # Draw OR Away win
    double_chance_12: float = 0.0  # Home win OR Away win
    # Goals predictions
    over_15_goals: float = 0.0     # Over 1.5 goals
    under_15_goals: float = 0.0    # Under 1.5 goals
    over_25_goals: float = 0.0     # Over 2.5 goals
    under_25_goals: float = 0.0    # Under 2.5 goals
    # Corners predictions
    over_85_corners: float = 0.0   # Over 8.5 corners
    under_85_corners: float = 0.0  # Under 8.5 corners

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
            'Eredivisie': 'DED',
        }
        
        # Initialize rate limiter with 9 requests per minute to avoid 429 errors
        self.rate_limiter = RateLimiter(max_requests_per_minute=9)
        
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
        
        # API configuration
        self.api_key = "c4d999e143b044f5a5d1b3d86fa01962"
        self.headers = {"X-Auth-Token": self.api_key}
        self.base_url = "https://api.football-data.org/v4"
        
        print("Enhanced Football Predictor initialized with rate limiting")
    
    def make_api_request(self, url, params=None, timeout=15):
        """Make API request with rate limiting and better error handling"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Apply rate limiting
                self.rate_limiter.wait_if_needed()
                
                print(f"Making API request (attempt {attempt + 1}/{max_retries})")
                
                response = requests.get(url, headers=self.headers, params=params, timeout=timeout)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limit
                    wait_time = 60 * (attempt + 1)  # Progressive backoff
                    print(f"Rate limit exceeded. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                elif response.status_code == 403:
                    print("API access denied. Check your API key and quota.")
                    return None
                elif response.status_code == 404:
                    print("Resource not found.")
                    return None
                else:
                    print(f"API error: {response.status_code}")
                    if attempt < max_retries - 1:
                        time.sleep(5)
            
            except requests.exceptions.Timeout:
                print(f"Request timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(5)
            except requests.exceptions.RequestException as e:
                print(f"Network error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
        
        return None
    
    def get_fixtures(self, league_code, days_ahead=7):
        """Get upcoming fixtures with rate limiting"""
        try:
            today = datetime.now()
            start_of_week = today - timedelta(days=today.weekday())
            end_of_week = start_of_week + timedelta(days=6)
            
            date_from = start_of_week.strftime('%Y-%m-%d')
            date_to = end_of_week.strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/competitions/{league_code}/matches"
            params = {
                'dateFrom': date_from,
                'dateTo': date_to,
                'status': 'SCHEDULED'
            }
            
            data = self.make_api_request(url, params)
            
            if data:
                fixtures = []
                for match in data.get('matches', []):
                    # Mock odds data (in real implementation, get from odds API)
                    home_odds = np.random.uniform(1.5, 4.0)
                    draw_odds = np.random.uniform(3.0, 4.5)
                    away_odds = np.random.uniform(1.8, 5.0)
                    
                    # Calculate double chance odds (typically lower than individual outcomes)
                    double_chance_1x = np.random.uniform(1.1, 1.8)  # Home or Draw
                    double_chance_x2 = np.random.uniform(1.2, 2.0)  # Draw or Away
                    double_chance_12 = np.random.uniform(1.1, 1.6)  # Home or Away
                    
                    # Goals markets odds
                    over_15_goals_odds = np.random.uniform(1.1, 1.4)  # Over 1.5 goals
                    under_15_goals_odds = np.random.uniform(2.5, 4.0)  # Under 1.5 goals
                    over_25_goals_odds = np.random.uniform(1.4, 2.2)  # Over 2.5 goals
                    under_25_goals_odds = np.random.uniform(1.6, 2.8)  # Under 2.5 goals
                    
                    # Corners markets odds
                    over_85_corners_odds = np.random.uniform(1.7, 2.3)  # Over 8.5 corners
                    under_85_corners_odds = np.random.uniform(1.5, 2.1)  # Under 8.5 corners
                    
                    fixture = {
                        'date': match['utcDate'],
                        'home_team': match['homeTeam']['name'],
                        'away_team': match['awayTeam']['name'],
                        'league': league_code,
                        'match_id': match['id'],
                        'odds': {
                            'home': home_odds,
                            'draw': draw_odds,
                            'away': away_odds,
                            'double_chance_1x': double_chance_1x,
                            'double_chance_x2': double_chance_x2,
                            'double_chance_12': double_chance_12,
                            'over_15_goals': over_15_goals_odds,
                            'under_15_goals': under_15_goals_odds,
                            'over_25_goals': over_25_goals_odds,
                            'under_25_goals': under_25_goals_odds,
                            'over_85_corners': over_85_corners_odds,
                            'under_85_corners': under_85_corners_odds
                        }
                    }
                    fixtures.append(fixture)
                
                return fixtures
            else:
                print(f"Failed to get fixtures for {league_code}")
                return []
                
        except Exception as e:
            print(f"Error getting fixtures for {league_code}: {e}")
            return []
    
    def get_historical_data(self, league_code, seasons=['2023', '2024']):
        """Get comprehensive historical data with improved error handling and rate limiting"""
        all_matches = []
        
        for season in seasons:
            print(f"Fetching {league_code} season {season}...")
            
            url = f"{self.base_url}/competitions/{league_code}/matches"
            params = {
                'season': season,
                'status': 'FINISHED'
            }
            
            data = self.make_api_request(url, params)
            
            if data:
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
                print(f" - Season {season}: {len(matches)} matches")
            else:
                print(f" - Failed to fetch {season} data for {league_code}")
        
        return all_matches
    
    def calculate_team_features(self, matches_df, team, home=True):
        """Calculate comprehensive team features"""
        team_matches = matches_df[
            (matches_df['home_team'] == team) if home else (matches_df['away_team'] == team)
        ].sort_values('date').tail(10)  # Last 10 matches
        
        if len(team_matches) == 0:
            return self._get_default_features()
        
        features = {}
        
        # Basic stats
        features['games_played'] = len(team_matches)
        features['wins'] = len(team_matches[team_matches['result'] == ('H' if home else 'A')])
        features['draws'] = len(team_matches[team_matches['result'] == 'D'])
        features['losses'] = features['games_played'] - features['wins'] - features['draws']
        
        # Win rate and form
        features['win_rate'] = features['wins'] / features['games_played']
        features['points_per_game'] = (features['wins'] * 3 + features['draws']) / features['games_played']
        
        # Goal statistics
        if home:
            features['goals_for'] = team_matches['home_score'].sum()
            features['goals_against'] = team_matches['away_score'].sum()
        else:
            features['goals_for'] = team_matches['away_score'].sum()
            features['goals_against'] = team_matches['home_score'].sum()
        
        features['goal_diff'] = features['goals_for'] - features['goals_against']
        features['goals_for_avg'] = features['goals_for'] / features['games_played']
        features['goals_against_avg'] = features['goals_against'] / features['games_played']
        
        # Recent form (last 5 games)
        recent_matches = team_matches.tail(5)
        if len(recent_matches) > 0:
            recent_wins = len(recent_matches[recent_matches['result'] == ('H' if home else 'A')])
            features['recent_form'] = recent_wins / len(recent_matches)
            
            # Recent goals
            if home:
                features['recent_goals_for'] = recent_matches['home_score'].mean()
                features['recent_goals_against'] = recent_matches['away_score'].mean()
            else:
                features['recent_goals_for'] = recent_matches['away_score'].mean()
                features['recent_goals_against'] = recent_matches['home_score'].mean()
        else:
            features['recent_form'] = 0.0
            features['recent_goals_for'] = 0.0
            features['recent_goals_against'] = 0.0
        
        # Elo rating
        features['elo_rating'] = self.elo_system.get_rating(team)
        
        return features
    
    def _get_default_features(self):
        """Return default features for teams with no match history"""
        return {
            'games_played': 0, 'wins': 0, 'draws': 0, 'losses': 0,
            'win_rate': 0.0, 'points_per_game': 0.0,
            'goals_for': 0, 'goals_against': 0, 'goal_diff': 0,
            'goals_for_avg': 0.0, 'goals_against_avg': 0.0,
            'recent_form': 0.0, 'recent_goals_for': 0.0, 'recent_goals_against': 0.0,
            'elo_rating': self.elo_system.initial_rating
        }
    
    def predict_goals_markets(self, home_features, away_features):
        """Predict over/under goals markets based on team features"""
        # Calculate expected total goals based on team averages
        home_goals_avg = home_features.get('goals_for_avg', 1.2)
        away_goals_avg = away_features.get('goals_for_avg', 1.0)
        home_concede_avg = home_features.get('goals_against_avg', 1.0)
        away_concede_avg = away_features.get('goals_against_avg', 1.2)
        
        # Expected goals = (team's scoring avg + opponent's conceding avg) / 2
        expected_home_goals = (home_goals_avg + away_concede_avg) / 2
        expected_away_goals = (away_goals_avg + home_concede_avg) / 2
        expected_total_goals = expected_home_goals + expected_away_goals
        
        # Adjust based on recent form
        home_recent_goals = home_features.get('recent_goals_for', 1.2)
        away_recent_goals = away_features.get('recent_goals_for', 1.0)
        recent_total = home_recent_goals + away_recent_goals
        
        # Weighted average (70% season avg, 30% recent form)
        adjusted_total = 0.7 * expected_total_goals + 0.3 * recent_total
        
        # Calculate probabilities using Poisson-like distribution assumptions
        # Over 1.5 goals (more than 1 goal)
        over_15_prob = min(0.95, max(0.05, (adjusted_total - 0.8) / 2.5))
        under_15_prob = 1.0 - over_15_prob
        
        # Over 2.5 goals (more than 2 goals)
        over_25_prob = min(0.90, max(0.05, (adjusted_total - 1.5) / 3.0))
        under_25_prob = 1.0 - over_25_prob
        
        return {
            'over_15': over_15_prob,
            'under_15': under_15_prob,
            'over_25': over_25_prob,
            'under_25': under_25_prob
        }
    
    def predict_corners_markets(self, home_features, away_features):
        """Predict over/under corners markets based on team style"""
        # Base corners expectation (average match has ~10 corners)
        base_corners = 10.0
        
        # Adjust based on team characteristics
        home_attack_style = home_features.get('goals_for_avg', 1.2) * 2  # Attacking teams get more corners
        away_attack_style = away_features.get('goals_for_avg', 1.0) * 2
        
        # Defensive teams also contribute to corners (through defending)
        home_defense_factor = home_features.get('goals_against_avg', 1.0) * 1.5
        away_defense_factor = away_features.get('goals_against_avg', 1.2) * 1.5
        
        # Recent form impact
        home_form = home_features.get('recent_form', 0.5)
        away_form = away_features.get('recent_form', 0.5)
        form_factor = (home_form + away_form) * 2
        
        # Expected corners calculation
        expected_corners = base_corners + home_attack_style + away_attack_style + \
                          (home_defense_factor + away_defense_factor) * 0.3 + form_factor
        
        # Cap the expected corners to reasonable range
        expected_corners = min(16.0, max(6.0, expected_corners))
        
        # Over 8.5 corners probability
        over_85_prob = min(0.85, max(0.15, (expected_corners - 7.0) / 6.0))
        under_85_prob = 1.0 - over_85_prob
        
        return {
            'over_85': over_85_prob,
            'under_85': under_85_prob
        }
    
    def prepare_features(self, matches_df):
        """Prepare features for machine learning"""
        features = []
        targets = []
        
        for _, match in matches_df.iterrows():
            # Update Elo ratings
            self.elo_system.update_ratings(
                match['home_team'], match['away_team'], match['result']
            )
            
            # Calculate features for both teams
            home_features = self.calculate_team_features(
                matches_df[matches_df['date'] < match['date']], 
                match['home_team'], home=True
            )
            away_features = self.calculate_team_features(
                matches_df[matches_df['date'] < match['date']], 
                match['away_team'], home=False
            )
            
            # Combine features
            combined_features = []
            for key in home_features.keys():
                combined_features.extend([
                    home_features[key],
                    away_features[key],
                    home_features[key] - away_features[key]  # Difference
                ])
            
            features.append(combined_features)
            targets.append(match['result'])
        
        return np.array(features), np.array(targets)
    
    def train_model(self, force_retrain=False):
        """Train the prediction model with all available data"""
        if self.trained and not force_retrain:
            print("Model already trained. Use force_retrain=True to retrain.")
            return
        
        print("Starting model training...")
        all_matches = []
        
        # Show rate limiter status
        status = self.rate_limiter.get_status()
        print(f"Rate limiter status: {status['requests_used']}/{status['requests_used'] + status['requests_remaining']} requests used")
        
        # Collect data from all leagues
        for league_name, league_code in self.leagues.items():
            print(f"\nFetching data for {league_name}...")
            matches = self.get_historical_data(league_code)
            all_matches.extend(matches)
            print(f"Total matches collected so far: {len(all_matches)}")
        
        if not all_matches:
            print("No matches found. Cannot train model.")
            return
        
        # Convert to DataFrame and sort by date
        df = pd.DataFrame(all_matches)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"\nPreparing features from {len(df)} matches...")
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        if len(X) == 0:
            print("No features prepared. Cannot train model.")
            return
        
        print(f"Training on {len(X)} samples with {X.shape[1]} features...")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Train model
        self.model.fit(X_scaled, y_encoded)
        
        # Calibrate probabilities
        self.calibrator = CalibratedClassifierCV(self.model, method='isotonic', cv=3)
        self.calibrator.fit(X_scaled, y_encoded)
        
        # Calculate performance metrics
        scores = []
        for train_idx, val_idx in tscv.split(X_scaled):
            if X_scaled is not None and y_encoded is not None:
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
            else:
                continue
            
            temp_model = xgb.XGBClassifier(**self.model.get_params())
            temp_model.fit(X_train, y_train)
            
            y_pred = temp_model.predict(X_val)
            scores.append(accuracy_score(y_val, y_pred))
        
        print(f"Cross-validation accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        
        self.trained = True
        self.match_history = all_matches
        print("Model training completed!")
    
    def predict_match(self, home_team, away_team, league):
        """Predict match outcome"""
        if not self.trained:
            print("Model not trained. Please train the model first.")
            return None
        
        # Get recent team performance
        df = pd.DataFrame(self.match_history)
        df['date'] = pd.to_datetime(df['date'])
        
        home_features = self.calculate_team_features(df, home_team, home=True)
        away_features = self.calculate_team_features(df, away_team, home=False)
        
        # Combine features
        combined_features = []
        for key in home_features.keys():
            combined_features.extend([
                home_features[key],
                away_features[key],
                home_features[key] - away_features[key]
            ])
        
        # Scale and predict
        X = np.array(combined_features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities
        try:
            if self.calibrator:
                probs = self.calibrator.predict_proba(X_scaled)[0]
            else:
                probs = self.model.predict_proba(X_scaled)[0]
        except:
            # Default probabilities if prediction fails
            probs = [0.4, 0.3, 0.3]
        
        # Map probabilities to outcomes
        outcomes = self.label_encoder.classes_
        if outcomes is not None and probs is not None:
            prob_dict = dict(zip(outcomes, probs))
            
            # Ensure all outcomes are present
            home_prob = prob_dict.get('H', 0.0)
            draw_prob = prob_dict.get('D', 0.0)
            away_prob = prob_dict.get('A', 0.0)
        else:
            # Default probabilities if prediction fails
            home_prob = 0.4
            draw_prob = 0.3
            away_prob = 0.3
        
        # Normalize probabilities
        total_prob = home_prob + draw_prob + away_prob
        if total_prob > 0:
            home_prob /= total_prob
            draw_prob /= total_prob
            away_prob /= total_prob
        
        # Calculate double chance probabilities
        double_chance_1x = home_prob + draw_prob  # Home win OR Draw
        double_chance_x2 = draw_prob + away_prob  # Draw OR Away win
        double_chance_12 = home_prob + away_prob  # Home win OR Away win (no draw)
        
        # Calculate goals and corners predictions
        goals_predictions = self.predict_goals_markets(home_features, away_features)
        corners_predictions = self.predict_corners_markets(home_features, away_features)
        
        # Determine predicted outcome
        max_prob = max(home_prob, draw_prob, away_prob)
        if max_prob == home_prob:
            predicted = 'H'
        elif max_prob == away_prob:
            predicted = 'A'
        else:
            predicted = 'D'
        
        return MatchPrediction(
            match_id=f"{home_team}_vs_{away_team}",
            home_team=home_team,
            away_team=away_team,
            league=league,
            date=datetime.now().strftime('%Y-%m-%d'),
            home_prob=home_prob,
            draw_prob=draw_prob,
            away_prob=away_prob,
            predicted_outcome=predicted,
            confidence_score=max_prob,
            odds={},
            double_chance_1x=double_chance_1x,
            double_chance_x2=double_chance_x2,
            double_chance_12=double_chance_12,
            over_15_goals=goals_predictions['over_15'],
            under_15_goals=goals_predictions['under_15'],
            over_25_goals=goals_predictions['over_25'],
            under_25_goals=goals_predictions['under_25'],
            over_85_corners=corners_predictions['over_85'],
            under_85_corners=corners_predictions['under_85']
        )
    
    def calculate_betting_value(self, prediction, odds):
        """Calculate betting value and recommendations"""
        recommendations = []
        
        # Regular outcomes
        outcomes = [
            ('H', prediction.home_prob, odds.get('home', 0)),
            ('D', prediction.draw_prob, odds.get('draw', 0)),
            ('A', prediction.away_prob, odds.get('away', 0))
        ]
        
        # Double chance outcomes
        double_chance_outcomes = [
            ('1X', prediction.double_chance_1x, odds.get('double_chance_1x', 0)),
            ('X2', prediction.double_chance_x2, odds.get('double_chance_x2', 0)),
            ('12', prediction.double_chance_12, odds.get('double_chance_12', 0))
        ]
        
        # Goals markets outcomes
        goals_outcomes = [
            ('O1.5', prediction.over_15_goals, odds.get('over_15_goals', 0)),
            ('U1.5', prediction.under_15_goals, odds.get('under_15_goals', 0)),
            ('O2.5', prediction.over_25_goals, odds.get('over_25_goals', 0)),
            ('U2.5', prediction.under_25_goals, odds.get('under_25_goals', 0))
        ]
        
        # Corners markets outcomes
        corners_outcomes = [
            ('OC8.5', prediction.over_85_corners, odds.get('over_85_corners', 0)),
            ('UC8.5', prediction.under_85_corners, odds.get('under_85_corners', 0))
        ]
        
        # Combine all outcomes
        all_outcomes = outcomes + double_chance_outcomes + goals_outcomes + corners_outcomes
        
        for outcome, prob, bookmaker_odds in all_outcomes:
            if bookmaker_odds <= 1.0:  # Invalid odds
                continue
            
            implied_prob = 1 / bookmaker_odds
            expected_value = (prob * bookmaker_odds) - 1
            
            if expected_value > self.min_ev_threshold:
                # Kelly criterion for bet sizing
                kelly_fraction = (prob * bookmaker_odds - 1) / (bookmaker_odds - 1)
                kelly_fraction = max(0, min(kelly_fraction, self.max_bet_fraction))
                
                bet_amount = self.bankroll * kelly_fraction
                
                confidence = "High" if expected_value > 0.15 else "Medium" if expected_value > 0.10 else "Low"
                
                recommendation = BettingRecommendation(
                    match_id=prediction.match_id,
                    home_team=prediction.home_team,
                    away_team=prediction.away_team,
                    outcome=outcome,
                    predicted_prob=prob,
                    bookmaker_odds=bookmaker_odds,
                    implied_prob=implied_prob,
                    expected_value=expected_value,
                    kelly_fraction=kelly_fraction,
                    bet_amount=bet_amount,
                    confidence=confidence
                )
                recommendations.append(recommendation)
        
        return recommendations
    
    def get_recommendations(self, league_codes=None):
        """Get betting recommendations for upcoming matches"""
        if not self.trained:
            print("Model not trained. Training now...")
            self.train_model()
        
        if league_codes is None:
            league_codes = list(self.leagues.values())
        
        all_recommendations = []
        
        for league_code in league_codes:
            print(f"\nAnalyzing {league_code} fixtures...")
            fixtures = self.get_fixtures(league_code)
            
            for fixture in fixtures:
                prediction = self.predict_match(
                    fixture['home_team'],
                    fixture['away_team'],
                    fixture['league']
                )
                
                if prediction:
                    recommendations = self.calculate_betting_value(prediction, fixture['odds'])
                    all_recommendations.extend(recommendations)
        
        # Sort by expected value
        all_recommendations.sort(key=lambda x: x.expected_value, reverse=True)
        
        return all_recommendations
    
    def save_recommendations_to_file(self, recommendations, filename="tips.txt"):
        """Save betting recommendations to a text file"""
        if not recommendations:
            print("No betting opportunities found.")
            with open(filename, 'w') as f:
                f.write("No betting opportunities found.\n")
            return
        
        outcome_map = {
            'H': 'Home Win', 
            'D': 'Draw', 
            'A': 'Away Win',
            '1X': 'Home Win or Draw',
            'X2': 'Draw or Away Win', 
            '12': 'Home Win or Away Win',
            'O1.5': 'Over 1.5 Goals',
            'U1.5': 'Under 1.5 Goals',
            'O2.5': 'Over 2.5 Goals',
            'U2.5': 'Under 2.5 Goals',
            'OC8.5': 'Over 8.5 Corners',
            'UC8.5': 'Under 8.5 Corners'
        }
        
        with open(filename, 'w') as f:
            f.write(f"{'='*80}\n")
            f.write(f"BETTING PREDICTIONS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"TOTAL OPPORTUNITIES: {len(recommendations)}\n")
            f.write(f"{'='*80}\n\n")
            
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec.home_team} vs {rec.away_team}\n")
                f.write(f"   Bet: {outcome_map[rec.outcome]}\n")
                f.write(f"   Odds: {rec.bookmaker_odds:.2f}\n")
                f.write(f"   Predicted Probability: {rec.predicted_prob:.1%}\n")
                f.write(f"   Expected Value: {rec.expected_value:.1%}\n")
                f.write(f"   Recommended Bet: ${rec.bet_amount:.2f}\n")
                f.write(f"   Confidence: {rec.confidence}\n\n")
        
        print(f"Betting tips saved to {filename}")
        print(f"Total recommendations: {len(recommendations)}")
    
    def print_recommendations(self, recommendations):
        """Print betting recommendations in a readable format"""
        if not recommendations:
            print("No betting opportunities found.")
            return
        
        print(f"\n{'='*80}")
        print(f"BETTING RECOMMENDATIONS ({len(recommendations)} opportunities)")
        print(f"{'='*80}")
        
        for i, rec in enumerate(recommendations, 1):
            outcome_map = {
                'H': 'Home Win', 
                'D': 'Draw', 
                'A': 'Away Win',
                '1X': 'Home Win or Draw',
                'X2': 'Draw or Away Win', 
                '12': 'Home Win or Away Win',
                'O1.5': 'Over 1.5 Goals',
                'U1.5': 'Under 1.5 Goals',
                'O2.5': 'Over 2.5 Goals',
                'U2.5': 'Under 2.5 Goals',
                'OC8.5': 'Over 8.5 Corners',
                'UC8.5': 'Under 8.5 Corners'
            }
            
            print(f"\n{i}. {rec.home_team} vs {rec.away_team}")
            print(f"   Bet: {outcome_map[rec.outcome]}")
            print(f"   Odds: {rec.bookmaker_odds:.2f}")
            print(f"   Predicted Probability: {rec.predicted_prob:.1%}")
            print(f"   Expected Value: {rec.expected_value:.1%}")
            print(f"   Recommended Bet: ${rec.bet_amount:.2f}")
            print(f"   Confidence: {rec.confidence}")

def main():
    """Main execution function"""
    predictor = AdvancedFootballPredictor()
    
    print("Starting Football Betting Predictor...")
    print("API calls are rate-limited to 9 per minute to avoid 429 timeout errors.")
    
    try:
        # Train the model
        print("\nTraining model with historical data...")
        predictor.train_model()
        
        # Get recommendations
        print("\nGetting betting recommendations...")
        recommendations = predictor.get_recommendations()
        
        # Save results to file and print summary
        predictor.save_recommendations_to_file(recommendations, "tips.txt")
        predictor.print_recommendations(recommendations)
        
        # Show final rate limiter status
        status = predictor.rate_limiter.get_status()
        print(f"\nFinal API usage: {status['requests_used']}/{status['requests_used'] + status['requests_remaining']} requests used")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()