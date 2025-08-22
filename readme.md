Enhanced Betting Predictor
Overview
This is an AI-powered football betting prediction system that combines machine learning models with real-time sports data to provide betting recommendations. The system features ELO rating calculations, bankroll management, rate-limited API integrations, and a web dashboard for monitoring predictions and performance metrics.

User Preferences
Preferred communication style: Simple, everyday language.

System Architecture
Machine Learning Pipeline

XGBoost ensemble models for match outcome prediction
Calibrated probability estimation using CalibratedClassifierCV
ELO rating system for team strength tracking
Feature engineering combining historical data, team statistics, and ratings
Time-series cross-validation for model evaluation
Data Management

Pickle-based persistence for model states and historical data
Thread-safe caching system for API responses
Automatic model state saving and loading
Match history tracking with feature extraction
API Integration Layer

Rate-limited API client with 9 requests per 60-second window
Exponential backoff strategy for failed requests
Thread-safe request management
Support for football data and odds APIs
Web Interface

Flask-based dashboard with real-time updates
Background analysis processing
AJAX-powered status monitoring
Bootstrap UI with responsive design
Configuration System

Dataclass-based configuration management
Separate configs for API, ML models, betting strategy, and system settings
Environment-based configuration loading
Betting Strategy Engine

Kelly Criterion-based bet sizing with conservative multipliers
Expected value calculations for betting opportunities
Bankroll management with configurable risk limits
Confidence thresholds for bet filtering
Error Handling & Logging

Comprehensive logging system with file and console output
Thread-safe operations throughout the system
Graceful degradation for API failures
Performance monitoring and metrics tracking
External Dependencies
APIs

Football data API for match information and team statistics
Odds API for betting market data and live odds
Rate limiting configured for API provider restrictions
Machine Learning Stack

XGBoost for gradient boosting models
scikit-learn for preprocessing, calibration, and evaluation
pandas/numpy for data manipulation
Threading support for concurrent operations

Pickle serialization for model persistence
File-based caching system
No database dependency (uses local file storage)
