"""
Demand Forecasting Module

This module implements time series forecasting models to predict future product demand
based on historical data from Amazon API and other market signals.
"""

import os
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose


class DemandForecaster:
    """
    Demand forecasting model for dynamic pricing.
    
    This class provides models for forecasting future product demand based on:
    1. Historical price and sales data
    2. Seasonal patterns and trends
    3. Market signals from competitor pricing
    4. Current product positioning and inventory levels
    5. Customer segment information
    """
    
    def __init__(self, forecast_horizon=30, use_advanced_features=True):
        """
        Initialize the demand forecaster
        
        Args:
            forecast_horizon: Number of days to forecast
            use_advanced_features: Whether to use advanced features from Amazon API
        """
        self.forecast_horizon = forecast_horizon
        self.use_advanced_features = use_advanced_features
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize models
        self.time_series_model = None
        self.feature_based_model = None
        self.ensemble_model = None
        
        # Model parameters
        self.arima_order = (1, 1, 1)  # Default ARIMA parameters (p,d,q)
        self.sarimax_order = (1, 1, 1)  # Default SARIMAX parameters (p,d,q)
        self.sarimax_seasonal_order = (1, 1, 1, 7)  # Default seasonal parameters (P,D,Q,s)
        
        # Seasonality parameters
        self.seasonal_periods = {
            'daily': 1,
            'weekly': 7,
            'monthly': 30,
            'quarterly': 90
        }
        
        # Load pre-trained models if available
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models if available"""
        feature_model_path = os.path.join(self.model_dir, 'demand_feature_model.pkl')
        
        if os.path.exists(feature_model_path):
            try:
                self.feature_based_model = joblib.load(feature_model_path)
                print(f"Loaded feature-based demand model from {feature_model_path}")
            except Exception as e:
                print(f"Error loading feature-based model: {e}")
    
    def _save_models(self):
        """Save trained models"""
        if self.feature_based_model is not None:
            feature_model_path = os.path.join(self.model_dir, 'demand_feature_model.pkl')
            
            try:
                joblib.dump(self.feature_based_model, feature_model_path)
                print(f"Saved feature-based demand model to {feature_model_path}")
            except Exception as e:
                print(f"Error saving feature-based model: {e}")
    
    def _create_time_series_features(self, df):
        """
        Create time-based features for demand forecasting
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            DataFrame with time features added
        """
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Basic time features
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        df['year'] = df['timestamp'].dt.year
        df['day_of_month'] = df['timestamp'].dt.day
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        
        # Is weekend or not
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Is holiday (simplistic approach, would need a proper holiday calendar)
        df['is_holiday'] = ((df['month'] == 12) & (df['day_of_month'] >= 20)).astype(int)
        
        # Season (Northern Hemisphere)
        df['season'] = pd.cut(
            df['month'], 
            bins=[0, 3, 6, 9, 12], 
            labels=['Winter', 'Spring', 'Summer', 'Fall'],
            include_lowest=True
        ).cat.codes
        
        return df
    
    def _create_lag_features(self, df, target_col, lag_days=[1, 7, 14, 30]):
        """
        Create lag features for time series forecasting
        
        Args:
            df: DataFrame with time series data
            target_col: Target column to create lags for
            lag_days: List of lag days to create
            
        Returns:
            DataFrame with lag features added
        """
        df_copy = df.copy()
        
        # Ensure data is sorted by timestamp
        df_copy = df_copy.sort_values('timestamp')
        
        # Create lag features
        for lag in lag_days:
            df_copy[f'{target_col}_lag_{lag}'] = df_copy[target_col].shift(lag)
        
        # Create rolling window features
        df_copy[f'{target_col}_rolling_mean_7'] = df_copy[target_col].rolling(window=7).mean()
        df_copy[f'{target_col}_rolling_mean_14'] = df_copy[target_col].rolling(window=14).mean()
        df_copy[f'{target_col}_rolling_std_7'] = df_copy[target_col].rolling(window=7).std()
        
        # Create expanding window features
        df_copy[f'{target_col}_expanding_mean'] = df_copy[target_col].expanding().mean()
        
        # Fill NaN values with 0 (appropriate for log-transformed data)
        df_copy = df_copy.fillna(0)
        
        return df_copy
    
    def _preprocess_time_series(self, df, target_col='demand'):
        """
        Preprocess time series data for forecasting
        
        Args:
            df: DataFrame with time series data
            target_col: Target column for forecasting
            
        Returns:
            Preprocessed DataFrame
        """
        # Convert to datetime if needed
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Handle missing values
        if df[target_col].isna().any():
            # Fill missing values with forward fill, then backward fill
            df[target_col] = df[target_col].fillna(method='ffill').fillna(method='bfill')
            
            # If still have NaNs, fill with column mean
            if df[target_col].isna().any():
                df[target_col] = df[target_col].fillna(df[target_col].mean())
        
        # Log transform to stabilize variance if data is strictly positive
        if (df[target_col] > 0).all():
            df[f'{target_col}_log'] = np.log1p(df[target_col])
        else:
            df[f'{target_col}_log'] = df[target_col]
        
        # Add time features
        df = self._create_time_series_features(df)
        
        # Create lag features
        df = self._create_lag_features(df, target_col)
        
        return df
    
    def fit_time_series_model(self, historical_data, target_col='demand', model_type='auto'):
        """
        Fit time series model to historical data
        
        Args:
            historical_data: DataFrame with historical time series data
            target_col: Target column for forecasting
            model_type: Type of time series model ('arima', 'sarimax', or 'auto')
            
        Returns:
            Fitted model and performance metrics
        """
        # Preprocess data
        processed_data = self._preprocess_time_series(historical_data, target_col)
        
        # Remove rows with NaN values (from lag creation)
        processed_data = processed_data.dropna()
        
        if len(processed_data) < 10:
            print("Warning: Not enough historical data for reliable time series modeling")
            return None, {'error': 'Insufficient historical data'}
        
        # Use log-transformed target if available
        target = processed_data[f'{target_col}_log'] if f'{target_col}_log' in processed_data.columns else processed_data[target_col]
        
        # Automatically determine best model type if 'auto'
        if model_type == 'auto':
            # Try different models and pick the best one
            models_to_try = ['arima', 'sarimax']
            best_model = None
            best_aic = float('inf')
            best_model_type = None
            
            for model in models_to_try:
                try:
                    if model == 'arima':
                        arima_model = ARIMA(target, order=self.arima_order)
                        arima_result = arima_model.fit()
                        if arima_result.aic < best_aic:
                            best_model = arima_result
                            best_aic = arima_result.aic
                            best_model_type = 'arima'
                    elif model == 'sarimax':
                        # Only use SARIMAX if we have enough data
                        if len(processed_data) >= 2 * self.sarimax_seasonal_order[3]:
                            sarimax_model = SARIMAX(
                                target, 
                                order=self.sarimax_order,
                                seasonal_order=self.sarimax_seasonal_order
                            )
                            sarimax_result = sarimax_model.fit(disp=False)
                            if sarimax_result.aic < best_aic:
                                best_model = sarimax_result
                                best_aic = sarimax_result.aic
                                best_model_type = 'sarimax'
                except Exception as e:
                    print(f"Error fitting {model} model: {e}")
                    continue
            
            self.time_series_model = {
                'model': best_model,
                'type': best_model_type,
                'aic': best_aic
            }
            
            if best_model is None:
                print("Warning: Could not fit any time series model")
                return None, {'error': 'Failed to fit time series models'}
                
            model_type = best_model_type
        else:
            # Fit the specified model type
            if model_type == 'arima':
                try:
                    arima_model = ARIMA(target, order=self.arima_order)
                    arima_result = arima_model.fit()
                    self.time_series_model = {
                        'model': arima_result,
                        'type': 'arima',
                        'aic': arima_result.aic
                    }
                except Exception as e:
                    print(f"Error fitting ARIMA model: {e}")
                    return None, {'error': f'Failed to fit ARIMA model: {str(e)}'}
            
            elif model_type == 'sarimax':
                try:
                    sarimax_model = SARIMAX(
                        target, 
                        order=self.sarimax_order,
                        seasonal_order=self.sarimax_seasonal_order
                    )
                    sarimax_result = sarimax_model.fit(disp=False)
                    self.time_series_model = {
                        'model': sarimax_result,
                        'type': 'sarimax',
                        'aic': sarimax_result.aic
                    }
                except Exception as e:
                    print(f"Error fitting SARIMAX model: {e}")
                    return None, {'error': f'Failed to fit SARIMAX model: {str(e)}'}
        
        # Generate in-sample predictions
        predictions = self.time_series_model['model'].predict()
        
        # If we used log transform, convert back
        if f'{target_col}_log' in processed_data.columns:
            predictions = np.expm1(predictions)
            actual_values = np.expm1(target)
        else:
            actual_values = target
        
        # Calculate performance metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(actual_values, predictions)),
            'mae': mean_absolute_error(actual_values, predictions),
            'model_type': model_type
        }
        
        return self.time_series_model, metrics
    
    def fit_feature_based_model(self, features_df, target_col='demand', model_type='random_forest'):
        """
        Fit feature-based machine learning model for demand forecasting
        
        Args:
            features_df: DataFrame with features
            target_col: Target column for forecasting
            model_type: ML model type ('random_forest', 'gradient_boosting', 'linear')
            
        Returns:
            Fitted model and performance metrics
        """
        # Clean the data
        data = features_df.copy()
        data = data.dropna()
        
        if len(data) < 10:
            print("Warning: Not enough data for feature-based modeling")
            return None, {'error': 'Insufficient data'}
        
        # Identify feature columns (exclude target and any timestamp/date columns)
        exclude_cols = [target_col]
        if 'timestamp' in data.columns:
            exclude_cols.append('timestamp')
        if 'date' in data.columns:
            exclude_cols.append('date')
            
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # Prepare features and target
        X = data[feature_cols]
        y = data[target_col]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Select and train model
        if model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif model_type == 'linear':
            model = LinearRegression()
        else:
            print(f"Unknown model type: {model_type}, using Random Forest")
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Fit model
        model.fit(X_scaled, y)
        
        # Generate predictions
        predictions = model.predict(X_scaled)
        
        # Calculate performance metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions),
            'r2': r2_score(y, predictions),
            'model_type': model_type
        }
        
        # Store model
        self.feature_based_model = {
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'type': model_type
        }
        
        # Save model
        self._save_models()
        
        return self.feature_based_model, metrics
    
    def forecast_demand(self, forecast_features, historical_data=None, forecast_days=30):
        """
        Generate demand forecast using the trained models
        
        Args:
            forecast_features: DataFrame with features for forecast period
            historical_data: Historical time series data (optional)
            forecast_days: Number of days to forecast
            
        Returns:
            DataFrame with forecasted demand
        """
        # If feature-based model not available, try to load it
        if self.feature_based_model is None:
            self._load_models()
        
        # Check if we have enough data for forecasting
        if self.feature_based_model is None and historical_data is None:
            return {'error': 'No models or historical data available for forecasting'}
        
        # Initialize results
        results = {}
        forecasts = []
        
        # Generate feature-based forecast if model is available
        if self.feature_based_model is not None:
            # Extract required features
            model_info = self.feature_based_model
            feature_cols = model_info['feature_cols']
            
            # Check if all required features are available
            missing_features = [col for col in feature_cols if col not in forecast_features.columns]
            
            if missing_features:
                print(f"Warning: Missing features for feature-based forecast: {missing_features}")
                features_forecast = None
            else:
                # Prepare features
                X = forecast_features[feature_cols]
                
                # Scale features
                X_scaled = model_info['scaler'].transform(X)
                
                # Generate predictions
                features_forecast = model_info['model'].predict(X_scaled)
                
                results['feature_based'] = {
                    'forecast': features_forecast,
                    'model_type': model_info['type']
                }
        else:
            features_forecast = None
        
        # Generate time series forecast if historical data is available
        if historical_data is not None and len(historical_data) > 10:
            # Fit time series model if not already done
            if self.time_series_model is None:
                self.fit_time_series_model(historical_data)
            
            if self.time_series_model is not None:
                model_info = self.time_series_model
                
                # Get the model
                model = model_info['model']
                
                # Generate forecast
                try:
                    # For ARIMA/SARIMAX models
                    forecast = model.forecast(steps=forecast_days)
                    
                    # If we used log transform, convert back
                    if 'demand_log' in historical_data.columns:
                        forecast = np.expm1(forecast)
                    
                    results['time_series'] = {
                        'forecast': forecast,
                        'model_type': model_info['type']
                    }
                except Exception as e:
                    print(f"Error generating time series forecast: {e}")
        
        # Combine forecasts (simple average if both are available)
        if 'feature_based' in results and 'time_series' in results:
            # Use weighted average based on model performance
            # Here we're just giving equal weight, but could be adjusted
            combined_forecast = (results['feature_based']['forecast'] + results['time_series']['forecast']) / 2
            results['combined'] = {
                'forecast': combined_forecast
            }
            
            # Use combined forecast
            final_forecast = combined_forecast
        elif 'feature_based' in results:
            # Use feature-based forecast
            final_forecast = results['feature_based']['forecast']
        elif 'time_series' in results:
            # Use time series forecast
            final_forecast = results['time_series']['forecast']
        else:
            # No forecast available
            return {'error': 'Could not generate forecast'}
        
        # Create forecast DataFrame
        start_date = datetime.now()
        dates = [start_date + timedelta(days=i) for i in range(forecast_days)]
        
        forecast_df = pd.DataFrame({
            'date': dates,
            'forecasted_demand': final_forecast[:forecast_days]  # Limit to requested days
        })
        
        # Add any additional forecast details
        if 'feature_based' in results:
            forecast_df['feature_forecast'] = results['feature_based']['forecast'][:forecast_days]
        
        if 'time_series' in results:
            forecast_df['time_series_forecast'] = results['time_series']['forecast'][:forecast_days]
        
        # Add forecast metadata
        forecast_meta = {
            'models_used': list(results.keys()),
            'forecast_days': forecast_days,
            'forecast_generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'forecast_horizon': forecast_days
        }
        
        return {
            'forecast': forecast_df,
            'metadata': forecast_meta
        }
    
    def analyze_seasonality(self, historical_data, target_col='demand', period=None):
        """
        Analyze seasonality patterns in historical data
        
        Args:
            historical_data: DataFrame with historical data
            target_col: Target column for analysis
            period: Seasonal period in days (default: auto-detect)
            
        Returns:
            Dictionary with seasonality components
        """
        # Ensure timestamp is datetime
        if 'timestamp' in historical_data.columns:
            historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
            historical_data = historical_data.sort_values('timestamp')
        
        # Ensure target column exists
        if target_col not in historical_data.columns:
            return {'error': f'Target column {target_col} not found in data'}
        
        # Fill missing values
        data = historical_data.copy()
        data[target_col] = data[target_col].fillna(method='ffill').fillna(method='bfill')
        
        # Detect period if not specified
        if period is None:
            # Try to auto-detect based on correlation analysis
            # Simple approach: try common periods and pick the one with highest autocorrelation
            periods_to_try = [7, 14, 30, 90]
            best_period = 7  # Default to weekly
            best_acf = 0
            
            for p in periods_to_try:
                if len(data) >= 2 * p:  # Need at least 2 full periods
                    # Calculate autocorrelation
                    acf = data[target_col].autocorr(lag=p)
                    if abs(acf) > best_acf:
                        best_acf = abs(acf)
                        best_period = p
            
            period = best_period
        
        # Minimum data requirements
        if len(data) < 2 * period:
            return {'error': f'Insufficient data for seasonality analysis with period {period}'}
        
        try:
            # Decompose the time series
            decomposition = seasonal_decompose(
                data[target_col],
                model='additive',
                period=period
            )
            
            # Extract components
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid
            
            # Calculate strength of seasonality
            seasonal_strength = 1 - (np.nanvar(residual) / np.nanvar(residual + seasonal))
            
            # Determine if data has significant seasonality
            has_seasonality = seasonal_strength > 0.3
            
            result = {
                'has_seasonality': has_seasonality,
                'seasonality_strength': seasonal_strength,
                'period': period,
                'period_name': next((name for name, days in self.seasonal_periods.items() 
                                    if days == period), f'{period}_days'),
                'components': {
                    'trend': trend.fillna(0).tolist(),
                    'seasonal': seasonal.fillna(0).tolist(),
                    'residual': residual.fillna(0).tolist()
                }
            }
            
            return result
        
        except Exception as e:
            print(f"Error in seasonality analysis: {e}")
            return {'error': f'Seasonality analysis failed: {str(e)}'}
    
    def get_demand_forecast_visualization(self, forecast_result):
        """
        Generate visualization data for demand forecast
        
        Args:
            forecast_result: Result from forecast_demand method
            
        Returns:
            Dictionary with visualization data
        """
        if 'error' in forecast_result:
            return {'error': forecast_result['error']}
        
        forecast_df = forecast_result['forecast']
        metadata = forecast_result['metadata']
        
        # Convert dates to string format for JSON serialization
        dates = [d.strftime('%Y-%m-%d') for d in forecast_df['date']]
        
        # Prepare visualization data
        viz_data = {
            'dates': dates,
            'forecast': forecast_df['forecasted_demand'].tolist(),
            'forecast_horizon': metadata['forecast_horizon'],
            'generated_at': metadata['forecast_generated']
        }
        
        # Add component forecasts if available
        if 'feature_forecast' in forecast_df.columns:
            viz_data['feature_forecast'] = forecast_df['feature_forecast'].tolist()
        
        if 'time_series_forecast' in forecast_df.columns:
            viz_data['time_series_forecast'] = forecast_df['time_series_forecast'].tolist()
        
        # Calculate forecast statistics
        viz_data['statistics'] = {
            'min': float(forecast_df['forecasted_demand'].min()),
            'max': float(forecast_df['forecasted_demand'].max()),
            'mean': float(forecast_df['forecasted_demand'].mean()),
            'total': float(forecast_df['forecasted_demand'].sum())
        }
        
        # Add model information
        viz_data['models_used'] = metadata['models_used']
        
        return viz_data
    
    def calculate_optimal_inventory(self, forecast_result, lead_time=7, service_level=0.95):
        """
        Calculate optimal inventory levels based on demand forecast
        
        Args:
            forecast_result: Result from forecast_demand method
            lead_time: Lead time for replenishment in days
            service_level: Desired service level (0-1)
            
        Returns:
            Dictionary with inventory recommendations
        """
        if 'error' in forecast_result:
            return {'error': forecast_result['error']}
        
        forecast_df = forecast_result['forecast']
        
        # Extract forecast for lead time period
        lead_time_forecast = forecast_df.head(lead_time)['forecasted_demand']
        
        # Calculate expected demand during lead time
        expected_demand = float(lead_time_forecast.sum())
        
        # Calculate standard deviation of forecast
        demand_std = float(lead_time_forecast.std())
        
        # Calculate safety stock using normal distribution quantile
        from scipy.stats import norm
        z_score = norm.ppf(service_level)
        safety_stock = z_score * demand_std * np.sqrt(lead_time)
        
        # Calculate recommended inventory level
        recommended_inventory = expected_demand + safety_stock
        
        return {
            'expected_demand': expected_demand,
            'safety_stock': safety_stock,
            'recommended_inventory': recommended_inventory,
            'lead_time': lead_time,
            'service_level': service_level
        }
    
    def analyze_price_elasticity(self, historical_data, target_col='demand', price_col='price'):
        """
        Analyze price elasticity of demand from historical data
        
        Args:
            historical_data: DataFrame with historical data
            target_col: Demand column name
            price_col: Price column name
            
        Returns:
            Dictionary with elasticity analysis
        """
        # Ensure required columns exist
        if target_col not in historical_data.columns or price_col not in historical_data.columns:
            return {'error': f'Required columns not found in data'}
        
        # Create log-transformed columns
        data = historical_data.copy()
        data['log_demand'] = np.log(data[target_col].replace(0, 0.1))  # Avoid log(0)
        data['log_price'] = np.log(data[price_col])
        
        # Simple approach: linear regression on log-transformed data
        # dlog(demand) = e * dlog(price) where e is elasticity
        try:
            # Calculate changes
            data['dlog_demand'] = data['log_demand'].diff()
            data['dlog_price'] = data['log_price'].diff()
            
            # Remove NaN values
            data = data.dropna()
            
            if len(data) < 3:
                return {'error': 'Insufficient data for elasticity calculation'}
            
            # Fit linear model
            X = data['dlog_price'].values.reshape(-1, 1)
            y = data['dlog_demand'].values
            
            model = LinearRegression(fit_intercept=False)
            model.fit(X, y)
            
            # Estimated elasticity is the coefficient
            elasticity = float(model.coef_[0])
            
            # Calculate 95% confidence interval
            predictions = model.predict(X)
            mse = mean_squared_error(y, predictions)
            n = len(data)
            se = np.sqrt(mse / (n - 1))
            
            # Confidence interval
            from scipy.stats import t
            t_value = t.ppf(0.975, n - 1)  # 95% CI
            ci_lower = elasticity - t_value * se
            ci_upper = elasticity + t_value * se
            
            # Categorize elasticity
            if abs(elasticity) < 0.5:
                category = 'inelastic'
            elif abs(elasticity) < 1.0:
                category = 'somewhat_inelastic'
            elif abs(elasticity) < 1.5:
                category = 'unit_elastic'
            else:
                category = 'elastic'
            
            result = {
                'elasticity': elasticity,
                'elasticity_category': category,
                'confidence_interval': [float(ci_lower), float(ci_upper)],
                'data_points': len(data),
                'r_squared': float(model.score(X, y))
            }
            
            return result
        
        except Exception as e:
            print(f"Error calculating elasticity: {e}")
            return {'error': f'Elasticity calculation failed: {str(e)}'}
    
    def prepare_forecast_features(self, product_data, price_ratio, segments=None):
        """
        Prepare features for demand forecasting
        
        Args:
            product_data: Product information dictionary
            price_ratio: Planned price ratio (planned price / current price)
            segments: Customer segment information (optional)
            
        Returns:
            DataFrame with features for forecasting
        """
        # Initialize feature dictionary
        features = {}
        
        # Basic product features
        features['product_type'] = product_data.get('product_type', 'unknown')
        features['product_group'] = product_data.get('product_group', 'unknown')
        features['price'] = product_data.get('price', 0) * price_ratio
        features['elasticity'] = product_data.get('elasticity', 1.0)
        features['rating'] = product_data.get('rating', 3.0)
        features['competitors_price'] = product_data.get('competitors_price', features['price'])
        
        # Calculate competition metrics
        features['price_competition_ratio'] = features['price'] / max(0.1, features['competitors_price'])
        
        # Incorporate time features
        today = datetime.now()
        features['day_of_week'] = today.weekday()
        features['month'] = today.month
        features['day_of_month'] = today.day
        features['is_weekend'] = 1 if today.weekday() >= 5 else 0
        
        # Add season (Northern Hemisphere)
        season_mapping = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 
                         7: 2, 8: 2, 9: 2, 10: 3, 11: 3, 12: 3}
        features['season'] = season_mapping.get(today.month, 0)
        
        # Add customer segment information if available
        if segments:
            for segment_name, segment_data in segments.items():
                # Extract key metrics
                features[f'segment_{segment_name}_weight'] = segment_data.get('weight', 0)
                features[f'segment_{segment_name}_price_sensitivity'] = segment_data.get('price_sensitivity', 1.0)
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        
        # One-hot encode categorical columns
        cat_columns = ['product_type', 'product_group']
        for col in cat_columns:
            if col in features_df.columns:
                # Create dummy column for this specific value
                value = features_df[col].iloc[0]
                features_df[f'{col}_{value}'] = 1
                features_df = features_df.drop(col, axis=1)
        
        return features_df
        
    def forecast_product_demand(self, product_data, price_ratio, historical_data=None, segments=None):
        """
        Generate demand forecast for a product at a specific price
        
        Args:
            product_data: Product information dictionary
            price_ratio: Planned price ratio (planned price / current price)
            historical_data: Historical time series data (optional)
            segments: Customer segment information (optional)
            
        Returns:
            Dictionary with demand forecast results
        """
        # Prepare features for forecasting
        forecast_features = self.prepare_forecast_features(product_data, price_ratio, segments)
        
        # Generate forecast
        forecast_result = self.forecast_demand(forecast_features, historical_data)
        
        # If forecast successful, add visualization data
        if 'error' not in forecast_result:
            # Generate visualization data
            viz_data = self.get_demand_forecast_visualization(forecast_result)
            forecast_result['visualization'] = viz_data
            
            # Add price elasticity impact
            elasticity = product_data.get('elasticity', 1.0)
            elasticity_impact = -elasticity * (price_ratio - 1) * 100  # Percentage change in demand
            
            forecast_result['price_impact'] = {
                'price_ratio': price_ratio,
                'elasticity': elasticity,
                'demand_change_pct': elasticity_impact
            }
            
            # Scale forecast based on price elasticity
            if 'forecast' in forecast_result:
                # Apply elasticity adjustment
                elasticity_multiplier = (1 + elasticity_impact/100)
                forecast_result['forecast']['forecasted_demand'] *= elasticity_multiplier
                
                # Update visualization
                viz_data['forecast'] = forecast_result['forecast']['forecasted_demand'].tolist()
                viz_data['statistics'] = {
                    'min': float(forecast_result['forecast']['forecasted_demand'].min()),
                    'max': float(forecast_result['forecast']['forecasted_demand'].max()),
                    'mean': float(forecast_result['forecast']['forecasted_demand'].mean()),
                    'total': float(forecast_result['forecast']['forecasted_demand'].sum())
                }
                
                forecast_result['visualization'] = viz_data
        
        return forecast_result 