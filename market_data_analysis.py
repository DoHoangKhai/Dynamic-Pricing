#!/usr/bin/env python
"""
Market data analysis module for extracting insights from collected Amazon API data.
Provides functions for time series analysis, trend detection, and feature extraction.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import stats

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
TIME_SERIES_DIR = os.path.join(DATA_DIR, 'time_series')
ANALYSIS_DIR = os.path.join(DATA_DIR, 'analysis')

# Create directories if they don't exist
os.makedirs(ANALYSIS_DIR, exist_ok=True)

class MarketDataAnalyzer:
    """Class for analyzing market data time series"""
    
    def __init__(self):
        """Initialize the analyzer"""
        self.data = {}
        
    def load_bestsellers_data(self, category="bestsellers", country="US", type_="best_sellers"):
        """Load bestsellers time series data"""
        ts_filename = f"bestsellers_ts_{category}_{country}_{type_}.csv"
        ts_filepath = os.path.join(TIME_SERIES_DIR, ts_filename)
        
        if os.path.exists(ts_filepath):
            df = pd.read_csv(ts_filepath)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Store in data dict
            self.data['bestsellers'] = df
            return df
        else:
            print(f"No bestsellers data found at {ts_filepath}")
            return None
    
    def load_product_data(self, asin, country="US"):
        """Load product time series data"""
        ts_filename = f"product_ts_{asin}_{country}.csv"
        ts_filepath = os.path.join(TIME_SERIES_DIR, ts_filename)
        
        if os.path.exists(ts_filepath):
            df = pd.read_csv(ts_filepath)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Store in data dict
            key = f"product_{asin}"
            self.data[key] = df
            return df
        else:
            print(f"No product data found at {ts_filepath}")
            return None
    
    def load_reviews_data(self, asin, country="US"):
        """Load reviews time series data"""
        ts_filename = f"reviews_ts_{asin}_{country}.csv"
        ts_filepath = os.path.join(TIME_SERIES_DIR, ts_filename)
        
        if os.path.exists(ts_filepath):
            df = pd.read_csv(ts_filepath)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Store in data dict
            key = f"reviews_{asin}"
            self.data[key] = df
            return df
        else:
            print(f"No reviews data found at {ts_filepath}")
            return None
    
    def load_deals_data(self, country="US"):
        """Load deals time series data"""
        ts_filename = f"deals_ts_{country}.csv"
        ts_filepath = os.path.join(TIME_SERIES_DIR, ts_filename)
        
        if os.path.exists(ts_filepath):
            df = pd.read_csv(ts_filepath)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Store in data dict
            self.data['deals'] = df
            return df
        else:
            print(f"No deals data found at {ts_filepath}")
            return None
    
    def load_sentiment_summary(self, asin, country="US"):
        """Load sentiment summary data"""
        summary_filename = f"sentiment_summary_{asin}_{country}.csv"
        summary_filepath = os.path.join(TIME_SERIES_DIR, summary_filename)
        
        if os.path.exists(summary_filepath):
            df = pd.read_csv(summary_filepath)
            
            # Convert collection_date to datetime
            df['collection_date'] = pd.to_datetime(df['collection_date'])
            
            # Store in data dict
            key = f"sentiment_{asin}"
            self.data[key] = df
            return df
        else:
            print(f"No sentiment data found at {summary_filepath}")
            return None
    
    def analyze_price_trends(self, asin, country="US", days=30):
        """
        Analyze price trends for a specific product
        
        Args:
            asin (str): Product ASIN
            country (str): Country code
            days (int): Number of days to analyze
            
        Returns:
            dict: Price trend analysis
        """
        key = f"product_{asin}"
        if key not in self.data:
            self.load_product_data(asin, country)
            
        if key not in self.data or self.data[key] is None:
            return None
        
        df = self.data[key]
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Filter data for date range
        filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        
        if len(filtered_df) < 2:
            return {
                'asin': asin,
                'insufficient_data': True,
                'message': f"Insufficient data for trend analysis (need at least 2 data points, have {len(filtered_df)})"
            }
        
        # Calculate statistics
        latest_price = filtered_df.iloc[-1]['current_price']
        min_price = filtered_df['current_price'].min()
        max_price = filtered_df['current_price'].max()
        mean_price = filtered_df['current_price'].mean()
        
        # Calculate day-to-day price volatility
        if len(filtered_df) > 1:
            filtered_df = filtered_df.sort_values('timestamp')
            filtered_df['price_change'] = filtered_df['current_price'].diff()
            filtered_df['price_change_pct'] = filtered_df['price_change'] / filtered_df['current_price'].shift(1) * 100
            volatility = filtered_df['price_change_pct'].std()
        else:
            volatility = 0
        
        # Calculate linear trend
        if len(filtered_df) > 1:
            # Convert timestamps to numeric (days since first observation)
            filtered_df = filtered_df.sort_values('timestamp')
            base_time = filtered_df['timestamp'].min()
            filtered_df['days_since'] = (filtered_df['timestamp'] - base_time).dt.total_seconds() / (24 * 3600)
            
            # Calculate linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                filtered_df['days_since'], 
                filtered_df['current_price']
            )
            
            # Determine trend direction
            if slope > 0.05:  # Allow for small fluctuations
                trend = "increasing"
            elif slope < -0.05:
                trend = "decreasing"
            else:
                trend = "stable"
                
            # Calculate daily percent change
            daily_pct_change = slope / intercept * 100
        else:
            trend = "unknown"
            slope = 0
            daily_pct_change = 0
            
        # Create analysis result
        analysis = {
            'asin': asin,
            'title': filtered_df.iloc[0]['title'] if len(filtered_df) > 0 else "Unknown",
            'current_price': latest_price,
            'min_price': min_price,
            'max_price': max_price,
            'mean_price': mean_price,
            'price_range': max_price - min_price,
            'volatility': volatility,
            'trend': trend,
            'daily_change_percent': daily_pct_change,
            'data_points': len(filtered_df)
        }
        
        return analysis
    
    def analyze_market_position(self, asin, category="bestsellers", country="US", type_="best_sellers"):
        """
        Analyze market position by comparing to bestsellers
        
        Args:
            asin (str): Product ASIN
            category (str): Category for bestsellers
            country (str): Country code
            type_ (str): Best seller type
            
        Returns:
            dict: Market position analysis
        """
        if 'bestsellers' not in self.data:
            self.load_bestsellers_data(category, country, type_)
            
        key = f"product_{asin}"
        if key not in self.data:
            self.load_product_data(asin, country)
            
        if 'bestsellers' not in self.data or key not in self.data:
            return None
        
        bestsellers_df = self.data['bestsellers']
        product_df = self.data[key]
        
        # Get latest product data
        latest_product = product_df.sort_values('timestamp', ascending=False).iloc[0]
        
        # Check if product is in bestsellers
        current_rank = None
        latest_bestsellers = bestsellers_df[bestsellers_df['timestamp'] == bestsellers_df['timestamp'].max()]
        
        if asin in latest_bestsellers['asin'].values:
            current_rank = latest_bestsellers[latest_bestsellers['asin'] == asin]['rank'].values[0]
        
        # Find similar products in bestsellers for price comparison
        product_title = latest_product['title'].lower()
        title_words = set(word.lower() for word in product_title.split() if len(word) > 3)
        
        similar_products = []
        
        for _, bestseller in latest_bestsellers.iterrows():
            bestseller_title = bestseller['title'].lower()
            bestseller_words = set(word.lower() for word in bestseller_title.split() if len(word) > 3)
            
            # Calculate Jaccard similarity
            if len(title_words) > 0 and len(bestseller_words) > 0:
                similarity = len(title_words & bestseller_words) / len(title_words | bestseller_words)
                
                if similarity > 0.2:  # Threshold for similarity
                    similar_products.append({
                        'asin': bestseller['asin'],
                        'title': bestseller['title'],
                        'rank': bestseller['rank'],
                        'price': bestseller['current_price'],
                        'similarity': similarity
                    })
        
        # Calculate price position relative to similar products
        if similar_products:
            similar_prices = [p['price'] for p in similar_products]
            avg_similar_price = sum(similar_prices) / len(similar_prices)
            
            price_position = latest_product['current_price'] / avg_similar_price
            
            # Determine if price is competitive
            if price_position < 0.9:
                price_status = "very competitive"
            elif price_position < 1.0:
                price_status = "competitive"
            elif price_position < 1.1:
                price_status = "average"
            elif price_position < 1.2:
                price_status = "premium"
            else:
                price_status = "very premium"
        else:
            avg_similar_price = None
            price_position = None
            price_status = "unknown"
        
        # Create analysis result
        analysis = {
            'asin': asin,
            'title': latest_product['title'],
            'current_price': latest_product['current_price'],
            'current_rank': current_rank,
            'in_bestsellers': current_rank is not None,
            'similar_products_count': len(similar_products),
            'avg_similar_price': avg_similar_price,
            'price_position': price_position,
            'price_status': price_status,
            'similar_products': similar_products
        }
        
        return analysis
    
    def analyze_review_sentiment(self, asin, country="US"):
        """
        Analyze review sentiment trends
        
        Args:
            asin (str): Product ASIN
            country (str): Country code
            
        Returns:
            dict: Sentiment analysis
        """
        sentiment_key = f"sentiment_{asin}"
        if sentiment_key not in self.data:
            self.load_sentiment_summary(asin, country)
            
        reviews_key = f"reviews_{asin}"
        if reviews_key not in self.data:
            self.load_reviews_data(asin, country)
            
        if sentiment_key not in self.data or reviews_key not in self.data:
            return None
        
        sentiment_df = self.data[sentiment_key]
        reviews_df = self.data[reviews_key]
        
        # Get the latest sentiment distribution
        latest_date = sentiment_df['collection_date'].max()
        latest_sentiment = sentiment_df[sentiment_df['collection_date'] == latest_date]
        
        total_count = latest_sentiment['count'].sum()
        
        sentiment_distribution = {}
        for _, row in latest_sentiment.iterrows():
            sentiment_distribution[row['sentiment']] = {
                'count': row['count'],
                'percentage': row['count'] / total_count * 100 if total_count > 0 else 0
            }
            
        # Calculate sentiment score (-1 to 1)
        positive_pct = sentiment_distribution.get('positive', {}).get('percentage', 0)
        negative_pct = sentiment_distribution.get('negative', {}).get('percentage', 0)
        
        sentiment_score = (positive_pct - negative_pct) / 100
        
        # Determine sentiment trend
        sentiment_trend = "stable"
        
        # Get recent reviews info
        recent_reviews = reviews_df.sort_values('timestamp', ascending=False).iloc[:5].to_dict('records')
        
        # Create analysis result
        analysis = {
            'asin': asin,
            'latest_date': latest_date.strftime('%Y-%m-%d') if isinstance(latest_date, pd.Timestamp) else latest_date,
            'sentiment_distribution': sentiment_distribution,
            'sentiment_score': sentiment_score,
            'sentiment_trend': sentiment_trend,
            'total_reviews_analyzed': total_count,
            'recent_reviews': recent_reviews
        }
        
        return analysis
    
    def analyze_deals_activity(self, country="US", days=30):
        """
        Analyze deals activity to understand market dynamics
        
        Args:
            country (str): Country code
            days (int): Number of days to analyze
            
        Returns:
            dict: Deals activity analysis
        """
        if 'deals' not in self.data:
            self.load_deals_data(country)
            
        if 'deals' not in self.data or self.data['deals'] is None:
            return None
        
        df = self.data['deals']
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Filter data for date range
        filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        
        if len(filtered_df) == 0:
            return {
                'country': country,
                'insufficient_data': True,
                'message': "No deals data available for the specified period"
            }
            
        # Calculate average discount
        avg_discount = filtered_df['discount'].mean()
        
        # Get deal types distribution
        deal_types = filtered_df['deal_type'].value_counts().to_dict()
        
        # Calculate deals frequency
        collection_dates = filtered_df['collection_date'].nunique()
        
        # Create analysis result
        analysis = {
            'country': country,
            'period_days': days,
            'total_deals': len(filtered_df),
            'unique_deals': filtered_df['deal_id'].nunique(),
            'avg_discount': avg_discount,
            'deal_types': deal_types,
            'collection_dates': collection_dates,
            'deals_frequency': len(filtered_df) / collection_dates if collection_dates > 0 else 0
        }
        
        return analysis
    
    def extract_pricing_features(self, asin, country="US"):
        """
        Extract features for pricing model
        
        Args:
            asin (str): Product ASIN
            country (str): Country code
            
        Returns:
            dict: Features for pricing model
        """
        # Collect all analyses
        price_trends = self.analyze_price_trends(asin, country)
        market_position = self.analyze_market_position(asin, country=country)
        sentiment = self.analyze_review_sentiment(asin, country)
        deals = self.analyze_deals_activity(country)
        
        if price_trends is None or market_position is None:
            return None
        
        # Extract key features for pricing
        features = {
            'asin': asin,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            
            # Basic product info
            'title': price_trends.get('title', ''),
            'current_price': price_trends.get('current_price', 0),
            
            # Price trend features
            'price_volatility': price_trends.get('volatility', 0),
            'price_trend': price_trends.get('trend', 'unknown'),
            'daily_price_change_percent': price_trends.get('daily_change_percent', 0),
            
            # Market position features
            'in_bestsellers': market_position.get('in_bestsellers', False),
            'bestseller_rank': market_position.get('current_rank', None),
            'price_position': market_position.get('price_position', 1.0),
            'price_status': market_position.get('price_status', 'average'),
            
            # Sentiment features (if available)
            'sentiment_score': sentiment.get('sentiment_score', 0) if sentiment else 0,
            
            # Deal activity features (if available)
            'market_avg_discount': deals.get('avg_discount', 0) if deals else 0,
            'market_deals_frequency': deals.get('deals_frequency', 0) if deals else 0
        }
        
        return features
    
    def generate_market_report(self, asins, category="bestsellers", country="US"):
        """
        Generate a comprehensive market report for multiple products
        
        Args:
            asins (list): List of product ASINs
            category (str): Category for bestsellers
            country (str): Country code
            
        Returns:
            dict: Comprehensive market report
        """
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'category': category,
            'country': country,
            'products': [],
            'market_summary': {},
            'deals_activity': None
        }
        
        # Get deals activity
        deals_activity = self.analyze_deals_activity(country)
        if deals_activity:
            report['deals_activity'] = deals_activity
        
        # Analyze each product
        for asin in asins:
            # Extract pricing features
            features = self.extract_pricing_features(asin, country)
            
            if features:
                report['products'].append(features)
        
        # Generate market summary
        if report['products']:
            avg_volatility = sum(p['price_volatility'] for p in report['products']) / len(report['products'])
            avg_price_position = sum(p['price_position'] for p in report['products'] if p['price_position']) / len(report['products'])
            
            # Count trends
            trends = [p['price_trend'] for p in report['products']]
            increasing = trends.count('increasing')
            decreasing = trends.count('decreasing')
            stable = trends.count('stable')
            
            # Determine market trend
            if increasing > decreasing and increasing > stable:
                market_trend = "increasing"
            elif decreasing > increasing and decreasing > stable:
                market_trend = "decreasing"
            else:
                market_trend = "stable"
            
            report['market_summary'] = {
                'products_analyzed': len(report['products']),
                'avg_price_volatility': avg_volatility,
                'avg_price_position': avg_price_position,
                'price_trends': {
                    'increasing': increasing,
                    'decreasing': decreasing,
                    'stable': stable
                },
                'market_trend': market_trend,
                'avg_market_discount': deals_activity.get('avg_discount', 0) if deals_activity else 0,
                'market_competitiveness': 'high' if avg_price_position < 1.05 else 'medium' if avg_price_position < 1.15 else 'low'
            }
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"market_report_{category}_{country}_{timestamp}.json"
        report_filepath = os.path.join(ANALYSIS_DIR, report_filename)
        
        with open(report_filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Market report saved to {report_filepath}")
        return report
    
    def get_latest_market_report(self, category="bestsellers", country="US"):
        """Get the latest market report"""
        # Find all report files
        report_files = [f for f in os.listdir(ANALYSIS_DIR) if f.startswith(f"market_report_{category}_{country}_")]
        
        if not report_files:
            return None
        
        # Sort by timestamp (which is in the filename)
        report_files.sort(reverse=True)
        
        # Load the latest report
        latest_report_path = os.path.join(ANALYSIS_DIR, report_files[0])
        
        with open(latest_report_path, 'r') as f:
            report = json.load(f)
        
        return report
    
    def visualize_price_trends(self, asin, country="US", days=30, save_path=None):
        """
        Visualize price trends for a product
        
        Args:
            asin (str): Product ASIN
            country (str): Country code
            days (int): Number of days to analyze
            save_path (str): Path to save the visualization
            
        Returns:
            str: Path to saved visualization
        """
        key = f"product_{asin}"
        if key not in self.data:
            self.load_product_data(asin, country)
            
        if key not in self.data or self.data[key] is None:
            return None
        
        df = self.data[key]
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Filter data for date range
        filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        
        if len(filtered_df) < 2:
            return None
        
        # Sort by timestamp
        filtered_df = filtered_df.sort_values('timestamp')
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot price trend
        plt.plot(filtered_df['timestamp'], filtered_df['current_price'], marker='o', linestyle='-')
        
        # Add linear trend line
        z = np.polyfit(np.arange(len(filtered_df)), filtered_df['current_price'], 1)
        p = np.poly1d(z)
        plt.plot(filtered_df['timestamp'], p(np.arange(len(filtered_df))), "r--", alpha=0.8)
        
        # Add title and labels
        plt.title(f"Price Trend: {filtered_df.iloc[0]['title']}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.2f}"))
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        # Tight layout
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            plt.close()
            return save_path
        else:
            # Create a default save path
            os.makedirs(os.path.join(ANALYSIS_DIR, 'visualizations'), exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_save_path = os.path.join(ANALYSIS_DIR, 'visualizations', f"price_trend_{asin}_{timestamp}.png")
            
            plt.savefig(default_save_path)
            plt.close()
            return default_save_path
    
    def visualize_sentiment_distribution(self, asin, country="US", save_path=None):
        """
        Visualize sentiment distribution for a product
        
        Args:
            asin (str): Product ASIN
            country (str): Country code
            save_path (str): Path to save the visualization
            
        Returns:
            str: Path to saved visualization
        """
        sentiment_key = f"sentiment_{asin}"
        if sentiment_key not in self.data:
            self.load_sentiment_summary(asin, country)
            
        if sentiment_key not in self.data or self.data[sentiment_key] is None:
            return None
        
        sentiment_df = self.data[sentiment_key]
        
        # Get the latest sentiment distribution
        latest_date = sentiment_df['collection_date'].max()
        latest_sentiment = sentiment_df[sentiment_df['collection_date'] == latest_date]
        
        if len(latest_sentiment) == 0:
            return None
        
        # Create a pivot table for sentiment counts
        pivot = latest_sentiment.pivot(index='collection_date', columns='sentiment', values='count')
        pivot = pivot.fillna(0)
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Define colors for sentiments
        colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
        
        # Create bar chart
        ax = pivot.plot(kind='bar', stacked=True, color=[colors.get(x, 'blue') for x in pivot.columns])
        
        # Add title and labels
        plt.title(f"Sentiment Distribution: {asin}")
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.legend(title="Sentiment")
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        # Tight layout
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            plt.close()
            return save_path
        else:
            # Create a default save path
            os.makedirs(os.path.join(ANALYSIS_DIR, 'visualizations'), exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_save_path = os.path.join(ANALYSIS_DIR, 'visualizations', f"sentiment_{asin}_{timestamp}.png")
            
            plt.savefig(default_save_path)
            plt.close()
            return default_save_path

# For manual testing
if __name__ == "__main__":
    analyzer = MarketDataAnalyzer()
    
    # Example usage:
    # analyzer.load_bestsellers_data()
    # analyzer.load_product_data('B07ZPKBL9V')
    # analyzer.analyze_price_trends('B07ZPKBL9V')
    # analyzer.visualize_price_trends('B07ZPKBL9V') 