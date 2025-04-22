"""
Enhanced Customer Segmentation Module

This module extends the basic CustomerSegmentation class with the ability to process
Amazon API data for more accurate customer segmentation based on real-time market data.
"""

import os
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from models.pricing_strategies import CustomerSegmentation

class EnhancedCustomerSegmentation(CustomerSegmentation):
    """
    Enhanced customer segmentation using Amazon API data
    
    This class extends the base CustomerSegmentation with capabilities to incorporate:
    1. Review sentiment and content analysis for identifying preference clusters
    2. Search data analysis for identifying intent patterns
    3. Purchase behavior analysis from order data
    4. Temporal segmentation (time of day, day of week patterns)
    """
    
    def __init__(self, segment_count=4, use_advanced_features=True):
        """
        Initialize enhanced customer segmentation
        
        Args:
            segment_count: Number of segments to use
            use_advanced_features: Whether to use advanced features from Amazon API
        """
        super().__init__(segment_count)
        self.use_advanced_features = use_advanced_features
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Extended segment attributes for more nuanced classifications
        self.extended_segments = {
            'bargain_hunters': {
                'weight': 0.25,
                'price_sensitivity': 2.8,
                'quality_importance': 0.4,
                'brand_loyalty': 0.2,
                'conversion_base': 0.07,
                'max_premium': 0.02,
                'search_keywords': ['cheap', 'discount', 'deal', 'low price', 'bargain'],
                'buying_time': 'weekend',
                'review_focus': 'price',
                'profit_per_conversion': 0.5
            },
            'value_seekers': {
                'weight': 0.35,
                'price_sensitivity': 1.5,
                'quality_importance': 1.2,
                'brand_loyalty': 0.6,
                'conversion_base': 0.12,
                'max_premium': 0.15,
                'search_keywords': ['best value', 'worth', 'quality for price'],
                'buying_time': 'weekday',
                'review_focus': 'value',
                'profit_per_conversion': 1.0
            },
            'quality_focused': {
                'weight': 0.25,
                'price_sensitivity': 0.7,
                'quality_importance': 1.8,
                'brand_loyalty': 0.8,
                'conversion_base': 0.15,
                'max_premium': 0.35,
                'search_keywords': ['best quality', 'premium', 'top rated'],
                'buying_time': 'any',
                'review_focus': 'features',
                'profit_per_conversion': 1.5
            },
            'premium_buyers': {
                'weight': 0.10,
                'price_sensitivity': 0.3,
                'quality_importance': 2.0,
                'brand_loyalty': 0.9,
                'conversion_base': 0.12,
                'max_premium': 0.60,
                'search_keywords': ['luxury', 'exclusive', 'premium'],
                'buying_time': 'evening',
                'review_focus': 'experience',
                'profit_per_conversion': 2.0
            },
            'trend_followers': {
                'weight': 0.05,
                'price_sensitivity': 1.0,
                'quality_importance': 1.2,
                'brand_loyalty': 0.4,
                'conversion_base': 0.10,
                'max_premium': 0.25,
                'search_keywords': ['popular', 'trending', 'best seller'],
                'buying_time': 'any',
                'review_focus': 'popularity',
                'profit_per_conversion': 1.2
            }
        }
        
        # Initialize customer clusters for advanced segmentation
        self.customer_clusters = None
        self.sentiment_patterns = None
        self.search_patterns = None
        
        # Load pre-trained models if available
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models if available"""
        customer_clusters_path = os.path.join(self.model_dir, 'customer_clusters.pkl')
        
        if os.path.exists(customer_clusters_path):
            try:
                with open(customer_clusters_path, 'rb') as f:
                    self.customer_clusters = pickle.load(f)
            except Exception as e:
                print(f"Error loading customer clusters: {e}")
    
    def _save_models(self):
        """Save trained models"""
        if self.customer_clusters is not None:
            customer_clusters_path = os.path.join(self.model_dir, 'customer_clusters.pkl')
            
            try:
                with open(customer_clusters_path, 'wb') as f:
                    pickle.dump(self.customer_clusters, f)
            except Exception as e:
                print(f"Error saving customer clusters: {e}")
    
    def analyze_review_data(self, reviews_data):
        """
        Analyze review data to extract customer segments
        
        Args:
            reviews_data: List of reviews from Amazon API
            
        Returns:
            Dictionary with segment insights
        """
        if not reviews_data or not isinstance(reviews_data, list):
            return None
            
        # Extract features from reviews
        review_features = []
        for review in reviews_data:
            if isinstance(review, dict):
                # Extract features
                rating = review.get('rating', 0)
                
                # Simple sentiment score based on rating
                sentiment = (rating - 3) / 2  # -1 to 1 scale
                
                # Additional features if available
                verified = 1 if review.get('verified_purchase', False) else 0
                helpful_votes = review.get('helpful_votes', 0)
                
                # Create feature vector
                feature = [rating, sentiment, verified, helpful_votes]
                review_features.append(feature)
        
        if not review_features:
            return None
            
        # Convert to numpy array
        X = np.array(review_features)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Cluster reviews to identify customer segments
        k = min(self.segment_count, len(X_scaled))
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Store model
        self.customer_clusters = {
            'kmeans': kmeans,
            'scaler': scaler
        }
        self._save_models()
        
        # Analyze clusters
        cluster_stats = {}
        for i in range(k):
            cluster_mask = clusters == i
            cluster_reviews = X[cluster_mask]
            
            if len(cluster_reviews) > 0:
                avg_rating = np.mean(cluster_reviews[:, 0])
                avg_sentiment = np.mean(cluster_reviews[:, 1])
                pct_verified = np.mean(cluster_reviews[:, 2])
                avg_helpful = np.mean(cluster_reviews[:, 3])
                
                cluster_stats[i] = {
                    'size': len(cluster_reviews),
                    'avg_rating': avg_rating,
                    'avg_sentiment': avg_sentiment,
                    'pct_verified': pct_verified,
                    'avg_helpful': avg_helpful
                }
        
        return {
            'clusters': cluster_stats,
            'segment_count': k
        }
    
    def analyze_search_data(self, search_data):
        """
        Analyze search data to extract customer intent patterns
        
        Args:
            search_data: Search data from Amazon API
            
        Returns:
            Dictionary with search patterns
        """
        if not search_data or not isinstance(search_data, dict):
            return None
            
        # Extract search results
        search_results = search_data.get('results', [])
        if not search_results:
            return None
            
        # Analyze price ranges in search results
        prices = []
        for item in search_results:
            if isinstance(item, dict) and 'price' in item:
                try:
                    price = float(item['price'].replace('$', '').replace(',', ''))
                    prices.append(price)
                except (ValueError, AttributeError):
                    continue
        
        if not prices:
            return None
            
        # Calculate price statistics
        min_price = min(prices)
        max_price = max(prices)
        avg_price = sum(prices) / len(prices)
        price_range = max_price - min_price
        
        # Calculate price distribution percentiles
        price_percentiles = {
            'p10': np.percentile(prices, 10),
            'p25': np.percentile(prices, 25),
            'p50': np.percentile(prices, 50),
            'p75': np.percentile(prices, 75),
            'p90': np.percentile(prices, 90)
        }
        
        return {
            'price_stats': {
                'min': min_price,
                'max': max_price,
                'avg': avg_price,
                'range': price_range
            },
            'price_percentiles': price_percentiles,
            'result_count': len(search_results)
        }
    
    def get_enhanced_segment_distribution(self, product_data, reviews_data=None, search_data=None):
        """
        Get enhanced segment distribution using additional Amazon API data
        
        Args:
            product_data: Product data dictionary
            reviews_data: Review data from Amazon API (optional)
            search_data: Search data from Amazon API (optional)
            
        Returns:
            Dictionary with adjusted segment weights
        """
        # Extract basic product information
        product_type = product_data.get('product_type', 'standard')
        product_group = product_data.get('product_group', 'general')
        rating = product_data.get('rating', 3.0)
        
        # Get base segment distribution
        base_segments = self.get_segment_distribution(product_type, product_group, rating)
        
        # If advanced features disabled or no additional data, return base segments
        if not self.use_advanced_features or (reviews_data is None and search_data is None):
            return base_segments
            
        # Analyze review data if available
        review_insights = None
        if reviews_data:
            review_insights = self.analyze_review_data(reviews_data)
            
        # Analyze search data if available
        search_insights = None
        if search_data:
            search_insights = self.analyze_search_data(search_data)
            
        # Adjust segment weights based on additional insights
        adjusted_segments = {k: v.copy() for k, v in base_segments.items()}
        
        # Apply review-based adjustments
        if review_insights and 'clusters' in review_insights:
            clusters = review_insights['clusters']
            
            # Look for clusters with very high/low ratings
            high_rating_cluster = False
            low_rating_cluster = False
            
            for cluster_id, stats in clusters.items():
                if stats['avg_rating'] >= 4.5 and stats['size'] >= 10:
                    high_rating_cluster = True
                elif stats['avg_rating'] <= 2.5 and stats['size'] >= 5:
                    low_rating_cluster = True
            
            # Adjust segments based on rating clusters
            if high_rating_cluster:
                # More quality-focused customers for highly-rated products
                if 'quality_focused' in adjusted_segments:
                    adjusted_segments['quality_focused']['weight'] *= 1.3
                if 'premium_buyers' in adjusted_segments:
                    adjusted_segments['premium_buyers']['weight'] *= 1.2
                if 'price_sensitive' in adjusted_segments:
                    adjusted_segments['price_sensitive']['weight'] *= 0.8
                    
            if low_rating_cluster:
                # More price-sensitive customers for poorly-rated products
                if 'price_sensitive' in adjusted_segments:
                    adjusted_segments['price_sensitive']['weight'] *= 1.4
                if 'premium_buyers' in adjusted_segments:
                    adjusted_segments['premium_buyers']['weight'] *= 0.6
        
        # Apply search-based adjustments
        if search_insights and 'price_stats' in search_insights:
            price_stats = search_insights['price_stats']
            price_percentiles = search_insights.get('price_percentiles', {})
            
            # Get product price
            product_price = product_data.get('price', 0)
            
            if product_price > 0 and price_stats['avg'] > 0:
                # Calculate price position relative to search results
                price_ratio = product_price / price_stats['avg']
                
                # Adjust segments based on price position
                if price_ratio <= 0.7:
                    # Low price - more bargain hunters and value seekers
                    if 'price_sensitive' in adjusted_segments:
                        adjusted_segments['price_sensitive']['weight'] *= 1.3
                    if 'value_seekers' in adjusted_segments:
                        adjusted_segments['value_seekers']['weight'] *= 1.1
                    if 'premium_buyers' in adjusted_segments:
                        adjusted_segments['premium_buyers']['weight'] *= 0.7
                        
                elif price_ratio >= 1.3:
                    # High price - more premium buyers and quality focused
                    if 'premium_buyers' in adjusted_segments:
                        adjusted_segments['premium_buyers']['weight'] *= 1.3
                    if 'quality_focused' in adjusted_segments:
                        adjusted_segments['quality_focused']['weight'] *= 1.2
                    if 'price_sensitive' in adjusted_segments:
                        adjusted_segments['price_sensitive']['weight'] *= 0.7
        
        # Normalize weights to sum to 1.0
        total_weight = sum(segment['weight'] for segment in adjusted_segments.values())
        for segment in adjusted_segments.values():
            segment['weight'] /= total_weight
        
        # Store current distribution
        self.current_segment_distribution = adjusted_segments
        return adjusted_segments
    
    def get_segment_visualization_data(self):
        """
        Get data for segment visualization
        
        Returns:
            Dictionary with visualization data
        """
        if self.current_segment_distribution is None:
            return None
            
        segments = []
        for name, data in self.current_segment_distribution.items():
            segments.append({
                'name': name.replace('_', ' ').title(),
                'weight': data['weight'],
                'price_sensitivity': data.get('price_sensitivity', 1.0),
                'conversion_rate': data.get('conversion_base', 0.1),
                'max_premium': data.get('max_premium', 0.1),
                'profit_factor': data.get('profit_per_conversion', 1.0)
            })
            
        # Sort by weight (descending)
        segments.sort(key=lambda x: x['weight'], reverse=True)
        
        return {
            'segments': segments
        }
    
    def calculate_segment_impact(self, price_ratio):
        """
        Calculate the financial impact of each segment at a given price
        
        Args:
            price_ratio: Price ratio (current price / reference price)
            
        Returns:
            Dictionary with segment impact data
        """
        if self.current_segment_distribution is None:
            return None
        
        segment_impacts = []
        total_weighted_conversion = 0
        total_profit_contribution = 0
        
        for name, segment in self.current_segment_distribution.items():
            # Calculate price effect based on segment price sensitivity
            price_effect = 1.0 - (price_ratio - 1.0) * segment['price_sensitivity']
            price_effect = max(0.01, min(1.0, price_effect))  # Limit to 0.01-1.0 range
            
            # Calculate conversion probability
            conversion_base = segment.get('conversion_base', 0.1)
            conversion_prob = conversion_base * price_effect
            
            # Calculate weighted conversion
            weighted_conversion = conversion_prob * segment['weight']
            
            # Calculate profit contribution
            profit_per_conversion = segment.get('profit_per_conversion', 1.0)
            profit_contribution = weighted_conversion * profit_per_conversion
            
            # Add to totals
            total_weighted_conversion += weighted_conversion
            total_profit_contribution += profit_contribution
            
            # Add segment impact
            segment_impacts.append({
                'name': name.replace('_', ' ').title(),
                'weight': segment['weight'],
                'conversion_rate': conversion_prob,
                'weighted_conversion': weighted_conversion,
                'profit_contribution': profit_contribution,
                'price_effect': price_effect
            })
        
        # Calculate percentage contributions
        for impact in segment_impacts:
            impact['conversion_pct'] = impact['weighted_conversion'] / total_weighted_conversion if total_weighted_conversion > 0 else 0
            impact['profit_pct'] = impact['profit_contribution'] / total_profit_contribution if total_profit_contribution > 0 else 0
        
        # Sort by profit contribution (descending)
        segment_impacts.sort(key=lambda x: x['profit_contribution'], reverse=True)
        
        return {
            'segments': segment_impacts,
            'total_conversion': total_weighted_conversion,
            'total_profit': total_profit_contribution
        } 