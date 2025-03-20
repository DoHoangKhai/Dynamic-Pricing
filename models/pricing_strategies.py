"""
Pricing Strategies Module for Dynamic Pricing Model

This module implements various dynamic pricing strategies that can be used
by the reinforcement learning model, including penetration pricing,
skimming, dynamic discounting, and competitive positioning.
"""

import numpy as np
import math

class PricingStrategy:
    """
    Dynamic pricing strategy that combines multiple approaches to optimize pricing.
    
    This module implements various pricing strategies including:
    1. Cost-plus pricing
    2. Value-based pricing
    3. Competition-based pricing
    4. Elasticity-based pricing
    
    These strategies are combined using a weighted approach to determine the optimal price.
    """
    
    def __init__(self):
        """Initialize the pricing strategy module"""
        self.strategies = {
            'cost_plus': self._cost_plus_pricing,
            'value_based': self._value_based_pricing,
            'competition_based': self._competition_based_pricing,
            'penetration': self._penetration_pricing,
            'premium': self._premium_pricing,
            'skimming': self._price_skimming,
            'dynamic_elasticity': self._elasticity_based_pricing,
            'psychological': self._psychological_pricing,
            'bundle': self._bundle_pricing,
            'promotional': self._promotional_pricing
        }
        
        # Default strategy weights - optimized for better market performance
        self.strategy_weights = {
            'cost_plus': 0.05,          # Reduced from 0.1 - less focus on simple cost-plus
            'value_based': 0.20,        # Increased from 0.15 - more focus on value
            'competition_based': 0.30,   # Increased from 0.25 - more competitive focus
            'penetration': 0.15,        # Increased from 0.10 - more aggressive market entry
            'premium': 0.10,            # Same
            'skimming': 0.05,           # Reduced from 0.1 - less skimming strategy
            'dynamic_elasticity': 0.15,  # Increased - more elasticity-based pricing
            'psychological': 0.00,       # Reduced from 0.05 - removed psychological pricing
            'bundle': 0.00,             # Removed bundle strategy
            'promotional': 0.00         # Removed promotional strategy
        }
        
        # Strategy confidence defaults
        self.strategy_confidence = {
            'cost_plus': 0.8,            # High confidence in cost data
            'value_based': 0.7,          # Moderate confidence in value perception
            'competition_based': 0.9,    # High confidence in competitive data
            'penetration': 0.75,         # Moderate-high confidence
            'premium': 0.7,              # Moderate confidence
            'skimming': 0.6,             # Lower confidence
            'dynamic_elasticity': 0.85,  # High confidence in elasticity modeling
            'psychological': 0.5,        # Low confidence
            'bundle': 0.5,               # Low confidence
            'promotional': 0.6           # Moderate-low confidence
        }
        
        # Market tracking
        self.observed_competitor_responses = []
        self.product_history = {}
        self.market_trends = {
            'price_trend': 0.0,  # positive = market prices rising
            'elasticity_trend': 0.0,  # positive = products becoming less elastic
            'competitive_intensity': 0.5  # 0 to 1, higher = more competitive
        }
    
    def _adjust_strategy_weights(self, product, market_info):
        """
        Adjust strategy weights based on product attributes and market information.
        
        Args:
            product: Dictionary with product information
            market_info: Dictionary with market information
            
        Returns:
            Dictionary of adjusted strategy weights
        """
        adjusted_weights = self.strategy_weights.copy()
        
        # Extract product information
        product_type = product.get('product_type', 'standard').lower()
        rating = product.get('rating', 3.0)
        elasticity = product.get('elasticity', 1.0)
        
        # Extract market information
        competitive_intensity = market_info.get('competitive_intensity', 0.5)
        price_trend = market_info.get('price_trend', 0.0)
        
        # Adjust weights based on product type
        if product_type in ['premium', 'luxury']:
            # For premium products, emphasize value and premium pricing
            adjusted_weights['value_based'] += 0.1
            adjusted_weights['premium'] += 0.15
            adjusted_weights['competition_based'] -= 0.15
            adjusted_weights['penetration'] -= 0.1
            
        elif product_type in ['commodity', 'basic', 'basics']:
            # For commodity products, emphasize competition and penetration
            adjusted_weights['competition_based'] += 0.2
            adjusted_weights['penetration'] += 0.1
            adjusted_weights['premium'] -= 0.1
            adjusted_weights['value_based'] -= 0.1
            adjusted_weights['skimming'] -= 0.05
            
        elif product_type in ['electronics', 'technology']:
            # For electronics, emphasize elasticity and competitive pricing
            adjusted_weights['dynamic_elasticity'] += 0.1
            adjusted_weights['competition_based'] += 0.05
            adjusted_weights['skimming'] -= 0.05
            adjusted_weights['penetration'] += 0.05
            
        elif product_type in ['clothing', 'fashion']:
            # For fashion, balance value and competition
            adjusted_weights['value_based'] += 0.05
            adjusted_weights['competition_based'] += 0.05
            adjusted_weights['psychological'] += 0.05
            adjusted_weights['cost_plus'] -= 0.1
            
        elif product_type in ['books', 'media']:
            # For media products, emphasize penetration and competition
            adjusted_weights['penetration'] += 0.15
            adjusted_weights['competition_based'] += 0.1
            adjusted_weights['premium'] -= 0.1
            adjusted_weights['skimming'] -= 0.05
        
        # Adjust weights based on product rating
        if rating >= 4.5:
            # For highly rated products, increase value and premium weights
            adjusted_weights['value_based'] += 0.05
            adjusted_weights['premium'] += 0.05
            adjusted_weights['penetration'] -= 0.05
            
        elif rating <= 3.0:
            # For lower rated products, increase competition and penetration
            adjusted_weights['competition_based'] += 0.1
            adjusted_weights['penetration'] += 0.1
            adjusted_weights['premium'] -= 0.1
            adjusted_weights['skimming'] -= 0.05
        
        # Adjust for competitive intensity
        if competitive_intensity > 0.7:
            # In highly competitive markets, emphasize competition and penetration
            adjusted_weights['competition_based'] += 0.15
            adjusted_weights['penetration'] += 0.1
            adjusted_weights['premium'] -= 0.1
            adjusted_weights['skimming'] -= 0.1
            
        elif competitive_intensity < 0.3:
            # In less competitive markets, increase premium and value pricing
            adjusted_weights['premium'] += 0.1
            adjusted_weights['value_based'] += 0.05
            adjusted_weights['skimming'] += 0.05
            adjusted_weights['competition_based'] -= 0.15
        
        # Adjust for price trends
        if price_trend > 0.05:
            # In rising markets, increase premium and skimming
            adjusted_weights['premium'] += 0.05
            adjusted_weights['skimming'] += 0.05
            adjusted_weights['penetration'] -= 0.05
            
        elif price_trend < -0.05:
            # In declining markets, increase penetration and competition
            adjusted_weights['penetration'] += 0.1
            adjusted_weights['competition_based'] += 0.05
            adjusted_weights['premium'] -= 0.05
            adjusted_weights['skimming'] -= 0.05
        
        # Elasticity adjustments
        if elasticity > 1.2:
            # For highly elastic products, emphasize penetration and elasticity pricing
            adjusted_weights['penetration'] += 0.1
            adjusted_weights['dynamic_elasticity'] += 0.1
            adjusted_weights['premium'] -= 0.1
            adjusted_weights['skimming'] -= 0.05
            
        elif elasticity < 0.8:
            # For inelastic products, emphasize value and premium
            adjusted_weights['value_based'] += 0.05
            adjusted_weights['premium'] += 0.05
            adjusted_weights['penetration'] -= 0.05
        
        # Ensure weights are non-negative and normalize
        for strategy in adjusted_weights:
            adjusted_weights[strategy] = max(0.0, adjusted_weights[strategy])
            
        # Normalize weights to sum to 1.0
        total_weight = sum(adjusted_weights.values())
        for strategy in adjusted_weights:
            adjusted_weights[strategy] /= total_weight
            
        return adjusted_weights
    
    def _competition_based_pricing(self, product, market_info):
        """
        Competition-based pricing strategy. More aggressive in finding optimal price points.
        
        Args:
            product: Dictionary with product information
            market_info: Dictionary with market information
            
        Returns:
            Dictionary with price recommendation and confidence
        """
        # Extract market information
        competitive_intensity = market_info.get('competitive_intensity', 0.5)
        price_trend = market_info.get('price_trend', 0.0)
        current_ratio = market_info.get('current_price_ratio', 1.0)
        
        # Extract product information
        rating = product.get('rating', 3.0)
        elasticity = product.get('elasticity', 1.0)
        ppi = product.get('ppi', 1.0)  # Price Position Index
        product_type = product.get('product_type', 'standard').lower()
        
        # Base price ratio - target the sweet spot of 3-8% below market
        base_ratio = 0.95  # Start with 5% below market as default
        
        # Adjust for price trend (follow the market trend)
        trend_adjustment = price_trend * 0.3  # Reduced impact of trend
        base_ratio += trend_adjustment
        
        # Quality adjustment based on rating
        quality_premium = (rating - 3.0) * 0.02  # Smaller premium for rating
        base_ratio += quality_premium
        
        # Elasticity adjustment (more elastic = more aggressive pricing)
        elasticity_adjustment = (1.0 - elasticity) * 0.05  # Smaller adjustment
        base_ratio += elasticity_adjustment
        
        # Adjust for product type
        if product_type in ['premium', 'luxury']:
            base_ratio += 0.03  # Smaller premium for premium products
        elif product_type in ['commodity', 'basic', 'basics']:
            base_ratio -= 0.03  # More aggressive for commodity products
        
        # Sweet spot for market-beating price points
        # Find optimal price points 3-8% below market for competitive advantage
        if competitive_intensity > 0.5 and elasticity > 0.9:
            # For elastic products in competitive markets, target 3-8% below market
            sweet_spot_adjustment = -0.03 * min(1.5, elasticity)  # More controlled discount
            base_ratio += sweet_spot_adjustment
        
        # Ensure the price ratio is reasonable and within target range
        base_ratio = max(0.92, min(1.05, base_ratio))
        
        # Calculate confidence
        confidence = self.strategy_confidence['competition_based']
        confidence *= (1.0 - (0.5 - competitive_intensity) * 0.5)  # Higher confidence with more competition data
        
        return {
            'price_ratio': base_ratio,
            'confidence': min(1.0, confidence),
            'strategy': 'competition_based'
        }
    
    def _penetration_pricing(self, product, market_info):
        """
        Penetration pricing strategy that aims to gain market share with below-market pricing.
        More aggressive in finding optimal below-market price points.
        
        Args:
            product: Dictionary with product information
            market_info: Dictionary with market information
            
        Returns:
            Dictionary with price recommendation and confidence
        """
        # Extract product information
        cost = product.get('cost', 60.0)
        price = product.get('price', 100.0)
        elasticity = product.get('elasticity', 1.0)
        rating = product.get('rating', 3.0)
        product_type = product.get('product_type', 'standard').lower()
        
        # Calculate minimum viable price (ensuring minimum margin)
        min_margin = 0.15  # Minimum 15% margin
        min_price_ratio = (cost * (1 + min_margin)) / price
        
        # Base penetration discount - more controlled
        base_ratio = 0.94  # Start with 6% below market (more conservative)
        
        # Adjust based on elasticity - more elastic products get more aggressive penetration
        elasticity_factor = min(1.3, elasticity * 1.1)  # Cap at 1.3, more controlled scaling
        penetration_strength = 0.08 * elasticity_factor  # Up to 10.4% penetration for highly elastic products
        
        # Calculate penetration price ratio
        penetration_ratio = 1.0 - penetration_strength
        
        # Adjust for product type
        if product_type in ['premium', 'luxury']:
            penetration_ratio += 0.05  # Less aggressive for premium
        elif product_type in ['commodity', 'basic', 'basics']:
            penetration_ratio -= 0.02  # More aggressive for commodity
        elif product_type in ['electronics', 'books', 'media']:
            penetration_ratio -= 0.01  # More aggressive for these categories
        
        # Quality adjustment - better products need less penetration
        quality_adjustment = (rating - 3.0) * 0.01
        penetration_ratio += quality_adjustment
        
        # Extract market information
        competitive_intensity = market_info.get('competitive_intensity', 0.5)
        
        # Adjust for competitive intensity - more competitive markets need more penetration
        if competitive_intensity > 0.6:
            penetration_ratio -= 0.02  # More aggressive in highly competitive markets
        
        # Ensure we don't go below minimum viable price
        penetration_ratio = max(min_price_ratio, penetration_ratio)
        
        # Ensure the price ratio is reasonable and within target range
        penetration_ratio = max(0.92, min(0.98, penetration_ratio))  # Cap at 0.98 to ensure below-market
        
        # Calculate confidence
        confidence = self.strategy_confidence['penetration']
        confidence *= (0.7 + elasticity * 0.3)  # Higher confidence for elastic products
        
        return {
            'price_ratio': penetration_ratio,
            'confidence': min(1.0, confidence),
            'strategy': 'penetration'
        }
    
    def get_price_recommendations(self, product, market_info=None):
        """
        Get pricing recommendations based on current strategies with enhanced
        focus on market-beating performance and profitability.
        
        Args:
            product: Product information dictionary
            market_info: Market information dictionary
            
        Returns:
            Dictionary with strategy recommendations and weighted suggestion
        """
        if market_info is None:
            market_info = self.market_trends
            
        # Product type-based weight adjustments
        product_type = product.get('product_type', 'standard')
        rating = product.get('rating', 3.0)
        elasticity = product.get('elasticity', 1.0)
        
        # Adjust strategy weights based on product type
        self._adjust_strategy_weights(product, market_info)
        
        # Further adjust weights based on product attributes
        if rating >= 4.5:  # Excellent quality products
            # Increase premium and skimming for high-quality products
            self.strategy_weights['premium'] += 0.10
            self.strategy_weights['skimming'] += 0.05
            self.strategy_weights['competition_based'] -= 0.10
            self.strategy_weights['penetration'] -= 0.05
        elif rating <= 2.5:  # Poor quality products
            # Increase competitive and penetration for lower-quality products
            self.strategy_weights['competition_based'] += 0.15
            self.strategy_weights['penetration'] += 0.10
            self.strategy_weights['premium'] -= 0.15
            self.strategy_weights['skimming'] -= 0.10
            
        # Adjust for elasticity
        if elasticity < 0.8:  # Inelastic products
            # Increase premium and skimming for inelastic products
            self.strategy_weights['premium'] += 0.05
            self.strategy_weights['skimming'] += 0.05
            self.strategy_weights['penetration'] -= 0.05
            self.strategy_weights['dynamic_elasticity'] -= 0.05
        elif elasticity > 1.2:  # Elastic products
            # Increase competitive and penetration for elastic products
            self.strategy_weights['competition_based'] += 0.10
            self.strategy_weights['penetration'] += 0.05
            self.strategy_weights['premium'] -= 0.10
            self.strategy_weights['skimming'] -= 0.05
            
        # Normalize weights to ensure they sum to 1.0
        total_weight = sum(self.strategy_weights.values())
        if total_weight > 0:
            for strategy in self.strategy_weights:
                self.strategy_weights[strategy] /= total_weight
        
        # Get recommendations from each strategy
        recommendations = {}
        for strategy_name, strategy_func in self.strategies.items():
            weight = self.strategy_weights.get(strategy_name, 0.0)
            # Only call strategies with non-zero weights to improve performance
            if weight > 0.001:
                recommendations[strategy_name] = strategy_func(product, market_info)
        
        # Calculate weighted recommendation
        weighted_recommendation = self._calculate_weighted_recommendation(recommendations)
        
        # Make sure we're returning the price ratio from the weighted recommendation
        price_ratio = weighted_recommendation.get('price_ratio', 1.0)
        
        return {
            'strategies': recommendations,
            'weighted_recommendation': weighted_recommendation,
            'strategy_weights': self.strategy_weights,
            'price_ratio': price_ratio  # Add this explicit key at the top level
        }
    
    def _calculate_weighted_recommendation(self, recommendations):
        """
        Calculate weighted recommendation based on strategy confidence and weights
        with enhanced focus on profit maximization and market-beating performance.
        
        Args:
            recommendations: Dictionary of strategy recommendations
            
        Returns:
            Dictionary with weighted recommendation details
        """
        weighted_price_ratio = 0.0
        total_confidence_weight = 0.0
        max_profit = 0.0
        max_profit_strategy = None
        profit_by_strategy = {}
        
        # Weight each strategy by its confidence and strategy weight
        for strategy, rec in recommendations.items():
            strategy_weight = self.strategy_weights.get(strategy, 0.0)
            confidence = rec.get('confidence', 0.0)
            price_ratio = rec.get('price_ratio', 1.0)
            expected_profit = rec.get('expected_profit', 0.0)
            
            # Track profit by strategy
            profit_by_strategy[strategy] = expected_profit
            
            # Track highest profit strategy
            if expected_profit > max_profit:
                max_profit = expected_profit
                max_profit_strategy = strategy
            
            # Apply profit-based weighting - strategies with higher profit get more influence
            profit_factor = 1.0
            if expected_profit > 0:
                # Scale profit factor based on relative profit performance
                profit_factor = 1.0 + (expected_profit / (max_profit + 0.01)) * 0.5
            
            # Apply enhanced weighting
            weighted_price_ratio += price_ratio * confidence * strategy_weight * profit_factor
            total_confidence_weight += confidence * strategy_weight * profit_factor
        
        # Default to 1.0 if no confident strategies
        if total_confidence_weight > 0:
            final_price_ratio = weighted_price_ratio / total_confidence_weight
        else:
            final_price_ratio = 1.0
        
        # Adjust final price ratio to favor higher profit strategies
        if max_profit_strategy and max_profit > 0:
            max_profit_price_ratio = recommendations[max_profit_strategy]['price_ratio']
            
            # Blend with the highest profit strategy (giving it 40% influence - reduced from 50%)
            profit_influence = 0.40
            final_price_ratio = (1 - profit_influence) * final_price_ratio + profit_influence * max_profit_price_ratio
        
        # Ensure minimum profitability - get cost ratio from first product (they should all be the same)
        first_strategy = next(iter(recommendations.values()))
        product_info = first_strategy.get('product_info', {})
        cost_ratio = product_info.get('cost_ratio', 0.6)
        min_profitable_ratio = cost_ratio * 1.25  # Ensure at least 25% margin over cost
        
        # Apply minimum profitability constraint
        final_price_ratio = max(min_profitable_ratio, final_price_ratio)
        
        # Ensure final price ratio is within the target range of 3-8% below market when appropriate
        # Only apply this constraint for non-premium products to avoid forcing discounts on premium items
        product_type = product_info.get('product_type', '').lower()
        if product_type not in ['premium', 'luxury'] and final_price_ratio < 1.0:
            # For below-market prices, ensure they're in the optimal range
            final_price_ratio = max(0.92, min(0.97, final_price_ratio))
        elif product_type not in ['premium', 'luxury'] and final_price_ratio > 1.0:
            # For above-market prices, keep them reasonable
            final_price_ratio = min(1.05, final_price_ratio)
        elif product_type in ['premium', 'luxury']:
            # For premium products, allow higher prices but still keep them reasonable
            final_price_ratio = min(1.15, final_price_ratio)
            
        return {
            'price_ratio': final_price_ratio,
            'total_confidence': total_confidence_weight,
            'max_profit_strategy': max_profit_strategy,
            'max_profit': max_profit,
            'profit_by_strategy': profit_by_strategy
        }
    
    def update_market_trends(self, price_data, competitor_responses):
        """
        Update market trend information based on recent data
        
        Args:
            price_data: List of recent price data points
            competitor_responses: List of recent competitor responses
        """
        if not price_data or len(price_data) < 2:
            return
        
        # Calculate price trend (positive = prices rising)
        price_changes = [price_data[i] - price_data[i-1] for i in range(1, len(price_data))]
        avg_change = sum(price_changes) / len(price_changes) if price_changes else 0
        
        # Update price trend with exponential smoothing
        self.market_trends['price_trend'] = 0.8 * self.market_trends['price_trend'] + 0.2 * avg_change
        
        # Update competitive intensity based on competitor responses
        if competitor_responses:
            recent_responses = competitor_responses[-min(len(competitor_responses), 5):]
            
            # Handle both float values and dictionaries
            response_strengths = []
            for r in recent_responses:
                if isinstance(r, float):
                    response_strengths.append(r)
                elif isinstance(r, dict) and 'response_strength' in r:
                    response_strengths.append(r['response_strength'])
                else:
                    response_strengths.append(0.0)
            
            avg_strength = sum(response_strengths) / len(response_strengths) if response_strengths else 0.0
            self.market_trends['competitive_intensity'] = 0.8 * self.market_trends['competitive_intensity'] + 0.2 * avg_strength
    
    def reset(self):
        """Reset the pricing strategy state"""
        # Reset market tracking
        self.observed_competitor_responses = []
        self.product_history = {}
        self.market_trends = {
            'price_trend': 0.0,  # positive = market prices rising
            'elasticity_trend': 0.0,  # positive = products becoming less elastic
            'competitive_intensity': 0.5  # 0 to 1, higher = more competitive
        }
        
        # Reset strategy weights to defaults
        self.strategy_weights = {
            'cost_plus': 0.05,          # Reduced from 0.1 - less focus on simple cost-plus
            'value_based': 0.20,        # Increased from 0.15 - more focus on value
            'competition_based': 0.30,   # Increased from 0.25 - more competitive focus
            'penetration': 0.15,        # Increased from 0.10 - more aggressive market entry
            'premium': 0.10,            # Same
            'skimming': 0.05,           # Reduced from 0.1 - less skimming strategy
            'dynamic_elasticity': 0.15,  # Increased - more elasticity-based pricing
            'psychological': 0.00,       # Reduced from 0.05 - removed psychological pricing
            'bundle': 0.00,             # Removed bundle strategy
            'promotional': 0.00         # Removed promotional strategy
        }
    
    def _cost_plus_pricing(self, product, market_info):
        """
        Cost-plus pricing strategy that adds a markup to the cost.
        
        Args:
            product: Dictionary with product information
            market_info: Dictionary with market information
            
        Returns:
            Dictionary with price recommendation and confidence
        """
        # Extract product information
        cost = product.get('cost', 60.0)
        price = product.get('price', 100.0)
        product_type = product.get('product_type', 'standard').lower()
        
        # Calculate cost ratio
        cost_ratio = cost / price
        
        # Base markup depends on product type
        if product_type in ['premium', 'luxury']:
            markup = 1.8  # 80% markup for premium products
        elif product_type in ['commodity', 'basic', 'basics']:
            markup = 1.3  # 30% markup for commodity products
        else:
            markup = 1.5  # 50% markup for standard products
        
        # Calculate price ratio
        price_ratio = (cost * markup) / price
        
        # Ensure price ratio is reasonable
        price_ratio = max(0.8, min(1.2, price_ratio))
        
        # Calculate confidence
        confidence = self.strategy_confidence['cost_plus']
        
        return {
            'price_ratio': price_ratio,
            'confidence': confidence,
            'strategy': 'cost_plus'
        }
    
    def _value_based_pricing(self, product, market_info):
        """
        Value-based pricing strategy that prices based on perceived value.
        
        Args:
            product: Dictionary with product information
            market_info: Dictionary with market information
            
        Returns:
            Dictionary with price recommendation and confidence
        """
        # Extract product information
        rating = product.get('rating', 3.0)
        elasticity = product.get('elasticity', 1.0)
        product_type = product.get('product_type', 'standard').lower()
        
        # Base value ratio
        base_ratio = 1.0
        
        # Adjust for product quality/rating
        quality_premium = (rating - 3.0) * 0.1  # 10% adjustment per rating point from baseline
        base_ratio += quality_premium
        
        # Adjust for product type
        if product_type in ['premium', 'luxury']:
            base_ratio += 0.1  # 10% premium for premium products
        elif product_type in ['commodity', 'basic', 'basics']:
            base_ratio -= 0.1  # 10% discount for commodity products
        
        # Adjust for elasticity - inelastic products can command higher value prices
        if elasticity < 0.8:
            base_ratio += 0.05  # 5% premium for inelastic products
        elif elasticity > 1.2:
            base_ratio -= 0.05  # 5% discount for elastic products
        
        # Ensure price ratio is reasonable
        base_ratio = max(0.8, min(1.3, base_ratio))
        
        # Calculate confidence - higher for products with strong ratings
        confidence = self.strategy_confidence['value_based']
        if rating >= 4.0:
            confidence *= 1.1  # Increase confidence for high-rated products
        
        return {
            'price_ratio': base_ratio,
            'confidence': min(1.0, confidence),
            'strategy': 'value_based'
        }
    
    def _price_skimming(self, product, market_info):
        """
        Price skimming strategy that starts with high prices.
        
        Args:
            product: Dictionary with product information
            market_info: Dictionary with market information
            
        Returns:
            Dictionary with price recommendation and confidence
        """
        # Extract product information
        rating = product.get('rating', 3.0)
        elasticity = product.get('elasticity', 1.0)
        product_type = product.get('product_type', 'standard').lower()
        cost = product.get('cost', 60.0)
        price = product.get('price', 100.0)
        
        # Base skimming ratio
        base_ratio = 1.1  # 10% premium as default
        
        # Adjust for product quality/rating
        quality_premium = (rating - 3.0) * 0.1  # 10% adjustment per rating point from baseline
        base_ratio += quality_premium
        
        # Adjust for product type
        if product_type in ['premium', 'luxury']:
            base_ratio += 0.1  # 10% additional premium for premium products
        elif product_type in ['commodity', 'basic', 'basics']:
            base_ratio = 1.02  # Minimal skimming for commodity products
        
        # Adjust for elasticity - skimming works poorly with elastic products
        if elasticity > 1.2:
            base_ratio = min(base_ratio, 1.05)  # Cap premium for elastic products
        elif elasticity < 0.8:
            base_ratio += 0.05  # Additional premium for inelastic products
        
        # Ensure minimum profitability
        cost_ratio = cost / price
        min_price_ratio = cost_ratio * 1.3  # Ensure at least 30% margin over cost
        base_ratio = max(min_price_ratio, base_ratio)
        
        # Ensure price ratio is reasonable
        base_ratio = max(1.02, min(1.4, base_ratio))
        
        # Calculate confidence
        confidence = self.strategy_confidence['skimming']
        if elasticity < 0.8 and rating > 4.0:
            confidence *= 1.2  # Increase confidence for ideal skimming conditions
        elif elasticity > 1.2:
            confidence *= 0.7  # Decrease confidence for elastic products
        
        return {
            'price_ratio': base_ratio,
            'confidence': min(1.0, confidence),
            'strategy': 'skimming'
        }
    
    def _premium_pricing(self, product, market_info):
        """
        Premium pricing strategy for high-quality products.
        
        Args:
            product: Dictionary with product information
            market_info: Dictionary with market information
            
        Returns:
            Dictionary with price recommendation and confidence
        """
        # Extract product information
        rating = product.get('rating', 3.0)
        elasticity = product.get('elasticity', 1.0)
        product_type = product.get('product_type', 'standard').lower()
        
        # Base premium ratio
        base_ratio = 1.05  # 5% premium as default
        
        # Adjust for product quality/rating - strong focus on quality
        quality_premium = (rating - 3.0) * 0.15  # 15% adjustment per rating point from baseline
        base_ratio += quality_premium
        
        # Adjust for product type - premium strategy works best with luxury products
        if product_type in ['premium', 'luxury']:
            base_ratio += 0.15  # 15% additional premium for premium products
        elif product_type in ['commodity', 'basic', 'basics']:
            base_ratio = 1.0  # No premium for commodity products
        
        # Adjust for elasticity - premium pricing works poorly with elastic products
        if elasticity > 1.2:
            base_ratio = min(base_ratio, 1.05)  # Cap premium for elastic products
        elif elasticity < 0.8:
            base_ratio += 0.1  # Additional premium for inelastic products
        
        # Ensure price ratio is reasonable
        base_ratio = max(1.0, min(1.4, base_ratio))
        
        # Calculate confidence
        confidence = self.strategy_confidence['premium']
        if product_type in ['premium', 'luxury'] and rating > 4.0:
            confidence *= 1.3  # Increase confidence for ideal premium conditions
        elif product_type in ['commodity', 'basic', 'basics']:
            confidence *= 0.5  # Decrease confidence for commodity products
        
        return {
            'price_ratio': base_ratio,
            'confidence': min(1.0, confidence),
            'strategy': 'premium'
        }
    
    def _elasticity_based_pricing(self, product, market_info):
        """
        Elasticity-based pricing that optimizes price based on demand elasticity.
        
        Args:
            product: Dictionary with product information
            market_info: Dictionary with market information
            
        Returns:
            Dictionary with price recommendation and confidence
        """
        # Extract product information
        elasticity = product.get('elasticity', 1.0)
        cost = product.get('cost', 60.0)
        price = product.get('price', 100.0)
        
        # Calculate cost ratio
        cost_ratio = cost / price
        
        # Calculate optimal price ratio based on elasticity theory
        # For profit maximization: price_ratio = cost_ratio / (1 - 1/elasticity)
        if elasticity != 1.0:
            optimal_ratio = cost_ratio / (1.0 - 1.0/elasticity)
        else:
            optimal_ratio = 1.0  # Unit elasticity - default to market price
        
        # Find optimal price point for elastic products with sweet spot
        # For elastic products, lower prices can significantly increase demand
        if elasticity > 1.0:
            # More aggressive pricing for elastic products - target 5-15% below market
            sweet_spot = 0.9 - (elasticity - 1.0) * 0.05  # Up to 15% below market for highly elastic
            optimal_ratio = min(optimal_ratio, sweet_spot)
        
        # Ensure price ratio is reasonable and profitable
        min_price_ratio = cost_ratio * 1.2  # Ensure at least 20% margin over cost
        optimal_ratio = max(min_price_ratio, optimal_ratio)
        optimal_ratio = max(0.75, min(1.3, optimal_ratio))
        
        # Calculate confidence - higher for products with clear elasticity data
        confidence = self.strategy_confidence['dynamic_elasticity']
        
        return {
            'price_ratio': optimal_ratio,
            'confidence': confidence,
            'strategy': 'dynamic_elasticity'
        }
    
    def _psychological_pricing(self, product, market_info):
        """
        Psychological pricing strategy using price points that appeal to consumers.
        
        Args:
            product: Dictionary with product information
            market_info: Dictionary with market information
            
        Returns:
            Dictionary with price recommendation and confidence
        """
        # Extract product information
        price = product.get('price', 100.0)
        
        # Start with market price ratio
        base_ratio = 1.0
        
        # Calculate psychological price
        psych_price = self._calculate_psychological_price(price * base_ratio)
        
        # Recalculate ratio based on psychological price
        psych_ratio = psych_price / price
        
        # Ensure price ratio is reasonable
        psych_ratio = max(0.8, min(1.2, psych_ratio))
        
        return {
            'price_ratio': psych_ratio,
            'confidence': self.strategy_confidence['psychological'],
            'strategy': 'psychological'
        }
    
    def _calculate_psychological_price(self, price):
        """
        Calculate a psychological price point (e.g., $9.99 instead of $10.00)
        
        Args:
            price: The calculated price
            
        Returns:
            Adjusted price with psychological pricing applied
        """
        # Round to nearest .99 or .95 ending
        if price >= 100:
            # For higher prices, use .99 ending
            base_price = math.floor(price)
            return base_price - 0.01
        elif price >= 10:
            # For medium prices, use .99 ending
            base_price = math.floor(price)
            return base_price - 0.01
        else:
            # For lower prices, use .95 ending
            base_price = math.floor(price)
            return base_price - 0.05
    
    def _bundle_pricing(self, product, market_info):
        """
        Bundle pricing strategy (dummy implementation).
        
        Args:
            product: Dictionary with product information
            market_info: Dictionary with market information
            
        Returns:
            Dictionary with price recommendation and confidence
        """
        # This is a simplified placeholder - bundle pricing requires specific bundle data
        return {
            'price_ratio': 0.95,  # 5% discount for bundles
            'confidence': self.strategy_confidence['bundle'],
            'strategy': 'bundle'
        }
    
    def _promotional_pricing(self, product, market_info):
        """
        Promotional pricing strategy (dummy implementation).
        
        Args:
            product: Dictionary with product information
            market_info: Dictionary with market information
            
        Returns:
            Dictionary with price recommendation and confidence
        """
        # This is a simplified placeholder - promotional pricing is time-limited
        return {
            'price_ratio': 0.9,  # 10% promotional discount
            'confidence': self.strategy_confidence['promotional'],
            'strategy': 'promotional'
        }


class CustomerSegmentation:
    """
    Customer Segmentation module for dynamic pricing model
    
    This module segments customers based on price sensitivity, product preferences,
    and other attributes to calculate segment-specific conversion rates and optimize pricing.
    """
    
    def __init__(self, segment_count=4):
        """
        Initialize the customer segmentation module
        
        Args:
            segment_count: Number of segments to use (default: 4)
        """
        self.segment_count = segment_count
        self.segments = self._create_default_segments(segment_count)
        self.current_segment_distribution = None
        
        # Add time of day and weekday/weekend factors
        self.time_factors = {
            'weekday': {
                'morning': 0.9,
                'afternoon': 1.0,
                'evening': 1.1,
                'night': 0.8
            },
            'weekend': {
                'morning': 1.0,
                'afternoon': 1.2,
                'evening': 1.3,
                'night': 0.9
            }
        }
    
    def _create_default_segments(self, count):
        """Create default customer segments with different price sensitivities"""
        if count == 4:
            return {
                'price_sensitive': {
                    'weight': 0.30,  # 30% of customers
                    'price_sensitivity': 2.5,  # Highly sensitive to price (increased from 2.0)
                    'quality_importance': 0.6,  # Care somewhat about quality
                    'brand_loyalty': 0.3,  # Low brand loyalty
                    'conversion_base': 0.08,  # Base conversion rate
                    'max_premium': 0.05,  # Maximum premium they'll pay (5%)
                    'profit_per_conversion': 0.7  # Lower profit per conversion (70% of reference)
                },
                'value_seekers': {
                    'weight': 0.40,  # 40% of customers
                    'price_sensitivity': 1.5,  # Moderately sensitive to price
                    'quality_importance': 1.0,  # Balance price and quality
                    'brand_loyalty': 0.6,  # Moderate brand loyalty
                    'conversion_base': 0.12,  # Base conversion rate
                    'max_premium': 0.15,  # Maximum premium they'll pay (15%)
                    'profit_per_conversion': 1.0  # Standard profit per conversion
                },
                'quality_focused': {
                    'weight': 0.20,  # 20% of customers
                    'price_sensitivity': 0.7,  # Less sensitive to price (decreased from 0.8)
                    'quality_importance': 1.8,  # Highly value quality
                    'brand_loyalty': 0.8,  # High brand loyalty
                    'conversion_base': 0.15,  # Base conversion rate
                    'max_premium': 0.35,  # Maximum premium they'll pay (increased from 30%)
                    'profit_per_conversion': 1.5  # Higher profit per conversion (150% of reference)
                },
                'premium_buyers': {
                    'weight': 0.10,  # 10% of customers
                    'price_sensitivity': 0.3,  # Very low price sensitivity (decreased from 0.4)
                    'quality_importance': 2.0,  # Extremely quality focused
                    'brand_loyalty': 0.9,  # Very high brand loyalty
                    'conversion_base': 0.12,  # Base conversion rate (increased from 0.10)
                    'max_premium': 0.60,  # Maximum premium they'll pay (increased from 50%)
                    'profit_per_conversion': 2.0  # Much higher profit per conversion (200% of reference)
                }
            }
        else:
            # Simplified model with fewer segments
            return {
                'budget': {
                    'weight': 0.5,
                    'price_sensitivity': 2.0,
                    'conversion_base': 0.10,
                    'max_premium': 0.10,
                    'profit_per_conversion': 0.8
                },
                'premium': {
                    'weight': 0.5,
                    'price_sensitivity': 0.6,  # Decreased from 0.7
                    'conversion_base': 0.15,
                    'max_premium': 0.45,  # Increased from 0.40
                    'profit_per_conversion': 1.7  # Added profit factor
                }
            }
    
    def get_segment_distribution(self, product_type, product_group, rating, price_point=None):
        """
        Get the distribution of customer segments for a specific product
        
        Args:
            product_type: Type of product (e.g., premium, basic)
            product_group: Product group/category
            rating: Product rating (1-5)
            price_point: Current price point (optional)
            
        Returns:
            Dictionary with adjusted segment weights
        """
        # Copy default segments
        adjusted_segments = {k: v.copy() for k, v in self.segments.items()}
        
        # Adjust based on product type
        if product_type == 'luxury' or product_type == 'premium':
            adjusted_segments['price_sensitive']['weight'] *= 0.6
            adjusted_segments['premium_buyers']['weight'] *= 1.8
            adjusted_segments['quality_focused']['weight'] *= 1.4
        elif product_type == 'basics' or product_type == 'commodity':
            adjusted_segments['price_sensitive']['weight'] *= 1.5
            adjusted_segments['premium_buyers']['weight'] *= 0.5
        elif product_type == 'new':
            # New products attract more early adopters (quality-focused and premium)
            adjusted_segments['quality_focused']['weight'] *= 1.3
            adjusted_segments['premium_buyers']['weight'] *= 1.2
        
        # Adjust for product rating
        if rating >= 4.5:
            # High-rated products attract quality-focused and premium buyers
            adjusted_segments['quality_focused']['weight'] *= 1.2
            adjusted_segments['premium_buyers']['weight'] *= 1.3
            adjusted_segments['price_sensitive']['weight'] *= 0.8
        elif rating <= 3.0:
            # Low-rated products mostly attract price-sensitive buyers
            adjusted_segments['price_sensitive']['weight'] *= 1.4
            adjusted_segments['premium_buyers']['weight'] *= 0.5
        
        # Normalize weights to sum to 1.0
        total_weight = sum(segment['weight'] for segment in adjusted_segments.values())
        for segment in adjusted_segments.values():
            segment['weight'] /= total_weight
        
        # Store current distribution
        self.current_segment_distribution = adjusted_segments
        return adjusted_segments
    
    def get_time_factor(self, is_weekend=False, time_of_day='afternoon'):
        """Get time-based adjustment factors for pricing"""
        day_type = 'weekend' if is_weekend else 'weekday'
        return self.time_factors[day_type].get(time_of_day, 1.0)
    
    def calculate_segment_conversion_probabilities(self, price_ratio, product):
        """
        Calculate conversion probabilities for each customer segment
        with enhanced focus on market-beating performance.
        
        Args:
            price_ratio: Current price ratio
            product: Product information dictionary
            
        Returns:
            Dictionary with segment conversion probabilities and weighted average
        """
        if self.current_segment_distribution is None:
            self.get_segment_distribution(
                product.get('product_type', 'standard'),
                product.get('product_group', 'general'),
                product.get('rating', 3.0)
            )
        
        rating = product.get('rating', 3.0)
        elasticity = product.get('elasticity', 1.0)
        
        result = {
            'segments': {},
            'weighted_conversion': 0.0,
            'expected_profit_multiplier': 0.0  # Track expected profit multiplier
        }
        
        total_weighted_conversion = 0.0
        total_profit_weighted_conversion = 0.0
        
        for segment_name, segment in self.current_segment_distribution.items():
            # Calculate price effect based on segment price sensitivity
            price_effect = 1.0 - (price_ratio - 1.0) * segment['price_sensitivity']
            
            # Calculate quality effect based on rating and segment quality importance
            quality_importance = segment.get('quality_importance', 1.0)
            quality_effect = 0.7 + (rating / 5.0) * quality_importance * 0.6
            
            # Calculate brand/loyalty effect
            brand_loyalty = segment.get('brand_loyalty', 0.5)
            if brand_loyalty > 0:
                # Loyal customers are less affected by price increases
                loyalty_dampening = brand_loyalty * 0.5
                price_effect = price_effect * (1 - loyalty_dampening) + loyalty_dampening
            
            # Calculate base conversion rate for this segment
            conversion_base = segment.get('conversion_base', 0.1)
            
            # Maximum premium this segment will tolerate
            max_premium = segment.get('max_premium', 0.2)
            if price_ratio > (1.0 + max_premium):
                # Sharp drop in conversion if price exceeds max premium
                price_effect *= 0.3  # More significant drop (was 0.5)
            
            # Enhanced premium segment response to quality
            if segment_name in ['premium_buyers', 'quality_focused'] and rating >= 4.0:
                # Premium buyers respond more strongly to high-quality products
                quality_effect *= 1.2  # 20% boost for high-quality products
                
                # Premium buyers prefer prices that signal quality
                if 1.0 < price_ratio <= (1.0 + max_premium * 0.6):
                    # Boost conversion for moderately premium prices on high-quality items
                    price_effect *= 1.15  # 15% boost for premium pricing that signals quality
            
            # Enhanced value segment response to good deals
            if segment_name in ['price_sensitive', 'value_seekers'] and price_ratio < 1.0:
                # Value-conscious buyers respond strongly to below-market prices
                discount_depth = 1.0 - price_ratio
                
                # Boost conversion for good deals, with diminishing returns for extreme discounts
                if discount_depth <= 0.15:  # Moderate discount (up to 15%)
                    price_effect *= (1.0 + discount_depth * 3.0)  # Up to 45% boost for 15% discount
                else:
                    # Maximum boost at 15% discount, then diminishing returns
                    price_effect *= 1.45  # 45% boost maximum
            
            # Calculate final conversion probability for this segment
            segment_conversion = conversion_base * price_effect * quality_effect
            
            # Ensure probability is in valid range
            segment_conversion = max(0.01, min(0.99, segment_conversion))
            
            # Get profit per conversion multiplier for this segment
            profit_multiplier = segment.get('profit_per_conversion', 1.0)
            
            # Store segment conversion probability
            result['segments'][segment_name] = segment_conversion
            
            # Add to weighted average calculations
            total_weighted_conversion += segment_conversion * segment['weight']
            total_profit_weighted_conversion += segment_conversion * segment['weight'] * profit_multiplier
        
        # Calculate overall conversion probability
        result['weighted_conversion'] = total_weighted_conversion
        
        # Calculate expected profit multiplier
        if total_weighted_conversion > 0:
            result['expected_profit_multiplier'] = total_profit_weighted_conversion / total_weighted_conversion
        else:
            result['expected_profit_multiplier'] = 1.0
        
        return result
    
    def reset(self):
        """Reset the customer segmentation state"""
        # Reset the current segment distribution
        self.current_segment_distribution = None
        # Reset segments to defaults
        self.segments = self._create_default_segments(self.segment_count)
    
    def get_segment_conversion_rates(self, product, price_ratio, base_conversion_prob=None):
        """
        Get conversion rates for different customer segments
        
        Args:
            product: Product information dictionary
            price_ratio: Current price ratio
            base_conversion_prob: Base conversion probability (optional)
            
        Returns:
            Dictionary with segment conversion rates and profit multiplier
        """
        # Calculate segment conversion probabilities
        conversion_data = self.calculate_segment_conversion_probabilities(price_ratio, product)
        
        # Extract segment data
        segment_conversions = conversion_data['segments']
        
        # Map to simplified segments for the environment
        simplified_segments = {
            'high_value': 0.0,
            'mid_value': 0.0,
            'low_value': 0.0,
            'overall': conversion_data['weighted_conversion'],
            'profit_multiplier': conversion_data['expected_profit_multiplier']  # Add profit multiplier
        }
        
        # Map detailed segments to simplified ones
        if 'premium_buyers' in segment_conversions:
            simplified_segments['high_value'] = segment_conversions['premium_buyers']
        
        if 'quality_focused' in segment_conversions:
            # Split quality focused between high and mid value
            simplified_segments['high_value'] = (simplified_segments['high_value'] + 
                                               segment_conversions['quality_focused']) / 2
            simplified_segments['mid_value'] = segment_conversions['quality_focused']
        
        if 'value_seekers' in segment_conversions:
            simplified_segments['mid_value'] = (simplified_segments['mid_value'] + 
                                              segment_conversions['value_seekers']) / 2
        
        if 'price_sensitive' in segment_conversions:
            simplified_segments['low_value'] = segment_conversions['price_sensitive']
        
        # If base conversion probability is provided, scale all values
        if base_conversion_prob is not None:
            scale_factor = base_conversion_prob / simplified_segments['overall']
            for key in simplified_segments:
                if key != 'profit_multiplier':  # Don't scale the profit multiplier
                    simplified_segments[key] *= scale_factor
        
        return simplified_segments
    
    def get_demand_modifier(self, product, price_ratio, margin):
        """
        Get a demand modifier based on customer segments and pricing strategy.
        This helps the market environment adjust demand based on segmentation.
        
        Args:
            product: Product information dictionary
            price_ratio: Current price ratio
            margin: Profit margin
            
        Returns:
            float: Demand modifier (multiplier)
        """
        # Default modifier
        modifier = 1.0
        
        # Get product attributes
        rating = product.get('rating', 3.0)
        product_type = product.get('product_type', 'standard')
        elasticity = product.get('elasticity', 1.0)
        num_orders = product.get('number_of_orders', 0)
        
        # Apply order volume impact on demand
        # Products with more orders have higher demand (popularity effect)
        if num_orders > 200:
            # Highly popular products get a significant boost
            order_boost = 0.25  # 25% boost for very popular products
        elif num_orders > 100:
            # Popular products get a moderate boost
            order_boost = 0.15  # 15% boost for popular products
        elif num_orders > 50:
            # Moderately popular products get a small boost
            order_boost = 0.08  # 8% boost for moderately popular products
        elif num_orders > 20:
            # Somewhat popular products get a tiny boost
            order_boost = 0.04  # 4% boost for somewhat popular products
        else:
            # New or unpopular products get a small penalty
            order_boost = -0.05  # 5% reduction for unpopular products
        
        # Apply the order volume boost/penalty
        modifier += order_boost
        
        # Enhanced modifier for market-beating pricing (below market)
        if price_ratio < 1.0:
            # Below-market pricing gets a demand boost
            discount_depth = 1.0 - price_ratio
            
            # Higher elasticity means stronger response to discounts
            elasticity_factor = min(1.6, elasticity * 1.2)  # Cap at 1.6
            
            # Boost is stronger for commodity products and price-elastic products
            if product_type.lower() in ['commodity', 'basics', 'basic']:
                modifier += discount_depth * 1.8 * elasticity_factor  # Up to 80% boost for deep discounts
            elif product_type.lower() in ['electronics', 'clothing', 'books', 'sports']:
                modifier += discount_depth * 1.6 * elasticity_factor  # Up to 60% boost for deep discounts
            else:
                modifier += discount_depth * 1.4 * elasticity_factor  # Up to 40% boost for deep discounts
                
            # Special sweet spot for optimal pricing - stronger boost for smaller discounts
            # This encourages finding the optimal price point slightly below market
            if 0.05 <= discount_depth <= 0.12:
                modifier *= 1.15  # Additional 15% boost for the sweet spot of 5-12% below market
            
            # But ensure profitability - reduce boost for low margins
            if margin < 0.15:
                margin_factor = max(0.5, margin / 0.15)  # Scales from 0.5 to 1.0 as margin approaches 15%
                modifier *= margin_factor  # Reduce boost for unprofitable pricing
        elif price_ratio > 1.0:
            # Above-market pricing reduces demand
            premium = price_ratio - 1.0
            
            # Reduction is less severe for premium/luxury products with good ratings
            if product_type.lower() in ['premium', 'luxury'] and rating >= 4.0:
                modifier -= premium * 0.7  # Less reduction for premium products
            elif product_type.lower() in ['premium', 'luxury'] and rating >= 3.5:
                modifier -= premium * 0.8  # Moderate reduction for good premium products
            else:
                modifier -= premium * 1.3  # Stronger reduction for non-premium products
                
            # Special handling for product types based on elasticity
            if elasticity > 1.2:  # Highly elastic products
                modifier -= premium * 0.4  # Additional reduction for elastic products
        
        # Apply rating-based adjustments
        if rating >= 4.5:
            modifier *= 1.1  # 10% boost for excellent products
        elif rating <= 2.5:
            modifier *= 0.85  # 15% reduction for poor products
        
        # Ensure modifier is reasonable but with expanded upper limit
        return max(0.4, min(1.8, modifier)) 