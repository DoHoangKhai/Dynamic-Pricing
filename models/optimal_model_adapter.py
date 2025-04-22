"""
FinalOptimalModel Adapter Module

This module adapts the high-performing FinalOptimalModel from the evaluation framework
to work with the existing pricing pipeline in the web application.
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FinalOptimalModel")

# Try to import from benchmark_models in the current directory first
try:
    from .benchmark_models.pricing_models import ProfitOptimizedModel
    logger.info("Successfully imported ProfitOptimizedModel from benchmark_models.pricing_models")
except ImportError as e:
    logger.warning(f"Could not import ProfitOptimizedModel from benchmark_models.pricing_models: {e}")
    try:
        # Try to import directly from the module
        from .benchmark_models import ProfitOptimizedModel
        logger.info("Successfully imported ProfitOptimizedModel from benchmark_models")
    except ImportError as e:
        logger.warning(f"Could not import ProfitOptimizedModel from benchmark_models: {e}")
        try:
            # Try to import from the root models directory
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
            from models.benchmark_models import ProfitOptimizedModel
            logger.info("Using ProfitOptimizedModel from root models directory")
        except ImportError as e:
            logger.warning(f"Could not import ProfitOptimizedModel: {e}")
            # Fallback: Define a stub version that matches the interface
            class ProfitOptimizedModel:
                """Stub implementation with equivalent interface"""
                def __init__(self, base_markup_pct=60, risk_threshold=0.8, 
                            elasticity_weight=0.6, competitor_weight=0.2, 
                            rating_factor=0.15, volume_factor=0.1):
                    self.base_markup_pct = base_markup_pct
                    self.risk_threshold = risk_threshold
                    self.elasticity_weight = elasticity_weight
                    self.competitor_weight = competitor_weight
                    self.rating_factor = rating_factor
                    self.volume_factor = volume_factor
                    logger.warning("Using FALLBACK ProfitOptimizedModel stub implementation")
                
                def predict(self, X):
                    """Simple predict method using cost-plus pricing"""
                    costs = X['cost'].values if isinstance(X, pd.DataFrame) else X
                    return costs * (1 + (self.base_markup_pct / 100))
                
                def fit(self, X, y=None):
                    return self

class FinalOptimalModel:
    """
    Final implementation of the Optimal Model for dynamic pricing.
    This model combines multiple pricing factors including:
    - Cost-plus pricing as a baseline
    - Rating-based adjustments
    - Competition-aware pricing
    - Elasticity-informed pricing
    - Volume-based pricing
    
    The model calculates a price recommendation based on these factors
    and applies constraints to ensure prices remain within reasonable bounds.
    """
    
    def __init__(self, 
                 base_markup_pct=95,  # Increased from 90 to 95 based on model evaluation
                 risk_threshold=0.65,  # Decreased from 0.70 to 0.65 for more aggressive pricing
                 elasticity_weight=0.8,  # Increased from 0.60 to 0.80 to better utilize elasticity
                 competitor_weight=0.25,  # Increased from 0.20 to 0.25 for competitive responsiveness
                 rating_factor=0.20,  # Increased from 0.15 to 0.20 for better quality adjustment
                 volume_factor=0.15):  # Increased from 0.00 to 0.15 to utilize volume data
        """
        Initialize the Final Optimal Model with model parameters.
        
        Args:
            base_markup_pct: Base markup percentage (default: 95%)
            risk_threshold: Threshold for risk-adjusted pricing (default: 0.65)
            elasticity_weight: Weight for elasticity in pricing (default: 0.8)
            competitor_weight: Weight for competitor pricing (default: 0.25)
            rating_factor: Impact of product rating (default: 0.20)
            volume_factor: Impact of order volume (default: 0.15)
        """
        self.base_markup_pct = base_markup_pct
        self.risk_threshold = risk_threshold
        self.elasticity_weight = elasticity_weight
        self.competitor_weight = competitor_weight
        self.rating_factor = rating_factor
        self.volume_factor = volume_factor
        
        logger.info(f"Initialized FinalOptimalModel with parameters: base_markup={base_markup_pct}, "
                   f"risk_threshold={risk_threshold}, elasticity_weight={elasticity_weight}, "
                   f"competitor_weight={competitor_weight}, rating_factor={rating_factor}, "
                   f"volume_factor={volume_factor}")
    
    def get_price_recommendations(self, product_data, market_data):
        """
        Generate price recommendations based on product and market data.
        
        Args:
            product_data: Dictionary with product data (price, cost, rating, orders)
            market_data: Dictionary with market data (competitor_price, competitive_intensity)
            
        Returns:
            Dictionary with price recommendation and additional information
        """
        # Extract product data
        current_price = product_data.get('price', 0)
        cost = product_data.get('cost', 0)
        rating = product_data.get('rating', 3.0)
        order_volume = product_data.get('number_of_orders', 0)
        product_type = product_data.get('product_type', 'standard')
        
        # Extract market data
        competitor_price = market_data.get('competitor_price', current_price * 1.1)
        competitive_intensity = market_data.get('competitive_intensity', 0.5)
        
        logger.info(f"Processing product: price={current_price}, cost={cost}, rating={rating}, "
                   f"volume={order_volume}, competitor_price={competitor_price}, "
                   f"competitive_intensity={competitive_intensity}")
        
        # Ensure valid cost data
        if cost <= 0:
            logger.warning(f"Invalid cost: {cost}. Using default margin estimate.")
            cost = current_price * 0.6  # Assume 40% margin if cost is invalid
        
        # Ensure valid current price
        if current_price <= 0:
            logger.warning(f"Invalid current price: {current_price}. Using cost-plus price.")
            current_price = cost * 1.5  # Default to 50% markup if current price is invalid
        
        # Calculate base price using cost-plus method with enhanced markup
        base_price = cost * (1 + (self.base_markup_pct / 100))
        logger.info(f"Base cost-plus price calculated: {base_price}")
        
        # Calculate elasticity-based price (if elasticity is provided)
        elasticity = product_data.get('elasticity', None)
        elasticity_price = None
        
        if elasticity:
            try:
                # Calculate optimal price based on elasticity
                # For elastic products (elasticity < -1), we reduce price
                # For inelastic products (elasticity > -1), we can increase price
                elasticity_adjustment = 1.0 + (0.1 * (-1 - elasticity))
                elasticity_price = current_price * elasticity_adjustment
                
                # Ensure price doesn't go below cost with reasonable margin
                min_elasticity_price = cost * 1.15  # At least 15% margin
                elasticity_price = max(elasticity_price, min_elasticity_price)
                
                logger.info(f"Elasticity-based price calculated: {elasticity_price}")
            except Exception as e:
                logger.warning(f"Error in elasticity calculation: {e}")
                elasticity_price = None
        
        # Handle case where elasticity price couldn't be calculated
        if elasticity_price is None:
            # Instead of falling back to base price, use a dynamic approach based on market positioning
            if competitor_price > current_price * 1.2:
                # Competitor price is much higher - opportunity to increase
                elasticity_price = current_price * 1.1
            elif competitor_price < current_price * 0.8:
                # Competitor price is much lower - need to adjust down
                elasticity_price = max(current_price * 0.9, cost * 1.15)
            else:
                # Competitor price is similar - use balanced approach
                elasticity_price = (current_price + (competitor_price * 0.8)) / 2
            
            logger.info(f"Calculated fallback elasticity price: {elasticity_price}")
        
        # Calculate initial price based on weighted average of base price and elasticity price
        if elasticity_price:
            initial_price = (base_price * (1 - self.elasticity_weight)) + (elasticity_price * self.elasticity_weight)
        else:
            initial_price = base_price
        
        logger.info(f"Initial price after elasticity weighting: {initial_price}")
        
        # Apply rating factor (product quality adjustment)
        rating_factor = self._calculate_rating_factor(rating)
        rating_adjusted_price = initial_price * rating_factor
        logger.info(f"Rating factor: {rating_factor}, adjusted price: {rating_adjusted_price}")
        
        # Apply volume factor (new implementation)
        volume_factor = self._calculate_volume_factor(order_volume)
        volume_adjusted_price = rating_adjusted_price * volume_factor
        logger.info(f"Volume factor: {volume_factor}, adjusted price: {volume_adjusted_price}")
        
        # Apply competitor price adjustment
        competitor_factor = self._calculate_competitor_factor(competitor_price, volume_adjusted_price)
        competitor_adjusted_price = volume_adjusted_price * competitor_factor
        logger.info(f"Competitor factor: {competitor_factor}, adjusted price: {competitor_adjusted_price}")
        
        # Apply market factor based on competitive intensity
        market_factor = self._calculate_market_factor(competitive_intensity)
        market_adjusted_price = competitor_adjusted_price * market_factor
        logger.info(f"Market factor: {market_factor}, adjusted price: {market_adjusted_price}")
        
        # Apply elasticity-based constraints on price change
        max_price_change_pct = 0.15  # Default max price change
        
        if elasticity:
            # Adjust max price change based on elasticity
            # More elastic products can have larger price changes
            abs_elasticity = abs(elasticity)
            if abs_elasticity > 1.5:
                max_price_change_pct = 0.25  # Allow 25% change for highly elastic products
            elif abs_elasticity > 1.0:
                max_price_change_pct = 0.20  # Allow 20% change for moderately elastic products
        
        # Apply rating-based adjustments to max price change
        if rating >= 4.5:
            max_price_change_pct += 0.05  # Allow additional 5% change for high-rated products
        
        # Apply competitive position adjustments to max price change
        comp_ratio = competitor_price / current_price if current_price > 0 else 1.0
        if comp_ratio > 1.2:  # Competitor price is much higher
            max_price_change_pct += 0.05  # Allow additional 5% change
        elif comp_ratio < 0.8:  # Competitor price is much lower
            max_price_change_pct += 0.05  # Allow additional 5% change
        
        # Apply competitive intensity adjustment to max price change
        if competitive_intensity > 0.7:
            max_price_change_pct += 0.05  # Allow additional 5% change in highly competitive markets
        
        # Calculate price constraints
        min_price = max(current_price * (1 - max_price_change_pct), cost * 1.15)  # Minimum 15% margin
        max_price = min(current_price * (1 + max_price_change_pct), competitor_price * 1.1)  # Maximum 10% above competitor
        
        # For high-rated products, adjust the ceiling constraint relative to competitor
        if rating >= 4.5:
            max_price = min(current_price * (1 + max_price_change_pct), competitor_price * 1.15)  # 15% above competitor for high-rated products
        
        logger.info(f"Calculated constraints: min_price={min_price}, max_price={max_price}")
        
        # Apply constraints to the final price
        final_price = max(min(market_adjusted_price, max_price), min_price)
        
        # If the final price is within 1% of min_price and it's a decrease, 
        # reconsider the constraints based on competitor positioning
        if abs(final_price - min_price) / min_price < 0.01 and final_price < current_price:
            comp_position = (competitor_price - current_price) / current_price
            
            # If competitor is priced significantly higher, don't enforce the floor price
            if comp_position > 0.1:  # Competitor price is >10% higher
                revised_min_price = max(cost * 1.15, current_price * (1 - (max_price_change_pct + 0.05)))
                final_price = max(market_adjusted_price, revised_min_price)
                logger.info(f"Relaxed min price constraint due to higher competitor pricing: {revised_min_price}")
        
        logger.info(f"Final recommended price: {final_price}")
        
        # Calculate segment impact
        segment_impact = self.calculate_segment_impact(final_price, current_price, product_data)
        
        # Calculate profit impact
        current_margin = current_price - cost
        new_margin = final_price - cost
        margin_change_pct = ((new_margin / current_margin) - 1) * 100 if current_margin > 0 else 0
        
        # Calculate price change
        price_change = final_price - current_price
        price_change_pct = (price_change / current_price) * 100 if current_price > 0 else 0
        
        # Round final price to two decimal places
        final_price = round(final_price, 2)
        
        # Return recommendation with detailed information
        recommendation = {
            'recommended_price': final_price,
            'original_price': current_price,
            'cost': cost,
            'price_change': round(price_change, 2),
            'price_change_pct': round(price_change_pct, 2),
            'margin_change_pct': round(margin_change_pct, 2),
            'segment_impact': segment_impact,
            'price_factors': {
                'base_price': round(base_price, 2),
                'elasticity_price': round(elasticity_price, 2) if elasticity_price else None,
                'rating_factor': round(rating_factor, 2),
                'volume_factor': round(volume_factor, 2),
                'competitor_factor': round(competitor_factor, 2),
                'market_factor': round(market_factor, 2)
            }
        }
        
        # Add timestamp
        recommendation['timestamp'] = datetime.now().isoformat()
        
        return recommendation
    
    def _calculate_rating_factor(self, rating):
        """
        Calculate the rating factor based on product rating.
        
        Args:
            rating: Product rating (1.0 to 5.0)
            
        Returns:
            Rating factor multiplier
        """
        # Default factor
        factor = 1.0
        
        # Adjust factor based on rating
        if rating is not None:
            if rating < 3.0:
                # Lower rated products get a discount
                factor = 0.95 - ((3.0 - max(1.0, rating)) * 0.025)
                logger.info(f"Low rating ({rating}) - applying discount factor: {factor}")
            elif rating >= 3.0 and rating < 4.0:
                # Average rated products get a slight discount
                factor = 0.97 + ((rating - 3.0) * 0.03)
                logger.info(f"Average rating ({rating}) - applying factor: {factor}")
            elif rating >= 4.0 and rating < 4.5:
                # Good rated products get a premium
                factor = 1.02 + ((rating - 4.0) * 0.04)
                logger.info(f"Good rating ({rating}) - applying premium factor: {factor}")
            elif rating >= 4.5:
                # Excellent rated products get a higher premium
                factor = 1.04 + ((rating - 4.5) * 0.08)
                logger.info(f"Excellent rating ({rating}) - applying premium factor: {factor}")
        
        # Ensure the factor is within reasonable bounds
        factor = max(0.9, min(factor, 1.15))
        
        return factor
    
    def _calculate_volume_factor(self, order_volume):
        """
        Calculate the volume factor based on order volume.
        Higher volumes generally allow for more aggressive pricing.
        
        Args:
            order_volume: Number of orders
            
        Returns:
            Volume factor multiplier
        """
        # Default factor
        factor = 1.0
        
        # No volume data available
        if order_volume is None or order_volume == 0:
            return factor
        
        # Apply volume-based factor
        if order_volume < 50:
            # Low volume products - be more conservative
            factor = 0.98
            logger.info(f"Low volume ({order_volume}) - applying conservative factor: {factor}")
        elif order_volume >= 50 and order_volume < 200:
            # Medium volume - slightly aggressive
            factor = 1.01
            logger.info(f"Medium volume ({order_volume}) - applying factor: {factor}")
        elif order_volume >= 200 and order_volume < 500:
            # High volume - more aggressive
            factor = 1.03
            logger.info(f"High volume ({order_volume}) - applying aggressive factor: {factor}")
        elif order_volume >= 500:
            # Very high volume - very aggressive
            factor = 1.05
            logger.info(f"Very high volume ({order_volume}) - applying very aggressive factor: {factor}")
        
        # Return the calculated factor
        return factor
    
    def _calculate_competitor_factor(self, competitor_price, current_price):
        """
        Calculate the competitor factor based on competitor pricing.
        
        Args:
            competitor_price: Competitor's price
            current_price: Current product price
            
        Returns:
            Competitor factor multiplier
        """
        # Default factor
        factor = 1.0
        
        # If competitor price is not available
        if competitor_price is None or competitor_price <= 0 or current_price <= 0:
            return factor
        
        # Calculate the ratio of competitor price to our price
        ratio = competitor_price / current_price
        
        # Apply competitor-based factor
        if ratio < 0.85:
            # Competitor is much cheaper - need to adjust down
            factor = 0.95
            logger.info(f"Competitor much cheaper (ratio: {ratio}) - applying factor: {factor}")
        elif ratio >= 0.85 and ratio < 0.95:
            # Competitor is cheaper - adjust down slightly
            factor = 0.98
            logger.info(f"Competitor cheaper (ratio: {ratio}) - applying factor: {factor}")
        elif ratio >= 0.95 and ratio < 1.05:
            # Competitor is similar - maintain price
            factor = 1.0
            logger.info(f"Competitor similar (ratio: {ratio}) - applying factor: {factor}")
        elif ratio >= 1.05 and ratio < 1.15:
            # Competitor is more expensive - adjust up slightly
            factor = 1.02
            logger.info(f"Competitor more expensive (ratio: {ratio}) - applying factor: {factor}")
        elif ratio >= 1.15:
            # Competitor is much more expensive - adjust up significantly
            factor = 1.05
            logger.info(f"Competitor much more expensive (ratio: {ratio}) - applying factor: {factor}")
        
        # Return the calculated factor
        return factor
    
    def _calculate_market_factor(self, competitive_intensity):
        """
        Calculate the market factor based on competitive intensity.
        
        Args:
            competitive_intensity: Market competitive intensity (0.0 to 1.0)
            
        Returns:
            Market factor multiplier
        """
        # Default factor
        factor = 1.0
        
        # If competitive intensity is not available
        if competitive_intensity is None:
            return factor
        
        # Apply market-based factor
        if competitive_intensity < 0.3:
            # Low competition - can increase prices
            factor = 1.03
            logger.info(f"Low competition ({competitive_intensity}) - applying factor: {factor}")
        elif competitive_intensity >= 0.3 and competitive_intensity < 0.6:
            # Moderate competition - maintain prices
            factor = 1.01
            logger.info(f"Moderate competition ({competitive_intensity}) - applying factor: {factor}")
        elif competitive_intensity >= 0.6 and competitive_intensity < 0.8:
            # High competition - need to be competitive
            factor = 0.99
            logger.info(f"High competition ({competitive_intensity}) - applying factor: {factor}")
        elif competitive_intensity >= 0.8:
            # Very high competition - need to be very competitive
            factor = 0.97
            logger.info(f"Very high competition ({competitive_intensity}) - applying factor: {factor}")
        
        # Return the calculated factor
        return factor
    
    def calculate_segment_impact(self, recommended_price, current_price, product_data):
        """
        Calculate the impact of price changes on different customer segments.
        
        Args:
            recommended_price: Recommended new price
            current_price: Current product price
            product_data: Dictionary with product data
            
        Returns:
            Dictionary with segment impact information
        """
        segments = {
            'price_sensitive': {
                'weight': 0.25,
                'elasticity': 1.8
            },
            'value_seekers': {
                'weight': 0.35,
                'elasticity': 1.2
            },
            'brand_focused': {
                'weight': 0.25,
                'elasticity': 0.8
            },
            'luxury': {
                'weight': 0.15,
                'elasticity': 0.5
            }
        }
        
        # Calculate price change
        price_change_pct = ((recommended_price / current_price) - 1) if current_price > 0 else 0
        
        # Calculate impact on each segment
        segment_impact = {}
        for segment_name, segment_data in segments.items():
            # Calculate demand change based on segment elasticity
            demand_change_pct = -1 * price_change_pct * segment_data['elasticity']
            
            # Calculate weighted impact
            weighted_impact = demand_change_pct * segment_data['weight']
            
            segment_impact[segment_name] = {
                'price_sensitivity': segment_data['elasticity'],
                'demand_change_pct': round(demand_change_pct * 100, 2),
                'weighted_impact': round(weighted_impact * 100, 2)
            }
        
        # Calculate total weighted impact
        total_impact = sum([data['weighted_impact'] for _, data in segment_impact.items()])
        
        return {
            'segments': segment_impact,
            'total_impact': round(total_impact, 2)
        }

class FinalOptimalModelAdapter:
    """
    Adapter for integrating the FinalOptimalModel with the pricing platform.
    This class provides the interface between the pricing model and the 
    application, handling data transformation and formatting.
    """
    
    def __init__(self):
        """Initialize the adapter with the pricing model."""
        self.model = FinalOptimalModel()
        logger.info("FinalOptimalModelAdapter initialized")
    
    def get_price_recommendations(self, product_data, market_data=None):
        """
        Get price recommendations for a product using the optimal model.
        
        Args:
            product_data: Dictionary with product data
            market_data: Dictionary with market data (optional)
            
        Returns:
            Dictionary with price recommendations and additional information
        """
        # Ensure market data exists
        if market_data is None:
            market_data = {}
        
        # Get recommendation from the model
        recommendation = self.model.get_price_recommendations(product_data, market_data)
        
        # Convert raw factors to percentage changes for UI display
        if 'price_factors' in recommendation:
            factors = recommendation['price_factors']
            recommendation['price_factors_pct'] = {
                'rating_impact': round((factors.get('rating_factor', 1.0) - 1.0) * 100, 1),
                'competitor_impact': round((factors.get('competitor_factor', 1.0) - 1.0) * 100, 1),
                'market_impact': round((factors.get('market_factor', 1.0) - 1.0) * 100, 1),
                'volume_impact': round((factors.get('volume_factor', 1.0) - 1.0) * 100, 1)
            }
        
        return recommendation
    
    def get_batch_recommendations(self, products_data, market_data=None):
        """
        Get batch price recommendations for multiple products.
        
        Args:
            products_data: List of dictionaries with product data
            market_data: Dictionary with market data (optional)
            
        Returns:
            List of dictionaries with price recommendations
        """
        # Ensure market data exists
        if market_data is None:
            market_data = {}
        
        recommendations = []
        for product_data in products_data:
            recommendation = self.get_price_recommendations(product_data, market_data)
            recommendations.append(recommendation)
        
        return recommendations
    
    def calculate_segment_impact(self, recommended_price, current_price, product_data):
        """
        Calculate the impact of price changes on different customer segments.
        
        Args:
            recommended_price: Recommended new price
            current_price: Current product price
            product_data: Dictionary with product data
            
        Returns:
            Dictionary with segment impact information
        """
        return self.model.calculate_segment_impact(recommended_price, current_price, product_data)
        
    def fit(self, X, y=None):
        """
        Fit the model to training data.
        
        Args:
            X: Training data
            y: Target values
            
        Returns:
            self
        """
        logger.info(f"Fitting FinalOptimalModel to {len(X)} data points")
        # Forward to underlying model if needed
        return self 