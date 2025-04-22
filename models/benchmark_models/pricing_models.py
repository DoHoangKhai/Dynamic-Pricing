"""
Benchmark pricing models for comparison with the RL pipeline.
"""

import numpy as np
import pandas as pd

class BasePricingModel:
    """Base class for pricing models"""
    
    def run_pipeline(self, product_data, market_info, historical_data=None):
        """
        Run pricing pipeline (adapter interface)
        
        Args:
            product_data: Dictionary with product information
            market_info: Dictionary with market information
            historical_data: Optional DataFrame with historical data
            
        Returns:
            Dictionary with pricing results
        """
        try:
            # Extract required data
            if isinstance(product_data, dict):
                price = product_data.get('price', 100)
                cost = product_data.get('cost', price * 0.6)
                elasticity = product_data.get('elasticity', 1.0)
            else:
                price = 100
                cost = 60
                elasticity = 1.0
                
            if isinstance(market_info, dict):
                competitor_price = market_info.get('competitor_price', price)
            else:
                competitor_price = price
            
            # Get price recommendation
            recommended_price = float(self.predict({'actual_price': price, 'cost': cost, 
                                                  'competitor_price': competitor_price,
                                                  'elasticity': elasticity})[0])
            
            # Calculate price ratio
            price_ratio = recommended_price / price
            
            # Calculate expected demand impact
            demand_ratio = price_ratio ** (-elasticity)
            
            return {
                'success': True,
                'pricing': {
                    'recommended_price': recommended_price,
                    'price_ratio': price_ratio,
                    'price_range': {
                        'min_price': recommended_price * 0.9,
                        'max_price': recommended_price * 1.1
                    },
                    'elasticity_category': 'standard',
                    'elasticity_factor': elasticity,
                    'expected_demand': demand_ratio,
                    'expected_profit': (recommended_price - cost) * demand_ratio,
                    'model_type': self.__class__.__name__
                },
                'segments': {
                    'primary_segment': 'unknown'
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'errors': [str(e)]
            }

class CostPlusPricingModel(BasePricingModel):
    """Cost plus pricing model"""
    
    def __init__(self, markup=0.5):
        """
        Initialize with markup percentage
        
        Args:
            markup: Markup percentage (e.g., 0.5 for 50% markup)
        """
        self.markup = markup
    
    def predict(self, X):
        """
        Generate prices using cost plus markup
        
        Args:
            X: DataFrame or dict with product attributes
            
        Returns:
            Array of recommended prices
        """
        if isinstance(X, pd.DataFrame):
            costs = X.get('cost', X['actual_price'] * 0.6)
            return costs * (1 + self.markup)
        elif isinstance(X, dict):
            cost = X.get('cost', X.get('actual_price', 100) * 0.6)
            return np.array([cost * (1 + self.markup)])
        else:
            raise ValueError("Input must be either a DataFrame or a dictionary")

class CompetitivePricingModel(BasePricingModel):
    """Competitive pricing model"""
    
    def __init__(self, discount=0.05):
        """
        Initialize with discount percentage
        
        Args:
            discount: Discount percentage (e.g., 0.05 for 5% below competition)
        """
        self.discount = discount
    
    def predict(self, X):
        """
        Generate prices relative to competition
        
        Args:
            X: DataFrame or dict with product attributes
            
        Returns:
            Array of recommended prices
        """
        if isinstance(X, pd.DataFrame):
            competitor_prices = X.get('competitor_price', X['actual_price'])
            return competitor_prices * (1 - self.discount)
        elif isinstance(X, dict):
            competitor_price = X.get('competitor_price', X.get('actual_price', 100))
            return np.array([competitor_price * (1 - self.discount)])
        else:
            raise ValueError("Input must be either a DataFrame or a dictionary")

class ElasticityPricingModel(BasePricingModel):
    """Elasticity-based pricing model"""
    
    def predict(self, X):
        """
        Generate prices based on price elasticity
        
        Args:
            X: DataFrame or dict with product attributes
            
        Returns:
            Array of recommended prices
        """
        if isinstance(X, pd.DataFrame):
            current_prices = X['actual_price']
            elasticities = X.get('elasticity', np.ones_like(current_prices))
            costs = X.get('cost', current_prices * 0.6)
        elif isinstance(X, dict):
            current_price = X.get('actual_price', 100)
            elasticity = X.get('elasticity', 1.0)
            cost = X.get('cost', current_price * 0.6)
            current_prices = np.array([current_price])
            elasticities = np.array([elasticity])
            costs = np.array([cost])
        else:
            raise ValueError("Input must be either a DataFrame or a dictionary")
        
        # Optimal price with constant elasticity demand
        optimal_prices = costs * (1 + 1/elasticities)
        
        # Ensure minimum profitability
        min_prices = costs * 1.15  # At least 15% margin
        return np.maximum(optimal_prices, min_prices)

class RuleBasedPricingModel(BasePricingModel):
    """Rule-based pricing model"""
    
    def __init__(self):
        """Initialize with pricing rules"""
        self.rules = [
            self._premium_rule,
            self._competitive_rule,
            self._elasticity_rule,
            self._cost_plus_rule,
            self._default_rule
        ]
    
    def predict(self, X):
        """
        Generate prices by applying business rules
        
        Args:
            X: DataFrame or dict with product attributes
            
        Returns:
            Array of recommended prices
        """
        if isinstance(X, pd.DataFrame):
            return X.apply(self._apply_rules, axis=1).values
        elif isinstance(X, dict):
            return np.array([self._apply_rules(X)])
        else:
            raise ValueError("Input must be either a DataFrame or a dictionary")
    
    def _apply_rules(self, data):
        """Apply rules in priority order until one matches"""
        for rule in self.rules:
            result = rule(data)
            if result is not None:
                return result
        
        # Fallback
        return data.get('actual_price', 100)
    
    def _premium_rule(self, data):
        """Premium price rule"""
        if data.get('rating', 0) >= 4.5:
            return data.get('actual_price', 100) * 1.1
        return None
    
    def _competitive_rule(self, data):
        """Competitive pricing rule"""
        if data.get('competitor_price', 0) > 0:
            return data.get('competitor_price', 100) * 0.95
        return None
    
    def _elasticity_rule(self, data):
        """Elasticity-based rule"""
        elasticity = data.get('elasticity', None)
        if elasticity and elasticity > 1.5:
            return data.get('actual_price', 100) * 0.9
        return None
    
    def _cost_plus_rule(self, data):
        """Cost plus rule"""
        cost = data.get('cost', None)
        if cost:
            return cost * 1.5
        return None
    
    def _default_rule(self, data):
        """Default rule"""
        return data.get('actual_price', 100)

class ProfitOptimizedModel(BasePricingModel):
    """
    Profit-optimized pricing model that uses multiple factors 
    to recommend prices that maximize profit.
    """
    
    def __init__(self, base_markup_pct=60, risk_threshold=0.8, 
                elasticity_weight=0.6, competitor_weight=0.2, 
                rating_factor=0.15, volume_factor=0.1):
        """
        Initialize the Profit Optimized Model with parameters
        
        Args:
            base_markup_pct: Base markup percentage (default: 60%)
            risk_threshold: Threshold for risk-adjusted pricing (default: 0.8)
            elasticity_weight: Weight for elasticity in pricing (default: 0.6)
            competitor_weight: Weight for competitor pricing (default: 0.2)
            rating_factor: Impact of product rating (default: 0.15)
            volume_factor: Impact of order volume (default: 0.1)
        """
        self.base_markup_pct = base_markup_pct
        self.risk_threshold = risk_threshold
        self.elasticity_weight = elasticity_weight
        self.competitor_weight = competitor_weight
        self.rating_factor = rating_factor
        self.volume_factor = volume_factor
    
    def predict(self, X):
        """
        Generate profit-optimized prices
        
        Args:
            X: DataFrame or dict with product attributes
            
        Returns:
            Array of recommended prices
        """
        if isinstance(X, pd.DataFrame):
            prices = []
            for _, row in X.iterrows():
                prices.append(self._calculate_price(row))
            return np.array(prices)
        elif isinstance(X, dict):
            return np.array([self._calculate_price(X)])
        else:
            raise ValueError("Input must be either a DataFrame or a dictionary")
    
    def _calculate_price(self, data):
        """Calculate optimized price for a single product"""
        # Get key values with defaults
        current_price = data.get('actual_price', 100)
        cost = data.get('cost', current_price * 0.6)
        elasticity = data.get('elasticity', 1.0)
        competitor_price = data.get('competitor_price', current_price * 1.1)
        rating = data.get('rating', 3.0)
        volume = data.get('volume', 50)
        
        # Base cost-plus price
        base_price = cost * (1 + (self.base_markup_pct / 100))
        
        # Elasticity-optimized price (profit maximizing)
        try:
            elasticity_price = cost * (elasticity / (elasticity - 1)) if elasticity != 1 else base_price
            # Ensure price is positive and above cost
            elasticity_price = max(elasticity_price, cost * 1.15)
        except:
            elasticity_price = base_price
        
        # Calculate initial price using weighted average
        initial_price = (base_price * (1 - self.elasticity_weight)) + (elasticity_price * self.elasticity_weight)
        
        # Apply rating adjustment
        rating_multiplier = 1.0 + (((rating - 3.0) / 5.0) * self.rating_factor)
        rating_adjusted = initial_price * rating_multiplier
        
        # Apply volume adjustment
        volume_multiplier = 1.0 + (((volume - 50) / 200) * self.volume_factor)
        volume_adjusted = rating_adjusted * volume_multiplier
        
        # Apply competitor adjustment
        competitor_ratio = competitor_price / current_price
        competitor_multiplier = 1.0
        if competitor_ratio > 1.2:  # Competitor much higher
            competitor_multiplier = 1.0 + (self.competitor_weight * 0.5)
        elif competitor_ratio < 0.8:  # Competitor much lower
            competitor_multiplier = 1.0 - (self.competitor_weight * 0.5)
        competitor_adjusted = volume_adjusted * competitor_multiplier
        
        # Cap price changes for safety
        min_price = current_price * 0.85  # Max 15% reduction
        max_price = current_price * 1.15  # Max 15% increase
        
        # Ensure minimum profit margin
        min_price = max(min_price, cost * 1.15)  # At least 15% margin
        
        # Final price within bounds
        final_price = max(min(competitor_adjusted, max_price), min_price)
        
        return final_price
    
    def fit(self, X, y=None):
        """Required for scikit-learn compatibility"""
        return self 