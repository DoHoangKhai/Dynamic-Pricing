import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .pricing_strategies import PricingStrategy, CustomerSegmentation

class DynamicPricingEnv(gym.Env):
    """Custom Environment for Dynamic Pricing"""
    metadata = {'render.modes': ['human']}

    def __init__(self, max_price=500, min_price=10, demand_intercept=100, 
                 demand_slope=0.5, noise_std=10, historical_data=None,
                 elasticity=1.0, ppi=1.0, rating=4.0, num_orders=50,
                 product_type='Electronics', product_group='Laptops'):
        super(DynamicPricingEnv, self).__init__()
        
        # Price range and step size
        self.min_price = min_price
        self.max_price = max_price
        
        # Number of discrete price levels - match the prototype environment
        self.n_price_levels = 21
        
        # Calculate price step based on range and levels
        self.price_step = (max_price - min_price) / (self.n_price_levels - 1)
        
        # Define action space (discrete price levels)
        self.action_space = spaces.Discrete(self.n_price_levels)
        
        # Basic market parameters
        self.demand_intercept = demand_intercept
        self.demand_slope = demand_slope
        self.noise_std = noise_std
        self.elasticity = elasticity
        self.ppi = ppi
        self.rating = rating
        self.num_orders = num_orders
        self.product_type = product_type
        self.product_group = product_group
        
        # Initialize pricing strategies and customer segmentation
        self.pricing_strategy = PricingStrategy()
        self.customer_segmentation = CustomerSegmentation()
        
        # Maximum steps per episode
        self.max_steps = 30
        self.step_count = 0
        
        # Historical data for state representation
        self.historical_data = historical_data
        self.price_history = np.zeros(3)  # Track last 3 prices
        
        # Create observation space with correct dimensions to match prototype environment
        # The prototype model expects exactly 251 dimensions
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(251,), dtype=np.float32
        )
        
    def _get_initial_state(self):
        """Generate the initial state observation"""
        # Create a zero-filled array for the state with 251 dimensions
        state = np.zeros(251, dtype=np.float32)
        
        # For compatibility with the prototype model, keep the same structure:
        # First 9 indices: Product type one-hot encoding
        state[0] = 1.0  # First product type
        
        # Next 207 indices: Product group one-hot encoding
        state[9] = 1.0  # First product group
        
        # Next 9 indices: Basic features
        basic_feat_start = 9 + 207
        state[basic_feat_start] = (self.rating - 1) / 4  # Normalized rating (1-5)
        state[basic_feat_start + 1] = (self.ppi - 0.5) / 1.0  # Normalized PPI (0.5-1.5)
        state[basic_feat_start + 2] = (self.elasticity - 0.5) / 1.0  # Normalized elasticity
        
        # Elasticity category indicators (one-hot)
        is_high_elasticity = 1.0 if self.elasticity > 1.2 else 0.0  # Price sensitive
        is_medium_elasticity = 1.0 if 0.8 <= self.elasticity <= 1.2 else 0.0  # Neutral
        is_low_elasticity = 1.0 if self.elasticity < 0.8 else 0.0  # Price insensitive
        
        state[basic_feat_start + 3] = is_high_elasticity
        state[basic_feat_start + 4] = is_medium_elasticity
        state[basic_feat_start + 5] = is_low_elasticity
        
        # Add optimal price ratio hint
        target_price = 1.0 / self.elasticity if self.elasticity > 0 else 1.0
        state[basic_feat_start + 6] = (target_price - 0.5) / 1.0
        
        # Add normalized number of orders
        state[basic_feat_start + 7] = self.num_orders / 200  # Normalized to max 200 orders
        
        # Default value for pages
        state[basic_feat_start + 8] = 0.5  # Mid-range for pages
        
        # Next 4 indices: Competitor features
        comp_start = basic_feat_start + 9
        state[comp_start:comp_start+4] = 0.5  # Neutral competitor values
        
        # Next 3 indices: Price history
        hist_start = comp_start + 4
        state[hist_start:hist_start+3] = 0.5  # Neutral price history
        
        # Next 5 indices: PPI categories (one-hot)
        ppi_start = hist_start + 3
        # Default to middle PPI category
        state[ppi_start + 2] = 1.0
        
        # Ensure the rest of the state array is zero-filled
        
        return state

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        # Reset step counter
        self.step_count = 0
        
        # Reset price history
        self.price_history = np.zeros(3)
        
        # Get initial state
        self.state = self._get_initial_state()
        
        # Support both old and new Gym API
        if options is not None:  # New Gym API
            return self.state, {}
        else:  # Old Gym API
            return self.state

    def step(self, action):
        """Take an action in the environment"""
        # Convert action to price
        price = self.min_price + action * self.price_step
        
        # Create product dictionary for pricing strategies
        product = {
            'product_id': 'test_product',
            'product_type': self.product_type,
            'product_group': self.product_group,
            'price': price,
            'cost': price * 0.6,  # Assume cost is 60% of price
            'elasticity': self.elasticity,
            'rating': self.rating,
            'ppi': self.ppi,
            'number_of_orders': self.num_orders
        }
        
        # Create market information
        market_info = {
            'competitive_intensity': 0.7,  # Default to moderately competitive
            'price_trend': 0.0,  # Default to stable prices
            'current_price_ratio': self.ppi
        }
        
        # Get customer segmentation
        segments = self.customer_segmentation.get_segment_distribution(
            self.product_type, self.product_group, self.rating
        )
        
        # Calculate price ratio compared to reference price
        price_ratio = price / (self.ppi * price)
        
        # Get segment conversion probabilities
        segment_conversion = self.customer_segmentation.calculate_segment_conversion_probabilities(
            price_ratio, product
        )
        
        # Get demand modifier from customer segmentation
        margin = (price - product['cost']) / price
        demand_modifier = self.customer_segmentation.get_demand_modifier(
            product, price_ratio, margin
        )
        
        # Calculate demand using enhanced approach
        # Base demand calculation using elasticity-influenced demand curve
        base_demand = self.demand_intercept - self.demand_slope * price * self.elasticity
        
        # Adjust demand using customer segmentation
        conversion_prob = segment_conversion['weighted_conversion']
        profit_multiplier = segment_conversion['expected_profit_multiplier']
        
        # Apply demand modifier
        modified_demand = base_demand * demand_modifier
        
        # Add randomness
        noise = np.random.normal(0, self.noise_std)
        actual_demand = max(0, modified_demand + noise)
        
        # Calculate revenue
        revenue = price * actual_demand
        
        # Calculate cost
        unit_cost = price * 0.6  # Assume cost is 60% of price
        total_cost = unit_cost * actual_demand
        
        # Calculate profit
        profit = revenue - total_cost
        
        # Calculate reward (profit) with quality and popularity bonuses
        # Products with higher ratings can command a premium
        rating_bonus = profit * (self.rating / 5) * 0.15
        # Products with more orders get a smaller bonus to encourage volume-based pricing
        order_bonus = profit * (min(200, self.num_orders) / 200) * 0.05
        
        # Apply profit multiplier from customer segmentation
        profit_bonus = profit * (profit_multiplier - 1.0) * 0.5
        
        # Calculate final reward with bonuses
        reward = profit + rating_bonus + order_bonus + profit_bonus
        
        # Update step counter
        self.step_count += 1
        
        # Update price history
        self.price_history = np.roll(self.price_history, 1)
        self.price_history[0] = price / self.max_price  # Normalize price
        
        # Update state
        self.state = self._get_initial_state()
        
        # Update price history in state
        hist_start = (9 + 207 + 9 + 4)
        self.state[hist_start:hist_start+3] = self.price_history
        
        # Check if episode is done
        done = self.step_count >= self.max_steps
        
        # New Gym API requires additional info dictionary
        info = {
            'price': price,
            'demand': actual_demand,
            'profit': profit,
            'revenue': revenue,
            'cost': total_cost,
            'conversion_prob': conversion_prob,
            'profit_multiplier': profit_multiplier,
            'demand_modifier': demand_modifier,
            'rating_bonus': rating_bonus,
            'order_bonus': order_bonus,
            'profit_bonus': profit_bonus
        }
        
        # Return state, reward, done, and info
        truncated = False  # Assume never truncated for simplicity
        return self.state, reward, done, truncated, info

    def render(self, mode='human'):
        """Render the environment"""
        # Not implemented, but required by the Gym interface
        pass

def make_env(historical_data=None, elasticity=1.0, ppi=1.0, rating=4.0, num_orders=50, 
            product_type='Electronics', product_group='Laptops'):
    """Create an instance of the DynamicPricingEnv"""
    return DynamicPricingEnv(
        historical_data=historical_data,
        elasticity=elasticity,
        ppi=ppi,
        rating=rating,
        num_orders=num_orders,
        product_type=product_type,
        product_group=product_group
    )