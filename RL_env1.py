import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class DynamicPricingEnv(gym.Env):
    """Custom Environment for Dynamic Pricing"""
    metadata = {'render.modes': ['human']}

    def __init__(self, max_price=500, min_price=10, demand_intercept=100, 
                 demand_slope=0.5, noise_std=10, historical_data=None,
                 elasticity=1.0, ppi=1.0, rating=4.0, num_orders=50):
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
        
        # Calculate demand using an elasticity-influenced demand curve with rating and orders influence
        # Higher elasticity means more sensitive to price changes
        # Higher rating increases base demand
        # Higher number of orders indicates product popularity and increases demand
        
        # Rating factor: 0.85 to 1.35 scale based on rating (1-5) - reduced influence
        rating_factor = 0.85 + (self.rating - 1) * 0.125  # Changed from 0.175 to 0.125
        
        # Order popularity factor: 0.9 to 1.1 scale based on number of orders (0-200) - significantly reduced influence
        order_factor = 0.9 + min(200, self.num_orders) / 1000  # Changed from 400 to 1000
        
        # Apply factors to base demand
        base_demand = self.demand_intercept * rating_factor * order_factor - self.demand_slope * price * self.elasticity
        noise = np.random.normal(0, self.noise_std)
        actual_demand = max(0, base_demand + noise)
        
        # Calculate revenue
        revenue = price * actual_demand
        
        # Calculate cost (simplified)
        unit_cost = price * 0.6  # Assume cost is 60% of price
        total_cost = unit_cost * actual_demand
        
        # Calculate profit
        profit = revenue - total_cost
        
        # Calculate reward (profit) with quality and popularity bonuses
        # Products with higher ratings can command a premium, incentivizing higher prices for quality products
        # Products with more orders get a smaller bonus to encourage volume-based pricing strategies
        rating_bonus = profit * (self.rating / 5) * 0.15  # Reduced from 0.2 to 0.15
        order_bonus = profit * (min(200, self.num_orders) / 200) * 0.05  # Reduced from 0.15 to 0.05
        
        # Calculate final reward with bonuses
        reward = profit + rating_bonus + order_bonus
        
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
            'rating_factor': rating_factor,
            'order_factor': order_factor,
            'rating_bonus': rating_bonus,
            'order_bonus': order_bonus
        }
        
        # Return state, reward, done, and info
        truncated = False  # Assume never truncated for simplicity
        return self.state, reward, done, truncated, info

    def render(self, mode='human'):
        """Render the environment"""
        # Not implemented, but required by the Gym interface
        pass

def make_env(historical_data=None, elasticity=1.0, ppi=1.0, rating=4.0, num_orders=50):
    """Create an instance of the DynamicPricingEnv"""
    return DynamicPricingEnv(
        historical_data=historical_data,
        elasticity=elasticity,
        ppi=ppi,
        rating=rating,
        num_orders=num_orders
    )