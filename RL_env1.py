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
        self.price_step = 10
        
        # Number of discrete price levels
        num_prices = int((max_price - min_price) / self.price_step) + 1
        
        # Define action space (discrete price levels)
        self.action_space = spaces.Discrete(num_prices)
        
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
        # Product type one-hot (9) + Product group one-hot (207) + Basic features (9) + 
        # Competitor features (4) + Price history (3) + PPI categories (5)
        total_obs_dim = 9 + 207 + 9 + 4 + 3 + 5
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(total_obs_dim,), dtype=np.float32
        )
        
    def _get_initial_state(self):
        """Generate the initial state observation"""
        # Create a zero-filled array for the state
        state = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # Set default values for product type (first type in one-hot encoding)
        state[9] = 1.0  # First product type
        
        # Set default values for product group (first group in one-hot encoding)
        state[9 + 207] = 1.0  # First product group
        
        # Set default values for basic features
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
        
        # Initialize competitor features
        comp_start = basic_feat_start + 9
        state[comp_start:comp_start+4] = 0.5  # Neutral competitor values
        
        # Initialize price history
        hist_start = comp_start + 4
        state[hist_start:hist_start+3] = 0.5  # Neutral price history
        
        # Initialize PPI categories (one-hot)
        ppi_start = hist_start + 3
        # Default to middle PPI category
        state[ppi_start + 2] = 1.0
        
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
        
        # Calculate demand using an elasticity-influenced demand curve
        # Higher elasticity means more sensitive to price changes
        base_demand = self.demand_intercept - self.demand_slope * price * self.elasticity
        noise = np.random.normal(0, self.noise_std)
        actual_demand = max(0, base_demand + noise)
        
        # Calculate revenue
        revenue = price * actual_demand
        
        # Calculate cost (simplified)
        unit_cost = price * 0.6  # Assume cost is 60% of price
        total_cost = unit_cost * actual_demand
        
        # Calculate profit
        profit = revenue - total_cost
        
        # Calculate reward (profit)
        reward = profit
        
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
            'cost': total_cost
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