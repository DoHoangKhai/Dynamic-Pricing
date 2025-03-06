import numpy as np
from gymnasium import Env, spaces

class DynamicPricingEnv(Env):
    metadata = {"render_modes": ["human"], "name": "dynamic_pricing_v1"}

    def __init__(self, max_price=500, min_price=10, demand_intercept=1000, demand_slope=2, noise_std=10, historical_data=None, render_mode=None):
        super().__init__()
        self.max_price = max_price
        self.min_price = min_price
        self.demand_intercept = demand_intercept
        self.demand_slope = demand_slope
        self.noise_std = noise_std
        self.render_mode = render_mode
        self.historical_data = historical_data

        # Action space: Discrete prices between min_price and max_price (steps of 10)
        self.action_space = spaces.Discrete((max_price - min_price) // 10 + 1)

        # State space: Features + sales history
        # Calculate expected feature dimension: 8 numerical + product_type one-hot + product_group one-hot
        # Based on data_loader.py analysis
        expected_features = 8 + 9 + 207  # 224 total features + 1 for sales = 225
        
        # State space: Current price + selected features + initial sales
        if historical_data is not None:
            # When historical data is provided, use its shape
            self.observation_space = spaces.Box(low=0, high=1, shape=(historical_data.shape[1] + 1,), dtype=np.float32)
        else:
            # When no historical data, use expected shape (225)
            self.observation_space = spaces.Box(low=0, high=1, shape=(expected_features + 1,), dtype=np.float32)

        # Episode tracking
        self.current_step = 0
        self.max_steps = 30  # 30 days per episode
        self.price = np.random.randint(min_price, max_price)
        self.state = self._get_initial_state()

    def _get_initial_state(self):
        if self.historical_data is not None:
            # Select a random row from historical data
            row_idx = np.random.randint(len(self.historical_data))
            row = self.historical_data[row_idx]
            return np.array(list(row) + [0], dtype=np.float32)  # Add initial sales
        else:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Important: Reset the random number generator
        self.current_step = 0
        self.price = np.random.randint(self.min_price, self.max_price)
        self.state = self._get_initial_state()
        return self.state, {}

    def step(self, action):
        self.price = 10 + action * 10  # Map action to price (10, 20, ..., 500)

        noise = np.random.normal(0, self.noise_std)
        demand = max(0, self.demand_intercept - self.demand_slope * self.price + noise)
        reward = self.price * demand
        self.state[-1] = demand  # Update sales in the last position

        observation = self.state

        self.current_step += 1
        done = self.current_step >= self.max_steps
        truncated = False

        return observation, reward, done, truncated, {}

    def render(self):
        pass

# Convert to Gymnasium-compatible environment
def make_env(historical_data=None):
    return DynamicPricingEnv(historical_data=historical_data)