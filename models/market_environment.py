#!/usr/bin/env python
"""
E-commerce Market Environment for Dynamic Pricing

This module provides a Gymnasium environment that simulates an e-commerce marketplace
for dynamic pricing. It uses a regression-based approach to model price elasticity
and customer demand based on product features.
"""

import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Any, Optional

# Import the pricing strategies
try:
    from models.pricing_strategies import PricingStrategy, CustomerSegmentation
except ImportError:
    try:
        from pricing_strategies import PricingStrategy, CustomerSegmentation
    except ImportError:
        try:
            from prototype.pricing_strategies import PricingStrategy, CustomerSegmentation
        except ImportError:
            print("Warning: pricing_strategies module not found. Using internal fallback.")
            PricingStrategy = None
            CustomerSegmentation = None


# Add MarketEnvironment class for app.py compatibility
class MarketEnvironment:
    """
    Wrapper class for the e-commerce market environment to provide a simplified interface
    for the web application.
    """
    
    def __init__(self, data_file=None, num_price_levels=21):
        """
        Initialize the market environment.
        
        Args:
            data_file: Path to the CSV data file with product information (optional)
            num_price_levels: Number of discrete price levels for the action space
        """
        self.env = EcommerceMarketEnv(
            data_file=data_file,
            n_price_levels=num_price_levels
        )
        
    def get_optimal_price(self, product_info, market_info=None):
        """
        Calculate the optimal price for a product.
        
        Args:
            product_info: Dictionary with product attributes
            market_info: Dictionary with market information (optional)
            
        Returns:
            Dictionary with price recommendation and strategy insights
        """
        # Convert input to the format expected by the environment
        if 'product_id' not in product_info:
            product_info['product_id'] = 'temp_product'
            
        # Setup environment with product info
        self.env.current_product = product_info
        
        # Use the environment's optimal price calculation
        optimal_price = self.env.get_optimal_price()
        
        # Return the price recommendation
        return {
            'recommended_price': optimal_price,
            'strategy': 'optimal',
            'confidence': 0.85
        }
    
    def evaluate_price(self, product_info, price, market_info=None):
        """
        Evaluate a given price for a product.
        
        Args:
            product_info: Dictionary with product attributes
            price: Price to evaluate
            market_info: Dictionary with market information (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Setup environment with product info
        self.env.current_product = product_info
        
        # Calculate price ratio
        base_price = product_info.get('price', 100.0)
        price_ratio = price / base_price if base_price > 0 else 1.0
        
        # Calculate expected demand
        demand = self.env._calculate_demand(price_ratio)
        
        # Calculate conversion probability
        conversion_prob = self.env._calculate_conversion_probability(price_ratio, demand)
        
        # Calculate profit
        cost = product_info.get('cost', base_price * 0.6)
        revenue = price * demand * conversion_prob
        profit = revenue - cost * demand * conversion_prob
        
        # Return evaluation metrics
        return {
            'demand': demand,
            'conversion_probability': conversion_prob,
            'revenue': revenue,
            'profit': profit,
            'margin': (price - cost) / price if price > 0 else 0.0
        }


class ElasticityPricingStrategy:
    """
    A strategy pattern class for handling elasticity-based pricing decisions
    
    This class implements different pricing strategies based on product elasticity,
    allowing for more nuanced pricing decisions that account for how price-sensitive
    a product's demand is.
    """
    
    @staticmethod
    def get_optimal_price_range(elasticity: float, base_price: float) -> tuple:
        """
        Determine the optimal price range for a product based on its elasticity
        
        Args:
            elasticity: Price elasticity of demand (higher = more sensitive to price)
            base_price: Base price of the product
            
        Returns:
            Tuple of (min_price, max_price, target_price) for the product
        """
        # For highly elastic products (price-sensitive)
        if elasticity > 1.2:
            # Price-sensitive products should be priced more competitively
            min_price = base_price * 0.8
            max_price = base_price * 1.0
            target_price = base_price * 0.9  # Target price below base price
            
        # For moderately elastic products
        elif 0.8 <= elasticity <= 1.2:
            # Moderately elastic products can be priced around their base price
            min_price = base_price * 0.9
            max_price = base_price * 1.1
            target_price = base_price  # Target at base price
            
        # For inelastic products (price-insensitive)
        else:
            # Price-insensitive products can sustain higher prices
            min_price = base_price * 0.95
            max_price = base_price * 1.2
            target_price = base_price * 1.1  # Target price above base price
            
        return min_price, max_price, target_price
    
    @staticmethod
    def calculate_elasticity_reward(price_ratio: float, elasticity: float, profit: float) -> float:
        """
        Calculate a reward component based on how well the price matches the optimal
        range for the product's elasticity
        
        Args:
            price_ratio: Price ratio chosen (relative to base price)
            elasticity: Price elasticity of demand
            profit: Current profit from this pricing decision
            
        Returns:
            Reward component based on elasticity-appropriate pricing
        """
        # Define ideal price ratios based on elasticity
        if elasticity > 1.2:  # High elasticity (price-sensitive)
            # Ideal range: 0.8 - 1.0, optimal at 0.9
            ideal_ratio = 0.9
            max_deviation = 0.1
        elif elasticity < 0.8:  # Low elasticity (price-insensitive)
            # Ideal range: 0.95 - 1.2, optimal at 1.1
            ideal_ratio = 1.1
            max_deviation = 0.15
        else:  # Moderate elasticity
            # Ideal range: 0.9 - 1.1, optimal at 1.0
            ideal_ratio = 1.0
            max_deviation = 0.1
            
        # Calculate deviation from ideal price ratio
        deviation = abs(price_ratio - ideal_ratio) / max_deviation
        
        # Calculate reward: higher when closer to ideal ratio
        # Scales from 0.2 (far from ideal) to 1.0 (at ideal)
        reward_factor = max(0.0, 1.0 - deviation)
        
        # Scale reward by profit (higher profits get higher rewards)
        elasticity_reward = profit * 0.2 * reward_factor
        
        return elasticity_reward


class CurriculumWrapper(gym.Wrapper):
    """
    A wrapper for the EcommerceMarketEnv that implements curriculum learning
    
    The curriculum progresses through stages of increasing difficulty:
    1. Initial Stage: Easy products, low noise, focused on conversions
    2. Intermediate Stage: Medium difficulty products, moderate noise
    3. Advanced Stage: Hard products, realistic noise
    4. Final Stage: All products, full noise, profit-focused
    
    Each stage gradually adjusts reward weights to shift emphasis from
    conversion-focused to profit-focused strategies.
    """
    
    def __init__(self, env, curriculum_stages=4, episodes_per_stage=50, 
                 include_profit_curriculum=True):
        """
        Initialize the curriculum wrapper
        
        Args:
            env: The environment to wrap
            curriculum_stages: Number of curriculum stages
            episodes_per_stage: Number of episodes per stage
            include_profit_curriculum: Whether to include profit-focused stages
        """
        super().__init__(env)
        
        # Curriculum parameters
        self.curriculum_stages = curriculum_stages
        self.episodes_per_stage = episodes_per_stage
        self.include_profit_curriculum = include_profit_curriculum
        
        # Track episodes and current stage
        self.episode_count = 0
        self.current_stage = 0
        
        # Define stage transitions
        self.min_price_ratio = 0.7  # Start with a narrower price range
        self.max_price_ratio = 1.3
        self.target_min_price_ratio = 0.4  # Target final price range
        self.target_max_price_ratio = 1.6
        
        # Define noise levels
        self.initial_noise = 0.05
        self.final_noise = 0.2
        
        # Define reward weight transitions
        self.initial_profit_weight = 0.5
        self.final_profit_weight = 2.0
        self.initial_conversion_weight = 3.0
        self.final_conversion_weight = 1.0
        
        # Track episode rewards for curriculum progression
        self.episode_rewards = []
        self.episode_profits = []
        self.episode_conversions = []
    
    def reset(self, **kwargs):
        """
        Reset the environment and update curriculum stage if needed
        
        Returns:
            observation, info
        """
        # Update current stage based on episode count
        self.current_stage = min(self.curriculum_stages - 1, 
                                self.episode_count // self.episodes_per_stage)
        
        # Calculate current parameters based on stage
        progress = self.current_stage / (self.curriculum_stages - 1)
        
        # Update price range
        price_range_progress = min(1.0, progress * 1.5)  # Expand price range faster
        price_min = self.min_price_ratio + (self.target_min_price_ratio - self.min_price_ratio) * price_range_progress
        price_max = self.max_price_ratio + (self.target_max_price_ratio - self.max_price_ratio) * price_range_progress
        
        # Update noise level
        noise_level = self.initial_noise + (self.final_noise - self.initial_noise) * progress
        
        # Update product difficulty
        product_difficulty = self._get_current_product_difficulty()
        
        # Update reward weights
        reward_weights = self._update_reward_weights(progress)
        
        # Print curriculum information for visibility
        print(f"\nCurriculum Stage {self.current_stage+1}/{self.curriculum_stages}")
        print(f"Price Range: {price_min:.2f}-{price_max:.2f}, Noise: {noise_level:.2f}, Product Difficulty: {product_difficulty}")
        print(f"Profit Weight: {reward_weights['profit_weight']:.2f}, Conversion Weight: {reward_weights['conversion_weight']:.2f}")
        
        # Set the environment parameters directly instead of passing them through options
        self.env.demand_noise = noise_level
        self.env.product_difficulty = product_difficulty
        self.env._reward_weights.update(reward_weights)
        
        # Reset the environment without passing options
        obs, info = self.env.reset(**kwargs)
        
        # Increment episode counter
        self.episode_count += 1
        
        return obs, info
    
    def step(self, action):
        """
        Take a step in the environment and track episode metrics
        
        Args:
            action: The action to take
            
        Returns:
            observation, reward, done, truncated, info
        """
        # Take the step in the wrapped environment
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Track episode metrics
        if done:
            self.episode_rewards.append(reward)
            self.episode_profits.append(info.get('episode_profit', 0.0))
            self.episode_conversions.append(info.get('episode_conversion_count', 0))
        
        return obs, reward, done, truncated, info
    
    def _update_reward_weights(self, progress):
        """
        Update reward weights based on curriculum progress
        
        Args:
            progress: Progress through the curriculum (0.0 to 1.0)
            
        Returns:
            updated_weights: Dictionary of updated reward weights
        """
        # Deep copy of the base reward weights
        updated_weights = self.env._reward_weights.copy()
        
        # Update profit and conversion weights
        if self.include_profit_curriculum:
            # Gradually increase profit weight and decrease conversion weight
            profit_weight = self.initial_profit_weight + (self.final_profit_weight - self.initial_profit_weight) * progress
            conversion_weight = self.initial_conversion_weight - (self.initial_conversion_weight - self.final_conversion_weight) * progress
            
            # Update the weights
            updated_weights['profit_weight'] = profit_weight
            updated_weights['conversion_weight'] = conversion_weight
            
            # Adjust other weights based on progress
            if progress > 0.5:
                # In later stages, reduce near-conversion weight
                updated_weights['near_conversion_weight'] *= (1.0 - (progress - 0.5) * 0.5)
                
                # Gradually increase optimal pricing weight
                updated_weights['optimal_price_weight'] *= (1.0 + (progress - 0.5) * 0.5)
                
                # Decrease exploration weight as training progresses
                updated_weights['exploration_weight'] *= (1.0 - (progress - 0.5) * 0.8)
        
        return updated_weights
    
    def _get_current_product_difficulty(self):
        """
        Get the current product difficulty based on curriculum stage
        
        Returns:
            difficulty: The difficulty level ("easy", "medium", or "hard")
        """
        if self.current_stage == 0:
            return "easy"
        elif self.current_stage == 1:
            return "medium"
        else:
            return "hard"


class EcommerceMarketEnv(gym.Env):
    """
    A Gymnasium environment for e-commerce dynamic pricing that models customer demand
    and price elasticity based on product attributes.
    
    The environment uses a regression model to simulate how demand changes in response
    to different prices, taking into account:
    - Product category and group
    - Product quality (rating)
    - Market conditions (Price Position Index)
    - Historical sales volume
    
    The agent's goal is to find optimal pricing strategies that maximize profit
    while maintaining reasonable and competitive prices.
    """
    
    metadata = {"render_modes": ["human"], "name": "ecommerce_market_v1"}
    
    def __init__(self, data_file=None, episode_length=20, n_price_levels=21, 
                 demand_noise=0.1, product_difficulty="medium"):
        """
        Initialize the e-commerce market environment
        
        Args:
            data_file: Path to the CSV data file with product information
            episode_length: Number of steps per episode
            n_price_levels: Number of discrete price levels
            demand_noise: Amount of noise to add to demand calculations
            product_difficulty: Difficulty level for product selection ("easy", "medium", "hard")
        """
        # Load and prepare data
        if data_file is None:
            # Use default path relative to the current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_file = os.path.join(current_dir, "../data/product_data.csv")
            
        # Load data
        self.data = self._load_data(data_file)
        
        # Store parameters
        self.episode_length = episode_length
        self.n_price_levels = n_price_levels
        self.demand_noise = demand_noise
        self.product_difficulty = product_difficulty
        
        # Setup spaces
        self._setup_environment()
        
        # Initialize advanced pricing modules
        if CustomerSegmentation is not None:
            self.customer_segmentation = CustomerSegmentation(segment_count=4)
            self.use_time_factors = True  # Enable time-based pricing factors
        else:
            self.customer_segmentation = None
            self.use_time_factors = False
        
        if PricingStrategy is not None:
            self.pricing_strategy = PricingStrategy()
        else:
            self.pricing_strategy = None
        
        # Initialize episode state
        self.reset()
        
    def _load_data(self, data_file):
        """
        Load product data from the specified path
        
        Args:
            data_file: Path to the CSV data file
            
        Returns:
            data: Pandas DataFrame with product data
        """
        # Check if the data file exists
        if not os.path.exists(data_file):
            raise ValueError(f"Data file not found: {data_file}")
        
        # Load the data
        data = pd.read_csv(data_file)
        
        # Basic validation - check for required columns
        required_columns = ['product_id']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns in data: {missing_columns}")
        
        # Add actual_price column if it doesn't exist
        if 'actual_price' not in data.columns:
            print("Actual price column not found. Generating prices...")
            # Generate prices based on PPI and quality factor if available
            if 'ppi' in data.columns and 'quality_factor' in data.columns:
                data['actual_price'] = data['ppi'] * data['quality_factor'] * 10
            else:
                # Generate random prices between 10 and 200
                data['actual_price'] = np.random.uniform(10.0, 200.0, size=len(data))
                data['actual_price'] = data['actual_price'].round(2)  # Round to 2 decimal places
        
        # Add elasticity column if it doesn't exist
        if 'elasticity' not in data.columns:
            print("Elasticity column not found. Calculating elasticity...")
            # Calculate elasticity for each product
            elasticities = []
            for _, row in data.iterrows():
                product_info = row.to_dict()
                elasticity = self._calculate_elasticity(product_info)
                elasticities.append(elasticity)
            
            # Add elasticity column to the data
            data['elasticity'] = elasticities
            print(f"Added elasticity column with values ranging from {min(elasticities):.2f} to {max(elasticities):.2f}")
        
        # Add cost column if it doesn't exist
        if 'cost' not in data.columns:
            print("Cost column not found. Generating costs...")
            # Generate costs as 60% of actual price
            data['cost'] = data['actual_price'] * 0.6
            data['cost'] = data['cost'].round(2)  # Round to 2 decimal places
        
        # Ensure product_type column exists
        if 'product_type' not in data.columns:
            print("Product type column not found. Adding default product types...")
            # Create random product types
            product_types = ['electronics', 'clothing', 'home', 'beauty', 'sports', 'books']
            data['product_type'] = np.random.choice(product_types, size=len(data))
        
        # Ensure product_group column exists
        if 'product_group' not in data.columns:
            print("Product group column not found. Adding default product groups...")
            # Create product groups based on product types
            data['product_group'] = data['product_type'].apply(lambda x: x[:3].upper())
        
        # Ensure rating column exists
        if 'rating' not in data.columns:
            print("Rating column not found. Adding default ratings...")
            # Create random ratings between 1 and 5
            data['rating'] = np.random.uniform(2.0, 5.0, size=len(data))
            data['rating'] = data['rating'].round(1)  # Round to 1 decimal place
        
        # Ensure ppi column exists
        if 'ppi' not in data.columns:
            print("PPI column not found. Calculating PPI...")
            # Calculate PPI as price / (6 - rating)
            data['ppi'] = data['actual_price'] / (6 - data['rating'])
        
        return data
        
    def _create_train_test_split(self):
        """Split the data into training and testing sets"""
        # Get unique product IDs
        product_ids = self.data['product_id'].unique()
        
        # Shuffle product IDs
        np.random.shuffle(product_ids)
        
        # Split into train and test
        split_idx = int(len(product_ids) * (1 - self.test_split))
        train_ids = product_ids[:split_idx]
        test_ids = product_ids[split_idx:]
        
        # Create train and test dataframes
        if self.mode == "train":
            self.data = self.data[self.data['product_id'].isin(train_ids)].copy()
        else:
            self.data = self.data[self.data['product_id'].isin(test_ids)].copy()
            
        # Calculate elasticity if it doesn't exist in the data
        if 'elasticity' not in self.data.columns:
            self._calculate_elasticity()
            
        # Store PPI categories
        if 'ppi' in self.data.columns:
            self.data['ppi_category'] = self.data['ppi'].apply(self._get_ppi_category)
            
        # Number of products
        self.n_products = len(self.data)
        
        # Initialize product index
        self.current_product_idx = 0

    def _calculate_elasticity(self, product_info=None):
        """
        Calculate price elasticity for a product based on its attributes
        
        Args:
            product_info: Dictionary with product information (optional)
            
        Returns:
            elasticity: The calculated price elasticity
        """
        # If product_info is not provided, use current product info
        if product_info is None:
            product_info = self.current_product_info
            
        # Start with a base elasticity centered around 1.0
        base_elasticity = 1.0
        
        # Adjust based on product rating if available
        # Higher rated products tend to have lower elasticity (less price sensitive)
        if 'rating' in product_info:
            rating = product_info['rating']
            # Adjust elasticity based on rating (higher rating = lower elasticity)
            rating_factor = 1.0 - (rating - 3.0) * 0.1  # +/-10% per rating point from baseline
            base_elasticity *= rating_factor
        
        # Adjust based on PPI (Price Performance Index) if available
        # Higher PPI products tend to have higher elasticity (more price sensitive)
        if 'ppi' in product_info:
            ppi = product_info['ppi']
            # Adjust elasticity based on PPI (higher PPI = higher elasticity)
            if ppi > 1.2:  # High PPI (expensive for its value)
                base_elasticity *= 1.2
            elif ppi < 0.8:  # Low PPI (good value)
                base_elasticity *= 0.8
        
        # Adjust based on product type if available
        if 'product_type' in product_info:
            product_type = product_info['product_type'].lower()
            
            # Different product types have different elasticity characteristics
            type_factors = {
                'electronics': np.random.uniform(0.9, 1.1),
                'luxury': np.random.uniform(0.7, 0.9),
                'clothing': np.random.uniform(1.0, 1.2),
                'food': np.random.uniform(1.1, 1.3),
                'household': np.random.uniform(0.9, 1.1),
                'books': np.random.uniform(0.8, 1.0),
                'beauty': np.random.uniform(0.9, 1.1),
                'sports': np.random.uniform(1.0, 1.2),
                'premium': np.random.uniform(0.7, 0.9),
                'commodity': np.random.uniform(1.1, 1.3),
                'basics': np.random.uniform(1.1, 1.3),
                'new': np.random.uniform(1.0, 1.2)
            }
            
            # Get the factor for this product type, or use a random factor if not found
            type_factor = type_factors.get(product_type, np.random.uniform(0.9, 1.1))
            base_elasticity *= type_factor
        
        # Add some random noise to create variation
        noise = np.random.uniform(0.9, 1.1)
        elasticity = base_elasticity * noise
        
        # Ensure elasticity is within reasonable bounds
        elasticity = max(0.5, min(1.5, elasticity))
        
        return elasticity
    
    def _setup_environment(self):
        """Set up the environment spaces and encoders"""
        # Define reward weights with default values
        self._reward_weights = {
            'profit_weight': 10.0,                  # Increased from 8.0 to emphasize profit more
            'conversion_weight': 0.6,               # Reduced from 0.8 to discourage unprofitable conversions
            'near_conversion_weight': 0.05,         # Further reduced from 0.1
            'optimal_price_weight': 3.0,            # Increased from 2.5 to encourage optimal pricing
            'elasticity_response_weight': 2.5,      # Increased from 2.0 for better elasticity response
            'exploration_weight': 0.03,             # Reduced from 0.05 to focus more on exploitation
            'competitor_response_weight': 1.0,       # Increased from 0.8 for better market awareness
            'product_specific_weight': 2.0,         # Increased from 1.5 for better product-type optimization
            'customer_lifetime_value_weight': 2.5,   # Increased from 2.0 to focus on valuable customers
            'segment_targeting_weight': 2.0,        # Increased from 1.8 for better segmentation
            'below_cost_penalty_weight': 20.0       # Significantly increased from 15.0 to prevent unprofitable pricing
        }
        
        # Initialize tracking variables
        self.reward_components = {}
        self.episode_profit = 0.0
        self.episode_conversion_count = 0
        self.episode_actions = []
        self.episode_profits = []
        
        # Define price ratio bounds and step size
        self.min_price_ratio = 0.4
        self.max_price_ratio = 1.6
        self.price_ratio_step = (self.max_price_ratio - self.min_price_ratio) / (self.n_price_levels - 1)
        
        # Define action space (discrete price levels)
        self.action_space = gym.spaces.Discrete(self.n_price_levels)
        
        # Define observation space
        self.observation_shape = 251  # Size of the observation vector (updated for enhanced features)
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(self.observation_shape,), dtype=np.float32
        )
        
        # Initialize encoders for categorical features
        # Get unique product types and groups
        product_types = self.data['product_type'].unique()
        product_groups = self.data['product_group'].unique()
        
        # Initialize and fit type encoder
        self.type_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.type_encoder.fit(product_types.reshape(-1, 1))
        
        # Initialize and fit group encoder
        self.group_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.group_encoder.fit(product_groups.reshape(-1, 1))
        
        # Initialize history trackers
        self.price_history = []
        self.demand_history = []
        self.profit_history = []
        self.conversion_history = []

    def _create_group_embeddings(self):
        """Create embeddings for product groups to reduce dimensionality"""
        # Get DataFrame with proper column names
        group_df = pd.DataFrame(
            self.data['product_group'].unique(), 
            columns=['product_group']
        )
        
        # Transform using proper DataFrame
        group_one_hot = self.group_encoder.transform(group_df)
        
        # Create a random projection matrix for the embeddings
        np.random.seed(self.seed_value)
        projection = np.random.normal(0, 1, (self.num_product_groups, self.group_embedding_dim))
        
        # Normalize the projection matrix
        projection = projection / np.sqrt(np.sum(projection**2, axis=1, keepdims=True))
        
        # Return the embedding matrix
        return projection
    
    def _get_transform_with_feature_names(self, encoder, feature_value, column_name):
        """Transform a single feature while maintaining feature names"""
        # Create DataFrame with proper column name
        df = pd.DataFrame([feature_value], columns=[column_name])
        
        # Transform and return first row
        return encoder.transform(df)[0]
    
    def _compute_category_elasticity(self):
        """
        Compute elasticity by product category
        
        For each product category, we calculate an average elasticity
        which helps set appropriate price ranges.
        """
        # If the elasticity column doesn't exist, create it
        if 'elasticity' not in self.data.columns:
            self._calculate_elasticity()
            
        # Compute average elasticity by product type
        self.category_elasticity = {}
        
        for product_type in self.data['product_type'].unique():
            category_data = self.data[self.data['product_type'] == product_type]
            self.category_elasticity[product_type] = category_data['elasticity'].mean()
    
    def _compute_group_elasticity(self):
        """Compute elasticity factors for each product group using PPI and sales data"""
        elasticity = {}
        
        for group in self.data['product_group'].unique():
            subset = self.data[self.data['product_group'] == group]
            
            if len(subset) > 10:  # Need enough samples
                # Correlation between PPI and orders can indicate elasticity
                # Higher correlation (more negative) means more elastic
                
                # Standardize the variables for better correlation
                subset_std = subset.copy()
                subset_std['ppi_std'] = (subset_std['ppi'] - subset_std['ppi'].mean()) / subset_std['ppi'].std()
                subset_std['orders_std'] = (subset_std['number_of_orders'] - subset_std['number_of_orders'].mean()) / subset_std['number_of_orders'].std()
                
                # Calculate correlation between PPI and orders
                # If PPI is higher (overpriced), orders should be lower (negative correlation)
                corr = subset_std[['ppi_std', 'orders_std']].corr().iloc[0, 1]
                
                # Transform correlation to elasticity value
                # Stronger negative correlation = higher elasticity
                elasticity_value = 1.0 - corr * 0.5  # Range from ~0.5 to ~1.5
                elasticity[group] = max(0.5, min(1.5, elasticity_value))
            else:
                # Default elasticity for groups with few samples
                elasticity[group] = 1.0
                
        self.group_elasticity = elasticity
        
    def _cluster_products(self):
        """Group products into clusters for more coherent pricing strategies"""
        # Extract key features for clustering
        if 'actual_price' in self.data.columns:
            # Use actual_price from unnormalized data if available
            price_mapping = self.data.set_index('product_id')['actual_price'].to_dict()
            self.data['temp_price'] = self.data['product_id'].map(price_mapping)
            features = self.data[['temp_price', 'rating', 'number_of_orders', 'ppi']].copy()
        else:
            # Use PPI and other available features
            features = self.data[['ppi', 'rating', 'number_of_orders']].copy()
        
        # Handle missing values
        features = features.fillna(features.mean())
        
        # Scale features for clustering
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Determine optimal number of clusters (between 3 and 8)
        max_clusters = min(8, len(self.data) // 100 + 2)
        n_clusters = max(3, min(max_clusters, 8))
        
        # Cluster products
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed_value, n_init=10)
        self.data['cluster'] = kmeans.fit_predict(scaled_features)
        
        # Calculate cluster statistics for strategy guidance
        cluster_stats = {}
        for cluster_id in range(n_clusters):
            cluster_data = self.data[self.data['cluster'] == cluster_id]
            
            # Calculate key statistics
            stats = {
                'size': len(cluster_data),
                'avg_ppi': cluster_data['ppi'].mean(),
                'avg_rating': cluster_data['rating'].mean(),
                'avg_orders': cluster_data['number_of_orders'].mean(),
                'ppi_std': cluster_data['ppi'].std(),
            }
            
            # Add actual_price stats if available
            if 'temp_price' in self.data.columns:
                stats['avg_price'] = cluster_data['temp_price'].mean()
                stats['price_std'] = cluster_data['temp_price'].std()
            
            cluster_stats[cluster_id] = stats
        
        # Clean up temporary columns
        if 'temp_price' in self.data.columns:
            self.data = self.data.drop('temp_price', axis=1)
            
        return cluster_stats
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to start a new episode with enhanced initialization
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        # Reset RNG state if seed is provided
        if seed is not None:
            np.random.seed(seed)
        
        # Reset episode state
        self.episode_step = 0
        self.episode_profit = 0.0
        self.episode_conversion_count = 0
        self.episode_actions = []
        self.episode_profits = []
        self.done = False
        
        # Reset history tracking
        self.price_history = []
        self.demand_history = []
        self.profit_history = []
        self.conversion_history = []
        
        # Reset reward components
        self.reward_components = {}
        
        # Reset customer segmentation if available
        if hasattr(self, 'customer_segmentation') and self.customer_segmentation is not None:
            self.customer_segmentation.reset()
        
        # Reset pricing strategy if available
        if hasattr(self, 'pricing_strategy') and self.pricing_strategy is not None:
            self.pricing_strategy.reset()
        
        # Select product based on difficulty
        self._select_product_for_episode()
        
        # Get initial observation
        self.current_observation = self._get_observation()
        
        # Create info dictionary with initial state
        product = self.current_product_info
        reference_price = product['price']
        
        info = {
            'product_type': str(product['product_type']),
            'product_group': str(product['product_group']),
            'reference_price': float(reference_price),
            'elasticity': float(product['elasticity']),
            'difficulty': str(self.product_difficulty)
        }
        
        # Add pricing strategy information if available
        if hasattr(self, 'pricing_strategy') and self.pricing_strategy is not None:
            # Create market info
            market_info = {
                'competitive_intensity': 0.5,  # Default value
                'price_trend': 0.0,           # Default stable trend
                'current_price_ratio': 1.0     # Default at reference price
            }
            
            # Get initial recommendations
            initial_recommendations = self.pricing_strategy.get_price_recommendations(product, market_info)
            
            # Add to info
            info['initial_strategy_recommendation'] = {
                'price_ratio': float(initial_recommendations['weighted_recommendation']['price_ratio']),
                'confidence': float(initial_recommendations['weighted_recommendation']['total_confidence'])
            }
        
        return self.current_observation, info
    
    def _select_next_product(self, difficulty="medium"):
        """
        Select the next product to price based on the difficulty level
        
        Args:
            difficulty: The difficulty level ("easy", "medium", or "hard")
            
        Returns:
            product_info: Dictionary with product information
        """
        # Pop a product from available products
        if not hasattr(self, 'available_products') or len(self.available_products) == 0:
            self.available_products = list(range(len(self.data)))
            np.random.shuffle(self.available_products)
            
        if len(self.available_products) > 0:
            self.current_product_idx = self.available_products.pop()
        else:
            # Fallback if available_products is empty (should not happen)
            self.current_product_idx = np.random.randint(0, len(self.data))
              
        # Implement difficulty-based product selection for curriculum learning
        if difficulty == "easy":
            # Easy: Focus on inelastic products (elasticity < 0.8)
            filtered_indices = self.data[self.data['elasticity'] < 0.8].index.tolist()
            if len(filtered_indices) > 0:
                self.current_product_idx = np.random.choice(filtered_indices)
        elif difficulty == "medium":
            # Medium: Focus on moderate elasticity products (elasticity between 0.6 and 1.2)
            filtered_indices = self.data[(self.data['elasticity'] >= 0.6) & 
                                       (self.data['elasticity'] <= 1.2)].index.tolist()
            if len(filtered_indices) > 0:
                self.current_product_idx = np.random.choice(filtered_indices)
        # Hard difficulty uses any product (default behavior)
        
        # Get the selected product info
        product_info = self.data.iloc[self.current_product_idx].to_dict()
        
        # Ensure product has necessary fields with defaults
        if 'elasticity' not in product_info:
            product_info['elasticity'] = self._calculate_elasticity(product_info)
            
        if 'product_type' not in product_info:
            product_info['product_type'] = 'unknown'
            
        if 'product_group' not in product_info:
            product_info['product_group'] = 'unknown'
            
        if 'rating' not in product_info:
            product_info['rating'] = 3.0  # Default neutral rating
            
        if 'ppi' not in product_info:
            # Default PPI as a function of price and rating
            product_info['ppi'] = product_info.get('actual_price', 100) / (6 - product_info.get('rating', 3))
            
        # Set the current product info
        self.current_product_info = product_info
        return product_info
    
    def _get_product_info(self, idx):
        """Extract information for the current product"""
        product = self.data.iloc[idx].copy()
        
        # Get the product cluster
        cluster_id = product.get('cluster', 0)  # Default to cluster 0 if not available
        
        # Calculate the base elasticity for this product
        category_elasticity = self.category_elasticity.get(product['product_type'], 1.0)
        group_elasticity = self.group_elasticity.get(product['product_group'], category_elasticity)
        
        # Combine elasticity factors with rating influence
        # Higher rated products tend to be less elastic (more brand loyalty)
        rating_factor = 1.0 - (product['rating'] / 5.0) * 0.3
        
        # Calculate final elasticity value
        elasticity = (category_elasticity * 0.3 + group_elasticity * 0.7) * rating_factor
        
        # Create a product info dictionary with all relevant data
        product_info = {
            'product_id': product['product_id'],
            'product_type': product['product_type'],
            'product_group': product['product_group'],
            'estimated_cost': product['estimated_cost'],
            'elasticity': elasticity,
            'ppi': product['ppi'],  # Include PPI in product info
            'ppi_category': product['ppi_category'],
            'rating': product['rating'],
            'number_of_orders': product['number_of_orders'],
            'cluster': cluster_id
        }
        
        # Add actual_price if available
        if 'actual_price' in product:
            product_info['actual_price'] = product['actual_price']
        
        return product_info
    
    def _get_observation(self):
        """
        Get the current state observation
        
        Returns:
            Observation vector
        """
        # Get current product data
        product = self.current_product_info
        
        # Safe access to product attributes with defaults
        product_type = product.get('product_type', 'unknown')
        product_group = product.get('product_group', 'unknown')
        elasticity = product.get('elasticity', 1.0)  # Default to neutral elasticity
        rating = product.get('rating', 3.0)  # Default to average rating
        ppi = product.get('ppi', 1.0)  # Default to neutral PPI
        orders = product.get('number_of_orders', 50)  # Default to moderate sales
        pages = product.get('number_of_pages', 1)  # Default to single page
        
        # Initialize observation vector
        observation = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        current_idx = 0
        
        # 1. Product type one-hot encoding (10 slots)
        type_vec = np.zeros(10)
        try:
            type_encoded = self.type_encoder.transform([[product_type]])
            # Copy the encoded values, up to 10 slots
            type_vec[:min(10, type_encoded.shape[1])] = type_encoded[0, :min(10, type_encoded.shape[1])]
        except:
            # If encoding fails, use default (first position)
            type_vec[0] = 1.0
            
        observation[current_idx:current_idx+10] = type_vec
        current_idx += 10
        
        # 2. Product group one-hot encoding (5 slots)
        group_vec = np.zeros(5)
        try:
            group_encoded = self.group_encoder.transform([[product_group]])
            # Copy the encoded values, up to 5 slots
            group_vec[:min(5, group_encoded.shape[1])] = group_encoded[0, :min(5, group_encoded.shape[1])]
        except:
            # If encoding fails, use default (first position)
            group_vec[0] = 1.0
            
        observation[current_idx:current_idx+5] = group_vec
        current_idx += 5
        
        # 3. Numeric features (normalized)
        # Elasticity (normalized around 1.0)
        observation[current_idx] = (elasticity - 1.0) * 2.0  # Scale to roughly [-1, 1]
        current_idx += 1
        
        # Rating (normalized to [-1, 1])
        observation[current_idx] = (rating - 3.0) / 2.0  # Scale from 1-5 to [-1, 1]
        current_idx += 1
        
        # PPI (normalized around 1.0)
        observation[current_idx] = (ppi - 1.0) * 2.0  # Scale to roughly [-1, 1]
        current_idx += 1
        
        # Orders (log-normalized)
        observation[current_idx] = np.log1p(orders) / 10.0  # Log-scale and normalize
        current_idx += 1
        
        # 4. Price history (last 10 steps, normalized)
        price_history = np.zeros(10)
        if hasattr(self, 'price_history') and len(self.price_history) > 0:
            # Fill with available price history, most recent first
            for i in range(min(10, len(self.price_history))):
                if i < len(self.price_history):
                    price_history[i] = (self.price_history[-(i+1)] - 1.0) * 2.0  # Normalize around 1.0
                    
        observation[current_idx:current_idx+10] = price_history
        current_idx += 10
        
        # 5. Demand history (last 10 steps, normalized)
        demand_history = np.zeros(10)
        if hasattr(self, 'demand_history') and len(self.demand_history) > 0:
            # Fill with available demand history, most recent first
            for i in range(min(10, len(self.demand_history))):
                if i < len(self.demand_history):
                    demand_history[i] = self.demand_history[-(i+1)] / 2.0  # Normalize to [0, 1] range
                    
        observation[current_idx:current_idx+10] = demand_history
        current_idx += 10
        
        # 6. Conversion history (last 10 steps)
        conversion_history = np.zeros(10)
        if hasattr(self, 'conversion_history') and len(self.conversion_history) > 0:
            # Fill with available conversion history, most recent first
            for i in range(min(10, len(self.conversion_history))):
                if i < len(self.conversion_history):
                    conversion_history[i] = float(self.conversion_history[-(i+1)])
                    
        observation[current_idx:current_idx+10] = conversion_history
        current_idx += 10
        
        # 7. Padding with zeros for the rest of the observation space
        # This ensures the observation vector has the expected size
        remaining_space = self.observation_space.shape[0] - current_idx
        if remaining_space > 0:
            observation[current_idx:] = 0.0
        
        return observation
    
    def step(self, action):
        """
        Take a step in the environment by applying a pricing action
        
        Args:
            action: Discrete action index
            
        Returns:
            observation: Next state observation
            reward: Reward for the action
            terminated: Whether the episode is done
            truncated: Whether the episode is truncated
            info: Additional information
        """
        # Convert action to price ratio
        price_ratio = self._action_to_price_ratio(action)
        
        # Get product attributes
        product = self.current_product_info
        base_price = product.get('actual_price', product.get('price', 100.0))
        cost = product.get('cost', 0.6 * base_price)  # Default cost is 60% of price
        
        # Apply price ratio to get new price
        price = base_price * price_ratio
        self.current_price = price
        self.current_price_ratio = price_ratio
        
        # Calculate demand and conversion probability
        demand = self._calculate_demand(price_ratio)
        conversion_prob, conversion = self._calculate_conversion(price_ratio, demand)
        
        # Calculate profit
        profit = 0.0
        if conversion:
            revenue = price * demand
            total_cost = cost * demand
            profit = revenue - total_cost
        
        # Store result in history
        self.price_history.append(price)
        self.demand_history.append(demand)
        self.profit_history.append(profit)
        self.conversion_history.append(conversion)
        
        # Update state
        self._update_state(price_ratio, demand, conversion, profit)
        
        # Calculate reward
        reward = self._calculate_reward(self.state, price, self.next_state)
        
        # Implement dynamic adjustment for conversions/non-conversions based on profit
        # This is a key improvement to make the model focus on profit and not just conversions
        if conversion:
            # For conversions, add a profit-scaled bonus instead of a fixed bonus
            profit_bonus = min(profit * 0.5, 100)  # Cap the bonus at 100
            reward += profit_bonus
        else:
            # For non-conversions, penalty is already in _calculate_reward, but we can refine
            # We want to penalize price choices that lead to no conversion AND have no profit potential
            # If the price is strategic (e.g., high margin premium positioning), reduce penalty
            strategic_factor = 0.0
            if price_ratio > 1.1 and product.get('rating', 3.0) >= 4.0:
                # Premium product with premium pricing - more strategic
                strategic_factor = 0.6
            elif price_ratio < 0.9 and self.episode_step < 10:
                # Early penetration pricing - more strategic
                strategic_factor = 0.4
                
            # Apply strategic adjustment to non-conversion penalty
            strategic_adjustment = 10 * strategic_factor
            reward += strategic_adjustment  # Reduce penalty for strategic pricing
        
        # Update episode step
        self.episode_step += 1
        
        # Check if episode is done
        done = self.episode_step >= self.episode_length
        
        # Get observation
        observation = self._get_observation()
        
        # Return step result
        info = {
            'product_id': product.get('product_id', ''),
            'product_type': product.get('product_type', ''),
            'product_group': product.get('product_group', ''),
            'elasticity': product.get('elasticity', 1.0),
            'price': price,
            'price_ratio': price_ratio,
            'demand': demand,
            'conversion': conversion,
            'profit': profit,
            'rating': product.get('rating', 3.0),
            'ppi': product.get('ppi', 1.0),
            'reference_price': base_price
        }
        
        return observation, reward, done, False, info
    
    def _simulate_competitor_response(self, price_ratio):
        """
        Simulate competitor response to our pricing decision with enhanced
        category-specific behavior and adaptive competitive dynamics.
        
        Args:
            price_ratio: Our chosen price ratio
            
        Returns:
            Dict with response information including strength and strategy
        """
        # Get product info
        product = self.current_product_info
        product_type = product.get('product_type', 'unknown').lower()
        elasticity = product.get('elasticity', 1.0)
        rating = product.get('rating', 3.0)
        
        # Get cost information
        reference_price = product.get('price', 100.0)
        cost = product.get('cost', 0.6 * reference_price)
        cost_ratio = cost / reference_price
        
        # Calculate profit margin
        margin_ratio = (price_ratio - cost_ratio) / cost_ratio if cost_ratio > 0 else 0
        
        # Initialize response data
        response_data = {
            'response_strength': 0.0,
            'price_change': 0.0,
            'strategy': 'neutral',
            'response_speed': 'normal',
            'competitive_intensity': 0.0
        }
        
        # Define category-specific competitive behaviors
        category_behaviors = {
            'electronics': {
                'intensity': 0.7,  # Very competitive market
                'response_speed': 1.3,  # Fast to respond
                'price_sensitivity': 1.2,  # Highly sensitive
                'thresholds': {'low': 0.85, 'moderate': 0.95, 'high': 1.05}
            },
            'luxury': {
                'intensity': 0.3,  # Less price-competitive
                'response_speed': 0.7,  # Slow to respond
                'price_sensitivity': 0.6,  # Less sensitive
                'thresholds': {'low': 0.7, 'moderate': 0.85, 'high': 1.15}
            },
            'premium': {
                'intensity': 0.4,  # Moderately competitive
                'response_speed': 0.8,  # Somewhat slow
                'price_sensitivity': 0.7,  # Less sensitive
                'thresholds': {'low': 0.75, 'moderate': 0.9, 'high': 1.1}
            },
            'commodity': {
                'intensity': 0.8,  # Extremely competitive
                'response_speed': 1.4,  # Very fast to respond
                'price_sensitivity': 1.4,  # Very sensitive
                'thresholds': {'low': 0.9, 'moderate': 0.98, 'high': 1.02}
            },
            'basic': {
                'intensity': 0.75,  # Highly competitive
                'response_speed': 1.2,  # Fast to respond
                'price_sensitivity': 1.3,  # Very sensitive
                'thresholds': {'low': 0.88, 'moderate': 0.96, 'high': 1.04}
            },
            'books': {
                'intensity': 0.6,  # Moderately competitive
                'response_speed': 0.9,  # Average response
                'price_sensitivity': 1.1,  # Sensitive
                'thresholds': {'low': 0.82, 'moderate': 0.92, 'high': 1.08}
            },
            'clothing': {
                'intensity': 0.65,  # Competitive
                'response_speed': 1.1,  # Somewhat fast
                'price_sensitivity': 1.0,  # Average sensitivity
                'thresholds': {'low': 0.8, 'moderate': 0.9, 'high': 1.1}
            },
            'home': {
                'intensity': 0.55,  # Moderately competitive
                'response_speed': 0.9,  # Average response
                'price_sensitivity': 0.9,  # Average sensitivity
                'thresholds': {'low': 0.78, 'moderate': 0.9, 'high': 1.1}
            },
            'beauty': {
                'intensity': 0.5,  # Balanced competition
                'response_speed': 1.0,  # Average response
                'price_sensitivity': 0.85,  # Below average sensitivity
                'thresholds': {'low': 0.8, 'moderate': 0.9, 'high': 1.1}
            },
            'toys': {
                'intensity': 0.6,  # Moderately competitive
                'response_speed': 1.0,  # Average response
                'price_sensitivity': 1.0,  # Average sensitivity
                'thresholds': {'low': 0.8, 'moderate': 0.9, 'high': 1.1}
            },
            'jewelry': {
                'intensity': 0.4,  # Less competitive
                'response_speed': 0.8,  # Slow response
                'price_sensitivity': 0.7,  # Below average sensitivity
                'thresholds': {'low': 0.75, 'moderate': 0.9, 'high': 1.1}
            },
            'garden': {
                'intensity': 0.5,  # Balanced competition
                'response_speed': 0.9,  # Average response
                'price_sensitivity': 0.9,  # Average sensitivity 
                'thresholds': {'low': 0.8, 'moderate': 0.9, 'high': 1.1}
            },
            'automotive': {
                'intensity': 0.6,  # Moderately competitive
                'response_speed': 0.95,  # Average response
                'price_sensitivity': 0.95,  # Average sensitivity
                'thresholds': {'low': 0.85, 'moderate': 0.95, 'high': 1.05}
            },
            'office': {
                'intensity': 0.65,  # Competitive
                'response_speed': 1.0,  # Average response
                'price_sensitivity': 1.05,  # Above average sensitivity
                'thresholds': {'low': 0.85, 'moderate': 0.95, 'high': 1.05}
            },
            'pets': {
                'intensity': 0.55,  # Moderately competitive
                'response_speed': 0.95,  # Average response
                'price_sensitivity': 0.9,  # Average sensitivity
                'thresholds': {'low': 0.8, 'moderate': 0.9, 'high': 1.1}
            },
            'sports': {
                'intensity': 0.6,  # Moderately competitive
                'response_speed': 1.0,  # Average response
                'price_sensitivity': 1.0,  # Average sensitivity
                'thresholds': {'low': 0.8, 'moderate': 0.9, 'high': 1.1}
            }
        }
        
        # Get category behavior or use default
        if product_type in category_behaviors:
            behavior = category_behaviors[product_type]
        else:
            # Default behavior
            behavior = {
                'intensity': 0.5,  # Medium competitive intensity
                'response_speed': 1.0,  # Normal response speed
                'price_sensitivity': 1.0,  # Average price sensitivity
                'thresholds': {'low': 0.8, 'moderate': 0.9, 'high': 1.1}
            }
        
        # Extract thresholds
        low_price_threshold = behavior['thresholds']['low']
        moderate_price_threshold = behavior['thresholds']['moderate']
        high_price_threshold = behavior['thresholds']['high']
        
        # Base response strength - competitive intensity affects baseline
        base_response = 0.05 * behavior['intensity']
        response_data['competitive_intensity'] = behavior['intensity']
        
        # Apply elasticity factor - more elastic markets see stronger responses
        elasticity_factor = 1.0 + (elasticity - 1.0) * 0.5  # Adjust impact of elasticity
        
        # Determine response based on price thresholds and category behavior
        if price_ratio < low_price_threshold:
            # Aggressive undercutting - strong response
            response_strength = base_response + (0.3 + (low_price_threshold - price_ratio) * 1.5) * behavior['price_sensitivity']
            response_data['strategy'] = 'match_aggressive'
            response_data['price_change'] = -0.05 - (low_price_threshold - price_ratio) * 0.6  # Follow downward but not as much
        
        elif price_ratio < moderate_price_threshold:
            # Moderate undercutting - proportional response
            response_strength = base_response + (0.1 + (moderate_price_threshold - price_ratio) * 1.5) * behavior['price_sensitivity']
            response_data['strategy'] = 'match_moderate'
            response_data['price_change'] = -0.02 - (moderate_price_threshold - price_ratio) * 0.4  # Small decrease
        
        elif price_ratio <= high_price_threshold:
            # Near market-level pricing - minimal response
            response_strength = base_response + 0.03 * behavior['price_sensitivity']
            response_data['strategy'] = 'neutral'
            response_data['price_change'] = 0.0  # No change
        
        else:
            # Premium pricing - opportunistic response
            premium_amount = price_ratio - high_price_threshold
            
            # Stronger response for products that can't command premium
            if product_type in ['commodity', 'basic'] or rating < 3.5:
                response_strength = base_response + (0.05 + premium_amount * 0.5) * behavior['price_sensitivity']
                response_data['strategy'] = 'undercut_premium'
                response_data['price_change'] = -0.03 - premium_amount * 0.3  # Undercut to gain market share
            else:
                # Premium products with good ratings see minimal response
                response_strength = base_response + 0.01 * behavior['price_sensitivity']
                response_data['strategy'] = 'maintain'
                response_data['price_change'] = 0.0  # No change for premium products
        
        # Apply elasticity factor
        response_strength *= elasticity_factor
        
        # Margin-based adjustments
        if 0.15 <= margin_ratio <= 0.35:
            # Reasonable margin - less threatening to competitors
            response_strength *= 0.7
        elif margin_ratio > 0.4:
            # Very high margin - might attract competition
            response_strength *= 1.2
            response_data['price_change'] *= 1.3  # Stronger price change for high margins
        
        # Add market conditions - seasonal effects, trends, etc.
        market_condition = np.random.choice(['normal', 'competitive', 'stable'], p=[0.7, 0.2, 0.1])
        if market_condition == 'competitive':
            response_strength *= 1.3
            response_data['price_change'] *= 1.2
        elif market_condition == 'stable':
            response_strength *= 0.7
            response_data['price_change'] *= 0.6
        
        # Add some randomness with reduced variance
        random_factor = np.random.normal(0, 0.03)
        response_strength += random_factor
        
        # Ensure response is in valid range
        response_strength = max(0.0, min(0.8, response_strength))  # Increased maximum from 0.6 to 0.8
        response_data['response_strength'] = response_strength
        
        # Assign response speed based on category behavior
        if behavior['response_speed'] > 1.1:
            response_data['response_speed'] = 'fast'
        elif behavior['response_speed'] < 0.9:
            response_data['response_speed'] = 'slow'
        else:
            response_data['response_speed'] = 'normal'
        
        return response_data
    
    def _calculate_demand(self, price_ratio, competitor_response=None):
        """
        Calculate demand for the product at the current price ratio with enhanced
        profitability focus and improved competitor response handling.
        
        Args:
            price_ratio: Ratio of current price to reference price
            competitor_response: Optional dict with competitor response data
        
        Returns:
            demand: Normalized demand (0.0 to 1.0)
        """
        # Get product info
        product = self.current_product_info
        elasticity = product.get('elasticity', 1.0)
        rating = product.get('rating', 3.0)
        product_type = product.get('product_type', 'standard')
        cost = product.get('cost', 60.0)
        reference_price = product.get('price', 100.0)
        
        # Enhanced profitability check
        price = reference_price * price_ratio
        if price < cost:
            # Stronger penalty for unprofitable pricing
            below_cost_penalty = 0.9  # Reduce demand by 90% (up from 80%)
            base_demand = max(0.01, 0.1 * np.power(price_ratio, -elasticity))  # Further reduced base demand
            return base_demand
        
        # Enhanced profit margin check
        margin_ratio = (price - cost) / price
        
        # Calculate base demand using enhanced formula
        # Higher elasticity = more sensitive to price
        # Enhanced to include quality multiplier based on rating
        quality_multiplier = 0.7 + (rating / 5.0) * 0.6  # Scale from 0.7 to 1.3 based on rating
        
        # Apply product type specific multipliers
        product_type_multiplier = 1.0
        if product_type.lower() in ['luxury', 'premium']:
            if rating >= 4.0 and price_ratio > 1.0:
                # Premium products with good ratings see higher demand at premium prices
                product_type_multiplier = 1.15
            elif price_ratio < 0.9:
                # Luxury products see lower demand at discount prices (too good to be true effect)
                product_type_multiplier = 0.85
        elif product_type.lower() in ['commodity', 'basic']:
            if price_ratio < 0.95:
                # Basic products see higher demand at discount prices
                product_type_multiplier = 1.2
            elif price_ratio > 1.05:
                # Basic products see much lower demand at premium prices
                product_type_multiplier = 0.7
        
        # Enhanced demand formula with price elasticity and quality
        base_demand = (1.0 + (1.0 - price_ratio) * elasticity) * quality_multiplier * product_type_multiplier
        
        # Apply customer segmentation demand modifier if available
        if hasattr(self, 'customer_segmentation'):
            segment_modifier = self.customer_segmentation.get_demand_modifier(
                product, price_ratio, margin_ratio)
            base_demand *= segment_modifier
        
        # Apply competitor response with enhanced handling
        if competitor_response is not None:
            if isinstance(competitor_response, dict):
                # Enhanced competitive impact calculation based on price change and response strength
                if 'price_change' in competitor_response:
                    competitor_price_change = competitor_response['price_change']
                    response_strength = competitor_response.get('response_strength', 0.5)
                    response_strategy = competitor_response.get('strategy', 'neutral')
                    
                    # Different impact based on response strategy
                    if competitor_price_change < 0:
                        # Competitors lowering prices - impact based on strategy
                        if response_strategy in ['match_aggressive', 'undercut_premium']:
                            # Strong competitive response - larger impact
                            competitive_impact = 1.0 + 3.0 * competitor_price_change * response_strength
                            base_demand *= max(0.3, competitive_impact)  # Stronger minimum impact
                        else:
                            # Moderate response - regular impact
                            competitive_impact = 1.0 + 2.5 * competitor_price_change * response_strength
                            base_demand *= max(0.4, competitive_impact)
                    elif competitor_price_change > 0:
                        # Competitors raising prices - boost for us
                        competitive_boost = 1.0 + 2.0 * competitor_price_change * response_strength
                        base_demand *= min(1.8, competitive_boost)  # Increased maximum boost
            elif isinstance(competitor_response, float):
                # Legacy support for float response strength
                if competitor_response < 0:
                    base_demand *= max(0.4, 1.0 + 2.0 * competitor_response)
        
        # Add random noise to demand
        noise_level = self.demand_noise if hasattr(self, 'demand_noise') else 0.1
        noise = 1.0 + noise_level * (2.0 * np.random.random() - 1.0)
        demand = base_demand * noise
        
        # Enhanced demand bounds
        demand = max(0.01, min(1.8, demand))  # Reduced maximum demand from 2.0 to 1.8
        
        # Store calculated demand for later use
        self.calculated_demand = demand
        
        return demand
    
    def _calculate_conversion_probability(self, price_ratio, demand):
        """
        Calculate the probability of conversion (purchase) based on price and demand
        
        Args:
            price_ratio: The ratio of price to reference price
            demand: The estimated demand
            
        Returns:
            conversion_prob: The probability of conversion
            conversion: Whether a conversion occurred (boolean)
        """
        product = self.current_product_info
        
        # Use customer segmentation to get more accurate conversion probabilities
        if hasattr(self, 'customer_segmentation'):
            segment_conversions = self.customer_segmentation.calculate_segment_conversion_probabilities(
                price_ratio, product
            )
            conversion_prob = segment_conversions['weighted_conversion']
            
            # Boost conversion probability for reasonable price ratios (0.8-1.1)
            if 0.8 <= price_ratio <= 1.1:
                conversion_prob *= 1.2  # 20% boost for well-priced products
            
            # Store segment-specific conversions for information
            self.segment_conversion_probs = segment_conversions['segments']
            
            # Add time-based adjustments if enabled
            if self.use_time_factors:
                # Determine if weekend and time of day (simplified)
                is_weekend = (self.episode_step % 7) >= 5  # Days 5,6 are weekend
                hour = (self.episode_step % 24)
                
                if hour < 5:
                    time_of_day = 'night'
                elif hour < 12:
                    time_of_day = 'morning'
                elif hour < 18:
                    time_of_day = 'afternoon'
                else:
                    time_of_day = 'evening'
                
                time_factor = self.customer_segmentation.get_time_factor(is_weekend, time_of_day)
                conversion_prob *= time_factor

# Original conversion probability calculation as fallback
        rating = product.get('rating', 3.0)
        elasticity = product.get('elasticity', 1.0)
        
        # Base conversion probability based on normalized demand
        base_conversion_prob = max(0.05, min(0.30, demand * 0.2))  # Increased base conversion range
        
        # Adjust for price elasticity (more elastic products are more sensitive to price)
        elasticity_factor = 1.0
        if elasticity > 0:
            elasticity_factor = 1.0 + (1.0 - price_ratio) * elasticity * 0.5
        
        # Adjust for product quality
        quality_factor = 0.8 + (rating / 5.0) * 0.6  # 0.8 to 1.4 based on rating (improved from 0.7-1.3)
        
        # Calculate final conversion probability
        conversion_prob = base_conversion_prob * elasticity_factor * quality_factor
        
        # Ensure probability is in valid range
        conversion_prob = max(0.05, min(0.95, conversion_prob))  # Increased minimum probability
        
        # Determine if conversion happens
        conversion = np.random.random() < conversion_prob
        
        return conversion_prob, conversion
    
    def _calculate_reward(self, state, action, next_state):
        """
        Calculate the reward for the current action with a stronger focus on profitability
        while maintaining market-beating performance and price reasonableness.
        """
        price = action
        cost = state['cost']
        market_price = state['market_price']
        
        # Early return for below-cost pricing
        if price <= cost:
            return -100
        
        # Calculate margins
        margin = (price - cost) / price
        market_margin = (market_price - cost) / market_price
        
        # Base reward starts at zero
        reward = 0
        
        # PROFIT COMPONENT - SIGNIFICANTLY STRENGTHENED
        # Direct profit reward is now the primary driver
        # This encourages finding the optimal price point for maximum profit
        profit = next_state.get('profit', 0)
        reward += profit * 2.5  # Increased multiplier from previous implicit value
        
        # Market performance component (balanced)
        margin_difference = margin - market_margin
        if margin_difference > 0:
            # Reward for beating market margins, but less exponential
            reward += 100 * margin_difference  # Less exponential scaling
        else:
            # More moderate penalty for underperforming
            reward -= 50 * abs(margin_difference)  # Reduced penalty
        
        # Price reasonableness component (maintained)
        price_diff_pct = abs(price - market_price) / market_price
        if price_diff_pct <= 0.20:
            reward += 30 * (1 - price_diff_pct)  # Reasonable reward for reasonable prices
        else:
            reward -= 80 * price_diff_pct  # Significant but not overwhelming penalty
        
        # Profitability bonus (enhanced but more gradual)
        if margin >= 0.40:
            reward += 90  # Very strong bonus for high margins
        elif margin >= 0.30:
            reward += 60  # Strong bonus for good margins
        elif margin >= 0.20:
            reward += 30  # Moderate bonus for acceptable margins
        elif margin >= 0.15:
            reward += 10  # Small bonus for minimum acceptable margins
        
        # Conversion penalty (REDUCED SIGNIFICANTLY)
        # This is a key change - we no longer heavily penalize non-conversions
        # Instead, we rely on the profit component to implicitly handle this
        conversion = next_state.get('conversion', False)
        if not conversion:
            reward -= 20  # Reduced from previous -200, creating more balanced incentives
        
        # Market competitiveness bonus (rebalanced)
        if price < market_price and margin >= 0.20:
            reward += 40  # Significant but not dominant bonus
        
        # Strategic pricing bonus
        product_type = state.get('product_type', 'standard')
        rating = state.get('rating', 3.0)
        
        # Premium product pricing strategy
        if product_type.lower() in ['luxury', 'premium']:
            if price > market_price and margin >= 0.30 and rating >= 4.0:
                reward += 30  # Bonus for premium products with premium pricing
        
        # Value product pricing strategy
        elif product_type.lower() in ['basic', 'economy']:
            if price < market_price and price > cost * 1.25:
                reward += 30  # Bonus for value products with competitive pricing
        
        return reward
    
    def _calculate_ppi_reward(self, effective_ppi, original_ppi, profit, rating, elasticity):
        """
        Calculate reward based on PPI positioning with stronger emphasis on profitability.
        
        Args:
            effective_ppi: Effective PPI after pricing decision
            original_ppi: Original PPI before pricing decision
            profit: Profit from this pricing decision
            rating: Product rating
            elasticity: Price elasticity of demand
            
        Returns:
            PPI-based reward component
        """
        
        # Base reward now starts with a profit factor regardless of PPI category
        # This shifts incentives towards maximizing profit rather than landing in a specific range
        base_profit_factor = 0.5  # Start with a base profit multiplier
        
        # PPI categories with adjusted rewards
        if effective_ppi <= 0.8:  # Significantly underpriced
            # Reduced penalty for underpricing compared to before
            quality_factor = rating / 5.0
            underpricing_penalty = -profit * 0.15 * (0.8 - effective_ppi) * quality_factor
            # But if original PPI was very high, reward moving toward better pricing
            if original_ppi > 1.3:
                correction_bonus = profit * 0.4 * min(1.0, (original_ppi - effective_ppi))
                return (profit * base_profit_factor) + underpricing_penalty + correction_bonus
            return (profit * base_profit_factor) + underpricing_penalty
            
        elif effective_ppi <= 0.95:  # Slightly underpriced
            # Increased bonus for slightly underpriced items, which often have high volume
            return profit * (base_profit_factor + 0.2)
            
        elif effective_ppi <= 1.1:  # Optimally priced
            # Strong bonus for optimal pricing, but less dominant than before
            optimal_bonus = profit * 0.3
            # Extra bonus for moving into optimal range from outside
            if original_ppi <= 0.8 or original_ppi > 1.3:
                optimal_bonus += profit * 0.2
            return (profit * base_profit_factor) + optimal_bonus
            
        elif effective_ppi <= 1.3:  # Moderately overpriced
            # Less penalty for moderate overpricing, especially for high-rated products
            quality_factor = rating / 5.0
            # The higher the quality, the more we reward premium pricing
            return profit * (base_profit_factor + 0.1 * quality_factor)
            
        else:  # Significantly overpriced
            # Elasticity-adjusted penalty for significant overpricing
            # The higher the elasticity, the more customers are price-sensitive
            overpricing_penalty = -profit * 0.25 * (effective_ppi - 1.3) * elasticity
            # But if original PPI was even higher, reward moving toward better pricing
            if original_ppi > effective_ppi:
                correction_bonus = profit * 0.3 * min(1.0, (original_ppi - effective_ppi))
                return (profit * base_profit_factor) + overpricing_penalty + correction_bonus
            return (profit * base_profit_factor) + overpricing_penalty
    
    def _calculate_strategy_reward(self, effective_category, original_category, 
                                 price_ratio, profit, rating, elasticity):
        """Calculate reward for specific pricing strategies"""
        # Initialize strategy reward
        strategy_reward = 0.0
        
        # Reward strategy 1: Optimal pricing for high-rated products
        if rating >= 4.0 and effective_category == "Optimally Priced":
            strategy_reward += profit * 0.15
            
        # Reward strategy 2: Discount high-elasticity products that are overpriced
        if elasticity > 1.1 and original_category in ["Moderately Overpriced", "Significantly Overpriced"] and price_ratio < 1.0:
            strategy_reward += profit * 0.2 * (1.0 - price_ratio)
            
        # Reward strategy 3: Premium pricing for low-elasticity, high-quality products
        if elasticity < 0.9 and rating > 4.0 and effective_category in ["Slightly Underpriced", "Optimally Priced"]:
            # Reward moving toward optimal from underpriced
            if original_category == "Significantly Underpriced" and price_ratio > 1.0:
                strategy_reward += profit * 0.18 * price_ratio
                
        # Reward strategy 4: Aggressive discounting for significantly overpriced, low-rated products
        if original_category == "Significantly Overpriced" and rating < 3.5 and price_ratio < 0.9:
            strategy_reward += profit * 0.25 * (1.0 - price_ratio)
            
        return strategy_reward
    
    def _calculate_entropy_bonus(self, price_ratio, profit):
        """Calculate entropy bonus to encourage price diversity"""
        # If we don't have enough history yet, return default bonus
        if len(self.price_history) < 10:
            return profit * 0.2  # 20% bonus on profit to encourage exploration
        
        # Get the last N price ratios
        recent_ratios = self.price_history[-10:]
        
        # Convert to price ratios
        recent_price_ratios = []
        for price in recent_ratios:
            product_idx = max(0, self.current_product_idx - len(recent_ratios))
            
            # Get reference price, with fallback if actual_price is not available
            if 'actual_price' in self.data.columns and not pd.isna(self.data.iloc[product_idx]['actual_price']):
                actual_price = self.data.iloc[product_idx]['actual_price']
            else:
                # Use current price as a fallback
                actual_price = price
            
            if actual_price > 0:
                recent_price_ratios.append(price / actual_price)
            else:
                recent_price_ratios.append(1.0)  # Default to 1.0 if actual price is 0
        
        # Discretize price ratios into bins
        bins = np.linspace(0.5, 1.5, 11)  # 10 bins covering the pricing range
        binned_ratios = np.digitize(recent_price_ratios, bins)
        
        # Count occurrences of each bin
        counts = np.bincount(binned_ratios, minlength=len(bins))
        probs = counts / len(recent_price_ratios)
        
        # Calculate entropy (-sum(p * log(p)))
        entropy = -np.sum([p * np.log(p) if p > 0 else 0 for p in probs])
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(len(bins))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Calculate novelty - reward setting prices in less-used bins
        current_bin = np.digitize([price_ratio], bins)[0]
        bin_prob = probs[current_bin] if current_bin < len(probs) else 0
        novelty_factor = 1.0 - bin_prob if bin_prob > 0 else 1.0
        
        # Calculate bonus based on entropy and novelty - strengthened
        entropy_bonus = (profit * 0.2 * normalized_entropy +  
                        profit * 0.15 * novelty_factor)  # Increased novelty factor
        
        # Add extra bonus for low-frequency bins
        if bin_prob < 0.1:  # Rarely used bins get an extra bonus
            entropy_bonus += profit * 0.1  # Additional 10% bonus
        
        return entropy_bonus
    
    def _get_optimal_price_ratio(self):
        """
        Calculate the optimal price ratio based on product attributes, elasticity,
        and market conditions with enhanced focus on profitability and reasonable pricing.
        
        Returns a price ratio that optimizes profit while maintaining market competitiveness.
        """
        product = self.current_product_info
        elasticity = product.get('elasticity', 1.0)
        rating = product.get('rating', 3.0)
        product_type = product.get('product_type', 'standard')
        cost = product.get('cost', 60.0)
        market_price = product.get('price', 100.0)
        
        # Calculate the cost-to-price ratio
        cost_ratio = cost / market_price if market_price > 0 else 0.6
        
        # Base optimal markup based on elasticity - ADJUSTED FOR HIGHER PROFITABILITY
        # For inelastic products (e < 1), we can price higher
        # For elastic products (e > 1), we need to be more price competitive
        # But we ensure all products maintain reasonable profitability
        if elasticity < 0.8:  # Highly inelastic
            base_markup = 1.18  # Increased from 1.15 (18% above market price)
        elif elasticity < 1.0:  # Moderately inelastic
            base_markup = 1.10  # Increased from 1.08 (10% above market price)
        elif elasticity < 1.2:  # Moderately elastic
            base_markup = 0.97  # Increased from 0.95 (3% below market price)
        else:  # Highly elastic
            base_markup = 0.94  # Increased from 0.92 (6% below market price)
        
        # Enhanced rating adjustment - higher quality products can command higher prices
        rating_adjustment = 0.0
        if rating >= 4.5:  # Excellent rating
            rating_adjustment = 0.15  # Increased from 0.12 (up to 15% premium)
        elif rating >= 4.0:  # Very good rating
            rating_adjustment = 0.10  # Increased from 0.08 (up to 10% premium)
        elif rating >= 3.5:  # Good rating
            rating_adjustment = 0.05  # Increased from 0.04 (up to 5% premium)
        elif rating <= 2.5:  # Poor rating
            rating_adjustment = -0.05  # Reduced discount from -0.08 (5% discount needed)
        
        # Product type adjustment
        type_adjustment = 0.0
        if product_type.lower() in ['luxury', 'premium']:
            type_adjustment = 0.12  # Increased from 0.10 (premium products have higher margins)
        elif product_type.lower() in ['basic', 'economy']:
            type_adjustment = -0.03  # Reduced discount from -0.05 (less discount on basic products)
        
        # Enhanced competitiveness adjustment based on cost ratio
        # Higher margin potential (lower cost ratio) allows more pricing flexibility
        margin_potential = 1.0 - cost_ratio
        competitiveness_adjustment = 0.0
        
        if margin_potential > 0.5:  # Very high margin potential
            competitiveness_adjustment = 0.10  # Increased from 0.08 (can price more aggressively)
        elif margin_potential > 0.3:  # Good margin potential
            competitiveness_adjustment = 0.07  # Increased from 0.05 (moderate pricing premium)
        else:  # Limited margin potential
            competitiveness_adjustment = -0.03  # Reduced discount from -0.05 (need more competitive pricing)
        
        # Calculate total adjustment
        total_adjustment = rating_adjustment + type_adjustment + competitiveness_adjustment
        
        # Apply adjustments to base markup, with constraints
        optimal_ratio = base_markup + total_adjustment
        
        # INCREASED MINIMUM PROFITABILITY - at least 25% margin over cost (increased from 20%)
        min_profitable_ratio = (cost * 1.33) / market_price
        
        # Cap maximum price at 25% above market to maintain reasonableness (increased from 20%)
        max_ratio = 1.25
        
        # Apply constraints
        optimal_ratio = max(min_profitable_ratio, min(optimal_ratio, max_ratio))
        
        # Additional boost for products with high margin potential and good ratings
        if margin_potential > 0.4 and rating >= 4.0:
            # Further optimize to beat market pricing while maintaining profitability
            optimal_ratio = min(optimal_ratio * 1.08, max_ratio)  # Increased from 1.05
        
        return optimal_ratio
    
    def get_optimal_price(self):
        """
        Calculate the optimal price for the current product based on its attributes,
        elasticity, and market conditions with enhanced focus on profitability.
        
        Returns:
            optimal_price: The calculated optimal price that maximizes profit
        """
        product = self.current_product_info
        market_price = product.get('price', 100.0)
        cost = product.get('cost', 60.0)
        elasticity = product.get('elasticity', 1.0)
        rating = product.get('rating', 3.0)
        product_type = product.get('product_type', 'standard')
        
        # Get optimal price ratio
        optimal_ratio = self._get_optimal_price_ratio()
        
        # Calculate optimal price
        optimal_price = market_price * optimal_ratio
        
        # Ensure minimum profitability - at least 33% margin over cost (increased from 25%)
        min_price = cost * 1.5  # Increased minimum markup for better profitability
        
        # Apply strategic adjustments based on product attributes
        if rating >= 4.5 and product_type.lower() in ['luxury', 'premium']:
            # Premium products with excellent ratings can command higher prices
            optimal_price = max(optimal_price, market_price * 1.10)  # Increased from 1.05
        elif rating >= 4.0:
            # Good quality products should maintain good margins
            optimal_price = max(optimal_price, cost * 1.50)  # Increased from 1.35
        elif product_type.lower() in ['basic', 'economy']:
            # Basic products need to be competitively priced but still profitable
            optimal_price = min(optimal_price, market_price * 0.97)  # Less discount from 0.95
            optimal_price = max(optimal_price, cost * 1.35)  # Increased from 1.25
        
        # Elasticity-based adjustments
        if elasticity < 0.8:  # Inelastic products
            # For inelastic products, we can price higher
            optimal_price = max(optimal_price, market_price * 1.08)  # Increased from 1.05
        elif elasticity > 1.5:  # Highly elastic products
            # For elastic products, we need to be more price competitive
            optimal_price = min(optimal_price, market_price * 0.96)  # Less discount from 0.95
            optimal_price = max(optimal_price, min_price)
        
        # Final bounds check
        optimal_price = max(min_price, optimal_price)
        
        # Ensure price is within reasonable bounds of market price
        max_price = market_price * 1.25  # Increased from 1.2
        optimal_price = min(optimal_price, max_price)
        
        return optimal_price
    
    def render(self):
        """
        Render the environment state
        
        Only supported in 'human' mode, which prints information
        about the current state to the console.
        """
        if self.render_mode != "human":
            return
            
        product = self.current_product_info
        print(f"\nProduct: {product['name'][:30]}... ({product['product_type']})")
        print(f"Actual Price: ${product['actual_price']:.2f} | Competitor: ${product['competitor_price']:.2f}")
        print(f"Current Price: ${self.current_price:.2f} (Ratio: {self.current_price_ratio:.2f})")
        print(f"Optimal Price Ratio: {self._get_optimal_price_ratio():.2f}")
        print(f"Rating: {product['rating']:.1f} | Elasticity: {product['elasticity']:.2f}")
        print(f"Last Demand: {self.demand_history[-1]:.1f} | Profit: ${self.profit_history[-1]:.2f}")
        print(f"Step: {self.episode_step}/{self.episode_length}")
        
    def close(self):
        """Clean up resources"""
        pass

    def _action_to_price_ratio(self, action):
        """
        Convert a discrete action to a price ratio
        
        Args:
            action: Discrete action index (0 to n_price_levels-1)
            
        Returns:
            price_ratio: The price ratio corresponding to the action
        """
        # Convert action index to price ratio in the range [min_price_ratio, max_price_ratio]
        price_ratio = self.min_price_ratio + action * self.price_ratio_step
        
        # Ensure the price ratio is within bounds
        price_ratio = max(self.min_price_ratio, min(self.max_price_ratio, price_ratio))
        
        return price_ratio

    def _normalize_value(self, value, min_val, max_val):
        """
        Normalize a value to the range [-1, 1]
        
        Args:
            value: The value to normalize
            min_val: The minimum expected value
            max_val: The maximum expected value
            
        Returns:
            Normalized value between -1 and 1
        """
        # Clip value to range
        value = max(min_val, min(max_val, value))
        
        # Normalize to [0, 1]
        normalized = (value - min_val) / (max_val - min_val)
        
        # Scale to [-1, 1]
        return normalized * 2 - 1

    def _calculate_exploration_bonus(self, price_ratio):
        """
        Calculate exploration bonus for trying different price levels
        
        Args:
            price_ratio: The price ratio chosen
            
        Returns:
            Exploration bonus
        """
        # If we don't have enough price history, provide base encouragement
        if len(self.price_history) < 5:
            return 100  # Base bonus when starting
        
        # Calculate price history statistics
        recent_prices = np.array(self.price_history[-10:]) if len(self.price_history) >= 10 else np.array(self.price_history)
        
        # Get reference price (actual or list)
        reference_price = 0
        if 'actual_price' in self.current_product_info and self.current_product_info['actual_price'] > 0:
            reference_price = self.current_product_info['actual_price']
        
        # Calculate recent price ratios if we have reference price
        if reference_price > 0:
            recent_ratios = [price/reference_price for price in recent_prices]
            mean_ratio = np.mean(recent_ratios)
            min_ratio = np.min(recent_ratios) if len(recent_ratios) > 0 else 1.0
            
            # Encourage diversity - higher bonus for ratios farther from recent mean
            diversity_bonus = 500 * abs(price_ratio - mean_ratio)
            
            # Special bonus for trying lower prices that haven't been tried before
            if price_ratio < min_ratio and price_ratio < 1.0:
                # The lower the price compared to previous minimum, the higher the bonus
                # Also higher for high elasticity products
                elasticity = self.current_product_info.get('elasticity', 1.0)
                discount_bonus = 1000 * (min_ratio - price_ratio) * min(2.0, elasticity)
                return diversity_bonus + discount_bonus
            
            # Bonus for exploring uncharted price levels
            is_novel = True
            for r in recent_ratios:
                if abs(r - price_ratio) < 0.05:  # Within 5% of a previously tried price
                    is_novel = False
                    break
            
            if is_novel:
                return diversity_bonus + 300  # Extra bonus for novel prices
            
            return diversity_bonus
        
        # Default exploration bonus if we can't calculate ratios
        return 100

    def get_performance_metrics(self):
        """
        Calculate performance metrics for the current episode
        
        Returns:
            metrics: Dictionary of performance metrics
        """
        # Initialize metrics dictionary
        metrics = {
            'total_profit': 0.0,
            'average_profit': 0.0,
            'conversion_rate': 0.0,
            'average_price_ratio': 0.0,
            'price_optimality': 0.0,
            'reward_components': {}
        }
        
        # Calculate total and average profit
        if len(self.profit_history) > 0:
            metrics['total_profit'] = sum(self.profit_history)
            metrics['average_profit'] = metrics['total_profit'] / len(self.profit_history)
        
        # Calculate conversion rate
        if len(self.conversion_history) > 0:
            metrics['conversion_rate'] = sum(self.conversion_history) / len(self.conversion_history)
        
        # Calculate average price ratio
        if len(self.price_history) > 0:
            metrics['average_price_ratio'] = sum(self.price_history) / len(self.price_history)
        
        # Calculate price optimality (how close to optimal prices)
        optimal_price_ratio = self._get_optimal_price_ratio()
        if len(self.price_history) > 0:
            avg_deviation = sum(abs(p - optimal_price_ratio) for p in self.price_history) / len(self.price_history)
            metrics['price_optimality'] = max(0.0, 1.0 - avg_deviation)
        
        # Include reward components for analysis
        metrics['reward_components'] = self.reward_components.copy()
        
        # Include episode totals
        metrics['episode_profit'] = self.episode_profit
        metrics['episode_conversion_count'] = self.episode_conversion_count
        
        return metrics

    def _update_state(self, price_ratio, profit, conversion):
        """Update the state after taking an action"""
        # Update history
        self.price_history.append(price_ratio)
        self.demand_history.append(self._calculate_demand(price_ratio, 0.0))
        self.profit_history.append(profit)
        self.conversion_history.append(conversion)
        
        # Update the observation
        self.current_observation = self._get_observation()

    def _select_product_for_episode(self):
        """
        Select a product for the current episode based on difficulty level
        """
        # Define product pools based on difficulty
        if not hasattr(self, 'product_pools') or not self.product_pools:
            # Initialize product pools by difficulty
            self.product_pools = self._create_product_difficulty_pools()
        
        # Select from appropriate pool based on difficulty
        pool = self.product_pools.get(self.product_difficulty, self.product_pools['medium'])
        
        if not pool:
            # Fallback: use all products if pool is empty
            pool = list(range(len(self.data)))
        
        # Select random product from the pool
        product_idx = np.random.choice(pool)
        
        # Store current product information
        self.current_product_idx = product_idx
        self.current_product_info = self.data.iloc[product_idx].to_dict()
        
        # Ensure all necessary fields are present
        if 'elasticity' not in self.current_product_info:
            self.current_product_info['elasticity'] = 1.0  # Default elasticity
        
        if 'price' not in self.current_product_info:
            # Use actual_price as price if available, otherwise set a default
            if 'actual_price' in self.current_product_info:
                self.current_product_info['price'] = self.current_product_info['actual_price']
            else:
                self.current_product_info['price'] = 100.0  # Default price
        
        if 'cost' not in self.current_product_info:
            # Estimate cost as 60% of price if not available
            self.current_product_info['cost'] = self.current_product_info['price'] * 0.6

    def _create_product_difficulty_pools(self):
        """
        Create pools of products based on difficulty levels
        
        Returns:
            pools: Dictionary of product pools by difficulty
        """
        pools = {
            'easy': [],
            'medium': [],
            'hard': []
        }
        
        # Create difficulty pools based on elasticity
        for idx, product in self.data.iterrows():
            elasticity = product.get('elasticity', 1.0)
            
            if 'elasticity' not in product:
                # Skip products without elasticity data
                continue
                
            if elasticity < 0.7 or elasticity > 1.3:
                # Products with extreme elasticity are hard
                pools['hard'].append(idx)
            elif 0.8 <= elasticity <= 1.2:
                # Products with moderate elasticity are easy
                pools['easy'].append(idx)
            else:
                # Everything else is medium difficulty
                pools['medium'].append(idx)
        
        # Ensure all pools have at least some products
        if not pools['easy']:
            # Use medium difficulty products as fallback for easy
            pools['easy'] = pools['medium']
        
        if not pools['hard']:
            # Create hard pool with 20% of products if none exist
            all_products = list(range(len(self.data)))
            np.random.shuffle(all_products)
            pools['hard'] = all_products[:max(1, int(len(all_products) * 0.2))]
        
        if not pools['medium'] and (pools['easy'] or pools['hard']):
            # Combine easy and hard for medium if needed
            pools['medium'] = pools['easy'] + pools['hard']
        
        # If still no products in any pool, use all products
        all_products = list(range(len(self.data)))
        if not any(pools.values()):
            for difficulty in pools:
                pools[difficulty] = all_products
        
        return pools


# Function to load and preprocess data for external use
def load_market_data(data_path="prototype/preprocessed_ppi.csv", test_split=0.2, seed=None):
    """
    Load and preprocess the e-commerce market data
    
    Args:
        data_path: Path to the dataset
        test_split: Fraction of data to use for testing
        seed: Random seed for reproducibility
        
    Returns:
        train_data: DataFrame with training data
        test_data: DataFrame with testing data
        preprocessor: Data preprocessing pipeline
    """
    # Create a dummy environment to use its preprocessing
    env = EcommerceMarketEnv(data_path=data_path, test_split=test_split, seed=seed)
    
    # Return the processed data
    return env.train_data, env.test_data, {
        'type_encoder': env.type_encoder,
        'group_encoder': env.group_encoder,
        'feature_scaler': env.feature_scaler,
        'group_embeddings': env.group_embeddings
    }


# Example usage
if __name__ == "__main__":
    # Create the environment
    env = EcommerceMarketEnv(render_mode="human")
    
    # Reset the environment
    state, _ = env.reset(seed=42)
    
    # Run a simple test with random actions
    total_reward = 0
    
    for _ in range(10):
        # Take a random action
        action = env.action_space.sample()
        state, reward, done, _, info = env.step(action)
        
        # Render the environment
        env.render()
        
        # Update total reward
        total_reward += reward
        
        if done:
            break
            
    print(f"\nTest complete. Total reward: ${total_reward:.2f}")
    
    # Close the environment
    env.close() 