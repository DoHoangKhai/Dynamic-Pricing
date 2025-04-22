"""
Pipeline Adapter Module

This module provides an adapter to integrate the three components of the pricing pipeline:
1. Customer segmentation
2. Reinforcement learning
3. Demand forecasting

The adapter provides a clean interface for using these components together in evaluation
frameworks and production environments.
"""

import os
import numpy as np
import pandas as pd
import json
import time
import logging
from datetime import datetime
import traceback

# Import pipeline components
from .pricing_strategies import PricingStrategy, CustomerSegmentation
from .demand_forecasting import DemandForecaster
try:
    # Try to import the RL components
    from .RL_env1 import DynamicPricingEnv, make_env
    import gymnasium as gym
    import stable_baselines3 as sb3
    from stable_baselines3 import PPO, DQN
    RL_AVAILABLE = True
except ImportError:
    # If imports fail, RL will be disabled
    RL_AVAILABLE = False
    logging.warning("RL components could not be imported. RL-based pricing will be disabled.")

class PricingPipelineAdapter:
    """
    Adapter class for integrating the pricing pipeline components.
    
    This class provides a unified interface to:
    1. Generate customer segments
    2. Run RL model for pricing optimization
    3. Forecast demand at recommended prices
    """
    
    def __init__(self, use_rl=True, model_path=None, log_level=logging.INFO):
        """
        Initialize the pricing pipeline adapter
        
        Args:
            use_rl: Whether to use RL for pricing optimization
            model_path: Path to saved RL model
            log_level: Logging level
        """
        # Set up logging
        self.logger = logging.getLogger("PricingPipeline")
        self.logger.setLevel(log_level)
        
        # If no handlers exist, add one
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Initialize components
        self.pricing_strategy = PricingStrategy()
        self.customer_segmentation = CustomerSegmentation()
        self.demand_forecaster = DemandForecaster()
        
        # RL settings
        self.use_rl = use_rl and RL_AVAILABLE
        self.model_path = model_path
        self.rl_model = None
        
        # Load RL model if available
        if self.use_rl and model_path and os.path.exists(model_path):
            try:
                self.logger.info(f"Loading RL model from {model_path}")
                # Try to load as DQN first
                try:
                    self.rl_model = DQN.load(model_path)
                    self.logger.info("Loaded DQN model")
                except:
                    # If that fails, try PPO
                    try:
                        self.rl_model = PPO.load(model_path)
                        self.logger.info("Loaded PPO model")
                    except Exception as e:
                        self.logger.error(f"Failed to load RL model: {str(e)}")
                        self.use_rl = False
            except Exception as e:
                self.logger.error(f"Failed to load RL model: {str(e)}")
                self.use_rl = False
    
    def run_pipeline(self, product_data, market_info, historical_data=None, asin=None):
        """
        Run the full pricing pipeline
        
        Args:
            product_data: Dictionary with product information
            market_info: Dictionary with market information
            historical_data: Optional dataframe with historical sales data
            asin: Optional product ASIN for API lookups
            
        Returns:
            Dictionary with pricing recommendations and supporting data
        """
        start_time = time.time()
        self.logger.info(f"Starting pricing pipeline for product: {product_data.get('product_id', 'unknown')}")
        
        results = {
            'success': False,
            'errors': [],
            'warnings': [],
            'timing': {},
            'product_data': product_data,
        }
        
        try:
            # Step 1: Generate customer segments
            segment_start = time.time()
            segments = self._run_segmentation(product_data)
            segment_time = time.time() - segment_start
            results['timing']['segmentation'] = segment_time
            results['segments'] = segments
            
            # Step 2: Generate price recommendations
            price_start = time.time()
            if self.use_rl and self.rl_model is not None:
                # Use RL model for pricing
                price_rec = self._run_rl_pricing(product_data, market_info, segments)
            else:
                # Use rule-based pricing strategy
                price_rec = self._run_rule_based_pricing(product_data, market_info)
            
            price_time = time.time() - price_start
            results['timing']['pricing'] = price_time
            results['pricing'] = price_rec
            
            # Extract price ratio for demand forecasting
            if 'price_ratio' in price_rec:
                price_ratio = price_rec['price_ratio']
            else:
                # Calculate ratio from recommended and current price
                current_price = product_data.get('price', 100)
                recommended_price = price_rec.get('recommended_price', current_price)
                price_ratio = recommended_price / current_price
            
            # Step 3: Generate demand forecast
            forecast_start = time.time()
            forecast = self._run_demand_forecast(product_data, price_ratio, historical_data, segments)
            forecast_time = time.time() - forecast_start
            results['timing']['forecasting'] = forecast_time
            results['forecast'] = forecast
            
            # Calculate segment conversion probabilities
            conversion_start = time.time()
            segment_conversion = self.customer_segmentation.calculate_segment_conversion_probabilities(
                price_ratio, product_data
            )
            conversion_time = time.time() - conversion_start
            results['timing']['conversion'] = conversion_time
            results['segment_conversion'] = segment_conversion
            
            # Calculate segment impact
            segment_impact = self._calculate_segment_impact(segment_conversion, price_ratio)
            results['segment_impact'] = segment_impact
            
            # Prepare visualization data
            viz_start = time.time()
            segment_viz_data = self._prepare_segment_visualization(segments)
            results['segment_visualization'] = segment_viz_data
            viz_time = time.time() - viz_start
            results['timing']['visualization'] = viz_time
            
            # Mark as successful
            results['success'] = True
            
            # Add total execution time
            total_time = time.time() - start_time
            results['timing']['total'] = total_time
            
            self.logger.info(f"Pricing pipeline completed successfully in {total_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Error in pricing pipeline: {str(e)}")
            traceback.print_exc()
            results['errors'].append(str(e))
            results['success'] = False
        
        return results
    
    def _run_segmentation(self, product_data):
        """Run customer segmentation"""
        try:
            product_type = product_data.get('product_type', 'standard')
            product_group = product_data.get('product_group', 'general')
            rating = product_data.get('rating', 3.0)
            price = product_data.get('price', 100)
            
            segments = self.customer_segmentation.get_segment_distribution(
                product_type, product_group, rating, price
            )
            
            self.logger.info(f"Generated customer segments for {product_type}/{product_group}")
            return segments
            
        except Exception as e:
            self.logger.error(f"Error in customer segmentation: {str(e)}")
            # Return default segments
            return self.customer_segmentation._create_default_segments(4)
    
    def _run_rl_pricing(self, product_data, market_info, segments):
        """Run RL-based pricing optimization"""
        try:
            # Extract product parameters
            product_type = product_data.get('product_type', 'Electronics')
            product_group = product_data.get('product_group', 'General')
            price = product_data.get('price', 100)
            cost = product_data.get('cost', price * 0.6)
            elasticity = product_data.get('elasticity', 1.0)
            ppi = product_data.get('ppi', 1.0)
            rating = product_data.get('rating', 3.0)
            num_orders = product_data.get('number_of_orders', 50)
            
            # Create environment with product data
            env = make_env(
                elasticity=elasticity,
                ppi=ppi, 
                rating=rating,
                num_orders=num_orders,
                product_type=product_type,
                product_group=product_group
            )
            
            # Get initial observation
            observation, _ = env.reset()
            
            # Get action from RL model
            action, _states = self.rl_model.predict(observation, deterministic=True)
            
            # Take action in environment to get price
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Extract price from info
            rl_price = info['price']
            price_ratio = rl_price / price
            
            # Calculate min and max prices (within 10% range)
            min_price = max(cost * 1.15, rl_price * 0.9)  # At least 15% margin
            max_price = rl_price * 1.1
            
            # Determine elasticity category
            if elasticity < 0.7:
                elasticity_category = "Inelastic"
            elif elasticity < 1.0:
                elasticity_category = "Somewhat Inelastic"
            elif elasticity < 1.3:
                elasticity_category = "Unit Elastic"
            else:
                elasticity_category = "Elastic"
            
            # Calculate expected profit
            expected_demand = info['demand']
            expected_profit = info['profit']
            
            self.logger.info(f"RL model recommended price: ${rl_price:.2f}, price ratio: {price_ratio:.2f}")
            
            return {
                'recommended_price': rl_price,
                'price_ratio': price_ratio,
                'price_range': {
                    'min_price': min_price,
                    'max_price': max_price
                },
                'elasticity_category': elasticity_category,
                'elasticity_factor': elasticity,
                'expected_demand': expected_demand,
                'expected_profit': expected_profit,
                'model_type': 'reinforcement_learning'
            }
            
        except Exception as e:
            self.logger.error(f"Error in RL pricing: {str(e)}")
            # Fall back to rule-based pricing
            self.logger.info("Falling back to rule-based pricing")
            return self._run_rule_based_pricing(product_data, market_info)
    
    def _run_rule_based_pricing(self, product_data, market_info):
        """Run rule-based pricing strategy"""
        try:
            # Get price recommendation from pricing strategy
            recommendation = self.pricing_strategy.get_price_recommendations(product_data, market_info)
            
            self.logger.info(f"Rule-based strategy recommended price: ${recommendation.get('recommended_price', 0):.2f}")
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Error in rule-based pricing: {str(e)}")
            # Return a safe default
            return {
                'recommended_price': product_data.get('price', 100),
                'price_ratio': 1.0,
                'price_range': {
                    'min_price': product_data.get('price', 100) * 0.9,
                    'max_price': product_data.get('price', 100) * 1.1
                },
                'elasticity_category': "Unknown",
                'elasticity_factor': 1.0,
                'model_type': 'fallback'
            }
    
    def _run_demand_forecast(self, product_data, price_ratio, historical_data=None, segments=None):
        """Run demand forecasting"""
        try:
            # Generate forecast
            forecast_result = self.demand_forecaster.forecast_product_demand(
                product_data, price_ratio, historical_data, segments
            )
            
            self.logger.info(f"Generated demand forecast at price ratio {price_ratio:.2f}")
            
            return forecast_result
            
        except Exception as e:
            self.logger.error(f"Error in demand forecasting: {str(e)}")
            # Return a minimal forecast result
            return {
                'error': str(e),
                'visualization': {
                    'dates': [(datetime.now() + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)],
                    'forecast': [0] * 7,
                    'statistics': {
                        'min': 0,
                        'max': 0,
                        'mean': 0,
                        'total': 0
                    }
                }
            }
    
    def _calculate_segment_impact(self, segment_conversion, price_ratio):
        """Calculate impact on different customer segments"""
        impact = {}
        
        # Calculate impact for each segment
        for segment_name, conversion_prob in segment_conversion.get('segments', {}).items():
            # Determine sentiment based on conversion probability
            if conversion_prob > 0.15:
                sentiment = "positive"
            elif conversion_prob > 0.10:
                sentiment = "neutral"
            else:
                sentiment = "negative"
            
            # Determine pricing feedback based on price ratio
            if segment_name == 'price_sensitive':
                if price_ratio > 1.05:
                    price_feedback = "too high"
                elif price_ratio < 0.95:
                    price_feedback = "attractive"
                else:
                    price_feedback = "acceptable"
            elif segment_name == 'premium_buyers':
                if price_ratio > 1.2:
                    price_feedback = "premium"
                elif price_ratio < 0.9:
                    price_feedback = "suspiciously low"
                else:
                    price_feedback = "acceptable"
            else:
                if price_ratio > 1.1:
                    price_feedback = "high"
                elif price_ratio < 0.9:
                    price_feedback = "good value"
                else:
                    price_feedback = "fair"
            
            impact[segment_name] = {
                'conversion': conversion_prob,
                'sentiment': sentiment,
                'price_feedback': price_feedback
            }
        
        return impact
    
    def _prepare_segment_visualization(self, segments):
        """Prepare segment data for visualization"""
        viz_data = []
        
        for segment_name, segment_data in segments.items():
            # Extract key properties for visualization
            viz_data.append({
                'name': segment_name.replace('_', ' ').title(),
                'weight': segment_data['weight'],
                'price_sensitivity': segment_data.get('price_sensitivity', 1.0),
                'conversion_base': segment_data.get('conversion_base', 0.1),
                'max_premium': segment_data.get('max_premium', 0.2),
                'profit_per_conversion': segment_data.get('profit_per_conversion', 1.0)
            })
        
        return viz_data 