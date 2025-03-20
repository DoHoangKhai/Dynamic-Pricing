#!/usr/bin/env python
"""
Test script to evaluate the dynamic pricing model with various test cases.
This script helps diagnose if the model is actually making dynamic pricing decisions.
"""

import os
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from RL_env1 import make_env
from pricing_strategies import PricingStrategy

def test_model_predictions():
    """Test model predictions with various test cases to see if it's responsive."""
    print("Testing dynamic pricing model predictions...")
    
    # Try to load the model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dynamic_pricing_dqn.zip")
    
    try:
        print(f"Loading model from {model_path}...")
        model = DQN.load(model_path)
        print("Model loaded successfully!")
        
        # Test cases with different elasticity, price position index (PPI), and ratings
        test_cases = [
            # Name, Elasticity, PPI, Rating, Orders
            ("Price Sensitive Budget Product", 1.5, 1.0, 3.5, 30),
            ("Price Sensitive Premium Product", 1.5, 1.0, 4.5, 80),
            ("Neutral Budget Product", 1.0, 1.0, 3.5, 30),
            ("Neutral Premium Product", 1.0, 1.0, 4.5, 80),
            ("Premium Budget Product", 0.7, 1.0, 3.5, 30),
            ("Premium Luxury Product", 0.7, 1.0, 4.5, 80),
            ("Underpriced Product", 1.0, 0.8, 4.0, 50),  # 20% below market
            ("Overpriced Product", 1.0, 1.2, 4.0, 50),   # 20% above market
        ]
        
        # Create the pricing strategy for comparison
        pricing_strategy = PricingStrategy()
        
        print("\n{:<30} | {:<15} | {:<15} | {:<15} | {:<15}".format(
            "Test Case", "Input Price", "Model Price", "Strategy Price", "% Change"
        ))
        print("-" * 100)
        
        # Run each test case
        for case_name, elasticity, ppi, rating, num_orders in test_cases:
            # Base input price (we'll use $100 for all tests)
            input_price = 100.0
            
            # Create environment with parameters
            env = make_env(
                elasticity=elasticity,
                ppi=ppi,
                rating=rating,
                num_orders=num_orders
            )
            
            # Get initial observation
            observation = env.reset()
            
            # Make prediction using DQN model
            action, _ = model.predict(observation, deterministic=True)
            
            # Convert action to price
            model_price = env.min_price + (action * env.price_step)
            
            # Also get the strategy price recommendation for comparison
            product = {
                'product_id': 'test_product',
                'product_type': 'Electronics',
                'product_group': 'Smartphones',
                'price': input_price,
                'cost': input_price * 0.6,
                'elasticity': elasticity,
                'rating': rating,
                'ppi': ppi,
                'number_of_orders': num_orders
            }
            
            market_info = {
                'competitive_intensity': 0.7,
                'price_trend': 0.0,
                'current_price_ratio': ppi
            }
            
            strategy_rec = pricing_strategy.get_price_recommendations(product, market_info)
            strategy_price = input_price * strategy_rec.get('price_ratio', 1.0)
            
            # Calculate percent change
            pct_change = ((model_price - input_price) / input_price) * 100
            
            # Print the results
            print("{:<30} | ${:<14.2f} | ${:<14.2f} | ${:<14.2f} | {:<+14.2f}%".format(
                case_name, input_price, model_price, strategy_price, pct_change
            ))
            
            # Take a step in the environment to see the reward
            observation, reward, done, truncated, info = env.step(action)
            print(f"   → Reward: {reward:.2f}, Demand: {info['demand']:.2f}, Profit: ${info['profit']:.2f}")
        
        print("\nTest complete. If the model is working correctly, you should see different pricing")
        print("recommendations based on the product characteristics.")
        return True
    
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_predictions() 