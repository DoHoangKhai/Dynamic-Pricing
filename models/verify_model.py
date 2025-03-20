#!/usr/bin/env python
"""
Verification script for testing the RL model integration in the test environment.
"""

import os
import numpy as np
from stable_baselines3 import DQN
from RL_env1 import make_env

def verify_model():
    """Verify that the model can be loaded and used correctly."""
    print("Starting model verification...")
    
    # Try to load the model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dynamic_pricing_dqn.zip")
    
    try:
        print(f"Loading model from {model_path}...")
        model = DQN.load(model_path)
        print("Model loaded successfully!")
        
        # Create test environment
        print("Creating test environment...")
        env = make_env(
            elasticity=1.2,
            ppi=1.0,
            rating=4.0,
            num_orders=50
        )
        
        # Get initial observation
        print("Resetting environment...")
        observation = env.reset()
        
        # Try making a prediction
        print("Making prediction...")
        action, _ = model.predict(observation, deterministic=True)
        
        # Convert action to price
        price = env.min_price + (action * env.price_step)
        
        print(f"Predicted action: {action}")
        print(f"Corresponding price: ${price:.2f}")
        
        # Take a step in the environment
        print("Taking step in environment...")
        observation, reward, done, truncated, info = env.step(action)
        
        print(f"Step result - Reward: {reward:.2f}")
        print(f"Step info: {info}")
        
        print("Verification complete! The model is working correctly.")
        return True
    
    except Exception as e:
        print(f"Verification failed with error: {e}")
        return False

if __name__ == "__main__":
    verify_model() 