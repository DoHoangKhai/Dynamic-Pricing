#!/usr/bin/env python
"""
Test script to evaluate the strategy-based pricing model with various input prices.
This helps to diagnose if the strategy model is correctly differentiating pricing.
"""

import os
import numpy as np
from pricing_strategies import PricingStrategy

def test_strategy_pricing():
    """Test strategy-based pricing with various input prices."""
    print("Testing strategy-based pricing with different input prices...")
    
    # Create the pricing strategy
    pricing_strategy = PricingStrategy()
    
    # Test with various input prices
    input_prices = [50, 100, 200, 500, 1000]
    
    # Test cases
    test_cases = [
        # Name, Elasticity, PPI, Rating
        ("Price Sensitive Product", 1.5, 1.0, 3.5),
        ("Neutral Product", 1.0, 1.0, 4.0),
        ("Premium Product", 0.7, 1.0, 4.5),
        ("Underpriced Product", 1.0, 0.8, 4.0),  # 20% below market
        ("Overpriced Product", 1.0, 1.2, 4.0)    # 20% above market
    ]
    
    # Loop through each test case
    for case_name, elasticity, ppi, rating in test_cases:
        print(f"\n=== {case_name} ===")
        print("{:<15} | {:<15} | {:<15} | {:<15}".format(
            "Input Price", "Strategy Price", "Price Ratio", "% Change"
        ))
        print("-" * 70)
        
        # Test with different input prices
        for input_price in input_prices:
            # Create product info
            product = {
                'product_id': 'test_product',
                'product_type': 'Electronics',
                'product_group': 'Smartphones',
                'price': input_price,
                'cost': input_price * 0.6,  # Assume cost is 60% of price
                'elasticity': elasticity,
                'rating': rating,
                'ppi': ppi,
                'number_of_orders': 50
            }
            
            # Create market info
            market_info = {
                'competitive_intensity': 0.7,
                'price_trend': 0.0,
                'current_price_ratio': ppi
            }
            
            # Get price recommendations
            recommendation = pricing_strategy.get_price_recommendations(product, market_info)
            price_ratio = recommendation.get('price_ratio', 1.0)
            strategy_price = input_price * price_ratio
            
            # Calculate percent change
            pct_change = (price_ratio - 1.0) * 100
            
            # Print results
            print("{:<15.2f} | {:<15.2f} | {:<15.4f} | {:<+15.2f}%".format(
                input_price, strategy_price, price_ratio, pct_change
            ))
    
    print("\nTest complete. If the strategy model is working correctly, you should see:")
    print("1. Different price ratios based on product characteristics")
    print("2. Consistent price ratios for each product type regardless of input price")
    print("3. Higher price ratios for premium products, lower for price-sensitive ones")
    
    return True

if __name__ == "__main__":
    test_strategy_pricing() 