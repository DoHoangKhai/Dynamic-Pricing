#!/usr/bin/env python
"""
Evaluation script for the dynamic pricing model, focusing on how different
inputs affect the final price recommendations.
"""

import os
import pandas as pd
import numpy as np
from pricing_strategies import PricingStrategy, CustomerSegmentation

def evaluate_pricing_sensitivity():
    """
    Evaluate how different input parameters affect price recommendations,
    particularly focusing on competitor price, number of orders, and elasticity.
    """
    print("Evaluating pricing model sensitivity to different inputs...")
    
    # Create pricing strategy and customer segmentation objects
    pricing_strategy = PricingStrategy()
    customer_segmentation = CustomerSegmentation()
    
    # Base product information
    base_product = {
        'product_id': 'test_product',
        'product_type': 'Electronics',
        'product_group': 'Smartphones',
        'price': 500.00,
        'cost': 300.00,
        'elasticity': 1.0,
        'rating': 4.0,
        'ppi': 1.0,
        'number_of_orders': 50
    }
    
    # Base market information
    base_market = {
        'competitive_intensity': 0.7,
        'price_trend': 0.0,
        'current_price_ratio': 1.0
    }
    
    # Test 1: Competitor Price Sensitivity
    print("\n=== Test 1: Competitor Price Sensitivity ===")
    print("{:<20} | {:<15} | {:<15} | {:<15}".format(
        "Competitor Ratio", "Input Price", "Recommended", "% Change"
    ))
    print("-" * 70)
    
    # Try different competitor price ratios
    for ratio in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
        # Clone the base product and market
        product = base_product.copy()
        market = base_market.copy()
        
        # Set competitor price ratio (PPI)
        product['ppi'] = ratio
        market['current_price_ratio'] = ratio
        
        # Get price recommendation
        recommendation = pricing_strategy.get_price_recommendations(product, market)
        price_ratio = recommendation.get('price_ratio', 1.0)
        strategy_price = product['price'] * price_ratio
        
        # Calculate percent change
        pct_change = (price_ratio - 1.0) * 100
        
        # Print results
        print("{:<20.2f} | ${:<14.2f} | ${:<14.2f} | {:<+14.2f}%".format(
            ratio, product['price'], strategy_price, pct_change
        ))
    
    # Test 2: Number of Orders Sensitivity
    print("\n=== Test 2: Number of Orders Sensitivity ===")
    print("{:<20} | {:<15} | {:<15} | {:<15}".format(
        "Number of Orders", "Input Price", "Recommended", "% Change"
    ))
    print("-" * 70)
    
    # Try different numbers of orders
    for orders in [0, 10, 50, 100, 200, 500, 1000]:
        # Clone the base product
        product = base_product.copy()
        
        # Set number of orders
        product['number_of_orders'] = orders
        
        # Get price recommendation
        recommendation = pricing_strategy.get_price_recommendations(product, base_market)
        price_ratio = recommendation.get('price_ratio', 1.0)
        strategy_price = product['price'] * price_ratio
        
        # Calculate demand modifier (which should be affected by orders)
        segments = customer_segmentation.get_segment_distribution(
            product['product_type'], product['product_group'], product['rating']
        )
        segment_conversion = customer_segmentation.calculate_segment_conversion_probabilities(
            price_ratio, product
        )
        profit_multiplier = segment_conversion.get('expected_profit_multiplier', 1.0)
        demand_modifier = customer_segmentation.get_demand_modifier(
            product, price_ratio, (strategy_price - product['cost']) / strategy_price
        )
        
        # Print results
        print("{:<20d} | ${:<14.2f} | ${:<14.2f} | {:<+14.2f}%".format(
            orders, product['price'], strategy_price, (price_ratio - 1.0) * 100
        ))
    
    # Test 3: Elasticity Sensitivity
    print("\n=== Test 3: Elasticity Sensitivity ===")
    print("{:<20} | {:<15} | {:<15} | {:<15} | {:<15}".format(
        "Elasticity", "Category", "Input Price", "Recommended", "% Change"
    ))
    print("-" * 90)
    
    # Try different elasticity values
    for elasticity in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]:
        # Clone the base product
        product = base_product.copy()
        
        # Set elasticity
        product['elasticity'] = elasticity
        
        # Get price recommendation
        recommendation = pricing_strategy.get_price_recommendations(product, base_market)
        price_ratio = recommendation.get('price_ratio', 1.0)
        strategy_price = product['price'] * price_ratio
        
        # Determine elasticity category
        if elasticity > 1.2:
            category = "high"
        elif elasticity < 0.8:
            category = "low"
        else:
            category = "medium"
        
        # Print results
        print("{:<20.2f} | {:<15} | ${:<14.2f} | ${:<14.2f} | {:<+14.2f}%".format(
            elasticity, category, product['price'], strategy_price, (price_ratio - 1.0) * 100
        ))
    
    # Test 4: Combined Factor Analysis
    print("\n=== Test 4: Combined Factor Analysis ===")
    print("{:<15} | {:<10} | {:<10} | {:<15} | {:<15} | {:<15}".format(
        "Elasticity", "PPI", "Orders", "Input Price", "Recommended", "% Change"
    ))
    print("-" * 90)
    
    # Test cases with combinations of factors
    test_cases = [
        # Elasticity, PPI, Orders
        (0.7, 0.8, 50),   # Low elasticity, underpriced
        (0.7, 1.2, 50),   # Low elasticity, overpriced
        (1.3, 0.8, 50),   # High elasticity, underpriced
        (1.3, 1.2, 50),   # High elasticity, overpriced
        (1.0, 1.0, 10),   # Medium elasticity, fair price, low orders
        (1.0, 1.0, 500),  # Medium elasticity, fair price, high orders
        (0.7, 1.0, 500),  # Low elasticity, fair price, high orders
        (1.3, 1.0, 500),  # High elasticity, fair price, high orders
    ]
    
    for elasticity, ppi, orders in test_cases:
        # Clone the base product and market
        product = base_product.copy()
        market = base_market.copy()
        
        # Set test parameters
        product['elasticity'] = elasticity
        product['ppi'] = ppi
        market['current_price_ratio'] = ppi
        product['number_of_orders'] = orders
        
        # Get price recommendation
        recommendation = pricing_strategy.get_price_recommendations(product, market)
        price_ratio = recommendation.get('price_ratio', 1.0)
        strategy_price = product['price'] * price_ratio
        
        # Print results
        print("{:<15.2f} | {:<10.2f} | {:<10d} | ${:<14.2f} | ${:<14.2f} | {:<+14.2f}%".format(
            elasticity, ppi, orders, product['price'], strategy_price, (price_ratio - 1.0) * 100
        ))
    
    print("\nEvaluation complete.")

if __name__ == "__main__":
    evaluate_pricing_sensitivity() 