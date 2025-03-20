#!/usr/bin/env python
"""
Test script to analyze how changing individual parameters affects the dynamic pricing model's recommendations.
This script varies one parameter at a time while keeping all others constant to isolate the effect.
"""

import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def test_parameter_sensitivity(parameter="competitorPrice"):
    """
    Test how changing a specific parameter affects the pricing recommendation.
    
    Args:
        parameter: The parameter to vary ("competitorPrice", "numberOfOrders", "elasticity")
    """
    print(f"Testing pricing sensitivity to {parameter}...")
    
    # API endpoint
    url = "http://localhost:5050/api/predict-price"
    
    # Base product configuration
    base_product = {
        "productType": "Electronics",
        "productGroup": "Headphones",
        "actualPrice": 100.00,
        "competitorPrice": 100.00,
        "rating": 4.0,
        "numberOfOrders": 50
    }
    
    # Define parameter ranges to test
    test_ranges = {
        "competitorPrice": [60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
        "numberOfOrders": [5, 10, 25, 50, 100, 150, 200, 300, 400, 500],
        "elasticity": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    }
    
    # If testing elasticity, we need to use product types with different elasticities
    elasticity_products = {
        0.7: {"productType": "Premium", "productGroup": "HighEndElectronics"},
        0.8: {"productType": "Premium", "productGroup": "LuxuryAudio"},
        0.9: {"productType": "Electronics", "productGroup": "Smartphones"},
        1.0: {"productType": "Electronics", "productGroup": "Tablets"},
        1.1: {"productType": "Electronics", "productGroup": "Headphones"},
        1.2: {"productType": "Electronics", "productGroup": "BluetoothSpeakers"},
        1.3: {"productType": "Basics", "productGroup": "KitchenBasics"},
        1.4: {"productType": "Basics", "productGroup": "OfficeBasics"},
        1.5: {"productType": "Commodity", "productGroup": "CableAccessories"}
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    results = []
    
    # For each test value
    for value in test_ranges[parameter]:
        # Create a copy of the base product
        product = base_product.copy()
        
        # Special handling for elasticity
        if parameter == "elasticity":
            # Update product type and group to match elasticity
            product.update(elasticity_products[value])
        else:
            # Update the parameter value
            product[parameter] = value
        
        try:
            # Send the request
            response = requests.post(url, headers=headers, json=product)
            
            # Check if request was successful
            if response.status_code == 200:
                data = response.json()
                
                # Extract relevant data for analysis
                debug = data.get("debug", {})
                
                result = {
                    "parameter_value": value,
                    "recommended_price": data["recommendedPrice"],
                    "price_change_pct": ((data["recommendedPrice"] - product["actualPrice"]) / product["actualPrice"]) * 100,
                    "competitor_impact": data.get("competitorImpact", 0.0),
                    "order_impact": data.get("orderImpact", 0.0),
                    "elasticity_impact": data.get("elasticityImpact", 0.0),
                    "model_price": debug.get("modelPredictedPrice", 0.0),
                    "strategy_price": debug.get("strategyPrice", 0.0),
                    "final_price": debug.get("finalPrice", 0.0),
                    "elasticity": debug.get("elasticity", 0.0),
                    "order_adjustment": debug.get("orderAdjustment", 1.0),
                }
                
                # Print the result
                print(f"{parameter} = {value}: Recommended Price = ${result['recommended_price']:.2f} ({result['price_change_pct']:+.2f}%)")
                
                results.append(result)
            else:
                print(f"{parameter} = {value} - Error: {response.status_code}, {response.text}")
                
        except Exception as e:
            print(f"{parameter} = {value} - Exception: {str(e)}")
    
    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    
    # Plot recommended price
    plt.subplot(2, 1, 1)
    plt.plot(results_df['parameter_value'], results_df['recommended_price'], marker='o', linewidth=2)
    plt.title(f'Effect of {parameter} on Recommended Price')
    plt.ylabel('Recommended Price ($)')
    plt.grid(True)
    
    # Plot price change percentage
    plt.subplot(2, 1, 2)
    plt.plot(results_df['parameter_value'], results_df['price_change_pct'], marker='o', linewidth=2)
    plt.title(f'Effect of {parameter} on Price Change %')
    plt.xlabel(parameter)
    plt.ylabel('Price Change %')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'parameter_sensitivity_{parameter}.png')
    print(f"Analysis complete. Results saved to parameter_sensitivity_{parameter}.png")
    
    return results_df

if __name__ == "__main__":
    # Test all three parameter types
    competitor_results = test_parameter_sensitivity("competitorPrice")
    print("\n" + "-"*50 + "\n")
    
    orders_results = test_parameter_sensitivity("numberOfOrders")
    print("\n" + "-"*50 + "\n")
    
    elasticity_results = test_parameter_sensitivity("elasticity") 