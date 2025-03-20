#!/usr/bin/env python
"""
Test script to send API requests with different product parameters
and check if the dynamic pricing model is making differentiated recommendations.
This enhanced version focuses on testing competitor price and order count impacts.
"""

import requests
import json

def test_api_pricing():
    """Test the pricing API with different product scenarios."""
    print("Testing API pricing recommendations with focus on competitors and orders...")
    
    # API endpoint
    url = "http://localhost:5050/api/predict-price"
    
    # Test cases specifically designed to test competitor price and order count impact
    test_cases = [
        # 1. Baseline case for comparison
        {
            "name": "Baseline Product",
            "productType": "Electronics",
            "productGroup": "Headphones",
            "actualPrice": 100.00,
            "competitorPrice": 100.00,
            "rating": 4.0,
            "numberOfOrders": 50
        },
        
        # Test cases for competitive price impact
        {
            "name": "Competitor Much Lower",
            "productType": "Electronics",
            "productGroup": "Headphones",
            "actualPrice": 100.00,
            "competitorPrice": 80.00,  # 20% lower
            "rating": 4.0,
            "numberOfOrders": 50
        },
        {
            "name": "Competitor Much Higher",
            "productType": "Electronics",
            "productGroup": "Headphones",
            "actualPrice": 100.00,
            "competitorPrice": 120.00,  # 20% higher
            "rating": 4.0,
            "numberOfOrders": 50
        },
        {
            "name": "Competitor Extremely Lower",
            "productType": "Electronics",
            "productGroup": "Headphones",
            "actualPrice": 100.00,
            "competitorPrice": 60.00,  # 40% lower
            "rating": 4.0,
            "numberOfOrders": 50
        },
        {
            "name": "Competitor Extremely Higher",
            "productType": "Electronics",
            "productGroup": "Headphones",
            "actualPrice": 100.00,
            "competitorPrice": 150.00,  # 50% higher
            "rating": 4.0,
            "numberOfOrders": 50
        },
        
        # Test cases for order count impact
        {
            "name": "Very Few Orders",
            "productType": "Electronics",
            "productGroup": "Headphones",
            "actualPrice": 100.00,
            "competitorPrice": 100.00,
            "rating": 4.0,
            "numberOfOrders": 5
        },
        {
            "name": "Moderate Orders",
            "productType": "Electronics",
            "productGroup": "Headphones",
            "actualPrice": 100.00,
            "competitorPrice": 100.00,
            "rating": 4.0,
            "numberOfOrders": 50
        },
        {
            "name": "High Orders",
            "productType": "Electronics",
            "productGroup": "Headphones",
            "actualPrice": 100.00,
            "competitorPrice": 100.00,
            "rating": 4.0,
            "numberOfOrders": 150
        },
        {
            "name": "Extremely High Orders",
            "productType": "Electronics",
            "productGroup": "Headphones",
            "actualPrice": 100.00,
            "competitorPrice": 100.00,
            "rating": 4.0,
            "numberOfOrders": 500
        },
        
        # Test cases for elasticity categorization
        {
            "name": "High Elasticity Product",
            "productType": "Basics",
            "productGroup": "KitchenBasics",
            "actualPrice": 100.00,
            "competitorPrice": 100.00,
            "rating": 4.0,
            "numberOfOrders": 50
        },
        {
            "name": "Low Elasticity Product",
            "productType": "Premium",
            "productGroup": "HighEndElectronics",
            "actualPrice": 100.00,
            "competitorPrice": 100.00,
            "rating": 4.0,
            "numberOfOrders": 50
        },
        
        # Combined test cases (multiple factors together)
        {
            "name": "Premium + High Orders",
            "productType": "Premium",
            "productGroup": "HighEndElectronics",
            "actualPrice": 100.00,
            "competitorPrice": 100.00,
            "rating": 4.5,
            "numberOfOrders": 300
        },
        {
            "name": "Premium + Higher Competitor",
            "productType": "Premium",
            "productGroup": "HighEndElectronics",
            "actualPrice": 100.00,
            "competitorPrice": 130.00,
            "rating": 4.0,
            "numberOfOrders": 50
        },
        {
            "name": "Basic + Lower Competitor",
            "productType": "Basics",
            "productGroup": "KitchenBasics",
            "actualPrice": 100.00,
            "competitorPrice": 70.00,
            "rating": 3.5,
            "numberOfOrders": 50
        },
        {
            "name": "Popular + Lower Competitor",
            "productType": "Electronics",
            "productGroup": "Headphones",
            "actualPrice": 100.00,
            "competitorPrice": 80.00,
            "rating": 4.0,
            "numberOfOrders": 200
        }
    ]
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print("\n{:<25} | {:<15} | {:<15} | {:<20} | {:<15} | {:<15}".format(
        "Test Case", "Competitor", "Orders", "Recommended", "% Change", "Elasticity"
    ))
    print("-" * 120)
    
    # Send requests for each test case
    for case in test_cases:
        try:
            # Extract test case name and prepare payload
            test_name = case.pop("name")
            payload = case
            
            # Send the request
            response = requests.post(url, headers=headers, json=payload)
            
            # Check if request was successful
            if response.status_code == 200:
                data = response.json()
                
                # Calculate price change percentage
                actual_price = payload["actualPrice"]
                competitor_price = payload["competitorPrice"]
                num_orders = payload["numberOfOrders"]
                recommended_price = data["recommendedPrice"]
                elasticity_category = data.get("elasticityCategory", "unknown")
                price_change_pct = ((recommended_price - actual_price) / actual_price) * 100
                
                # Print the results
                print("{:<25} | ${:<14.2f} | {:<15d} | ${:<19.2f} | {:<+14.2f}% | {:<15}".format(
                    test_name, 
                    competitor_price,
                    int(num_orders),
                    recommended_price, 
                    price_change_pct,
                    elasticity_category
                ))
                
                # Print the impact factors
                print("   → Rating Impact: {:.1f}%, Order Impact: {:.1f}%, Competitor Impact: {:.1f}%, Market Impact: {:.1f}%".format(
                    data.get("ratingImpact", 0.0),
                    data.get("orderImpact", 0.0),
                    data.get("competitorImpact", 0.0),
                    data.get("marketImpact", 0.0)
                ))
                
                # Print debug information
                debug = data.get("debug", {})
                if debug:
                    print("   → DEBUG: Model: ${:.2f}, Strategy: ${:.2f}, Blended: ${:.2f}, Final: ${:.2f}".format(
                        debug.get("modelPredictedPrice", 0.0),
                        debug.get("strategyPrice", 0.0),
                        debug.get("blendedPrice", 0.0),
                        debug.get("finalPrice", 0.0)
                    ))
                    print("   → ADJUSTMENTS: Order Factor: {:.3f}, Model Weight: {:.3f}, Elasticity: {:.3f}, PPI: {:.3f}".format(
                        debug.get("orderAdjustment", 1.0),
                        debug.get("modelWeight", 0.0),
                        debug.get("elasticity", 1.0),
                        debug.get("ppi", 1.0)
                    ))
            else:
                print(f"{test_name} - Error: {response.status_code}, {response.text}")
                
        except Exception as e:
            print(f"{test_name} - Exception: {str(e)}")
    
    print("\nAPI test complete. If the model is working correctly, you should see:")
    print("1. Lower recommended prices when competitor prices are significantly lower")
    print("2. Higher recommended prices when competitor prices are significantly higher")
    print("3. Higher recommended prices for products with more orders (popular products)")
    print("4. High elasticity for basic products and low elasticity for premium products")
    print("5. Combined effects: Premium products with high orders get the highest price premiums")
    print("6. Combined effects: Basic products facing low-priced competitors get the largest price reductions")
    print("7. Combined effects: Popular products can maintain higher prices even against lower-priced competitors")
    return True

if __name__ == "__main__":
    test_api_pricing() 