from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import json
import traceback
from RL_env1 import DynamicPricingEnv

# Import our market data API integration
from api_market_data import register_api_endpoints
from market_data_analysis import MarketDataAnalyzer

app = Flask(__name__)

# Initialize pricing environment
env = DynamicPricingEnv()

# Initialize market data analyzer
market_analyzer = MarketDataAnalyzer()

# Register market data API endpoints
register_api_endpoints(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict-price', methods=['POST'])
def predict_price():
    try:
        # Get input data from request
        data = request.get_json()

        # Extract input features
        product_type = data.get('productType', 'Electronics')
        product_group = data.get('productGroup', 'Laptops')
        actual_price = float(data.get('actualPrice', 1000))
        competitor_price = float(data.get('competitorPrice', 1000))
        rating = float(data.get('rating', 4.5))
        number_of_orders = int(data.get('numberOfOrders', 100))
        
        # Get additional market features if available
        asin = data.get('asin')
        market_features = {}
        
        if asin:
            # Try to get market insights for this product
            try:
                features = market_analyzer.extract_pricing_features(asin)
                if features:
                    market_features = features
            except Exception as e:
                # If market data fetch fails, continue without it
                print(f"Warning: Could not fetch market data for {asin}: {str(e)}")
        
        # Determine elasticity classification
        if actual_price <= 50:
            elasticity_category = "Low-price"
            elasticity_value = 1.2
        elif actual_price <= 200:
            elasticity_category = "Mid-price"
            elasticity_value = 1.0
        else:
            elasticity_category = "Premium"
            elasticity_value = 0.8
        
        # Prepare input for the model - default values to avoid NaN issues
        price_ratio = actual_price / max(1, competitor_price)
        
        # Incorporate market features if available
        market_adjustment = 1.0
        market_reason = ""
        
        if market_features:
            # Adjust pricing based on market position
            price_status = market_features.get('price_status', 'average')
            price_trend = market_features.get('price_trend', 'stable')
            sentiment_score = market_features.get('sentiment_score', 0)
            
            # Sentiment-based adjustment
            if sentiment_score > 0.5:
                market_adjustment *= 1.05
                market_reason += "Strong positive sentiment allows premium pricing. "
            elif sentiment_score < -0.3:
                market_adjustment *= 0.95
                market_reason += "Negative sentiment suggests reducing price. "
            
            # Price trend adjustment
            if price_trend == "increasing":
                market_adjustment *= 1.02
                market_reason += "Upward market price trend detected. "
            elif price_trend == "decreasing":
                market_adjustment *= 0.98
                market_reason += "Downward market price trend detected. "
            
            # Market position adjustment
            if price_status == "very competitive":
                market_adjustment *= 0.97
                market_reason += "Market is very price competitive. "
            elif price_status == "premium" or price_status == "very premium":
                if sentiment_score > 0.2:  # Only if sentiment allows it
                    market_adjustment *= 1.03
                    market_reason += "Product can command premium pricing. "
        
        # Calculate rating factor (influence of rating on price)
        rating_factor = 0.05 * (rating - 3) if rating > 3 else -0.1 * (3 - rating)
        
        # Calculate order factor (influence of order volume on price)
        order_factor = 0.02 * min(1, number_of_orders / 200)  # Capped at 200 orders
        
        # Calculate recommended price based on elasticity, rating, orders, and market data
        price_adjustment = 1 + rating_factor + order_factor
        recommended_price = actual_price * (price_adjustment * market_adjustment)
        
        # Calculate min and max recommended prices (±10% range)
        min_price = recommended_price * 0.9
        max_price = recommended_price * 1.1
        
        # Formulate explanation
        explanation = f"Based on the {elasticity_category} product category, a rating of {rating} stars, and {number_of_orders} recent orders, "
        
        if actual_price > competitor_price * 1.1:
            explanation += "your product is currently priced higher than competitors. "
        elif actual_price < competitor_price * 0.9:
            explanation += "your product is currently priced below competitors. "
        else:
            explanation += "your product is priced similarly to competitors. "
        
        if rating_factor > 0:
            explanation += f"The strong rating suggests potential for a price increase. "
        elif rating_factor < 0:
            explanation += f"The rating indicates a need for competitive pricing. "
        
        if order_factor > 0.01:
            explanation += f"Strong sales volume allows for pricing confidence. "
        
        if market_reason:
            explanation += f"Market analysis shows: {market_reason}"
        
        # Return the response
        response = {
            'recommendedPrice': round(recommended_price, 2),
            'minPrice': round(min_price, 2),
            'maxPrice': round(max_price, 2),
            'elasticityCategory': elasticity_category,
            'explanation': explanation,
            'ratingImpact': round(rating_factor * 100, 1),  # as percentage
            'orderImpact': round(order_factor * 100, 1),   # as percentage
            'marketImpact': round((market_adjustment - 1) * 100, 1)  # as percentage
        }
        
        # Include market features in response if available
        if market_features:
            response['marketInsights'] = {
                'priceTrend': market_features.get('price_trend', 'stable'),
                'priceVolatility': market_features.get('price_volatility', 0),
                'marketPosition': market_features.get('price_status', 'average'),
                'sentimentScore': market_features.get('sentiment_score', 0)
            }
        
        return jsonify(response)
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/product-types', methods=['GET'])
def get_product_types():
    # Return a list of supported product types
    product_types = [
        "Electronics", "Computers&Accessories", "MusicalInstruments", 
        "OfficeProducts", "Home&Kitchen", "HomeImprovement", 
        "Toys&Games", "Car&Motorbike", "Health&PersonalCare"
    ]
    return jsonify(product_types)

@app.route('/api/product-groups', methods=['GET'])
def get_product_groups():
    # Get the product type from query parameter
    product_type = request.args.get('productType', 'Electronics')
    
    # Define product groups for each product type
    product_groups_map = {
        "Electronics": ["Smartphones", "Cameras", "Headphones", "Tablets", "Wearables"],
        "Computers&Accessories": ["Laptops", "Desktops", "Monitors", "Keyboards", "Mice"],
        "MusicalInstruments": ["Guitars", "Keyboards", "Drums", "Accessories", "Recording Equipment"],
        "OfficeProducts": ["Printers", "Scanners", "Paper", "Stationery", "Calculators"],
        "Home&Kitchen": ["Appliances", "Cookware", "Utensils", "Furniture", "Decor"],
        "HomeImprovement": ["Tools", "Hardware", "Electrical", "Plumbing", "Flooring"],
        "Toys&Games": ["Action Figures", "Board Games", "Outdoor Play", "Educational Toys", "Electronic Toys"],
        "Car&Motorbike": ["Parts", "Accessories", "Tools", "Electronics", "Care Products"],
        "Health&PersonalCare": ["Vitamins", "Medical Supplies", "Personal Care", "Beauty", "Wellness"]
    }
    
    # Return product groups for the selected product type
    return jsonify(product_groups_map.get(product_type, []))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)
