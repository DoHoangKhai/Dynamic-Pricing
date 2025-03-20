from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import json
import traceback
from models.RL_env1 import DynamicPricingEnv
import os

# Import our market data API integration
from api.api_market_data import register_api_endpoints
from utils.market_data_analysis import MarketDataAnalyzer

# Import enhanced pricing model components
from models.pricing_strategies import PricingStrategy, CustomerSegmentation
from models.market_environment import MarketEnvironment

app = Flask(__name__)

# Initialize pricing environment
env = DynamicPricingEnv()

# Initialize pricing strategy and customer segmentation
pricing_strategy = PricingStrategy()
customer_segmentation = CustomerSegmentation()

# Initialize market data analyzer
market_analyzer = MarketDataAnalyzer()

# Register market data API endpoints
register_api_endpoints(app)

# Add route to directly access our new web scraper endpoints
@app.route('/market/analyze')
def market_analyze_proxy():
    """Proxy route to market data analyze endpoint"""
    from api.api_market_data import analyze_product
    return analyze_product()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict-price', methods=['POST'])
def predict_price():
    try:
        # Get input data from request
        data = request.get_json()
        print(f"[DEBUG] Received request data: {data}")

        # Extract input features
        product_type = data.get('productType', 'Electronics')
        product_group = data.get('productGroup', 'Laptops')
        actual_price = float(data.get('actualPrice', 1000))
        competitor_price = float(data.get('competitorPrice', 1000))
        rating = float(data.get('rating', 4.5))
        number_of_orders = int(data.get('numberOfOrders', 100))
        
        print(f"[DEBUG] Extracted features: type={product_type}, group={product_group}, price={actual_price}, competitor={competitor_price}, rating={rating}, orders={number_of_orders}")
        
        # Get additional market features if available
        asin = data.get('asin')
        market_features = {}
        
        if asin:
            # Try to get market insights for this product
            try:
                print(f"[DEBUG] Attempting to extract pricing features for ASIN: {asin}")
                features = market_analyzer.extract_pricing_features(asin)
                print(f"[DEBUG] Extracted features: {features}")
                if features:
                    market_features = features
            except Exception as e:
                # If market data fetch fails, continue without it
                print(f"[ERROR] Could not fetch market data for {asin}: {str(e)}")
                traceback.print_exc()
        
        # NEW: Get market-wide deals data for pricing adjustment
        deals_adjustment = 1.0
        deals_reason = ""
        try:
            # Import fetch_deals function from amazon_api
            from api.amazon_api import fetch_deals
            
            print(f"[DEBUG] Fetching deals data for price adjustment")
            
            # Fetch current deals data
            deals_data = fetch_deals("US")
            
            if not deals_data:
                print("[WARNING] fetch_deals returned None or empty response")
                deals_data = {}
            
            if "error" in deals_data:
                print(f"[WARNING] Error in deals data: {deals_data['error']}")
            
            print(f"[DEBUG] Deals data received: {deals_data.keys() if isinstance(deals_data, dict) else 'Not a dict'}")
            
            if deals_data and "data" in deals_data and "deals" in deals_data["data"]:
                deals = deals_data["data"]["deals"]
                
                # Calculate average discount
                total_discount = 0
                discount_count = 0
                
                for deal in deals:
                    if isinstance(deal, dict) and "savings_percentage" in deal:
                        try:
                            discount = deal["savings_percentage"]
                            if isinstance(discount, (int, float)):
                                total_discount += discount
                                discount_count += 1
                            elif isinstance(discount, str):
                                discount_value = float(discount.replace('%', ''))
                                total_discount += discount_value
                                discount_count += 1
                        except (ValueError, AttributeError) as e:
                            print(f"[WARNING] Could not parse discount: {e}")
                            continue
                
                avg_discount = total_discount / discount_count if discount_count > 0 else 0
                print(f"[DEBUG] Calculated avg_discount: {avg_discount:.2f}% from {discount_count} deals")
                
                # Apply deals-based adjustment
                if avg_discount > 20 and len(deals) > 30:
                    # High discount environment - be more competitive
                    deals_adjustment = 0.96  # 4% reduction
                    deals_reason = f"High market discounting detected (avg {avg_discount:.1f}% across {len(deals)} deals). "
                elif avg_discount > 10 and len(deals) > 20:
                    # Moderate discount environment
                    deals_adjustment = 0.98  # 2% reduction
                    deals_reason = f"Moderate market discounting detected (avg {avg_discount:.1f}% across {len(deals)} deals). "
                elif avg_discount < 5 and len(deals) < 10:
                    # Low discount environment - opportunity for higher margins
                    deals_adjustment = 1.01  # 1% increase
                    deals_reason = "Low market discounting activity detected. "
        except Exception as e:
            print(f"[ERROR] Error getting deals data: {str(e)}")
            traceback.print_exc()
        
        print(f"[DEBUG] Final deals_adjustment: {deals_adjustment}, reason: {deals_reason}")
        
        # Determine elasticity
        elasticity = 1.0
        if product_type.lower() in ['premium', 'luxury']:
            elasticity = 0.7
        elif product_type.lower() in ['commodity', 'basics', 'basic']:
            elasticity = 1.3
        elif actual_price <= 50:
            elasticity = 1.2  # Low-price items tend to be more elastic
        elif actual_price >= 200:
            elasticity = 0.8  # High-price items tend to be less elastic
        
        # Prepare product information for pricing strategy
        product = {
            'product_id': asin or 'temp_product',
            'product_type': product_type,
            'product_group': product_group,
            'price': actual_price,
            'cost': actual_price * 0.6,  # Estimate cost as 60% of price
            'elasticity': elasticity,
            'rating': rating,
            'ppi': actual_price / max(1, competitor_price),
            'number_of_orders': number_of_orders
        }
        
        # Prepare market information
        market_info = {
            'competitive_intensity': 0.7,  # Default to moderately competitive
            'price_trend': 0.0,  # Default to stable prices
            'current_price_ratio': actual_price / max(1, competitor_price)
        }
        
        # Incorporate market features if available
        if market_features:
            # Extract competition intensity from market features
            if market_features.get('price_status') == 'very competitive':
                market_info['competitive_intensity'] = 0.9
            elif market_features.get('price_status') == 'competitive':
                market_info['competitive_intensity'] = 0.7
            elif market_features.get('price_status') == 'premium':
                market_info['competitive_intensity'] = 0.4
                
            # Extract price trend
            if market_features.get('price_trend') == 'increasing':
                market_info['price_trend'] = 0.05
            elif market_features.get('price_trend') == 'decreasing':
                market_info['price_trend'] = -0.05
        
        # Get price recommendation from our enhanced pricing strategy
        strategy_recommendation = pricing_strategy.get_price_recommendations(product, market_info)
        strategy_price_ratio = strategy_recommendation.get('price_ratio', 1.0)
        
        # Calculate recommended price
        recommended_price = actual_price * strategy_price_ratio
        
        # Get customer segmentation data
        segments = customer_segmentation.get_segment_distribution(product_type, product_group, rating)
        segment_conversion = customer_segmentation.calculate_segment_conversion_probabilities(
            strategy_price_ratio, product
        )
        profit_multiplier = segment_conversion.get('expected_profit_multiplier', 1.0)
        
        # Calculate demand modifier for market adjustment
        demand_modifier = customer_segmentation.get_demand_modifier(
            product, strategy_price_ratio, (recommended_price - product['cost']) / recommended_price
        )
        
        # Apply market adjustment from API data
        market_adjustment = 1.0
        market_reason = ""
        
        if market_features:
            # Adjust pricing based on market position
            price_status = market_features.get('price_status', 'average')
            price_trend = market_features.get('price_trend', 'stable')
            sentiment_score = market_features.get('sentiment_score', 0)
            
            # Sentiment-based adjustment
            if sentiment_score > 0.5:
                market_adjustment *= 1.03
                market_reason += "Strong positive sentiment allows premium pricing. "
            elif sentiment_score < -0.3:
                market_adjustment *= 0.97
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
        
        # Apply market adjustment and deals adjustment to the recommended price
        final_price = recommended_price * market_adjustment * deals_adjustment
        
        # Calculate min and max recommended prices (±5% range)
        min_price = final_price * 0.95
        max_price = final_price * 1.05
        
        # Generate elasticity category for UI
        if elasticity > 1.2:
            elasticity_category = "high"
        elif elasticity < 0.8:
            elasticity_category = "low"
        else:
            elasticity_category = "medium"
        
        # Get top customer segment for explanation
        top_segment = max(segments.items(), key=lambda x: x[1]['weight'])
        segment_explanation = f"Your top customer segment is '{top_segment[0]}' ({int(top_segment[1]['weight']*100)}% of customers). "
        
        # Create pricing explanation
        if strategy_price_ratio < 0.97:
            strategy_explanation = "Our model recommends a competitive price below market to maximize sales volume and profit. "
        elif strategy_price_ratio > 1.03:
            strategy_explanation = "Our model recommends a premium price to maximize profit margins based on product attributes. "
        else:
            strategy_explanation = "Our model recommends staying close to market price for this product. "
        
        # Combine explanations
        explanation = strategy_explanation + segment_explanation
        
        if market_reason:
            explanation += f"Market analysis shows: {market_reason}"
            
        if deals_reason:
            explanation += deals_reason
        
        # Calculate impact factors for UI
        rating_impact = (rating - 3.0) * 5.0  # 5% impact per star above 3
        order_impact = min(5.0, number_of_orders / 50.0)  # Up to 5% for orders
        market_impact = (market_adjustment * deals_adjustment - 1.0) * 100.0  # Combined market impact
        
        # Return the response
        response = {
            'recommendedPrice': round(final_price, 2),
            'minPrice': round(min_price, 2),
            'maxPrice': round(max_price, 2),
            'elasticityCategory': elasticity_category,
            'explanation': explanation,
            'ratingImpact': round(rating_impact, 1),
            'orderImpact': round(order_impact, 1),
            'marketImpact': round(market_impact, 1)
        }
        
        # Include market features in response if available
        if market_features:
            response['marketInsights'] = {
                'priceTrend': market_features.get('price_trend', 'stable'),
                'priceVolatility': market_features.get('price_volatility', 0),
                'marketPosition': market_features.get('price_status', 'average'),
                'sentimentScore': market_features.get('sentiment_score', 0)
            }
            
        # Add deals information to market insights
        if deals_adjustment != 1.0:
            if 'marketInsights' not in response:
                response['marketInsights'] = {}
            response['marketInsights']['dealsActivity'] = {
                'adjustmentPercent': round((deals_adjustment - 1.0) * 100, 1),
                'reason': deals_reason.strip()
            }
        
        print(f"[DEBUG] Returning response: {response}")
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
        "Toys&Games", "Car&Motorbike", "Health&PersonalCare",
        "Premium", "Luxury", "Basics", "Commodity"  # Added new types for better classification
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
        "Health&PersonalCare": ["Vitamins", "Medical Supplies", "Personal Care", "Beauty", "Wellness"],
        "Premium": ["HighEndElectronics", "LuxuryWatches", "DesignerClothing", "GourmetFood"],
        "Luxury": ["FineJewelry", "DesignerAccessories", "CollectibleItems", "LuxuryAppliances"],
        "Basics": ["EssentialClothing", "OfficeSupplies", "KitchenBasics", "HomeEssentials"],
        "Commodity": ["Groceries", "CleaningSupplies", "PaperProducts", "BulkItems"]
    }
    
    # Return product groups for the selected product type
    return jsonify(product_groups_map.get(product_type, []))

@app.route('/api/market-deals', methods=['GET'])
def get_market_deals():
    """Get current deals data for price adjustment"""
    try:
        country = request.args.get('country', 'US')
        
        # Import here to avoid circular imports
        from api.amazon_api import fetch_deals
        
        print(f"[DEBUG] Fetching deals for country: {country}")
        
        # Fetch current deals data
        deals_data = fetch_deals(country)
        
        if not deals_data:
            print("[ERROR] fetch_deals returned None")
            return jsonify({
                'success': False,
                'message': "No response from deals API"
            }), 500
            
        if "error" in deals_data:
            print(f"[ERROR] Deal API error: {deals_data.get('error')}")
            return jsonify({
                'success': False,
                'message': f"Error fetching deals data: {deals_data.get('error', 'Unknown error')}"
            }), 500
            
        # Ensure the expected structure exists
        if "data" not in deals_data or "deals" not in deals_data.get("data", {}):
            print(f"[ERROR] Unexpected API response structure: {deals_data.keys()}")
            return jsonify({
                'success': False,
                'message': "Invalid API response structure"
            }), 500
            
        # Get the actual deals list
        deals = deals_data.get("data", {}).get("deals", [])
        
        if not deals:
            return jsonify({
                'success': True,
                'message': 'No active deals found',
                'deals_count': 0,
                'avg_discount': 0
            })
            
        # Calculate average discount
        total_discount = 0
        discount_count = 0
        
        for deal in deals:
            if isinstance(deal, dict) and "savings_percentage" in deal:
                try:
                    discount = deal["savings_percentage"]
                    if isinstance(discount, (int, float)):
                        total_discount += discount
                        discount_count += 1
                    elif isinstance(discount, str):
                        discount_value = float(discount.replace('%', ''))
                        total_discount += discount_value
                        discount_count += 1
                except (ValueError, AttributeError) as e:
                    print(f"[DEBUG] Error parsing discount: {e}")
                    continue
        
        avg_discount = total_discount / discount_count if discount_count > 0 else 0
        
        print(f"[DEBUG] Deals count: {len(deals)}, Avg discount: {avg_discount:.2f}%")
        
        return jsonify({
            'success': True,
            'deals_count': len(deals),
            'avg_discount': round(avg_discount, 2),
            'deals_summary': {
                'total_deals': len(deals),
                'average_discount': round(avg_discount, 2),
                'market_activity': 'high' if len(deals) > 50 else 'medium' if len(deals) > 20 else 'low'
            }
        })
        
    except Exception as e:
        import traceback
        print(f"[ERROR] Error processing deals data: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'Error processing deals data: {str(e)}'
        }), 500

# Load product details for market analysis
def get_product_details(asin):
    from api.api_market_data import analyze_product
    try:
        product_data = analyze_product(asin)
        return product_data
    except Exception as e:
        print(f"Error getting product details: {e}")
        return None

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)
