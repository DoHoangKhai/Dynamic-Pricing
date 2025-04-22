#!/usr/bin/env python
"""
Web application for QuickPrice
This app provides a web interface to run and visualize the dynamic pricing model.
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import numpy as np
import pandas as pd
import json
import traceback
import logging
from models.RL_env1 import DynamicPricingEnv
import os
from datetime import datetime, timedelta
import requests
import argparse
import random

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import our market data API integration
from api.api_market_data import register_api_endpoints
from utils.market_data_analysis import MarketDataAnalyzer

# Import our Amazon Data Service
from api.amazon_data_service import amazon_data_service, market_analyzer

# Import enhanced pricing model components
from models.pricing_strategies import PricingStrategy, CustomerSegmentation
from models.market_environment import MarketEnvironment
from models.enhanced_customer_segmentation import EnhancedCustomerSegmentation
from models.demand_forecasting import DemandForecaster
# Import our FinalOptimalModel adapter
from models.optimal_model_adapter import FinalOptimalModelAdapter

app = Flask(__name__)
# Set a secret key for session
app.secret_key = os.environ.get('SECRET_KEY', 'dev_secret_key_for_dynamic_pricing')

# Configure the application for subdirectory deployment if needed
application_root = os.environ.get('APPLICATION_ROOT', '/')
if application_root != '/':
    app.config['APPLICATION_ROOT'] = application_root
    
    # If deployed in a subdirectory, ensure URLs work correctly
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app)

# Initialize pricing environment
env = DynamicPricingEnv()

# Initialize pricing strategy and customer segmentation
pricing_strategy = PricingStrategy()
# Initialize our FinalOptimalModel
final_optimal_model = FinalOptimalModelAdapter()
# Replace basic customer segmentation with enhanced version
customer_segmentation = EnhancedCustomerSegmentation()
# Initialize demand forecaster
demand_forecaster = DemandForecaster()

# Register market data API endpoints
register_api_endpoints(app)

# RapidAPI credentials (update these with our new Amazon Data Service credentials)
RAPIDAPI_KEY = "2a6b802feamsh0f78b4cd091e889p149b18jsn9e0eda3f6870"
RAPIDAPI_HOST = "axesso-axesso-amazon-data-service-v1.p.rapidapi.com"

# Add route to directly access our new web scraper endpoints
@app.route('/market/analyze')
def market_analyze_proxy():
    """Proxy route to market data analyze endpoint"""
    from api.api_market_data import analyze_product_route
    return analyze_product_route()

@app.route('/')
def index():
    """Render the landing page as the main entry point"""
    return render_template('landing.html')

@app.route('/dashboard')
def dashboard():
    """Render the main dashboard"""
    return render_template('index.html')

@app.route('/landing')
def landing():
    """Render the landing page - keeping for backward compatibility"""
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    # For GET requests, just render the login page
    if request.method == 'GET':
        return render_template('login.html')
    
    # For POST requests, handle login
    if request.method == 'POST':
        try:
            data = request.get_json()
            app.logger.info(f"Login POST data: {data}")
            
            if data:
                email = data.get('email')
                
                # Set user in session
                session['user'] = {
                    'email': email,
                    'name': data.get('name', 'User'),
                    'role': data.get('role', 'User')
                }
                
                app.logger.info(f"User set in session: {session['user']}")
                return jsonify({'success': True, 'redirect': '/'})
            
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        except Exception as e:
            app.logger.error(f"Login error: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    # Clear the session
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    # Handle both GET and potential POST requests
    if request.method == 'POST':
        # Process signup data if needed
        pass
    return render_template('signup.html')

@app.route('/profile')
def profile():
    # Get session user
    session_user = session.get('user')
    
    # For debugging
    app.logger.info(f"Session user: {session_user}")
    
    # Check if user is logged in via session
    if session_user:
        return render_template('profile.html')
    
    # If no session, the client-side check in the template will handle redirection
    return render_template('profile.html')

@app.route('/check-auth', methods=['GET'])
def check_auth():
    """API endpoint to check authentication status"""
    # Log the current session state for debugging
    app.logger.info(f"Session check: {dict(session)}")
    
    if 'user' in session:
        app.logger.info(f"User authenticated in session: {session['user']}")
        return jsonify({
            'authenticated': True,
            'user': session['user']
        })
    
    app.logger.info("No user in session")
    return jsonify({
        'authenticated': False
    })

@app.route('/calculate-price', methods=['POST'])
def calculate_price():
    try:
        # Get request data
        data = request.get_json()
        product_type = data.get('productType')
        star_rating = float(data.get('starRating', 0))
        orders_per_month = int(data.get('ordersPerMonth', 0))
        competitor_price = float(data.get('competitorPrice', 0))
        product_cost = float(data.get('productCost', 0))
        actual_price = float(data.get('actualPrice', 0))
        
        # Basic validation
        if not all([product_type, star_rating, orders_per_month, competitor_price, product_cost, actual_price]):
            return jsonify({'error': 'Missing required parameters'}), 400

        # Create model with no parameters
        model = FinalOptimalModelAdapter()
        
        # Create product and market info dictionaries
        product = {
            'product_type': product_type,
            'rating': star_rating,
            'number_of_orders': orders_per_month,
            'price': actual_price,
            'cost': product_cost
        }
        
        market_info = {
            'competitor_price': competitor_price,
            'competitive_intensity': 0.5  # Default medium competitive intensity
        }
        
        # Get price recommendation using get_price_recommendations
        result = model.get_price_recommendations(product, market_info)
        
        # Map min_price and max_price to top level for frontend compatibility
        if 'price_range' in result:
            result['min_price'] = result['price_range'].get('min_price')
            result['max_price'] = result['price_range'].get('max_price')
        
        # Ensure pricing_factors are included in the response
        if 'pricing_factors' not in result:
            result['pricing_factors'] = {
                'rating_factor': 0,
                'competitor_factor': 0,
                'market_factor': 0,
                'volume_factor': 0,
                'product_type_factor': 0
            }
            
        # Add the percentage values to the response for UI display
        if 'price_factors_pct' in result:
            result['rating_impact'] = result['price_factors_pct']['rating_impact']
            result['competitor_impact'] = result['price_factors_pct']['competitor_impact']
            result['market_impact'] = result['price_factors_pct']['market_impact']
            result['volume_impact'] = result['price_factors_pct']['volume_impact']
        
        # Log the result for debugging
        app.logger.info(f"Price calculation result: {result}")
        
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Error calculating price: {str(e)}")
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
    product_type = request.args.get('product_type')
    
    if not product_type:
        return jsonify({'error': 'Missing product_type parameter'}), 400
    
    # Mapping of product type to available product groups
    product_groups_map = {
        'Electronics': ['Headphones', 'Speakers', 'Laptops', 'Cameras', 'Smartphones', 'Tablets', 'Wearables'],
        'Home': ['Furniture', 'Appliances', 'Kitchenware', 'Bedding', 'Lighting', 'Décor'],
        'Books': ['Fiction', 'Non-fiction', 'Children', 'Educational', 'Comics', 'Biography'],
        'Fashion': ['Men', 'Women', 'Kids', 'Footwear', 'Accessories', 'Sportswear'],
        'Luxury': ['Watches', 'Jewelry', 'Designer', 'Premium'],
        'Toys': ['Board Games', 'Action Figures', 'Dolls', 'Educational', 'Outdoor', 'Electronic']
    }
    
    # Return product groups for the selected product type
    return jsonify(product_groups_map.get(product_type, []))

@app.route('/api/best-sellers', methods=['GET'])
def get_best_sellers():
    asin = request.args.get('asin', '')
    category = request.args.get('category', 'Electronics')
    
    try:
        # Use our new Amazon Data Service
        response = amazon_data_service.fetch_best_sellers(category)
        
        if response.get("success"):
            return jsonify(response)
        else:
            logger.error(f"API request failed: {response.get('error', 'Unknown error')}")
            return jsonify({"error": f"Failed to fetch best sellers data", "data": generate_dummy_best_sellers()}), 200
    
    except Exception as e:
        logger.exception(f"Error fetching best sellers data: {str(e)}")
        return jsonify({"error": f"Exception occurred: {str(e)}", "data": generate_dummy_best_sellers()}), 200

@app.route('/api/product-details', methods=['GET'])
def get_product_details():
    asin = request.args.get('asin', '')
    if not asin:
        return jsonify({"error": "ASIN parameter is required"}), 400
    
    try:
        # Use our new Amazon Data Service
        response = amazon_data_service.fetch_product_details(asin)
        
        if response.get("success"):
            return jsonify(response)
        else:
            logger.error(f"API request failed: {response.get('error', 'Unknown error')}")
            return jsonify({"error": f"Failed to fetch product details", "data": generate_dummy_product_details(asin)}), 200
    
    except Exception as e:
        logger.exception(f"Error fetching product details: {str(e)}")
        return jsonify({"error": f"Exception occurred: {str(e)}", "data": generate_dummy_product_details(asin)}), 200

@app.route('/api/reviews', methods=['GET'])
def get_reviews():
    asin = request.args.get('asin', '')
    if not asin:
        return jsonify({"error": "ASIN parameter is required"}), 400
    
    try:
        # Use our new Amazon Data Service
        response = amazon_data_service.fetch_product_reviews(asin)
        
        if response.get("success"):
            return jsonify(response)
        else:
            logger.error(f"API request failed: {response.get('error', 'Unknown error')}")
            return jsonify({"error": f"Failed to fetch reviews", "data": generate_dummy_reviews(asin)}), 200
    
    except Exception as e:
        logger.exception(f"Error fetching reviews: {str(e)}")
        return jsonify({"error": f"Exception occurred: {str(e)}", "data": generate_dummy_reviews(asin)}), 200

@app.route('/api/deals', methods=['GET'])
def get_deals():
    try:
        # Use our new Amazon Data Service
        response = amazon_data_service.fetch_deals()
        
        if response.get("success"):
            return jsonify(response)
        else:
            logger.error(f"API request failed: {response.get('error', 'Unknown error')}")
            return jsonify({"error": f"Failed to fetch deals", "data": generate_dummy_deals()}), 200
    
    except Exception as e:
        logger.exception(f"Error fetching deals: {str(e)}")
        return jsonify({"error": f"Exception occurred: {str(e)}", "data": generate_dummy_deals()}), 200

@app.route('/api/market-deals', methods=['GET'])
def get_market_deals():
    """Get current market-wide deals data with pricing adjustment suggestion"""
    try:
        # Get country parameter
        country = request.args.get('country', 'US')
        
        # Use our new Amazon Data Service
        try:
            deals_data = amazon_data_service.fetch_deals(country)
            
            if not deals_data or deals_data.get("success") == False:
                raise Exception(deals_data.get("error", "Error fetching deals"))
                
        except Exception as e:
            # API error handling
            print(f"[WARNING] Deals API fetch failed: {str(e)}")
            return jsonify({
                'success': False,
                'error': f"API error: {str(e)}",
                'fallback_used': True,
                'deals_data': {
                    'deals_adjustment': 1.0,
                    'adjustment_reason': 'Market data unavailable',
                    'market_discount_avg': 0,
                    'deal_count': 0
                }
            })
        
        # Process deals data if available
        deals = []
        total_discount = 0
        discount_count = 0
        deals_adjustment = 1.0
        adjustment_reason = "No significant market trend detected"
        
        if deals_data and "data" in deals_data and "deals" in deals_data["data"]:
            deals = deals_data["data"]["deals"]
            
            # Calculate average discount
            for deal in deals:
                if isinstance(deal, dict) and "discount_percentage" in deal:
                    try:
                        if isinstance(deal["discount_percentage"], str):
                            savings = float(deal["discount_percentage"].replace("%", ""))
                        else:
                            savings = float(deal["discount_percentage"])
                        total_discount += savings
                        discount_count += 1
                    except (ValueError, TypeError) as e:
                        print(f"[WARNING] Error parsing discount percentage: {e}")
            
            # Calculate average discount and determine adjustment
            if discount_count > 0:
                avg_discount = total_discount / discount_count
                
                # Set adjustment based on market conditions
                if avg_discount > 30:
                    deals_adjustment = 0.95
                    adjustment_reason = f"High discount marketplace (avg. {avg_discount:.1f}% off)"
                elif avg_discount > 20:
                    deals_adjustment = 0.98
                    adjustment_reason = f"Moderate discount marketplace (avg. {avg_discount:.1f}% off)"
                elif avg_discount < 10 and discount_count < 20:
                    deals_adjustment = 1.02
                    adjustment_reason = f"Low discount marketplace (avg. {avg_discount:.1f}% off)"
                else:
                    adjustment_reason = f"Standard marketplace discounts (avg. {avg_discount:.1f}% off)"
        
        # Prepare response
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'deals_data': {
                'deals_adjustment': deals_adjustment,
                'adjustment_reason': adjustment_reason,
                'market_discount_avg': total_discount / discount_count if discount_count > 0 else 0,
                'deal_count': len(deals)
            }
        })
        
    except Exception as e:
        print(f"[ERROR] Exception in get_market_deals: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'fallback_used': True,
            'deals_data': {
                'deals_adjustment': 1.0,
                'adjustment_reason': 'Error processing request',
                'market_discount_avg': 0,
                'deal_count': 0
            }
        })

# Load product details for market analysis
def get_product_details(asin):
    try:
        # Use our new Amazon Data Service
        product_data = market_analyzer.analyze_product(asin)
        return product_data
    except Exception as e:
        print(f"Error getting product details: {e}")
        return None

# Helper functions to generate dummy data
def generate_dummy_best_sellers():
    """Generate dummy best sellers data"""
    import random
    
    # Generate random bestseller products
    product_categories = ["Electronics", "Home & Kitchen", "Books", "Toys"]
    product_templates = [
        "Wireless Headphones",
        "Smart Speaker",
        "Bluetooth Earbuds",
        "Fitness Tracker",
        "Robot Vacuum",
        "Air Fryer",
        "Coffee Maker",
        "Best-selling Novel",
        "Self-help Book",
        "Educational Toy"
    ]
    
    results = []
    for i in range(10):
        rank = i + 1
        product_name = f"{random.choice(product_templates)} {chr(65 + i)}"
        category = random.choice(product_categories)
        price = round(random.uniform(20, 300), 2)
        
        results.append({
            "rank": rank,
            "asin": f"B0{i}XXXXXX",
            "product_name": product_name,
            "category": category,
            "price": price,
            "rating": round(random.uniform(3.5, 5.0), 1),
            "reviews_count": random.randint(50, 5000)
        })
    
    return {
        "success": True,
        "data": {
            "category": "bestsellers",
            "results": results
        }
    }

def generate_dummy_product_details(asin):
    """Generate dummy product details data"""
    import random
    from datetime import datetime, timedelta
    
    # Generate price history for the last 30 days
    price_history = []
    base_price = round(random.uniform(50, 300), 2)
    
    for i in range(30):
        date = (datetime.now() - timedelta(days=30-i)).strftime("%b %d")
        price_variation = round(random.uniform(-20, 20), 2)
        price = max(10, base_price + price_variation)
        market_avg = price * random.uniform(0.9, 1.2)
        
        price_history.append({
            "date": date,
            "price": round(price, 2),
            "marketAverage": round(market_avg, 2)
        })
    
    return {
        "success": True,
        "data": {
            "asin": asin,
            "product_name": f"Product {asin}",
            "brand": "Brand Name",
            "current_price": base_price,
            "currency": "USD",
            "availability": "In Stock",
            "features": ["Feature 1", "Feature 2", "Feature 3"],
            "rating": round(random.uniform(3.0, 5.0), 1),
            "reviews_count": random.randint(50, 3000),
            "images": ["https://example.com/image1.jpg"],
            "categories": ["Electronics", "Accessories"],
            "priceHistory": price_history
        }
    }

def generate_dummy_reviews(asin):
    """Generate dummy reviews data"""
    import random
    from datetime import datetime, timedelta
    
    # Review templates
    positive_templates = [
        "Great product, exactly what I needed!",
        "Highly recommend this, works perfectly.",
        "Very satisfied with my purchase, will buy again.",
        "Quality is excellent, fast shipping too.",
        "Does everything as advertised, happy customer."
    ]
    
    neutral_templates = [
        "It's okay, not great but not bad either.",
        "Does the job but could be better.",
        "Average product for the price.",
        "Some good features but also a few issues.",
        "Expected more but it's acceptable."
    ]
    
    negative_templates = [
        "Disappointed with the quality.",
        "Broke after a few uses, not durable.",
        "Not worth the money, avoid buying.",
        "Did not work as expected, returning it.",
        "Poor customer service, had issues with delivery."
    ]
    
    # Generate mock reviews
    reviews = []
    review_count = random.randint(5, 20)
    
    for i in range(review_count):
        rating = random.randint(1, 5)
        
        if rating >= 4:
            review_text = random.choice(positive_templates)
        elif rating >= 3:
            review_text = random.choice(neutral_templates)
        else:
            review_text = random.choice(negative_templates)
        
        days_ago = random.randint(1, 90)
        date = (datetime.now() - timedelta(days=days_ago)).strftime("%B %d, %Y")
        
        reviews.append({
            "id": f"review_{i}",
            "rating": rating,
            "title": f"Review {i+1}",
            "text": review_text,
            "date": date,
            "verified_purchase": random.choice([True, False]),
            "helpfulness": f"{random.randint(0, 20)} people found this helpful"
        })
    
    # Calculate sentiment percentages
    ratings = [review["rating"] for review in reviews]
    sentiment = {
        "positive": len([r for r in ratings if r >= 4]) / len(ratings) * 100 if ratings else 0,
        "neutral": len([r for r in ratings if r == 3]) / len(ratings) * 100 if ratings else 0,
        "negative": len([r for r in ratings if r <= 2]) / len(ratings) * 100 if ratings else 0
    }
    
    return {
        "success": True,
        "data": {
            "asin": asin,
            "product_name": f"Product {asin}",
            "total_reviews": review_count,
            "average_rating": round(sum(ratings) / len(ratings), 1) if ratings else 0,
            "sentiment": sentiment,
            "rating_breakdown": {
                "5_star": f"{random.randint(40, 70)}%",
                "4_star": f"{random.randint(10, 30)}%",
                "3_star": f"{random.randint(5, 15)}%",
                "2_star": f"{random.randint(1, 10)}%",
                "1_star": f"{random.randint(1, 10)}%"
            },
            "reviews": reviews
        }
    }

def generate_dummy_deals():
    """Generate dummy deals data"""
    import random
    from datetime import datetime, timedelta
    
    # Generate mock deals
    deals = []
    deal_count = random.randint(5, 15)
    
    product_templates = [
        "Wireless Headphones",
        "Smart Speaker",
        "Bluetooth Earbuds",
        "Fitness Tracker",
        "Robot Vacuum",
        "Air Fryer",
        "Coffee Maker",
        "Best-selling Novel",
        "Self-help Book",
        "Educational Toy"
    ]
    
    for i in range(deal_count):
        regular_price = random.randint(50, 300)
        savings_percentage = random.randint(10, 50)
        sale_price = regular_price * (1 - savings_percentage/100)
        
        expiry_days = random.randint(1, 7)
        expiry_date = (datetime.now() + timedelta(days=expiry_days)).strftime("%Y-%m-%d")
        
        deals.append({
            "title": random.choice(product_templates),
            "asin": f"B0{i}XXXXXX",
            "url": f"https://example.com/product/B0{i}XXXXXX",
            "regular_price": regular_price,
            "deal_price": round(sale_price, 2),
            "currency": "USD",
            "discount_percentage": savings_percentage,
            "savings_amount": round(regular_price - sale_price, 2),
            "expiry_date": expiry_date,
            "deal_type": random.choice(["Lightning Deal", "Deal of the Day", "Limited Time Offer"]),
            "rating": round(random.uniform(3.5, 5.0), 1),
            "reviews_count": random.randint(10, 1000)
        })
    
    return {
        "success": True,
        "data": {
            "deals_count": deal_count,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "deals": deals
        }
    }

@app.route('/api/market/price-history/<asin>')
def get_price_history(asin):
    """API endpoint for fetching price history data."""
    logging.info(f"Fetching price history for ASIN: {asin}")
    
    # Skip validation - assuming any ASIN is valid for API purposes
    try:
        # Get basic product data
        product_data = {
            'asin': asin,
            'title': f'Product {asin}',
            'current_price': 999.99
        }
        
        logging.info(f"Product data fetched for price history: {product_data}")
        
        # Generate synthetic price history data
        price_history = []
        base_price = float(product_data['current_price'])
        
        # Generate 90 days of price history
        for i in range(90):
            date = datetime.now() - timedelta(days=90-i)
            # Add some random variation to prices
            your_price = base_price * (1 + ((random.random() * 0.12) - 0.06))
            market_avg = base_price * 0.9 * (1 + ((random.random() * 0.16) - 0.08))
            
            price_history.append({
                'date': date.strftime('%Y-%m-%d'),
                'your_price': round(your_price, 2),
                'market_avg': round(market_avg, 2)
            })
        
        return jsonify({
            'success': True,
            'asin': asin,
            'product_title': product_data['title'],
            'current_price': product_data['current_price'],
            'price_history': price_history
        })
        
    except Exception as e:
        logging.error(f"Error fetching price history: {str(e)}")
        return jsonify({
            'success': False,
            'message': f"Failed to fetch price history: {str(e)}"
        })

@app.route('/api/market/competitive-position/<asin>')
def get_competitive_position(asin):
    """API endpoint for fetching competitive position data."""
    logging.info(f"Fetching competitive position for ASIN: {asin}")
    
    # Skip validation - assuming any ASIN is valid for API purposes
    try:
        # Current product price
        current_price = 999.99
        
        # Simulating market data
        market_avg_price = current_price * 0.9  # 10% lower
        percentile = random.randint(65, 95)
        
        # Generate price distribution
        price_ranges = [
            f"${round(market_avg_price * 0.7, 2)} - ${round(market_avg_price * 0.8, 2)}",
            f"${round(market_avg_price * 0.8, 2)} - ${round(market_avg_price * 0.9, 2)}",
            f"${round(market_avg_price * 0.9, 2)} - ${round(market_avg_price, 2)}",
            f"${round(market_avg_price, 2)} - ${round(market_avg_price * 1.1, 2)}",
            f"${round(market_avg_price * 1.1, 2)} - ${round(market_avg_price * 1.2, 2)}"
        ]
        
        price_distribution = {
            'brackets': [
                {'range': price_ranges[0], 'count': random.randint(5, 15)},
                {'range': price_ranges[1], 'count': random.randint(10, 25)},
                {'range': price_ranges[2], 'count': random.randint(30, 50)},
                {'range': price_ranges[3], 'count': random.randint(15, 35)},
                {'range': price_ranges[4], 'count': random.randint(5, 20)}
            ]
        }
        
        # Calculate total competitors
        competitor_count = sum(bracket['count'] for bracket in price_distribution['brackets'])
        
        # Generate pricing insights
        if percentile > 90:
            insights = [
                "Your price is significantly higher than most competitors.",
                "Consider lowering price to increase competitiveness.",
                "Premium positioning may require additional value justification."
            ]
        elif percentile > 75:
            insights = [
                "Your price is higher than most competitors.",
                "Ensure product quality justifies the premium pricing.",
                "Consider promotional offers to increase appeal."
            ]
        elif percentile > 50:
            insights = [
                "Your price is in the upper mid-range of the market.",
                "Competitive but not the cheapest option.",
                "Highlight value proposition to justify pricing."
            ]
        elif percentile > 25:
            insights = [
                "Your price is in the lower mid-range of the market.",
                "Good value positioning relative to competitors.",
                "Consider testing slight price increases."
            ]
        else:
            insights = [
                "Your price is lower than most competitors.",
                "Potential to increase profit margins.",
                "Consider positioning as value leader."
            ]
        
        return jsonify({
            'success': True,
            'asin': asin,
            'competitive_position': {
                'percentile': percentile,
                'avg_market_price': round(market_avg_price, 2),
                'competitor_count': competitor_count
            },
            'price_distribution': price_distribution,
            'pricing_context': {
                'insights': insights
            }
        })
        
    except Exception as e:
        logging.error(f"Error fetching competitive position: {str(e)}")
        return jsonify({
            'success': False,
            'message': f"Failed to fetch competitive position: {str(e)}"
        })

@app.route('/api/market/data-status')
def get_market_data_status():
    """API endpoint for checking market data availability."""
    asin = request.args.get('asin', '')
    
    # Skip validation - assuming any ASIN is valid
    try:
        # For demo purposes, always return that data is available
        return jsonify({
            'success': True,
            'asin': asin,
            'product_available': True,
            'price_history': True,
            'competitor_data': True,
            'reviews_data': True
        })
        
    except Exception as e:
        logging.error(f"Error fetching market data status: {str(e)}")
        return jsonify({
            'success': False,
            'message': f"Failed to fetch market data status: {str(e)}"
        })

@app.route('/api/market/competitive-position-by-keyword/<keyword>', methods=['GET'])
def get_competitive_position_by_keyword(keyword):
    """Get competitive position data using product keyword instead of ASIN"""
    try:
        logger.info(f"Getting competitive position data for keyword: {keyword}")
        
        # Create a new instance of MarketDataAnalyzer
        from utils.market_data_analysis import MarketDataAnalyzer
        keyword_analyzer = MarketDataAnalyzer()
        
        # Use MarketDataAnalyzer to get competitive position data based on keyword
        result = keyword_analyzer.analyze_keyword(keyword)
        
        if not result.get('success', False):
            return jsonify({
                'success': False,
                'message': result.get('error', 'Failed to analyze keyword')
            }), 500
        
        # Return the actual result from the analyzer
        return jsonify(result)
    except Exception as e:
        logger.exception(f"Error getting competitive position by keyword: {str(e)}")
        return jsonify({
            'success': False,
            'message': f"Error processing request: {str(e)}"
        }), 500

@app.route('/api/search-product', methods=['GET'])
def search_product():
    """API endpoint for searching products by keyword"""
    keyword = request.args.get('keyword', '')
    
    if not keyword:
        return jsonify({'error': 'Keyword parameter is required'}), 400
    
    try:
        app.logger.info(f"Searching products with keyword: {keyword}")
        
        # Use the competitive position by keyword endpoint functionality
        from utils.market_data_analysis import MarketDataAnalyzer
        analyzer = MarketDataAnalyzer()
        
        # Get the result directly from the analyzer
        result = analyzer.analyze_keyword(keyword)
        
        app.logger.info(f"Search results for '{keyword}': {result.get('success', False)}")
        
        if not result.get('success', False):
            app.logger.error(f"Failed to get search results: {result.get('error', 'Unknown error')}")
            return jsonify({
                'success': False,
                'message': result.get('error', 'Failed to retrieve search results'),
                'keyword': keyword
            }), 500
        
        # Ensure we have the expected data structure
        if 'products' not in result and 'competitive_analysis' in result and 'competitors' in result.get('competitive_analysis', {}):
            # Transform competitors data into products format for backward compatibility
            products = []
            for comp in result['competitive_analysis']['competitors']:
                products.append({
                    'asin': comp.get('asin', ''),
                    'title': comp.get('name', ''),
                    'price': comp.get('price', 0),
                    'rating': comp.get('rating', 0),
                    'image': comp.get('image', '')
                })
            result['products'] = products
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Error searching products: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f"Error searching products: {str(e)}",
            'keyword': keyword
        }), 500

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='QuickPrice Dashboard')
    parser.add_argument('--port', type=int, default=os.environ.get('PORT', 5050),
                        help='Port to run the server on')
    args = parser.parse_args()
    
    # Get debug mode from environment variable
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    
    # Start the app
    app.run(debug=debug_mode, host='0.0.0.0', port=args.port) 