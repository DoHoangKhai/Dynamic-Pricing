#!/usr/bin/env python
"""
Market data API endpoints for the dynamic pricing system.
Provides access to market data collection, analysis, and visualization.
"""

import os
import json
import requests
from flask import Blueprint, jsonify, request, send_file
from datetime import datetime, timedelta
import traceback
import sys

# Add parent directory to path to allow imports from sibling modules
sys.path.append('..')

# Import our new Amazon Data Service
from api.amazon_data_service import amazon_data_service, market_analyzer, analyze_product

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
TIME_SERIES_DIR = os.path.join(DATA_DIR, 'time_series')
ANALYSIS_DIR = os.path.join(DATA_DIR, 'analysis')
VISUALIZATIONS_DIR = os.path.join(ANALYSIS_DIR, 'visualizations')

# Create directories if they don't exist
for directory in [DATA_DIR, TIME_SERIES_DIR, ANALYSIS_DIR, VISUALIZATIONS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Create blueprint
market_data_bp = Blueprint('market_data', __name__)

# Initialize analyzer from the new module
# We'll still provide the existing routes for backward compatibility

@market_data_bp.route('/market/analyze', methods=['GET'])
def analyze_product_route():
    """Analyze a product using live Amazon data"""
    try:
        asin = request.args.get('asin')
        country = request.args.get('country', 'US')
        
        if not asin:
            return jsonify({
                'success': False,
                'message': 'ASIN parameter is required'
            }), 400
        
        # Use our new analyze_product function
        response = analyze_product(asin, country)
        
        # Save the analysis to file for future reference
        analysis_file = os.path.join(ANALYSIS_DIR, f"analysis_{asin}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
        with open(analysis_file, 'w') as f:
            json.dump(response, f, indent=2)
        
        return jsonify(response)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Error analyzing product: {str(e)}'
        }), 500

@market_data_bp.route('/market/refresh', methods=['POST'])
def refresh_market_data():
    """Trigger a refresh of market data"""
    try:
        data = request.get_json() or {}
        
        # Default parameters
        category = data.get('category', 'bestsellers')
        country = data.get('country', 'US')
        asin = data.get('asin')
        
        # If ASIN provided, focus on that product
        if asin:
            # Use our new Amazon Data Service
            product_details = amazon_data_service.get_product_details(asin)
            product_reviews = amazon_data_service.get_product_reviews(asin)
            
            # Save raw data
            raw_data_dir = os.path.join(DATA_DIR, 'raw')
            os.makedirs(raw_data_dir, exist_ok=True)
            
            with open(os.path.join(raw_data_dir, f"{asin}_details_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"), 'w') as f:
                json.dump(product_details, f, indent=2)
                
            with open(os.path.join(raw_data_dir, f"{asin}_reviews_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"), 'w') as f:
                json.dump(product_reviews, f, indent=2)
                
            # Return success message
            return jsonify({
                'success': True,
                'message': 'Market data refresh completed',
                'refreshed_data': {
                    'product_details': bool(product_details),
                    'product_reviews': bool(product_reviews)
                }
            })
        else:
            # If no ASIN, refresh category data
            best_sellers = amazon_data_service.get_best_sellers(category)
            deals = amazon_data_service.get_deals()
            
            # Save raw data
            raw_data_dir = os.path.join(DATA_DIR, 'raw')
            os.makedirs(raw_data_dir, exist_ok=True)
            
            with open(os.path.join(raw_data_dir, f"{category}_bestsellers_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"), 'w') as f:
                json.dump(best_sellers, f, indent=2)
                
            with open(os.path.join(raw_data_dir, f"deals_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"), 'w') as f:
                json.dump(deals, f, indent=2)
                
            # Return success message
            return jsonify({
                'success': True,
                'message': 'Market data refresh completed',
                'refreshed_data': {
                    'best_sellers': bool(best_sellers),
                    'deals': bool(deals)
                }
            })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Error refreshing market data: {str(e)}'
        }), 500

@market_data_bp.route('/market/status', methods=['GET'])
def market_data_status():
    """Get status of market data collection"""
    try:
        # Check data directories
        raw_data_dir = os.path.join(DATA_DIR, 'raw')
        os.makedirs(raw_data_dir, exist_ok=True)
        
        # Count files in each directory
        raw_files = len(os.listdir(raw_data_dir)) if os.path.exists(raw_data_dir) else 0
        ts_files = len(os.listdir(TIME_SERIES_DIR)) if os.path.exists(TIME_SERIES_DIR) else 0
        analysis_files = len(os.listdir(ANALYSIS_DIR)) if os.path.exists(ANALYSIS_DIR) else 0
        
        # Get timestamps of latest files if any exist
        latest_raw = None
        latest_ts = None
        latest_analysis = None
        
        if raw_files > 0:
            raw_files_list = os.listdir(raw_data_dir)
            if raw_files_list:
                latest_raw = max([os.path.getmtime(os.path.join(raw_data_dir, f)) for f in raw_files_list])
                latest_raw = datetime.fromtimestamp(latest_raw).strftime('%Y-%m-%d %H:%M:%S')
        
        if ts_files > 0:
            ts_files_list = os.listdir(TIME_SERIES_DIR)
            if ts_files_list:
                latest_ts = max([os.path.getmtime(os.path.join(TIME_SERIES_DIR, f)) for f in ts_files_list])
                latest_ts = datetime.fromtimestamp(latest_ts).strftime('%Y-%m-%d %H:%M:%S')
        
        if analysis_files > 0:
            analysis_files_list = [f for f in os.listdir(ANALYSIS_DIR) if os.path.isfile(os.path.join(ANALYSIS_DIR, f))]
            if analysis_files_list:
                latest_analysis = max([os.path.getmtime(os.path.join(ANALYSIS_DIR, f)) for f in analysis_files_list])
                latest_analysis = datetime.fromtimestamp(latest_analysis).strftime('%Y-%m-%d %H:%M:%S')
            
        # Get API status
        try:
            # Test API with a minimal request to check availability
            test_asin = "B07PXGQC1Q"  # Example ASIN (this is an arbitrary product)
            api_status = amazon_data_service.get_product_details(test_asin) is not None
        except Exception:
            api_status = False
            
        return jsonify({
            'success': True,
            'api_status': {
                'available': api_status,
                'last_checked': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'status': {
                'raw_files': raw_files,
                'time_series_files': ts_files,
                'analysis_files': analysis_files,
                'latest_raw_update': latest_raw,
                'latest_ts_update': latest_ts,
                'latest_analysis_update': latest_analysis
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Error getting market data status: {str(e)}'
        }), 500

@market_data_bp.route('/market/product-analysis/<asin>', methods=['GET'])
def product_analysis(asin):
    """Get analysis for a specific product"""
    try:
        country = request.args.get('country', 'US')
        
        # Use our market analyzer for comprehensive analysis
        result = market_analyzer.analyze_product(asin, "Electronics")
        
        if result.get('status') == 'error':
            return jsonify({
                'success': False,
                'message': 'Failed to analyze product',
                'errors': result.get('errors', [])
            }), 500
        
        # Create a simplified response structure
        response = {
            'success': True,
            'asin': asin,
            'product_info': result.get('product_info', {}),
            'price_analysis': result.get('price_analysis', {}),
            'competitive_analysis': result.get('competitive_analysis', {}),
            'pricing_context': result.get('pricing_context', {})
        }
        
        # Save analysis results
        analysis_file = os.path.join(ANALYSIS_DIR, f"analysis_{asin}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
        with open(analysis_file, 'w') as f:
            json.dump(response, f, indent=2)
            
        return jsonify(response)
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Error analyzing product {asin}: {str(e)}'
        }), 500

@market_data_bp.route('/market/pricing-features/<asin>', methods=['GET'])
def pricing_features(asin):
    """Get pricing features for a specific product"""
    try:
        country = request.args.get('country', 'US')
        
        # Use our market analyzer for comprehensive analysis
        result = market_analyzer.analyze_product(asin, "Electronics")
        
        if result.get('status') == 'error':
            return jsonify({
                'success': False,
                'message': 'Failed to extract pricing features',
                'errors': result.get('errors', [])
            }), 500
        
        # Extract pricing insights
        pricing_context = result.get('pricing_context', {})
        competitive_analysis = result.get('competitive_analysis', {})
        product_info = result.get('product_info', {})
        
        # Create a simplified response with pricing features
        response = {
            'success': True,
            'features': {
                'market_position': pricing_context.get('price_position', 'unknown'),
                'position_percentile': competitive_analysis.get('position_percentile', 50),
                'avg_market_price': competitive_analysis.get('avg_market_price'),
                'price_range': competitive_analysis.get('price_range', {}),
                'current_price': product_info.get('current_price'),
                'rating': product_info.get('rating'),
                'insights': pricing_context.get('insights', [])
            }
        }
            
        return jsonify(response)
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Error extracting pricing features for {asin}: {str(e)}'
        }), 500

@market_data_bp.route('/market/pricing-recommendations/<asin>', methods=['GET'])
def pricing_recommendations(asin):
    """Get pricing recommendations based on market analysis"""
    try:
        # Get optional parameters
        current_price = request.args.get('price')
        if current_price:
            try:
                current_price = float(current_price)
            except ValueError:
                current_price = None
        
        # Get product analysis
        result = market_analyzer.analyze_product(asin, "Electronics")
        
        if result.get('status') == 'error':
            return jsonify({
                'success': False,
                'message': 'Failed to generate pricing recommendations',
                'errors': result.get('errors', [])
            }), 500
        
        # Extract data for recommendations
        product_info = result.get('product_info', {})
        competitive_analysis = result.get('competitive_analysis', {})
        pricing_context = result.get('pricing_context', {})
        
        # Use current price from parameter or product info
        if not current_price:
            current_price = product_info.get('current_price')
            
        if not current_price:
            return jsonify({
                'success': False,
                'message': 'Current price not available',
            }), 400
        
        # Get market price data
        avg_market_price = competitive_analysis.get('avg_market_price')
        price_min = competitive_analysis.get('price_range', {}).get('min')
        price_max = competitive_analysis.get('price_range', {}).get('max')
        
        # Generate recommendations
        recommendations = []
        
        # Only generate if we have market data
        if all([avg_market_price, price_min, price_max]):
            # Competitive pricing recommendation
            if current_price > avg_market_price * 1.1:
                competitive_price = round(avg_market_price * 0.95, 2)
                recommendations.append({
                    'strategy': 'competitive',
                    'price': competitive_price,
                    'change_pct': round(((competitive_price - current_price) / current_price) * 100, 2),
                    'rationale': 'Aligning with market average to increase competitiveness'
                })
            
            # Premium pricing recommendation if rating is high
            rating = product_info.get('rating', 0)
            if rating and rating >= 4.5:
                premium_price = round(avg_market_price * 1.1, 2)
                recommendations.append({
                    'strategy': 'premium',
                    'price': premium_price,
                    'change_pct': round(((premium_price - current_price) / current_price) * 100, 2),
                    'rationale': 'High rating supports premium positioning'
                })
            
            # Value pricing recommendation
            if current_price < avg_market_price * 0.85:
                value_price = round(avg_market_price * 0.9, 2)
                recommendations.append({
                    'strategy': 'value',
                    'price': value_price,
                    'change_pct': round(((value_price - current_price) / current_price) * 100, 2),
                    'rationale': 'Increase price while maintaining value position'
                })
        
        response = {
            'success': True,
            'current_price': current_price,
            'market_price': avg_market_price,
            'price_range': {
                'min': price_min,
                'max': price_max
            },
            'recommendations': recommendations,
            'insights': pricing_context.get('insights', [])
        }
            
        return jsonify(response)
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Error generating pricing recommendations for {asin}: {str(e)}'
        }), 500

@market_data_bp.route('/market/competitive-position/<asin>', methods=['GET'])
def competitive_position(asin):
    """Get competitive positioning data for visualizations"""
    try:
        # Get product analysis
        result = market_analyzer.analyze_product(asin, "Electronics")
        
        if result.get('status') == 'error':
            return jsonify({
                'success': False,
                'message': 'Failed to analyze competitive position',
                'errors': result.get('errors', [])
            }), 500
        
        # Extract data for visualization
        product_info = result.get('product_info', {})
        competitive_analysis = result.get('competitive_analysis', {})
        current_price = product_info.get('current_price')
        
        # Get competitor data
        competitors = competitive_analysis.get('competitors', [])
        
        # Prepare visualization data
        competitors_data = []
        for comp in competitors:
            if 'price' in comp and 'title' in comp:
                competitors_data.append({
                    'name': comp['title'][:30] + '...' if len(comp['title']) > 30 else comp['title'],
                    'price': comp['price'],
                    'rating': comp.get('rating', 0)
                })
        
        # Add current product
        if current_price and 'title' in product_info:
            competitors_data.append({
                'name': product_info['title'][:30] + '...' if len(product_info['title']) > 30 else product_info['title'],
                'price': current_price,
                'rating': product_info.get('rating', 0),
                'is_current': True
            })
        
        # Sort by price
        competitors_data.sort(key=lambda x: x.get('price', 0))
        
        # Get price distribution
        price_distribution = competitive_analysis.get('price_distribution', {})
        
        response = {
            'success': True,
            'competitive_position': {
                'percentile': competitive_analysis.get('position_percentile', 50),
                'avg_market_price': competitive_analysis.get('avg_market_price'),
                'competitor_count': competitive_analysis.get('competitor_count', 0)
            },
            'competitors': competitors_data,
            'price_distribution': price_distribution,
            'product_info': {
                'title': product_info.get('title', ''),
                'price': current_price,
                'rating': product_info.get('rating', 0)
            }
        }
            
        return jsonify(response)
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Error analyzing competitive position for {asin}: {str(e)}'
        }), 500

def register_api_endpoints(app):
    """Register all API endpoints with the Flask app"""
    app.register_blueprint(market_data_bp, url_prefix='/api') 