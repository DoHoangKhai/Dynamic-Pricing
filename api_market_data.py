#!/usr/bin/env python
"""
API endpoints for market data visualization and pricing features.
Integrates with app.py to provide market insights for pricing model.
"""

import os
import json
from flask import Blueprint, jsonify, request, send_file
import pandas as pd
from datetime import datetime

# Import our market data modules
from market_data_scheduler import collect_data_job, load_config
from market_data_analysis import MarketDataAnalyzer

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

# Initialize analyzer
analyzer = MarketDataAnalyzer()

@market_data_bp.route('/market/refresh', methods=['POST'])
def refresh_market_data():
    """Trigger a refresh of market data"""
    try:
        data = request.get_json() or {}
        
        # Default parameters
        category = data.get('category', 'bestsellers')
        country = data.get('country', 'US')
        asin = data.get('asin')
        
        # Load configuration
        config = load_config()
        
        # If ASIN provided, focus on that product
        if asin:
            config['products'] = [asin]
        
        # Run data collection
        result = collect_data_job(config)
        
        return jsonify({
            'success': True,
            'message': 'Market data refresh started',
            'result': result
        })
    except Exception as e:
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
        
        # Count files in each directory
        raw_files = len(os.listdir(raw_data_dir)) if os.path.exists(raw_data_dir) else 0
        ts_files = len(os.listdir(TIME_SERIES_DIR)) if os.path.exists(TIME_SERIES_DIR) else 0
        analysis_files = len(os.listdir(ANALYSIS_DIR)) if os.path.exists(ANALYSIS_DIR) else 0
        
        # Get timestamps of latest files
        latest_raw = max([os.path.getmtime(os.path.join(raw_data_dir, f)) for f in os.listdir(raw_data_dir)]) if raw_files > 0 else None
        latest_ts = max([os.path.getmtime(os.path.join(TIME_SERIES_DIR, f)) for f in os.listdir(TIME_SERIES_DIR)]) if ts_files > 0 else None
        latest_analysis = max([os.path.getmtime(os.path.join(ANALYSIS_DIR, f)) for f in os.listdir(ANALYSIS_DIR) if os.path.isfile(os.path.join(ANALYSIS_DIR, f))]) if analysis_files > 0 else None
        
        # Format timestamps
        if latest_raw:
            latest_raw = datetime.fromtimestamp(latest_raw).strftime('%Y-%m-%d %H:%M:%S')
        if latest_ts:
            latest_ts = datetime.fromtimestamp(latest_ts).strftime('%Y-%m-%d %H:%M:%S')
        if latest_analysis:
            latest_analysis = datetime.fromtimestamp(latest_analysis).strftime('%Y-%m-%d %H:%M:%S')
            
        # Get config
        config = load_config()
        
        return jsonify({
            'success': True,
            'status': {
                'raw_files': raw_files,
                'time_series_files': ts_files,
                'analysis_files': analysis_files,
                'latest_raw_update': latest_raw,
                'latest_ts_update': latest_ts,
                'latest_analysis_update': latest_analysis
            },
            'config': config
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting market data status: {str(e)}'
        }), 500

@market_data_bp.route('/market/product-analysis/<asin>', methods=['GET'])
def product_analysis(asin):
    """Get analysis for a specific product"""
    try:
        country = request.args.get('country', 'US')
        days = int(request.args.get('days', 30))
        
        # Analyze price trends
        price_trends = analyzer.analyze_price_trends(asin, country, days)
        
        # Analyze market position
        market_position = analyzer.analyze_market_position(asin, country=country)
        
        # Analyze review sentiment
        sentiment = analyzer.analyze_review_sentiment(asin, country)
        
        # Create response
        response = {
            'success': True,
            'asin': asin,
            'price_trends': price_trends,
            'market_position': market_position,
            'sentiment': sentiment
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error analyzing product {asin}: {str(e)}'
        }), 500

@market_data_bp.route('/market/price-visualization/<asin>', methods=['GET'])
def price_visualization(asin):
    """Get visualization for price trends"""
    try:
        country = request.args.get('country', 'US')
        days = int(request.args.get('days', 30))
        
        # Generate visualization
        viz_path = analyzer.visualize_price_trends(asin, country, days)
        
        if not viz_path:
            return jsonify({
                'success': False,
                'message': f'No data available for price visualization of {asin}'
            }), 404
            
        return send_file(viz_path, mimetype='image/png')
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error generating price visualization for {asin}: {str(e)}'
        }), 500

@market_data_bp.route('/market/sentiment-visualization/<asin>', methods=['GET'])
def sentiment_visualization(asin):
    """Get visualization for sentiment distribution"""
    try:
        country = request.args.get('country', 'US')
        
        # Generate visualization
        viz_path = analyzer.visualize_sentiment_distribution(asin, country)
        
        if not viz_path:
            return jsonify({
                'success': False,
                'message': f'No data available for sentiment visualization of {asin}'
            }), 404
            
        return send_file(viz_path, mimetype='image/png')
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error generating sentiment visualization for {asin}: {str(e)}'
        }), 500

@market_data_bp.route('/market/report', methods=['GET'])
def market_report():
    """Get latest market report"""
    try:
        category = request.args.get('category', 'bestsellers')
        country = request.args.get('country', 'US')
        
        # Get latest report
        report = analyzer.get_latest_market_report(category, country)
        
        if not report:
            return jsonify({
                'success': False,
                'message': f'No market report available for {category} in {country}'
            }), 404
            
        return jsonify({
            'success': True,
            'report': report
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error retrieving market report: {str(e)}'
        }), 500

@market_data_bp.route('/market/generate-report', methods=['POST'])
def generate_market_report():
    """Generate a new market report"""
    try:
        data = request.get_json() or {}
        
        # Get parameters
        category = data.get('category', 'bestsellers')
        country = data.get('country', 'US')
        asins = data.get('asins', [])
        
        if not asins:
            # Load config to get default products
            config = load_config()
            asins = config.get('products', [])
            
            if not asins:
                return jsonify({
                    'success': False,
                    'message': 'No products specified for report generation'
                }), 400
        
        # Generate report
        report = analyzer.generate_market_report(asins, category, country)
        
        return jsonify({
            'success': True,
            'message': 'Market report generated successfully',
            'report': report
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error generating market report: {str(e)}'
        }), 500

@market_data_bp.route('/market/pricing-features/<asin>', methods=['GET'])
def pricing_features(asin):
    """Get pricing features for a specific product"""
    try:
        country = request.args.get('country', 'US')
        
        # Extract pricing features
        features = analyzer.extract_pricing_features(asin, country)
        
        if not features:
            return jsonify({
                'success': False,
                'message': f'No data available for pricing features of {asin}'
            }), 404
            
        return jsonify({
            'success': True,
            'features': features
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error extracting pricing features for {asin}: {str(e)}'
        }), 500

def register_api_endpoints(app):
    """Register all API endpoints with the Flask app"""
    app.register_blueprint(market_data_bp, url_prefix='/api') 