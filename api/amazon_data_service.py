#!/usr/bin/env python
"""
Amazon Data Service API adapter for the dynamic pricing system.
Provides a unified interface to the Axesso Amazon Data Service API.
"""

import os
import json
import requests
import time
from typing import Dict, Optional, List, Any
from datetime import datetime
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AmazonDataService:
    """Main service class for Amazon Data API interactions"""
    
    def __init__(self):
        self.headers = {
            "x-rapidapi-key": "2a6b802feamsh0f78b4cd091e889p149b18jsn9e0eda3f6870",
            "x-rapidapi-host": "axesso-axesso-amazon-data-service-v1.p.rapidapi.com"
        }
        self.base_url = "https://axesso-axesso-amazon-data-service-v1.p.rapidapi.com/amz"
        self.request_delay = 1  # Delay between requests in seconds
        self.cache = {}  # Simple in-memory cache
        self.cache_ttl = 3600  # Cache TTL in seconds (1 hour)
        
    def _make_request(self, endpoint: str, params: Dict = None, use_cache: bool = True) -> Optional[Dict]:
        """Make API request with error handling, rate limiting and caching"""
        cache_key = f"{endpoint}:{json.dumps(params or {})}"
        
        # Check cache first
        if use_cache and cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            cache_time = cache_entry.get('timestamp', 0)
            if time.time() - cache_time < self.cache_ttl:
                logger.info(f"Using cached data for {endpoint}")
                return cache_entry.get('data')
        
        try:
            time.sleep(self.request_delay)  # Rate limiting
            url = f"{self.base_url}/{endpoint}"
            logger.info(f"Making API request to {url} with params {params}")
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Cache the result
            if use_cache:
                self.cache[cache_key] = {
                    'data': data,
                    'timestamp': time.time()
                }
                
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None

    def get_product_details(self, asin: str) -> Optional[Dict]:
        """Get details for a specific product by ASIN"""
        logging.info(f"Getting product details for ASIN: {asin}")
        
        # Make a direct API call to the Amazon Data Service
        params = {"url": f"https://www.amazon.com/dp/{asin}/"}
        
        try:
            url = f"{self.base_url}/amazon-lookup-product"
            logging.info(f"Making API request to {url} with params {params}")
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Log the raw response for debugging
            logging.info(f"API response: {data}")
            
            # Return the data directly without any transformation
            return data
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            return {"success": False, "error": str(e)}

    def get_competitive_data(self, keyword: str, page: int = 1) -> Optional[Dict]:
        """Get competitive product data for market analysis"""
        params = {
            "domainCode": "com",
            "keyword": keyword,
            "page": str(page),
            "excludeSponsored": "true",
            "sortBy": "relevanceblender",
            "withCache": "true"
        }
        return self._make_request("amazon-search-by-keyword-asin", params)

    def get_price_offers(self, asin: str) -> Optional[Dict]:
        """Get current price offers for a product"""
        params = {
            "page": "1",
            "domainCode": "com",
            "asin": asin
        }
        return self._make_request("amazon-lookup-prices", params)
        
    def get_product_reviews(self, asin: str, page: int = 1) -> Optional[Dict]:
        """Get product reviews"""
        params = {
            "page": str(page),
            "domainCode": "com",
            "asin": asin,
            "sortBy": "recent"
        }
        return self._make_request("amazon-lookup-reviews", params)
        
    def get_best_sellers(self, category: str = None) -> Optional[Dict]:
        """Get best seller products"""
        if category:
            url = f"https://www.amazon.com/gp/movers-and-shakers/{category.lower().replace(' ', '-')}"
        else:
            url = "https://www.amazon.com/gp/movers-and-shakers"
            
        params = {
            "url": url,
            "page": "1"
        }
        return self._make_request("amazon-best-sellers-list", params)
        
    def get_deals(self) -> Optional[Dict]:
        """Get current product deals"""
        params = {
            "domainCode": "com",
            "page": "1"
        }
        return self._make_request("v2/amazon-search-deals", params)

    # Legacy API adapters (for backward compatibility)
    def fetch_product_details(self, asin: str, country: str = "US") -> Dict:
        """Legacy adapter for the old API format"""
        result = self.get_product_details(asin)
        
        if not result:
            return {
                "success": False,
                "error": "Failed to fetch product details"
            }
            
        # Transform to match the old API format
        return {
            "success": True,
            "data": {
                "asin": asin,
                "product_name": result.get("title", ""),
                "brand": result.get("brand", ""),
                "current_price": result.get("price", 0),
                "currency": "USD",
                "availability": result.get("availability", ""),
                "features": result.get("feature_bullets", []),
                "rating": result.get("rating", 0),
                "reviews_count": result.get("reviewsCount", 0),
                "images": result.get("imageUrlList", []),
                "categories": [result.get("category", "")],
                "priceHistory": []  # No price history in this API
            }
        }
        
    def fetch_product_reviews(self, asin: str, country: str = "US") -> Dict:
        """Legacy adapter for the old API format"""
        result = self.get_product_reviews(asin)
        
        if not result:
            return {
                "success": False,
                "error": "Failed to fetch product reviews"
            }
            
        # Transform to match the old API format
        reviews = []
        
        if "reviews" in result:
            for review in result["reviews"]:
                reviews.append({
                    "id": review.get("id", ""),
                    "rating": review.get("rating", 0),
                    "title": review.get("title", ""),
                    "text": review.get("review", ""),
                    "date": review.get("date", ""),
                    "verified_purchase": review.get("verifiedPurchase", False),
                    "helpfulness": f"{review.get('helpful', 0)} people found this helpful"
                })
        
        # Calculate sentiment percentages
        ratings = [review.get("rating", 0) for review in reviews]
        sentiment = {
            "positive": len([r for r in ratings if r >= 4]) / len(ratings) * 100 if ratings else 0,
            "neutral": len([r for r in ratings if r == 3]) / len(ratings) * 100 if ratings else 0,
            "negative": len([r for r in ratings if r <= 2]) / len(ratings) * 100 if ratings else 0
        }
        
        return {
            "success": True,
            "data": {
                "asin": asin,
                "product_name": "",  # Not available in this API response
                "total_reviews": len(reviews),
                "average_rating": sum(ratings) / len(ratings) if ratings else 0,
                "sentiment": sentiment,
                "rating_breakdown": {
                    "5_star": f"{len([r for r in ratings if r == 5]) / len(ratings) * 100 if ratings else 0}%",
                    "4_star": f"{len([r for r in ratings if r == 4]) / len(ratings) * 100 if ratings else 0}%",
                    "3_star": f"{len([r for r in ratings if r == 3]) / len(ratings) * 100 if ratings else 0}%",
                    "2_star": f"{len([r for r in ratings if r == 2]) / len(ratings) * 100 if ratings else 0}%",
                    "1_star": f"{len([r for r in ratings if r == 1]) / len(ratings) * 100 if ratings else 0}%"
                },
                "reviews": reviews
            }
        }
        
    def fetch_deals(self, country: str = "US") -> Dict:
        """Legacy adapter for the old API format"""
        result = self.get_deals()
        
        if not result:
            return {
                "success": False,
                "error": "Failed to fetch deals"
            }
            
        # Transform to match the old API format
        deals = []
        
        if "deals" in result:
            for deal in result["deals"]:
                deals.append({
                    "title": deal.get("title", ""),
                    "asin": deal.get("asin", ""),
                    "url": deal.get("url", ""),
                    "regular_price": deal.get("listPrice", 0),
                    "deal_price": deal.get("price", 0),
                    "currency": "USD",
                    "discount_percentage": deal.get("discountPercentage", 0),
                    "savings_amount": deal.get("savings", 0),
                    "expiry_date": "",  # Not available in this API
                    "deal_type": "Deal",
                    "rating": deal.get("stars", 0),
                    "reviews_count": deal.get("numberOfReviews", 0)
                })
        
        return {
            "success": True,
            "data": {
                "deals_count": len(deals),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "deals": deals
            }
        }
        
    def fetch_best_sellers(self, category: str = None, country: str = "US") -> Dict:
        """Legacy adapter for the old API format"""
        result = self.get_best_sellers(category)
        
        if not result:
            return {
                "success": False,
                "error": "Failed to fetch best sellers"
            }
            
        # Transform to match the old API format
        results = []
        
        if "bestsellers" in result:
            for i, item in enumerate(result["bestsellers"]):
                results.append({
                    "rank": i + 1,
                    "asin": item.get("asin", ""),
                    "product_name": item.get("title", ""),
                    "category": category or "bestsellers",
                    "price": item.get("price", 0),
                    "rating": item.get("rating", 0),
                    "reviews_count": item.get("reviews_count", 0)
                })
        
        return {
            "success": True,
            "data": {
                "category": category or "bestsellers",
                "results": results
            }
        }


class MarketDataAnalyzer:
    """Analyze market data for pricing insights"""
    
    def __init__(self):
        self.amazon_service = AmazonDataService()
        self.cache = {}  # Simple in-memory cache for analysis results
        
    def analyze_product(self, asin: str, category: str = "Electronics") -> Dict:
        """Analyze product and gather relevant market data"""
        # Check cache first
        cache_key = f"analyze:{asin}:{category}"
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            cache_time = cache_entry.get('timestamp', 0)
            if time.time() - cache_time < 3600:  # 1 hour TTL
                logger.info(f"Using cached analysis for {asin}")
                return cache_entry.get('data')
        
        market_data = {
            "timestamp": datetime.now().isoformat(),
            "asin": asin,
            "category": category,
            "market_analysis": {},
            "status": "success",
            "errors": []
        }
        
        try:
            # Get core product data
            product_details = self.amazon_service.get_product_details(asin)
            if product_details:
                market_data["product_info"] = self._extract_product_info(product_details)
            else:
                market_data["errors"].append("Failed to fetch product details")
            
            # Get competitive pricing data
            if market_data.get("product_info", {}).get("title"):
                competitive_data = self.amazon_service.get_competitive_data(
                    market_data["product_info"]["title"]
                )
                if competitive_data:
                    market_data["competitive_analysis"] = self._analyze_competition(competitive_data)
                else:
                    market_data["errors"].append("Failed to fetch competitive data")
            
            # Get current offers
            price_offers = self.amazon_service.get_price_offers(asin)
            if price_offers:
                market_data["price_analysis"] = self._analyze_prices(price_offers)
            else:
                market_data["errors"].append("Failed to fetch price offers")
                
            # Get reviews data
            reviews_data = self.amazon_service.get_product_reviews(asin)
            if reviews_data:
                market_data["reviews_analysis"] = self._analyze_reviews(reviews_data)
            else:
                market_data["errors"].append("Failed to fetch reviews")
            
            if market_data["errors"]:
                market_data["status"] = "partial_success"
                
            # Add pricing recommendations context
            market_data["pricing_context"] = self._generate_pricing_context(market_data)
                
        except Exception as e:
            market_data["status"] = "error"
            market_data["errors"].append(str(e))
            logger.error(f"Error analyzing product {asin}: {e}")
            traceback.print_exc()
        
        # Cache the result
        self.cache[cache_key] = {
            'data': market_data,
            'timestamp': time.time()
        }
        
        return market_data
    
    def _extract_product_info(self, data: Dict) -> Dict:
        """Extract relevant product information"""
        return {
            "title": data.get("title"),
            "current_price": data.get("price"),
            "rating": data.get("rating"),
            "review_count": data.get("reviewsCount"),
            "category": data.get("category"),
            "images": data.get("imageUrlList", []),
            "features": data.get("feature_bullets", []),
            "brand": data.get("brand")
        }
    
    def _analyze_competition(self, data: Dict) -> Dict:
        """Analyze competitive product data"""
        products = data.get("products", [])
        prices = [p.get("price") for p in products if p.get("price")]
        
        # Calculate price position percentile
        position_percentile = 0
        if prices and "product_info" in data and "current_price" in data["product_info"]:
            current_price = data["product_info"]["current_price"]
            lower_prices = sum(1 for p in prices if p < current_price)
            position_percentile = (lower_prices / len(prices)) * 100 if len(prices) > 0 else 50
        
        result = {
            "avg_market_price": sum(prices) / len(prices) if prices else None,
            "price_range": {
                "min": min(prices) if prices else None,
                "max": max(prices) if prices else None
            },
            "competitor_count": len(products),
            "position_percentile": position_percentile,
            "price_distribution": self._calculate_price_distribution(prices)
        }
        
        # Add competitor product details
        result["competitors"] = [
            {
                "title": p.get("title", ""),
                "price": p.get("price", 0),
                "rating": p.get("rating", 0),
                "asin": p.get("asin", "")
            }
            for p in products[:5]  # Only include top 5 competitors
        ]
        
        return result
    
    def _analyze_prices(self, data: Dict) -> Dict:
        """Analyze price offers data"""
        offers = data.get("offers", [])
        prices = [o.get("price") for o in offers if o.get("price")]
        
        return {
            "current_offers": len(offers),
            "price_distribution": {
                "min": min(prices) if prices else None,
                "max": max(prices) if prices else None,
                "avg": sum(prices) / len(prices) if prices else None
            },
            "offers": offers[:5]  # Only include top 5 offers
        }
    
    def _analyze_reviews(self, data: Dict) -> Dict:
        """Analyze product reviews"""
        reviews = data.get("reviews", [])
        ratings = [float(r.get("rating", 0)) for r in reviews if r.get("rating")]
        
        # Calculate sentiment
        positive = len([r for r in ratings if r >= 4])
        neutral = len([r for r in ratings if r == 3])
        negative = len([r for r in ratings if r <= 2])
        total = len(ratings) if ratings else 1  # Avoid division by zero
        
        sentiment = {
            "positive": (positive / total) * 100,
            "neutral": (neutral / total) * 100,
            "negative": (negative / total) * 100
        }
        
        return {
            "total_reviews": len(reviews),
            "average_rating": sum(ratings) / len(ratings) if ratings else 0,
            "sentiment": sentiment,
            "recent_reviews": reviews[:3]  # Only include 3 most recent reviews
        }
    
    def _calculate_price_distribution(self, prices: List[float]) -> Dict:
        """Calculate price distribution for visualization"""
        if not prices:
            return {}
            
        # Create price ranges
        min_price = min(prices)
        max_price = max(prices)
        
        # Create 5 price brackets
        bracket_size = (max_price - min_price) / 5 if max_price > min_price else 1
        brackets = []
        
        for i in range(5):
            lower = min_price + (i * bracket_size)
            upper = min_price + ((i + 1) * bracket_size)
            count = sum(1 for p in prices if lower <= p < upper)
            brackets.append({
                "range": f"${lower:.2f} - ${upper:.2f}",
                "count": count,
                "percentage": (count / len(prices)) * 100
            })
            
        # Add the max price to the last bracket
        brackets[-1]["range"] = f"${min_price + (4 * bracket_size):.2f} - ${max_price:.2f}"
        
        return {
            "brackets": brackets,
            "total": len(prices)
        }
    
    def _generate_pricing_context(self, data: Dict) -> Dict:
        """Generate pricing context and recommendations based on market data"""
        product_info = data.get("product_info", {})
        competitive_analysis = data.get("competitive_analysis", {})
        price_analysis = data.get("price_analysis", {})
        
        current_price = product_info.get("current_price", 0)
        avg_market_price = competitive_analysis.get("avg_market_price")
        price_min = competitive_analysis.get("price_range", {}).get("min")
        price_max = competitive_analysis.get("price_range", {}).get("max")
        
        # Default insights
        insights = ["No market data available for pricing context"]
        price_position = "unknown"
        
        if all([current_price, avg_market_price, price_min, price_max]):
            # Determine price position
            if current_price < avg_market_price * 0.9:
                price_position = "below_average"
                insights = ["Price is below market average", "Consider testing price increases"]
            elif current_price > avg_market_price * 1.1:
                price_position = "above_average"
                insights = ["Price is above market average", "Monitor conversion rates carefully"]
            else:
                price_position = "average"
                insights = ["Price is near market average", "Consider testing small adjustments"]
                
            # Add competitive positioning
            position_percentile = competitive_analysis.get("position_percentile", 50)
            if position_percentile < 20:
                insights.append("Price is among the lowest 20% of competitors")
            elif position_percentile > 80:
                insights.append("Price is among the highest 20% of competitors")
                
        # Add rating context if available
        rating = product_info.get("rating")
        if rating:
            if rating >= 4.5:
                insights.append("High product rating may support premium pricing")
            elif rating < 3.5:
                insights.append("Lower rating may require competitive pricing")
                
        return {
            "price_position": price_position,
            "insights": insights,
            "similar_price_range": {
                "min": price_min,
                "max": price_max,
                "avg": avg_market_price
            }
        }

# Create singleton instances for use across the application
amazon_data_service = AmazonDataService()
market_analyzer = MarketDataAnalyzer()

# Legacy API compatibility functions
def fetch_product_details(asin: str, country: str = "US") -> Dict:
    return amazon_data_service.fetch_product_details(asin, country)

def fetch_product_reviews(asin: str, country: str = "US") -> Dict:
    return amazon_data_service.fetch_product_reviews(asin, country)

def fetch_deals(country: str = "US") -> Dict:
    return amazon_data_service.fetch_deals(country)

def fetch_best_sellers(category: str = None, country: str = "US") -> Dict:
    return amazon_data_service.fetch_best_sellers(category, country)

def analyze_product(asin: str = None, country: str = "US") -> Dict:
    """API function to analyze a product"""
    if not asin:
        # Get from request parameters in Flask context
        from flask import request
        asin = request.args.get('asin')
        country = request.args.get('country', 'US')
    
    if not asin:
        return {
            'success': False,
            'message': 'ASIN parameter is required'
        }
    
    result = market_analyzer.analyze_product(asin)
    
    # Transform to old API format for compatibility
    return {
        'success': result.get('status') != 'error',
        'timestamp': result.get('timestamp'),
        'asin': asin,
        'product_details': {
            'success': True,
            'title': result.get('product_info', {}).get('title', ''),
            'category': result.get('product_info', {}).get('category', '')
        },
        'price_trends': {
            'success': True,
            'price_history': []  # No price history in new API
        },
        'market_position': {
            'position': result.get('pricing_context', {}).get('price_position', 'unknown'),
            'percentile': result.get('competitive_analysis', {}).get('position_percentile', 50),
            'similar_products': result.get('competitive_analysis', {}).get('competitors', []),
            'competitive_index': len(result.get('competitive_analysis', {}).get('competitors', []))
        }
    } 