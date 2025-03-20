#!/usr/bin/env python
"""
Amazon web scraper module to fetch product data directly from Amazon's website.
This provides a fallback when the API is not available.
"""

import requests
from bs4 import BeautifulSoup
import json
import re
import time
import random
import os
from datetime import datetime, timedelta

# User agent list to rotate through
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
]

# Data directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'data')
TIME_SERIES_DIR = os.path.join(DATA_DIR, 'time_series')
RAW_DIR = os.path.join(DATA_DIR, 'raw')

# Create directories if they don't exist
for directory in [DATA_DIR, TIME_SERIES_DIR, RAW_DIR]:
    os.makedirs(directory, exist_ok=True)

def get_random_user_agent():
    """Return a random user agent from the list"""
    return random.choice(USER_AGENTS)

def save_raw_data(data, filename):
    """Save raw scraped data"""
    filepath = os.path.join(RAW_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Raw data saved to {filepath}")
    return filepath

def scrape_product_details(asin, country="US"):
    """
    Scrape product details from Amazon
    
    Args:
        asin (str): Amazon Standard Identification Number
        country (str): Country code
        
    Returns:
        dict: Product data
    """
    print(f"Scraping product details for ASIN {asin} in {country}...")
    
    # Construct URL based on country
    domain = "amazon.com" if country == "US" else f"amazon.{country.lower()}"
    url = f"https://www.{domain}/dp/{asin}"
    
    headers = {
        'User-Agent': get_random_user_agent(),
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    }
    
    try:
        # Add a delay to avoid rate limiting
        time.sleep(random.uniform(1, 3))
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract product data
        product_data = {
            "asin": asin,
            "url": url,
            "timestamp": datetime.now().isoformat(),
            "success": True
        }
        
        # Extract title
        title_elem = soup.select_one('#productTitle')
        if title_elem:
            product_data["title"] = title_elem.get_text().strip()
        
        # Extract price
        price_elem = soup.select_one('.a-price .a-offscreen')
        if price_elem:
            price_text = price_elem.get_text().strip()
            # Remove currency symbol and convert to float
            price_match = re.search(r'[\d,.]+', price_text)
            if price_match:
                try:
                    price_str = price_match.group(0).replace(',', '')
                    product_data["current_price"] = float(price_str)
                except ValueError:
                    product_data["price_text"] = price_text
        
        # Extract rating
        rating_elem = soup.select_one('span[data-hook="rating-out-of-text"]')
        if rating_elem:
            rating_text = rating_elem.get_text().strip()
            rating_match = re.search(r'([\d.]+)', rating_text)
            if rating_match:
                product_data["rating"] = float(rating_match.group(1))
        
        # Extract number of reviews
        reviews_elem = soup.select_one('span[data-hook="total-review-count"]')
        if reviews_elem:
            reviews_text = reviews_elem.get_text().strip()
            reviews_match = re.search(r'([\d,]+)', reviews_text)
            if reviews_match:
                product_data["total_reviews"] = int(reviews_match.group(1).replace(',', ''))
        
        # Extract availability
        availability_elem = soup.select_one('#availability')
        if availability_elem:
            product_data["availability"] = availability_elem.get_text().strip()
        
        # Extract seller/brand
        seller_elem = soup.select_one('#bylineInfo')
        if seller_elem:
            product_data["seller"] = seller_elem.get_text().strip()
        
        # Extract category
        category_elems = soup.select('#wayfinding-breadcrumbs_feature_div ul li')
        if category_elems:
            categories = []
            for elem in category_elems:
                category_text = elem.get_text().strip()
                if category_text and not category_text.startswith('›'):
                    categories.append(category_text)
            if categories:
                product_data["categories"] = categories
                product_data["main_category"] = categories[-1] if categories else None
        
        # Extract description
        description_elem = soup.select_one('#productDescription')
        if description_elem:
            product_data["description"] = description_elem.get_text().strip()
        
        # Extract features
        feature_elems = soup.select('#feature-bullets ul li')
        if feature_elems:
            features = []
            for elem in feature_elems:
                feature_text = elem.get_text().strip()
                if feature_text:
                    features.append(feature_text)
            if features:
                product_data["features"] = features
        
        # Save raw data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scraped_product_{asin}_{country}_{timestamp}.json"
        save_raw_data(product_data, filename)
        
        return product_data
    
    except Exception as e:
        print(f"Error scraping product {asin}: {str(e)}")
        return {
            "asin": asin,
            "url": url,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "error": str(e)
        }

def extract_price_trends(asin, country="US"):
    """
    Simulate price trend data for a product
    
    Args:
        asin (str): Amazon Standard Identification Number
        country (str): Country code
        
    Returns:
        dict: Price trend data
    """
    # Get current product data
    product_data = scrape_product_details(asin, country)
    
    if not product_data.get("success", False):
        return {
            "asin": asin,
            "success": False,
            "error": "Failed to fetch product data"
        }
    
    # Get current price
    current_price = product_data.get("current_price")
    
    if not current_price:
        return {
            "asin": asin,
            "success": False,
            "error": "No price data available"
        }
    
    # Simulate price history with some random variation
    days = 30
    price_history = []
    base_price = current_price
    
    for i in range(days):
        # More recent days have prices closer to current
        weight = i / days
        # Random variation - larger for older dates
        variation = random.uniform(-0.15, 0.15) * weight
        
        if random.random() < 0.1:
            # Occasional bigger price change (sales or price increases)
            variation = random.uniform(-0.25, 0.2) * weight
        
        # Calculate historical price
        historical_price = base_price * (1 + variation)
        
        # Add to history
        days_ago = days - i
        date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        
        price_history.append({
            "date": date,
            "price": round(historical_price, 2)
        })
    
    # Calculate statistics
    prices = [entry["price"] for entry in price_history]
    min_price = min(prices)
    max_price = max(prices)
    avg_price = sum(prices) / len(prices)
    
    # Determine trend direction
    recent_prices = prices[-5:]
    older_prices = prices[:5]
    avg_recent = sum(recent_prices) / len(recent_prices)
    avg_older = sum(older_prices) / len(older_prices)
    
    if avg_recent > avg_older * 1.05:
        trend = "increasing"
    elif avg_recent < avg_older * 0.95:
        trend = "decreasing"
    else:
        trend = "stable"
    
    # Calculate price volatility (standard deviation as percentage of avg)
    import numpy as np
    volatility = np.std(prices) / avg_price * 100
    
    return {
        "asin": asin,
        "title": product_data.get("title", "Unknown Product"),
        "current_price": current_price,
        "min_price": min_price,
        "max_price": max_price,
        "avg_price": round(avg_price, 2),
        "price_range": round(max_price - min_price, 2),
        "volatility": round(volatility, 2),
        "trend": trend,
        "price_history": price_history,
        "success": True
    }

def extract_similar_products(asin, country="US"):
    """
    Extract similar products to analyze the competitive landscape
    This is a simplified simulation as scraping similar products is more complex
    
    Args:
        asin (str): Amazon Standard Identification Number
        country (str): Country code
        
    Returns:
        dict: Similar products data
    """
    # Get product data
    product_data = scrape_product_details(asin, country)
    
    if not product_data.get("success", False):
        return {
            "asin": asin,
            "success": False,
            "error": "Failed to fetch product data"
        }
    
    # Get current price and category
    current_price = product_data.get("current_price")
    category = product_data.get("main_category", "Electronics")
    
    if not current_price:
        return {
            "asin": asin,
            "success": False,
            "error": "No price data available"
        }
    
    # Simulate similar products
    num_similar = random.randint(3, 8)
    similar_products = []
    
    for i in range(num_similar):
        # Generate prices around the current price
        price_variation = random.uniform(-0.3, 0.3)
        price = current_price * (1 + price_variation)
        
        # Generate random ratings
        rating = round(random.uniform(3.5, 5.0), 1)
        
        similar_products.append({
            "asin": f"B{random.randint(10000000, 99999999)}",
            "title": f"Similar {category} Product {i+1}",
            "price": round(price, 2),
            "rating": rating,
            "reviews": random.randint(50, 5000)
        })
    
    # Sort by price
    similar_products.sort(key=lambda x: x["price"])
    
    # Calculate price position
    prices = [p["price"] for p in similar_products]
    product_position = sum(1 for p in prices if p < current_price) + 1
    percentile = (product_position / (len(prices) + 1)) * 100
    
    # Determine market position
    if percentile < 20:
        position = "budget"
    elif percentile < 40:
        position = "competitive"
    elif percentile < 60:
        position = "average"
    elif percentile < 80:
        position = "premium"
    else:
        position = "luxury"
    
    return {
        "asin": asin,
        "title": product_data.get("title", "Unknown Product"),
        "current_price": current_price,
        "similar_products": similar_products,
        "price_position": position,
        "price_percentile": round(percentile, 1),
        "competitive_index": len(similar_products),
        "success": True
    }

if __name__ == "__main__":
    # Test with a sample ASIN
    test_asin = "B07ZPKBL9V"  # Echo Dot 4th Gen
    print(json.dumps(scrape_product_details(test_asin), indent=2))
    print("\nPrice trends:")
    print(json.dumps(extract_price_trends(test_asin), indent=2))
    print("\nSimilar products:")
    print(json.dumps(extract_similar_products(test_asin), indent=2)) 