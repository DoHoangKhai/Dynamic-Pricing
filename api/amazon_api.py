#!/usr/bin/env python
"""
Amazon API data collection module.
Fetches data from Amazon API and processes it into time series format.
"""

import os
import json
import requests
import time
import datetime
from datetime import datetime
import pandas as pd
import traceback

# API Configuration
API_KEY = "2a6b802feamsh0f78b4cd091e889p149b18jsn9e0eda3f6870"
API_HOST = "real-time-amazon-data.p.rapidapi.com"
BASE_URL = "https://real-time-amazon-data.p.rapidapi.com"

# Headers for API requests
headers = {
    "x-rapidapi-key": API_KEY,
    "x-rapidapi-host": API_HOST
}

# Data directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'data')
TIME_SERIES_DIR = os.path.join(DATA_DIR, 'time_series')
RAW_DIR = os.path.join(DATA_DIR, 'raw')

# Example ASINs for testing
EXAMPLE_ASINS = [
    'B07ZPKBL9V',  # Echo Dot
    'B08N5KWB9H',  # Fire TV Stick
    'B094DXMVRP',  # AirPods Pro 
    'B08L5TNJHG'   # Apple Watch SE
]

# Create directories if they don't exist
for directory in [DATA_DIR, TIME_SERIES_DIR, RAW_DIR]:
    os.makedirs(directory, exist_ok=True)

def make_api_request(endpoint, params=None):
    """
    Make a request to the Amazon API
    
    Args:
        endpoint (str): API endpoint path
        params (dict): Query parameters
        
    Returns:
        dict: API response as JSON
    """
    url = f"{BASE_URL}/{endpoint}"
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        traceback.print_exc()
        return {"error": str(e)}

def save_raw_data(data, filename):
    """Save raw API response data"""
    filepath = os.path.join(RAW_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Raw data saved to {filepath}")
    return filepath

def fetch_best_sellers(category="bestsellers", country="US", type_="best_sellers"):
    """
    Fetch best sellers data from Amazon API
    
    Args:
        category (str): Category name
        country (str): Country code
        type_ (str): Best seller type
        
    Returns:
        dict: API response
    """
    print(f"Fetching best sellers data for {category} in {country}...")
    
    params = {
        "category": category,
        "country": country,
        "type": type_
    }
    
    response = make_api_request("best-sellers", params)
    
    if "error" in response:
        print(f"Error fetching best sellers: {response['error']}")
        return None
    
    # Save raw data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"bestsellers_{category}_{country}_{type_}_{timestamp}.json"
    save_raw_data(response, filename)
    
    return response

def fetch_product_details(asin, country="US"):
    """
    Fetch product details by ASIN
    
    Args:
        asin (str): Amazon Standard Identification Number
        country (str): Country code
        
    Returns:
        dict: API response
    """
    print(f"Fetching product details for ASIN {asin} in {country}...")
    
    params = {
        "asin": asin,
        "country": country
    }
    
    response = make_api_request("product-details", params)
    
    if "error" in response:
        print(f"Error fetching product details: {response['error']}")
        return None
    
    # Save raw data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"product_{asin}_{country}_{timestamp}.json"
    save_raw_data(response, filename)
    
    return response

def fetch_product_reviews(asin, country="US"):
    """
    Fetch product reviews for analysis
    
    Args:
        asin (str): Amazon Standard Identification Number
        country (str): Country code
        
    Returns:
        dict: API response
    """
    print(f"Fetching product reviews for ASIN {asin} in {country}...")
    
    params = {
        "asin": asin,
        "country": country
    }
    
    response = make_api_request("product-reviews", params)
    
    if "error" in response:
        print(f"Error fetching product reviews: {response['error']}")
        return None
    
    # Save raw data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reviews_{asin}_{country}_{timestamp}.json"
    save_raw_data(response, filename)
    
    return response

def fetch_deals(country="US"):
    """
    Fetch current Amazon deals
    
    Args:
        country (str): Country code
        
    Returns:
        dict: API response
    """
    print(f"Fetching deals for {country}...")
    
    params = {
        "country": country
    }
    
    try:
        response = make_api_request("deals-v2", params)
        
        if not response:
            print("[ERROR] make_api_request returned None")
            return {"error": "API request failed with no response"}
            
        if "error" in response:
            error_msg = response.get('error', 'Unknown error')
            print(f"[ERROR] Error fetching deals: {error_msg}")
            return {"error": f"API error: {error_msg}"}
        
        # Validate response structure
        if "data" not in response or "deals" not in response.get("data", {}):
            print(f"[ERROR] Invalid API response structure: {list(response.keys())}")
            return {"error": "Invalid API response structure"}
            
        deals_count = len(response.get("data", {}).get("deals", []))
        print(f"[SUCCESS] Successfully fetched {deals_count} deals")
        
        # Save raw data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"deals_{country}_{timestamp}.json"
        save_raw_data(response, filename)
        
        return response
        
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Exception in fetch_deals: {error_msg}")
        traceback.print_exc()
        return {"error": f"Exception: {error_msg}"}

def process_best_sellers_time_series(category="bestsellers", country="US", type_="best_sellers"):
    """
    Process best sellers data into time series format
    
    Args:
        category (str): Category name
        country (str): Country code
        type_ (str): Best seller type
        
    Returns:
        str: Path to time series file
    """
    # Get all raw bestseller data files for this category/country/type
    raw_files = [f for f in os.listdir(RAW_DIR) if f.startswith(f"bestsellers_{category}_{country}_{type_}")]
    
    if not raw_files:
        print(f"No raw data found for bestsellers_{category}_{country}_{type_}")
        return None
    
    # Process each file and build a comprehensive time series
    all_data = []
    
    for raw_file in raw_files:
        with open(os.path.join(RAW_DIR, raw_file), 'r') as f:
            data = json.load(f)
        
        # Extract timestamp from filename
        timestamp_str = raw_file.split('_')[-2] + '_' + raw_file.split('_')[-1].replace('.json', '')
        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        collection_date = timestamp.strftime("%Y-%m-%d")
        
        # Process each bestseller item
        if "bestsellers" in data and isinstance(data["bestsellers"], list):
            for rank, item in enumerate(data["bestsellers"], 1):
                if not isinstance(item, dict):
                    continue
                    
                # Extract key information
                asin = item.get("asin", "")
                title = item.get("title", "")
                current_price = item.get("current_price")
                
                # Convert price to float if possible
                try:
                    if isinstance(current_price, str) and current_price.startswith('$'):
                        current_price = float(current_price[1:].replace(',', ''))
                    else:
                        current_price = float(current_price) if current_price else None
                except (ValueError, TypeError):
                    current_price = None
                
                # Add to the dataset
                all_data.append({
                    "timestamp": timestamp,
                    "collection_date": collection_date,
                    "rank": rank,
                    "asin": asin,
                    "title": title,
                    "current_price": current_price,
                    "category": category,
                    "country": country
                })
    
    if not all_data:
        print("No bestseller data to process")
        return None
    
    # Create pandas DataFrame
    df = pd.DataFrame(all_data)
    
    # Save to CSV
    ts_filename = f"bestsellers_ts_{category}_{country}_{type_}.csv"
    ts_filepath = os.path.join(TIME_SERIES_DIR, ts_filename)
    df.to_csv(ts_filepath, index=False)
    
    print(f"Bestsellers time series saved to {ts_filepath}")
    return ts_filepath

def process_product_time_series(asin, country="US"):
    """
    Process product details into time series format
    
    Args:
        asin (str): Amazon Standard Identification Number
        country (str): Country code
        
    Returns:
        str: Path to time series file
    """
    # Get all raw product data files for this ASIN/country
    raw_files = [f for f in os.listdir(RAW_DIR) if f.startswith(f"product_{asin}_{country}")]
    
    if not raw_files:
        print(f"No raw data found for product_{asin}_{country}")
        return None
    
    # Process each file and build a time series
    all_data = []
    
    for raw_file in raw_files:
        with open(os.path.join(RAW_DIR, raw_file), 'r') as f:
            data = json.load(f)
        
        # Extract timestamp from filename
        timestamp_str = raw_file.split('_')[-2] + '_' + raw_file.split('_')[-1].replace('.json', '')
        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        collection_date = timestamp.strftime("%Y-%m-%d")
        
        # Extract key product information
        product_data = data.get("product", {})
        if not product_data:
            continue
            
        # Extract pricing information
        price_data = product_data.get("price", {})
        current_price = None
        if isinstance(price_data, dict):
            current_price_str = price_data.get("current_price")
            if current_price_str:
                try:
                    if isinstance(current_price_str, str) and current_price_str.startswith('$'):
                        current_price = float(current_price_str[1:].replace(',', ''))
                    else:
                        current_price = float(current_price_str)
                except (ValueError, TypeError):
                    pass
        
        # Extract other product details
        title = product_data.get("title", "")
        rating = product_data.get("rating", {}).get("rating")
        rating_count = product_data.get("rating", {}).get("rating_count")
        
        # Convert rating to float if possible
        try:
            rating = float(rating) if rating else None
            rating_count = int(rating_count.replace(',', '')) if isinstance(rating_count, str) else rating_count
        except (ValueError, TypeError):
            rating = None
            rating_count = None
        
        # Add to the dataset
        all_data.append({
            "timestamp": timestamp,
            "collection_date": collection_date,
            "asin": asin,
            "title": title,
            "current_price": current_price,
            "rating": rating,
            "rating_count": rating_count,
            "country": country
        })
    
    if not all_data:
        print("No product data to process")
        return None
    
    # Create pandas DataFrame
    df = pd.DataFrame(all_data)
    
    # Save to CSV
    ts_filename = f"product_ts_{asin}_{country}.csv"
    ts_filepath = os.path.join(TIME_SERIES_DIR, ts_filename)
    df.to_csv(ts_filepath, index=False)
    
    print(f"Product time series saved to {ts_filepath}")
    return ts_filepath

def process_reviews_sentiment_time_series(asin, country="US"):
    """
    Process product reviews into time series format with sentiment analysis
    
    Args:
        asin (str): Amazon Standard Identification Number
        country (str): Country code
        
    Returns:
        str: Path to time series file
    """
    # Get all raw review data files for this ASIN/country
    raw_files = [f for f in os.listdir(RAW_DIR) if f.startswith(f"reviews_{asin}_{country}")]
    
    if not raw_files:
        print(f"No raw data found for reviews_{asin}_{country}")
        return None
    
    # Process each file and build time series
    all_reviews = []
    sentiment_summary = []
    
    for raw_file in raw_files:
        with open(os.path.join(RAW_DIR, raw_file), 'r') as f:
            data = json.load(f)
        
        # Extract timestamp from filename
        timestamp_str = raw_file.split('_')[-2] + '_' + raw_file.split('_')[-1].replace('.json', '')
        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        collection_date = timestamp.strftime("%Y-%m-%d")
        
        # Extract reviews
        reviews = data.get("reviews", [])
        if not reviews:
            continue
        
        # Process each review
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for review in reviews:
            if not isinstance(review, dict):
                continue
                
            # Extract key information
            review_title = review.get("title", "")
            review_text = review.get("review", "")
            review_rating = review.get("rating")
            
            # Convert rating to float
            try:
                review_rating = float(review_rating) if review_rating else None
            except (ValueError, TypeError):
                review_rating = None
            
            # Simple sentiment analysis based on rating
            sentiment = "neutral"
            if review_rating is not None:
                if review_rating >= 4:
                    sentiment = "positive"
                    positive_count += 1
                elif review_rating <= 2:
                    sentiment = "negative"
                    negative_count += 1
                else:
                    neutral_count += 1
            
            # Add to reviews dataset
            all_reviews.append({
                "timestamp": timestamp,
                "collection_date": collection_date,
                "asin": asin,
                "review_title": review_title,
                "review_text": review_text,
                "rating": review_rating,
                "sentiment": sentiment,
                "country": country
            })
        
        # Add to sentiment summary
        sentiment_summary.extend([
            {
                "collection_date": collection_date,
                "asin": asin,
                "sentiment": "positive",
                "count": positive_count,
                "country": country
            },
            {
                "collection_date": collection_date,
                "asin": asin,
                "sentiment": "neutral",
                "count": neutral_count,
                "country": country
            },
            {
                "collection_date": collection_date,
                "asin": asin,
                "sentiment": "negative",
                "count": negative_count,
                "country": country
            }
        ])
    
    if not all_reviews:
        print("No review data to process")
        return None
    
    # Create pandas DataFrames
    reviews_df = pd.DataFrame(all_reviews)
    sentiment_df = pd.DataFrame(sentiment_summary)
    
    # Save to CSV
    reviews_ts_filename = f"reviews_ts_{asin}_{country}.csv"
    reviews_ts_filepath = os.path.join(TIME_SERIES_DIR, reviews_ts_filename)
    reviews_df.to_csv(reviews_ts_filepath, index=False)
    
    sentiment_summary_filename = f"sentiment_summary_{asin}_{country}.csv"
    sentiment_summary_filepath = os.path.join(TIME_SERIES_DIR, sentiment_summary_filename)
    sentiment_df.to_csv(sentiment_summary_filepath, index=False)
    
    print(f"Reviews time series saved to {reviews_ts_filepath}")
    print(f"Sentiment summary saved to {sentiment_summary_filepath}")
    return reviews_ts_filepath

def process_deals_time_series(country="US"):
    """
    Process deals data into time series format
    
    Args:
        country (str): Country code
        
    Returns:
        str: Path to time series file
    """
    # Get all raw deals data files for this country
    raw_files = [f for f in os.listdir(RAW_DIR) if f.startswith(f"deals_{country}")]
    
    if not raw_files:
        print(f"No raw data found for deals_{country}")
        return None
    
    # Process each file and build time series
    all_deals = []
    
    for raw_file in raw_files:
        with open(os.path.join(RAW_DIR, raw_file), 'r') as f:
            data = json.load(f)
        
        # Extract timestamp from filename
        timestamp_str = raw_file.split('_')[-2] + '_' + raw_file.split('_')[-1].replace('.json', '')
        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        collection_date = timestamp.strftime("%Y-%m-%d")
        
        # Extract deals
        deals = data.get("deals", [])
        if not deals:
            continue
        
        # Process each deal
        for deal in deals:
            if not isinstance(deal, dict):
                continue
                
            # Extract key information
            deal_id = deal.get("id", "")
            title = deal.get("title", "")
            current_price_str = deal.get("deal_price", "")
            original_price_str = deal.get("list_price", "")
            discount_str = deal.get("discount_percentage", "")
            
            # Convert price strings to float
            current_price = None
            original_price = None
            discount = None
            
            try:
                if isinstance(current_price_str, str) and current_price_str.startswith('$'):
                    current_price = float(current_price_str[1:].replace(',', ''))
                else:
                    current_price = float(current_price_str) if current_price_str else None
                    
                if isinstance(original_price_str, str) and original_price_str.startswith('$'):
                    original_price = float(original_price_str[1:].replace(',', ''))
                else:
                    original_price = float(original_price_str) if original_price_str else None
                    
                if isinstance(discount_str, str) and discount_str.endswith('%'):
                    discount = float(discount_str[:-1])
                else:
                    discount = float(discount_str) if discount_str else None
            except (ValueError, TypeError):
                pass
            
            # Deal type classification
            deal_type = "standard"
            if discount is not None:
                if discount >= 50:
                    deal_type = "clearance"
                elif discount >= 25:
                    deal_type = "major"
                elif discount >= 10:
                    deal_type = "minor"
            
            # Add to the dataset
            all_deals.append({
                "timestamp": timestamp,
                "collection_date": collection_date,
                "deal_id": deal_id,
                "title": title,
                "current_price": current_price,
                "original_price": original_price,
                "discount": discount,
                "deal_type": deal_type,
                "country": country
            })
    
    if not all_deals:
        print("No deals data to process")
        return None
    
    # Create pandas DataFrame
    df = pd.DataFrame(all_deals)
    
    # Save to CSV
    ts_filename = f"deals_ts_{country}.csv"
    ts_filepath = os.path.join(TIME_SERIES_DIR, ts_filename)
    df.to_csv(ts_filepath, index=False)
    
    print(f"Deals time series saved to {ts_filepath}")
    return ts_filepath

def get_market_insights(asin, category="bestsellers", country="US"):
    """
    Get comprehensive market insights for a product
    
    Args:
        asin (str): Amazon Standard Identification Number
        category (str): Category for bestsellers
        country (str): Country code
        
    Returns:
        dict: Market insights
    """
    # Ensure we have the latest data
    product_data = fetch_product_details(asin, country)
    bestsellers_data = fetch_best_sellers(category, country)
    reviews_data = fetch_product_reviews(asin, country)
    deals_data = fetch_deals(country)
    
    # Process into time series
    process_product_time_series(asin, country)
    process_best_sellers_time_series(category, country)
    process_reviews_sentiment_time_series(asin, country)
    process_deals_time_series(country)
    
    # Extract key insights
    insights = {
        "asin": asin,
        "country": country,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Product information
    if product_data and "product" in product_data:
        product = product_data["product"]
        insights["title"] = product.get("title", "")
        
        # Price information
        if "price" in product:
            price = product["price"]
            insights["current_price"] = price.get("current_price", "")
            insights["currency"] = price.get("currency", "$")
        
        # Rating information
        if "rating" in product:
            rating_data = product["rating"]
            insights["rating"] = rating_data.get("rating", "")
            insights["rating_count"] = rating_data.get("rating_count", "")
    
    # Market position
    if bestsellers_data and "bestsellers" in bestsellers_data:
        # Check if product is in bestsellers
        bestsellers = bestsellers_data["bestsellers"]
        in_bestsellers = False
        rank = None
        
        for i, item in enumerate(bestsellers, 1):
            if item.get("asin") == asin:
                in_bestsellers = True
                rank = i
                break
        
        insights["in_bestsellers"] = in_bestsellers
        insights["bestseller_rank"] = rank
    
    # Return insights
    return insights

def collect_market_data(asins=None, category="bestsellers", country="US"):
    """
    Collect market data for all specified ASINs
    
    Args:
        asins (list): List of ASINs to collect data for
        category (str): Category for bestsellers
        country (str): Country code
        
    Returns:
        dict: Collection results
    """
    if asins is None:
        asins = []
    
    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "success": True,
        "message": "Market data collection completed",
        "asins_processed": len(asins),
        "insights": []
    }
    
    # Collect bestsellers and deals data (common for all products)
    try:
        bestsellers_data = fetch_best_sellers(category, country)
        process_best_sellers_time_series(category, country)
    except Exception as e:
        print(f"Error collecting bestsellers data: {e}")
        traceback.print_exc()
        result["success"] = False
        result["message"] = f"Error collecting bestsellers data: {str(e)}"
    
    try:
        deals_data = fetch_deals(country)
        process_deals_time_series(country)
    except Exception as e:
        print(f"Error collecting deals data: {e}")
        traceback.print_exc()
    
    # Collect product-specific data
    for asin in asins:
        try:
            # Get market insights for this product
            insights = get_market_insights(asin, category, country)
            result["insights"].append(insights)
        except Exception as e:
            print(f"Error collecting data for {asin}: {e}")
            traceback.print_exc()
    
    # Save collection results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"collection_result_{timestamp}.json"
    filepath = os.path.join(DATA_DIR, filename)
    
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"Collection results saved to {filepath}")
    return result

if __name__ == "__main__":
    # Example usage
    asins = ["B07ZPKBL9V", "B08N5KWB9H"]  # Example ASINs
    collect_market_data(asins) 