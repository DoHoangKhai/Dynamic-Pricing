#!/usr/bin/env python
"""
Scheduler for collecting market data from Amazon API.
This script can be run continuously in the background or as a cron job.
"""

import time
import logging
import os
import argparse
import json
from datetime import datetime
import schedule
import sys

# Add parent directory to path to allow imports from sibling modules
sys.path.append('..')

# Import our Amazon API module
from api.amazon_api import collect_market_data, EXAMPLE_ASINS

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'market_data_scheduler.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('market_scheduler')

# Path for config file
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'market_data_config.json')

# Default configuration
DEFAULT_CONFIG = {
    "collection_interval_hours": 6,  # Collect data every 6 hours
    "categories": ["bestsellers", "electronics"],  # Categories to track
    "country": "US",  # Country to track
    "product_asins": EXAMPLE_ASINS  # ASINs to track
}

def load_config():
    """Load configuration from file or create default if it doesn't exist"""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {CONFIG_PATH}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Using default configuration")
            return DEFAULT_CONFIG
    else:
        # Create default config file
        with open(CONFIG_PATH, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        logger.info(f"Default configuration created at {CONFIG_PATH}")
        return DEFAULT_CONFIG

def collect_data_job():
    """Job to collect market data"""
    logger.info("Starting market data collection job")
    config = load_config()
    
    try:
        # Collect data for each configured category
        for category in config["categories"]:
            logger.info(f"Collecting data for category: {category}")
            insights = collect_market_data(
                category=category,
                country=config["country"],
                product_asins=config["product_asins"]
            )
            logger.info(f"Successfully collected data for category: {category}")
            
        logger.info("Market data collection job completed successfully")
    except Exception as e:
        logger.error(f"Error in market data collection job: {e}")

def run_scheduler():
    """Run the scheduler"""
    config = load_config()
    interval_hours = config.get("collection_interval_hours", 6)
    
    logger.info(f"Starting scheduler with {interval_hours} hour interval")
    
    # Schedule the job to run at the specified interval
    schedule.every(interval_hours).hours.do(collect_data_job)
    
    # Run the job once at startup
    collect_data_job()
    
    # Keep the scheduler running
    while True:
        schedule.run_pending()
        time.sleep(60)

def run_once():
    """Run data collection once and exit"""
    logger.info("Running one-time data collection")
    collect_data_job()
    logger.info("One-time data collection completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Market data collection scheduler')
    parser.add_argument('--once', action='store_true', help='Run data collection once and exit')
    args = parser.parse_args()
    
    if args.once:
        run_once()
    else:
        run_scheduler() 