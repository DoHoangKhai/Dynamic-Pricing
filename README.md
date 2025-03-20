# Dynamic Pricing Dashboard

This is a web-based dashboard for dynamic pricing analysis and market data visualization. The application allows you to analyze products, calculate optimal prices, and visualize market trends.

## Directory Structure

The application is organized into these main directories:

- **api/** - API integrations and data fetching services
  - `amazon_api.py` - Amazon product data API integration
  - `amazon_web_scraper.py` - Web scraper for product details
  - `api_market_data.py` - Market data API endpoints
  - `api.py` - General API endpoints

- **models/** - Pricing models and algorithms
  - `market_environment.py` - Market environment simulation
  - `pricing_strategies.py` - Pricing algorithms and strategies
  - `RL_env1.py` - Reinforcement learning environment
  - `verify_drift.py` - Model drift detection
  - `verify_model.py` - Model verification tools

- **static/** - Static assets for the web application
  - `category_revenue.js` - Revenue visualization
  - `market_analyzer.js` - Market data visualization
  - `script.js` - Main application scripts
  - `style.css` - Main styles
  - `styles.css` - Additional styles
  - **img/** - Image assets

- **templates/** - HTML templates
  - `index.html` - Main dashboard template

- **utils/** - Utility scripts
  - `market_data_analysis.py` - Market data analysis tools
  - `market_data_scheduler.py` - Scheduler for data collection

- **tests/** - Test scripts for different components
  - Various test files for different parts of the system

## Getting Started

1. Install required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python app.py
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Replacing the Dynamic Pricing Model

The pricing model is isolated in the `models/` directory, making it easy to replace:

1. To replace the pricing algorithm, modify or replace `models/pricing_strategies.py`
2. To adjust the market environment simulation, modify `models/market_environment.py`
3. If using reinforcement learning, update `models/RL_env1.py`

After replacing the model, update any imports in `app.py` if the new model uses different class or function names.

## API Documentation

The application provides several API endpoints:

- `/api/predict-price` - Calculate optimal price
- `/api/market-deals` - Get current market deals
- `/api/product-info` - Get product information
- `/market/analyze` - Analyze a product

## Data Sources

- Amazon product data (via RapidAPI)
- Market trend data (synthetic and real)
- Price history data (where available)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 