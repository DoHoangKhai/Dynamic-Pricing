# Market Data Collection and Analysis Module

This module integrates real-time market data collection from the Amazon API into the dynamic pricing system, enabling data-driven pricing decisions based on current market conditions and trends.

## Features

- **Real-Time Data Collection**: Scheduled data collection from various Amazon API endpoints
- **Time Series Processing**: Conversion of raw API data into structured time series format
- **Market Analysis**: Advanced analysis of price trends, sentiment, and market position
- **Visualizations**: Visual representations of price trends and sentiment distribution
- **Pricing Integration**: Incorporation of market insights into the dynamic pricing model

## Directory Structure

```
/test
  ├── amazon_api.py          # Amazon API access and data collection
  ├── market_data_scheduler.py # Scheduled data collection
  ├── market_data_analysis.py  # Analysis and feature extraction
  ├── api_market_data.py      # API endpoints for market data
  ├── data/                  # Data storage directory
  │   ├── raw/               # Raw API responses
  │   ├── time_series/       # Processed time series data
  │   └── analysis/          # Analysis outputs and visualizations
  └── README_MARKET_DATA.md  # This file
```

## Setup

1. Ensure you have the required dependencies installed:
   ```
   pip install -r requirements.txt
   ```

2. Set up your Amazon API key:
   - Create a `.env` file in the `/test` directory
   - Add your Amazon API key:
     ```
     AMAZON_API_KEY=your_api_key_here
     AMAZON_API_HOST=amazon-api-host.com
     ```

3. Create the necessary data directories:
   ```
   mkdir -p test/data/raw test/data/time_series test/data/analysis
   ```

## Usage

### Data Collection

There are two ways to collect market data:

1. **Manual Collection via API**:
   - Use the `/api/market/refresh` endpoint with POST request
   - Provide parameters: `asin`, `category`, `country`

2. **Scheduled Collection**:
   - Run the scheduler script:
     ```
     python market_data_scheduler.py
     ```
   - Configure collection interval and products in the script

### Viewing Market Analysis

1. **Web Dashboard**:
   - Navigate to the "Market Analysis" tab in the web interface
   - Use the controls to refresh data, analyze products, and generate reports

2. **API Endpoints**:
   - `/api/market/product-analysis/<asin>`: Get detailed analysis for a product
   - `/api/market/price-visualization/<asin>`: Get price trend visualization
   - `/api/market/sentiment-visualization/<asin>`: Get sentiment visualization
   - `/api/market/report`: Get the latest market report
   - `/api/market/pricing-features/<asin>`: Get pricing features for a product

### Integration with Pricing

The market data is automatically integrated with the pricing model through:

1. **Price Prediction API**:
   - When an ASIN is provided in the price prediction request, market data is incorporated
   - The response includes a `marketInsights` section with key market indicators

2. **Pricing Factors**:
   - Market trends influence the recommended price
   - Sentiment affects the premium/discount applied
   - Market position determines competitiveness adjustments

## Configuration

The data collection behavior can be configured in `market_data_scheduler.py`:

- `collection_interval`: How often to collect data (in hours)
- `categories`: Which product categories to track
- `country`: Target marketplace (e.g., "US", "UK")
- `products`: List of ASINs to track

## Data Analysis Features

The module provides several types of analysis:

1. **Price Trend Analysis**:
   - Historical price tracking
   - Volatility calculation
   - Trend detection (increasing, stable, decreasing)
   - Min/max/average prices

2. **Market Position Analysis**:
   - Comparison with bestsellers
   - Similar product detection
   - Price position relative to similar products
   - Competitive status determination

3. **Sentiment Analysis**:
   - Review sentiment tracking
   - Sentiment score calculation
   - Recent review monitoring
   - Sentiment trend detection

4. **Deal Activity Analysis**:
   - Monitoring market-wide discounts
   - Deal frequency tracking
   - Promotional patterns identification
   - Average discount calculation

## Adding New Data Sources

To add new Amazon API endpoints for data collection:

1. Add a new fetch function in `amazon_api.py`
2. Create a corresponding processing function for time series data
3. Update the `collect_market_data` function to call the new endpoint
4. Add relevant analysis methods in `market_data_analysis.py`

## Troubleshooting

- **Missing Data**: Check if the data collection has been run for the specified ASIN
- **API Errors**: Verify your API key and request parameters
- **Visualization Errors**: Ensure matplotlib and its dependencies are correctly installed
- **Scheduler Issues**: Check if the scheduler is running and properly configured

## Dependencies

- Flask: Web framework
- Pandas: Data processing
- Matplotlib: Visualization
- Requests: API communication
- Schedule: Task scheduling
- Scikit-learn: Data analysis 