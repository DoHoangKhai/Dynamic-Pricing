# QuickPrice Test Environment

This is a standalone test environment for the improved QuickPrice model. It allows you to test the `FinalOptimalModelAdapter` model independently of the main application.

## Features

- Clean, standalone test environment
- Web interface for testing pricing recommendations
- Visualization tools for pricing factors
- Easy setup and configuration

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:5050
   ```

## Usage

The test environment provides a simplified interface to the QuickPrice system:

1. Enter product details in the form
2. View pricing recommendations
3. Explore pricing factors and visualizations
4. Test different product scenarios

## Configuration

You can configure the test environment by editing the following files:

- `app.py`: Main application settings
- `models/optimal_model_adapter.py`: Pricing model configuration
- `api/api_market_data.py`: Market data settings

## Key Features

- **Standalone Testing**: Test the pricing model without dependencies on the main application
- **API Server**: A simple Flask API server for interacting with the model
- **Test Scripts**: Scripts to verify model functionality and API endpoints
- **Comprehensive Documentation**: Detailed deployment and usage instructions

## Directory Structure

- **models/** - Core pricing models
  - `optimal_model_adapter.py` - Main pricing model implementation
  - `benchmark_models/` - Model benchmarking components

- **api/** - API integration components
  - `api_market_data.py` - Market data API endpoints
  - `api.py` - Main API functions

- **app.py** - Flask API server implementation
- **run_test.py** - Script to test the model directly
- **test_api.py** - Script to test the API endpoints
- **test_improved_model.py** - Comprehensive model test script
- **run.sh** - Server startup script
- **model_improvements.md** - Documentation of model improvements
- **DEPLOYMENT.md** - Detailed deployment instructions

## Getting Started

1. Install required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the standalone test to verify model functionality:
   ```
   python run_test.py
   ```

3. Start the API server:
   ```
   ./run.sh
   ```
   Or manually:
   ```
   python app.py --port 5005
   ```

4. Test the API endpoints:
   ```
   python test_api.py
   ```

## API Endpoints

- `GET /` - Health check and version information
- `POST /api/optimal/recommend` - Generate price recommendations

Sample request to the recommend endpoint:
```json
{
  "product_id": "test123",
  "price": 199.99,
  "cost": 120.0,
  "rating": 4.5,
  "number_of_orders": 500,
  "elasticity": 1.2,
  "competitor_price": 189.99,
  "competitive_intensity": 0.7
}
```

## Model Improvements

The improved dynamic pricing model (`FinalOptimalModelAdapter`) includes several key enhancements:

1. **Fixed elasticity handling**: Properly calculates elasticity-based prices
2. **Added volume impact**: Adjusts pricing based on order volume data
3. **Dynamic constraints**: Uses dynamic price constraints based on market conditions
4. **Rating factor refinement**: Improved quality adjustments based on ratings
5. **Competitive responsiveness**: Better responsiveness to competitor pricing

For more details on the improvements, see [model_improvements.md](model_improvements.md).

## For More Information

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions and troubleshooting. 