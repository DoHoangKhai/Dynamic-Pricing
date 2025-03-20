# Save this file as api.py
from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import os
import pandas as pd
from stable_baselines3 import DQN
from RL_env1 import make_env
from pricing_strategies import PricingStrategy, CustomerSegmentation
from market_environment import MarketEnvironment
from datetime import datetime, timedelta
import json
import traceback
import sys

# Add parent directory to path to allow imports from sibling modules
sys.path.append('..')

app = Flask(__name__)

# Global variable to store the loaded model and strategies
MODEL = None
pricing_strategy = PricingStrategy()
customer_segmentation = CustomerSegmentation()

# Add a route for the root URL
@app.route('/')
def index():
    return render_template('index.html')

# Try to load the model at startup
try:
    # Use the absolute path to the model file
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dynamic_pricing_dqn.zip")
    if os.path.exists(model_path):
        MODEL = DQN.load(model_path)
        print(f"Model loaded successfully at startup from {model_path}")
    else:
        print(f"Model file not found at {model_path}")
        print("Will attempt to load model on first request")
except Exception as e:
    print(f"Could not load model at startup: {e}")
    print("Will attempt to load model on first request")

# Map product types to indices
PRODUCT_TYPE_MAP = {
    "Computers&Accessories": 0,
    "Electronics": 1,
    "MusicalInstruments": 2,
    "OfficeProducts": 3,
    "Home&Kitchen": 4,
    "HomeImprovement": 5,
    "Toys&Games": 6,
    "Car&Motorbike": 7,
    "Health&PersonalCare": 8
}

# Map product groups to their respective product types
PRODUCT_GROUP_MAP = {
    "Computers&Accessories": [
      "GamingLaptops", "Keyboards", "Mouse", "MousePads", "Monitors", "GraphicsCards",
      "ExternalHardDrives", "NetworkingDevices", "RAMs", "Routers", "WirelessAccessPoint",
      "USBFlashDrives", "CPUs", "Motherboards", "PCCasings", "Barebone", "PowerSupplies",
      "SoundCards", "WiredNetworkAdapters", "Webcams", "CoolingPads", "LaptopAccessories", "InternalSolidStateDrives",
      "MultimediaSpeakerSystems", "DataCards&Dongles", "LaptopChargers&PowerSupplies",
      "PCSpeakers", "InternalHardDrives", "Printers", "SATACables", "PCHeadsets",
      "GamingKeyboards", "ExternalSolidStateDrives", "PowerLANAdapters", "Caddies",
      "TraditionalLaptops"
    ],
    "Electronics": [
      "USBCables", "WirelessUSBAdapters", "HDMICables", "SmartTelevisions",
      "RemoteControls", "StandardTelevisions", "TVWall&CeilingMounts", "RCACables",
      "Mounts", "OpticalCables", "Projectors", "Adapters", "SatelliteReceivers",
      "DVICables", "SpeakerCables", "StreamingClients", "AVReceivers&Amplifiers",
      "TowerSpeakers", "3DGlasses", "SmartWatches", "PowerBanks", "Smartphones",
      "MicroSD", "BasicMobiles", "In-Ear", "AutomobileChargers", "Cradles",
      "WallChargers", "OTGAdapters", "Tripods", "SelfieSticks", "Stands",
      "CableConnectionProtectors", "ScreenProtectors", "StylusPens", "BasicCases",
      "HandlebarMounts", "On-Ear", "CameraPrivacyCovers", "PhoneCharms", "Shower&WallMounts",
      "PenDrives", "DisposableBatteries", "Repeaters&Extenders", "TripodLegs",
      "VideoCameras", "Tabletop&TravelTripods", "Over-Ear", "BluetoothSpeakers",
      "GeneralPurposeBatteries&BatteryChargers", "RechargeableBatteries", "BluetoothAdapters",
      "USBtoUSBAdapters", "CompleteTripodUnits", "Film", "Lamps", "CleaningKits",
      "DomeCameras", "Basic", "OutdoorSpeakers", "SelfieLights", "BatteryChargers",
      "SoundbarSpeakers", "Earpads", "Headsets", "Tablets"
    ],
    "MusicalInstruments": ["Condenser"],
    "OfficeProducts": [
      "GelInkRollerballPens", "Tape", "InkjetInkCartridges", "WireboundNotebooks",
      "Notepads&MemoBooks", "BottledInk", "CompositionNotebooks", "RetractableBallpointPens",
      "ColouredPaper", "StickBallpointPens", "WoodenPencils", "Pens", "InkjetPrinters",
      "ColouringPens&Markers", "InkjetInkRefills&Kits", "Notebooks,WritingPads&Diaries",
      "BackgroundSupports", "Financial&Business", "TonerCartridges", "LiquidInkRollerballPens",
      "FountainPens"
    ],
    "Home&Kitchen": [
      "Décor", "Bedstand&DeskMounts", "ElectricKettles", "ElectricHeaters", "FanHeaters",
      "LintShavers", "DigitalKitchenScales", "Choppers", "InductionCooktop", "HandBlenders",
      "DryIrons", "MixerGrinders", "InstantWaterHeaters", "RoomHeaters", "Kettle&ToasterSets",
      "StorageWaterHeaters", "ImmersionRods", "AirFryers", "LaundryBaskets", "SteamIrons",
      "JuicerMixerGrinders", "HandheldVacuums", "EggBoilers", "SandwichMakers",
      "MiniFoodProcessors&Choppers", "DigitalScales", "VacuumSealers", "CeilingFans",
      "CanisterVacuums", "PressureWashers,Steam&WindowCleaners", "HalogenHeaters",
      "Pop-upToasters", "HeatConvectors", "ElectricGrinders", "ExhaustFans",
      "DripCoffeeMachines", "WaterPurifierAccessories", "WaterCartridges",
      "Rice&PastaCookers", "AirPurifiers&Ionizers", "Wet-DryVacuums", "HEPAAirPurifiers",
      "WaterFilters&Purifiers", "LaundryBags", "Sewing&EmbroideryMachines", "SprayBottles",
      "HandMixers", "WetGrinders", "OvenToasterGrills", "Juicers", "SmallKitchenAppliances",
      "DigitalBathroomScales", "EspressoMachines", "TableFans", "MilkFrothers",
      "Humidifiers", "StandMixerAccessories", "RoboticVacuums", "YogurtMakers",
      "ColdPressJuicers", "Split-SystemAirConditioners", "SmallApplianceParts&Accessories",
      "WaffleMakers&Irons", "StovetopEspressoPots", "MeasuringSpoons", "CoffeePresses",
      "RotiMakers", "FanParts&Accessories", "StandMixers", "PedestalFans", "HandheldBags"
    ],
    "HomeImprovement": [
      "Paints", "PaintingMaterials", "Adapters&Multi-Outlets", "SurgeProtectors", "CordManagement"
    ],
    "Toys&Games": [],
    "Car&Motorbike": [],
    "Health&PersonalCare": ["Scientific"]
}

# Define elasticity values for product groups (example values)
PRODUCT_GROUP_ELASTICITY = {
    "Computers&Accessories": {
        "GamingLaptops": 0.8,      # Less elastic - premium product
        "TraditionalLaptops": 1.2,  # More elastic - more competition
        "Mouse": 1.4,               # More elastic - commodity item
        "Keyboards": 1.3,           # More elastic - commodity item
        "Monitors": 1.1,            # Moderate elasticity
        "ExternalHardDrives": 1.2,  # More elastic
        "InternalSolidStateDrives": 0.9, # Less elastic - specialty item
        "GraphicsCards": 0.7,       # Low elasticity - specialized hardware
    },
    "Electronics": {
        "SmartTelevisions": 1.1,    # Moderate elasticity
        "Smartphones": 0.9,         # Less elastic - premium devices
        "PowerBanks": 1.3,          # More elastic - commodity
        "BluetoothSpeakers": 1.2,   # More elastic
        "Tablets": 1.0,             # Neutral elasticity
        "Headsets": 1.1             # Moderate elasticity
    },
    "Home&Kitchen": {
        "AirFryers": 1.2,           # More elastic - competitive market
        "ElectricKettles": 1.4,     # More elastic - commodity
        "RoboticVacuums": 0.8,      # Less elastic - premium product
        "CeilingFans": 1.0          # Neutral elasticity
    }
}

# Default elasticity for product types
PRODUCT_TYPE_ELASTICITY = {
    "Computers&Accessories": 1.1,
    "Electronics": 1.0,
    "MusicalInstruments": 0.8,
    "OfficeProducts": 1.3,
    "Home&Kitchen": 1.2,
    "HomeImprovement": 1.1,
    "Toys&Games": 1.0,
    "Car&Motorbike": 0.9,
    "Health&PersonalCare": 0.8
}

# Flatten all product groups
ALL_PRODUCT_GROUPS = []
for product_type, groups in PRODUCT_GROUP_MAP.items():
    for group in groups:
        if group not in ALL_PRODUCT_GROUPS:
            ALL_PRODUCT_GROUPS.append(group)

# Create a map of product group to index
PRODUCT_GROUP_INDEX_MAP = {group: idx for idx, group in enumerate(ALL_PRODUCT_GROUPS)}

def get_elasticity(product_type, product_group):
    """
    Get elasticity value based on product type and group.
    Higher elasticity means more price sensitive.
    """
    # Default elasticity
    elasticity = 1.0
    
    # Adjust based on product type
    product_type_lower = product_type.lower()
    if "premium" in product_type_lower or "luxury" in product_type_lower:
        elasticity = 0.7  # Premium products are less price sensitive
    elif "basic" in product_type_lower or "commodity" in product_type_lower:
        elasticity = 1.3  # Basic products are more price sensitive

    # Adjust based on specific product groups
    product_group_lower = product_group.lower()
    if "gaming" in product_group_lower or "laptop" in product_group_lower:
        elasticity = 0.9  # Gaming and laptops are less price sensitive
    elif "accessory" in product_group_lower or "cable" in product_group_lower:
        elasticity = 1.2  # Accessories and cables are more price sensitive
    
    return elasticity

def get_price_ranges(elasticity, base_price):
    """
    Get optimal price and price ranges based on elasticity.
    
    Args:
        elasticity: Product price elasticity
        base_price: Base price from model prediction
        
    Returns:
        dict: Dictionary with optimal price, min price, and max price
    """
    # Default price range = ±10%
    if elasticity > 1.2:  # High elasticity (price sensitive)
        min_price = base_price * 0.90
        max_price = base_price * 1.05
        
        # Slightly favor lower prices for price-sensitive products
        optimal_price = base_price * 0.97
    elif elasticity < 0.8:  # Low elasticity (premium)
        min_price = base_price * 0.95
        max_price = base_price * 1.15
        
        # Slightly favor higher prices for premium products
        optimal_price = base_price * 1.05
    else:  # Medium elasticity
        min_price = base_price * 0.92
        max_price = base_price * 1.08
        optimal_price = base_price
    
    return {
        "optimal_price": round(optimal_price, 2),
        "min_price": round(min_price, 2),
        "max_price": round(max_price, 2)
    }

@app.route('/api/predict-price', methods=['POST'])
def predict_price():
    global MODEL, pricing_strategy, customer_segmentation
    
    # Reset pricing strategy and customer segmentation to avoid price drift between requests
    pricing_strategy.reset()
    customer_segmentation.reset()
    
    # Check if we have received data
    if not request.json:
        return jsonify({"error": "No data provided"}), 400
    
    try:
        data = request.json
        
        # Print received data for debugging
        print("Received data:", data)
        
        # Try to load model if not already loaded
        if MODEL is None:
            try:
                print("Attempting to load model...")
                model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dynamic_pricing_dqn.zip")
                MODEL = DQN.load(model_path)
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {e}")
                return jsonify({"error": f"Failed to load model: {e}"}), 500
        
        # Extract features from the request
        # Match the exact field names as sent from frontend
        product_type = data.get('productType', '')
        product_group = data.get('productGroup', '')
        
        # Get elasticity with more precise calculation
        elasticity = get_elasticity(product_type, product_group)
        
        # Look up product-group specific elasticity if available
        if product_type in PRODUCT_GROUP_ELASTICITY and product_group in PRODUCT_GROUP_ELASTICITY[product_type]:
            elasticity = PRODUCT_GROUP_ELASTICITY[product_type][product_group]
        
        # Derive PPI (Price Position Index) with more emphasis
        actual_price = float(data.get('actualPrice', 0))
        competitor_price = float(data.get('competitorPrice', 0))
        
        # Calculate PPI (Price Position Index)
        ppi = 1.0
        if competitor_price > 0:
            ppi = actual_price / competitor_price
            
            # Adjust elasticity based on PPI
            # If price is much higher than competitors, increase elasticity (more price-sensitive)
            # If price is much lower than competitors, decrease elasticity (less price-sensitive)
            if ppi > 1.2:  # More than 20% above competitor
                elasticity = elasticity * 1.2  # Increase elasticity by 20%
            elif ppi < 0.8:  # More than 20% below competitor
                elasticity = elasticity * 0.9  # Decrease elasticity by 10%
        
        # Get rating and number of orders
        rating = float(data.get('rating', 4.0))
        num_orders = float(data.get('numberOfOrders', 0))
        
        print(f"Processing with: elasticity={elasticity}, ppi={ppi}, rating={rating}, num_orders={num_orders}")
        
        # Create environment with our custom parameters
        env = make_env(
            elasticity=elasticity,
            ppi=ppi,
            rating=rating,
            num_orders=num_orders
        )
        
        # Get initial observation
        observation = env.reset()
        
        # Make prediction using DQN model
        action, _ = MODEL.predict(observation, deterministic=True)
        
        # Convert action to price (assuming action maps to price as in environment)
        model_price = env.min_price + (action * env.price_step)
        
        # Define product information for enhanced pricing strategy
        product = {
            'product_id': 'test_product',
            'product_type': product_type,
            'product_group': product_group,
            'price': actual_price,
            'cost': actual_price * 0.6,  # Estimate cost as 60% of price
            'elasticity': elasticity,
            'rating': rating,
            'ppi': ppi,
            'number_of_orders': num_orders
        }
        
        # Calculate competitive intensity based on competitor price
        # If the competitor price is very different from yours, competition is more intense
        competitive_intensity = 0.7  # Default
        if competitor_price > 0:
            price_diff_ratio = abs(actual_price - competitor_price) / competitor_price
            # Higher difference = more competitive pressure
            competitive_intensity = min(0.9, 0.6 + price_diff_ratio)
        
        # Define market information with stronger competitor influence
        market_info = {
            'competitive_intensity': competitive_intensity,
            'price_trend': 0.0,  # Default to stable prices
            'current_price_ratio': ppi
        }
        
        # Get enhanced price recommendation using our pricing strategy
        strategy_recommendation = pricing_strategy.get_price_recommendations(product, market_info)
        strategy_price_ratio = strategy_recommendation.get('price_ratio', 1.0)
        
        # Calculate strategy-based price
        strategy_price = actual_price * strategy_price_ratio
        
        # Determine model weight based on sanity check of model price
        # If model price is too far off from actual price, give it less weight
        model_weight = 0.3  # Default weight for model
        price_diff_pct = abs(model_price - actual_price) / actual_price if actual_price > 0 else 0
        
        if price_diff_pct > 0.40:  # If model price is more than 40% different from actual
            model_weight = 0.05  # Give model very little weight
        elif price_diff_pct > 0.25:  # If model price is 25-40% different
            model_weight = 0.10  # Give model little weight
        elif price_diff_pct > 0.15:  # If model price is 15-25% different
            model_weight = 0.20  # Give model moderate weight
        
        # Weighted blend that prioritizes strategy price when model price is unrealistic
        blended_price = (model_price * model_weight) + (strategy_price * (1 - model_weight))
        
        # Apply adjustment based on competitor price (stronger influence)
        if competitor_price > 0:
            # If competitor price is significantly different, adjust our price towards it
            competitor_impact_factor = 0.0
            price_diff = competitor_price - blended_price
            price_diff_ratio = price_diff / blended_price if blended_price > 0 else 0
            
            # Stronger influence when competitor price differs greatly
            if abs(price_diff_ratio) > 0.20:  # >20% difference
                competitor_impact_factor = 0.30  # 30% weight to competitor direction
            elif abs(price_diff_ratio) > 0.10:  # >10% difference
                competitor_impact_factor = 0.25  # 25% weight to competitor direction
            else:  # Small difference
                competitor_impact_factor = 0.15  # 15% weight to competitor direction
            
            # Apply the competitor influence
            competitor_adjusted_price = blended_price + (price_diff * competitor_impact_factor)
            
            # Ensure the adjusted price is within reasonable bounds
            min_bound = blended_price * 0.8
            max_bound = blended_price * 1.2
            blended_price = max(min_bound, min(max_bound, competitor_adjusted_price))
                
        # Apply adjustment based on order count (popularity pricing)
        # This ensures order count directly affects the price
        order_adjustment = 1.0
        if num_orders > 200:
            # Very popular products can command higher prices
            order_adjustment = 1.07  # 7% premium for very popular products
        elif num_orders > 100:
            # Popular products get a small premium
            order_adjustment = 1.04  # 4% premium for popular products
        elif num_orders > 50:
            # Moderately popular products get a tiny premium
            order_adjustment = 1.02  # 2% premium for moderately popular products
        elif num_orders < 20:
            # Less popular products need lower prices
            order_adjustment = 0.97  # 3% discount
        elif num_orders < 10:
            # Very unpopular products need significant discounts
            order_adjustment = 0.93  # 7% discount for unpopular products
            
        # Apply order adjustment to blended price
        final_price = blended_price * order_adjustment
        
        # Get customer segmentation data for explanation purposes
        segments = customer_segmentation.get_segment_distribution(product_type, product_group, rating)
        segment_conversion = customer_segmentation.calculate_segment_conversion_probabilities(
            strategy_price_ratio, product
        )
        profit_multiplier = segment_conversion.get('expected_profit_multiplier', 1.0)
        
        # Calculate demand modifier for explanation
        demand_modifier = customer_segmentation.get_demand_modifier(
            product, final_price / actual_price, (final_price - product['cost']) / final_price
        )
        
        # Calculate min and max prices based on adjusted price (±5%)
        min_price = final_price * 0.95
        max_price = final_price * 1.05
        optimal_price = final_price
        
        # Generate elasticity category and explanation
        if elasticity > 1.2:
            explanation = "This product has high price elasticity, making it price-sensitive. We recommend competitive pricing to maximize sales volume."
            elasticity_category = "high"
        elif elasticity < 0.8:
            explanation = "This product has low price elasticity, suggesting premium positioning. You can prioritize higher margins with less impact on demand."
            elasticity_category = "low"
        else:
            explanation = "This product has moderate price elasticity. We recommend balanced pricing that considers both competitive positioning and profit margins."
            elasticity_category = "medium"
        
        # Calculate price change percentage from current price
        price_change_pct = ((optimal_price - actual_price) / actual_price) * 100 if actual_price > 0 else 0
        
        # Compare with competitor
        competitor_comparison = "lower than" if optimal_price < competitor_price else "higher than"
        if abs(optimal_price - competitor_price) < 0.05 * competitor_price:
            competitor_comparison = "comparable to"
        
        # Add segment information to explanation
        top_segment = max(segments.items(), key=lambda x: x[1]['weight'])
        segment_explanation = f" Our analysis shows your top customer segment is '{top_segment[0]}' ({int(top_segment[1]['weight']*100)}% of customers)."
        
        # Create comprehensive explanation
        comprehensive_explanation = (
            f"{explanation}{segment_explanation} The recommended price of ${optimal_price:.2f} is "
            f"{abs(round(price_change_pct, 1))}% {'higher' if price_change_pct > 0 else 'lower'} than your current price "
            f"and {competitor_comparison} your competitor's price of ${competitor_price}."
        )
        
        # Create explanation for each impact factor
        rating_impact = (rating - 3.0) * 5.0  # 5% impact per star above 3
        order_impact = min(10.0, num_orders / 20.0)  # Up to 10% for orders (increased impact)
        elasticity_impact = (1.0 - elasticity) * 10.0  # Impact based on elasticity
        
        # Calculate competitor impact more directly
        competitor_impact = 0.0
        if competitor_price > 0:
            # Calculate impact based on how much competitor price affected our final price
            initial_strategy_price = actual_price * strategy_price_ratio
            competitor_price_effect = optimal_price - initial_strategy_price
            competitor_impact = (competitor_price_effect / actual_price) * 100.0
            
            # Cap the competitor impact at ±20%
            competitor_impact = max(-20.0, min(20.0, competitor_impact))
        
        # Calculate market impact from demand modifier
        market_impact = (demand_modifier - 1.0) * 100.0
        
        # Prepare response
        response = {
            "recommendedPrice": round(optimal_price, 2),
            "minPrice": round(min_price, 2),
            "maxPrice": round(max_price, 2),
            "elasticityCategory": elasticity_category,
            "explanation": comprehensive_explanation,
            "ratingImpact": round(rating_impact, 1),
            "orderImpact": round(order_impact, 1),
            "competitorImpact": round(competitor_impact, 1),
            "elasticityImpact": round(elasticity_impact, 1),
            "marketImpact": round(market_impact, 1),
            "model_info": "Enhanced Dynamic Pricing Model with Customer Segmentation",
            # Add debug info to see internal price calculations
            "debug": {
                "initialPrice": actual_price,
                "modelPredictedPrice": round(model_price, 2),
                "strategyPrice": round(strategy_price, 2),
                "modelWeight": round(model_weight, 3),
                "blendedPrice": round(blended_price, 2),
                "finalPrice": round(final_price, 2),
                "competitorPrice": round(competitor_price, 2),
                "orderCount": int(num_orders),
                "elasticity": round(elasticity, 4),
                "orderAdjustment": round(order_adjustment, 4),
                "ppi": round(ppi, 4)
            }
        }
        
        # Add market insights if available
        # These could come from the market_data_analyzer in a real implementation
        asin = data.get('asin')
        if asin:
            response["marketInsights"] = {
                "priceTrend": "stable",
                "priceVolatility": 2.5,
                "marketPosition": "average",
                "sentimentScore": 0.35
            }
        
        print("Sending response:", response)
        return jsonify(response)
        
    except Exception as e:
        import traceback
        print(f"Error during prediction: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/category-revenue', methods=['GET'])
def get_category_revenue():
    """
    Get revenue data for a specific product category
    """
    category = request.args.get('category', 'Electronics')
    
    # Generate simulated revenue data (since we don't have real data)
    current_date = datetime.now()
    data = []
    
    # Generate 6 months of simulated data
    for i in range(6):
        month_date = current_date - timedelta(days=30 * i)
        month_str = month_date.strftime('%b')
        
        # Base revenue by category
        base_revenue = {
            'Electronics': 42000,
            'Computers&Accessories': 38000,
            'MusicalInstruments': 15000,
            'OfficeProducts': 22000,
            'Home&Kitchen': 30000,
            'HomeImprovement': 28000,
            'Toys&Games': 18000,
            'Car&Motorbike': 24000,
            'Health&PersonalCare': 26000
        }.get(category, 20000)
        
        # Add some random variation
        revenue = base_revenue * (0.8 + 0.4 * np.random.random())
        
        # Add seasonality (higher in recent months)
        seasonality = 1.0 + (0.2 * (5 - i) / 5)
        revenue = revenue * seasonality
        
        data.append({
            'month': month_str,
            'revenue': round(revenue, 2)
        })
    
    # Reverse to get chronological order
    data.reverse()
    
    return jsonify(data)

# Add the following routes to serve static files
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/api/market-deals', methods=['GET'])
def get_market_deals():
    """Get current deals data for price adjustment"""
    try:
        country = request.args.get('country', 'US')
        
        # Import here to avoid circular imports
        from amazon_api import fetch_deals
        
        print(f"[DEBUG] Fetching deals for country: {country}")
        
        # Fetch current deals data
        deals_data = fetch_deals(country)
        
        if not deals_data:
            print("[ERROR] fetch_deals returned None")
            return jsonify({
                'success': False,
                'message': "No response from deals API"
            }), 500
            
        if "error" in deals_data:
            print(f"[ERROR] Deal API error: {deals_data.get('error')}")
            return jsonify({
                'success': False,
                'message': f"Error fetching deals data: {deals_data.get('error', 'Unknown error')}"
            }), 500
            
        # Ensure the expected structure exists
        if "data" not in deals_data or "deals" not in deals_data.get("data", {}):
            print(f"[ERROR] Unexpected API response structure: {deals_data.keys()}")
            return jsonify({
                'success': False,
                'message': "Invalid API response structure"
            }), 500
            
        # Get the actual deals list
        deals = deals_data.get("data", {}).get("deals", [])
        
        if not deals:
            return jsonify({
                'success': True,
                'message': 'No active deals found',
                'deals_count': 0,
                'avg_discount': 0
            })
            
        # Calculate average discount
        total_discount = 0
        discount_count = 0
        
        for deal in deals:
            if isinstance(deal, dict) and "savings_percentage" in deal:
                try:
                    discount = deal["savings_percentage"]
                    if isinstance(discount, (int, float)):
                        total_discount += discount
                        discount_count += 1
                    elif isinstance(discount, str):
                        discount_value = float(discount.replace('%', ''))
                        total_discount += discount_value
                        discount_count += 1
                except (ValueError, AttributeError) as e:
                    print(f"[DEBUG] Error parsing discount: {e}")
                    continue
        
        avg_discount = total_discount / discount_count if discount_count > 0 else 0
        
        print(f"[DEBUG] Deals count: {len(deals)}, Avg discount: {avg_discount:.2f}%")
        
        return jsonify({
            'success': True,
            'deals_count': len(deals),
            'avg_discount': round(avg_discount, 2),
            'deals_summary': {
                'total_deals': len(deals),
                'average_discount': round(avg_discount, 2),
                'market_activity': 'high' if len(deals) > 50 else 'medium' if len(deals) > 20 else 'low'
            }
        })
        
    except Exception as e:
        import traceback
        print(f"[ERROR] Error processing deals data: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'Error processing deals data: {str(e)}'
        }), 500

if __name__ == "__main__":
    app.run(debug=True, port=5050)