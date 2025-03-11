# Save this file as api.py
from flask import Flask, request, jsonify
import numpy as np
import os
from stable_baselines3 import DQN
from RL_env1 import make_env

app = Flask(__name__)

# Global variable to store the loaded model
MODEL = None

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
    """Calculate elasticity based on product type and group"""
    # If we have specific elasticity for this product group
    if (product_type in PRODUCT_GROUP_ELASTICITY and 
        product_group in PRODUCT_GROUP_ELASTICITY[product_type]):
        return PRODUCT_GROUP_ELASTICITY[product_type][product_group]
    
    # Otherwise use the product type elasticity
    if product_type in PRODUCT_TYPE_ELASTICITY:
        return PRODUCT_TYPE_ELASTICITY[product_type]
    
    # Default elasticity
    return 1.0

def get_price_ranges(elasticity, base_price):
    """Calculate price ranges based on elasticity"""
    if elasticity > 1.2:  # High elasticity - price sensitive
        min_price = base_price * 0.7
        max_price = base_price * 0.9
        optimal_price = base_price * 0.8
        sensitivity = "Price sensitive"
    elif elasticity < 0.8:  # Low elasticity - price insensitive
        min_price = base_price * 0.9
        max_price = base_price * 1.3
        optimal_price = base_price * 1.2
        sensitivity = "Premium pricing"
    else:  # Medium elasticity - neutral
        min_price = base_price * 0.8
        max_price = base_price * 1.1
        optimal_price = base_price
        sensitivity = "Balanced pricing"
    
    return {
        "min_price": round(min_price, 2),
        "max_price": round(max_price, 2),
        "optimal_price": round(optimal_price, 2),
        "sensitivity": sensitivity
    }

@app.route('/api/predict-price', methods=['POST'])
def predict_price():
    global MODEL
    
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
        
        # Get elasticity
        elasticity = get_elasticity(product_type, product_group)
        
        # Derive PPI (Price Position Index) - simplified for now
        actual_price = float(data.get('actualPrice', 0))
        competitor_price = float(data.get('competitorPrice', 0))
        
        ppi = 1.0
        if competitor_price > 0:
            ppi = actual_price / competitor_price
        
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
        
        # Make prediction
        action, _ = MODEL.predict(observation, deterministic=True)
        
        # Convert action to price (assuming action maps to price as in environment)
        base_price = env.min_price + (action * env.price_step)
        
        # Get price ranges based on elasticity
        price_ranges = get_price_ranges(elasticity, base_price)
        
        # Generate explanation based on elasticity
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
        price_change_pct = ((price_ranges["optimal_price"] - actual_price) / actual_price) * 100 if actual_price > 0 else 0
        
        # Compare with competitor
        competitor_comparison = "lower than" if price_ranges["optimal_price"] < competitor_price else "higher than"
        if abs(price_ranges["optimal_price"] - competitor_price) < 0.05 * competitor_price:
            competitor_comparison = "comparable to"
        
        # Create comprehensive explanation
        comprehensive_explanation = (
            f"{explanation} The recommended price of ${price_ranges['optimal_price']} is "
            f"{abs(round(price_change_pct, 1))}% {'higher' if price_change_pct > 0 else 'lower'} than your current price "
            f"and {competitor_comparison} your competitor's price of ${competitor_price}."
        )
        
        # Prepare response
        response = {
            "recommended_price": price_ranges["optimal_price"],
            "min_price": price_ranges["min_price"],
            "max_price": price_ranges["max_price"],
            "elasticity": elasticity_category,
            "explanation": comprehensive_explanation,
            "model_info": "Enhanced DQN Model with Elasticity Awareness"
        }
        
        print("Sending response:", response)
        return jsonify(response)
        
    except Exception as e:
        import traceback
        print(f"Error during prediction: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)