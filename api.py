# Save this file as api.py
from flask import Flask, request, jsonify
import numpy as np
import os
from stable_baselines3 import DQN
from RL_env1 import make_env

app = Flask(__name__)

# Load the model once at startup
MODEL = None
try:
    MODEL = DQN.load("dynamic_pricing_dqn")
    print("Model loaded successfully at startup")
except Exception as e:
    print(f"Warning: Could not load model at startup: {e}")
    # We'll try to load again when needed

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

# Create a flattened list of all product groups
ALL_PRODUCT_GROUPS = []
PRODUCT_GROUP_MAP = {
    "Computers&Accessories": [
      "Mice", "GraphicTablets", "Lapdesks", "NotebookComputerStands", "Keyboards",
      "Keyboard&MouseSets", "ExternalHardDisks", "DustCovers", "GamingMice", "MousePads",
      "HardDiskBags", "NetworkingDevices", "Routers", "Monitors", "Gamepads",
      "USBHubs", "PCMicrophones", "LaptopSleeves&Slipcases", "ExternalMemoryCardReaders",
      "EthernetCables", "Memory", "UninterruptedPowerSupplies", "Cases", "SecureDigitalCards",
      "Webcams", "CoolingPads", "LaptopAccessories", "InternalSolidStateDrives",
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

# Flatten all product groups
for product_type, groups in PRODUCT_GROUP_MAP.items():
    for group in groups:
        if group not in ALL_PRODUCT_GROUPS:
            ALL_PRODUCT_GROUPS.append(group)

# Create a map of product group to index
PRODUCT_GROUP_INDEX_MAP = {group: idx for idx, group in enumerate(ALL_PRODUCT_GROUPS)}

@app.route('/api/predict-price', methods=['POST'])
def predict_price():
    global MODEL
    
    # Check if we have received data
    if not request.json:
        return jsonify({"error": "No data provided"}), 400
    
    try:
        data = request.json
        
        # Try to load model if not already loaded
        if MODEL is None:
            try:
                print("Attempting to load model...")  # Debug log
                MODEL = DQN.load("dynamic_pricing_dqn")
                print("Model loaded successfully")  # Debug log
            except Exception as e:
                print(f"Error loading model: {e}")  # Debug log
                return jsonify({"error": f"Failed to load model: {e}"}), 500
        
        # Create the environment to get observation space
        env = make_env()
        
        # Extract features from the request
        product_type = data.get('productType')
        product_group = data.get('productGroup')
        
        # Prepare numerical features
        numerical_features = np.array([
            float(data.get('discountedPrice', 0)),
            float(data.get('actualPrice', 0)),
            float(data.get('discountPercentage', 0)) / 100,  # Convert to decimal
            float(data.get('rating', 0)),
            float(data.get('averagePrice', 0)),
            float(data.get('averageShippingValue', 0)),
            float(data.get('numberOfOrders', 0)),
            float(data.get('competitorPrice', 0))
        ], dtype=np.float32)
        
        # One-hot encode product type (9 dimensions)
        product_type_idx = PRODUCT_TYPE_MAP.get(product_type, 0)
        product_type_onehot = np.zeros(9, dtype=np.float32)
        product_type_onehot[product_type_idx] = 1
        
        # One-hot encode product group (207 dimensions)
        product_group_idx = PRODUCT_GROUP_INDEX_MAP.get(product_group, 0)
        product_group_onehot = np.zeros(207, dtype=np.float32)
        if product_group in PRODUCT_GROUP_INDEX_MAP and product_group_idx < 207:
            product_group_onehot[product_group_idx] = 1
        
        # Combine all features
        features = np.concatenate([
            numerical_features,
            product_type_onehot,
            product_group_onehot
        ])
        
        # Add sales history feature (making it 225 dimensions total as in test.py)
        features = np.append(features, 0).astype(np.float32)
        
        # Ensure feature vector has correct shape
        expected_shape = env.observation_space.shape
        if features.shape != expected_shape:
            if len(features) < expected_shape[0]:
                features = np.pad(features, (0, expected_shape[0] - len(features)))
            elif len(features) > expected_shape[0]:
                features = features[:expected_shape[0]]
        
        # Make prediction
        action, _ = MODEL.predict(features, deterministic=True)
        
        # Convert action to price (assuming action maps to price as in test.py)
        predicted_price = 10 + (action * 10)  # Map action to price (10, 20, ..., 500)
        
        return jsonify({
            "predicted_price": float(predicted_price),
            "model_info": "DQN RL model",
            "action_index": int(action)
        })
        
    except Exception as e:
        import traceback
        print(f"Error during prediction: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)