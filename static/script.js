const productGroupMap = {
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
    "MusicalInstruments": [
      "Condenser"
    ],
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
    "Health&PersonalCare": [
      "Scientific"
    ]
};

// Switch between tabs
function switchTab(tabName) {
  const tabs = document.querySelectorAll('.tab');
  const tabContents = document.querySelectorAll('.tab-content');
  
  tabs.forEach(tab => tab.classList.remove('active'));
  tabContents.forEach(content => content.classList.add('hidden'));
  
  if (tabName === 'dashboard') {
    document.getElementById('dashboard-tab').classList.remove('hidden');
    document.querySelector('.tab:nth-child(1)').classList.add('active');
  } else if (tabName === 'pricing') {
    document.getElementById('pricing-tab').classList.remove('hidden');
    document.querySelector('.tab:nth-child(2)').classList.add('active');
  }
}

// Update product groups based on selected product type
function updateProductGroups() {
  const productType = document.getElementById('productType').value;
  const productGroupSelect = document.getElementById('productGroup');
  const searchInput = document.getElementById('groupSearchInput');
  
  // Clear existing options and search
  productGroupSelect.innerHTML = '';
  searchInput.value = '';
  
  if (productType) {
    // Enable select and search
    productGroupSelect.disabled = false;
    searchInput.disabled = false;
    
    // Add default option
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = 'Select Product Group';
    productGroupSelect.appendChild(defaultOption);
    
    // Add product groups for selected type
    const groups = productGroupMap[productType] || [];
    
    if (groups.length === 0) {
      const noGroupOption = document.createElement('option');
      noGroupOption.value = '';
      noGroupOption.textContent = 'No product groups available';
      productGroupSelect.appendChild(noGroupOption);
      productGroupSelect.disabled = true;
      searchInput.disabled = true;
    } else {
      groups.forEach(group => {
        const option = document.createElement('option');
        option.value = group;
        // Format the group name for display
        option.textContent = group
          .replace(/([A-Z])/g, ' $1') // Add space before capital letters
          .replace(/&/g, ' & ') // Add spaces around ampersands
          .replace(/\s{2,}/g, ' '); // Remove extra spaces
        productGroupSelect.appendChild(option);
      });
    }
  } else {
    // Disable and reset if no product type selected
    productGroupSelect.disabled = true;
    searchInput.disabled = true;
    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = 'Select Product Type First';
    productGroupSelect.appendChild(placeholder);
  }
}

// Filter product groups based on search input
function filterProductGroups() {
  const searchInput = document.getElementById('groupSearchInput');
  const productGroupSelect = document.getElementById('productGroup');
  const filter = searchInput.value.toLowerCase();
  const options = productGroupSelect.options;
  
  for (let i = 0; i < options.length; i++) {
    const text = options[i].text.toLowerCase();
    if (text.indexOf(filter) > -1) {
      options[i].style.display = '';
    } else {
      options[i].style.display = 'none';
    }
  }
}

// Calculate discount percentage dynamically
function calculateDiscountPercentage() {
  const actualPrice = parseFloat(document.getElementById('actualPrice').value);
  const discountedPrice = parseFloat(document.getElementById('discountedPrice').value);
  
  if (!isNaN(actualPrice) && !isNaN(discountedPrice) && actualPrice > 0) {
    const discountPercentage = ((actualPrice - discountedPrice) / actualPrice * 100).toFixed(2);
    document.getElementById('discountPercentage').value = discountPercentage;
  }
}

// Calculate discounted price dynamically
function calculateDiscountedPrice() {
  const actualPrice = parseFloat(document.getElementById('actualPrice').value);
  const discountPercentage = parseFloat(document.getElementById('discountPercentage').value);
  
  if (!isNaN(actualPrice) && !isNaN(discountPercentage) && actualPrice > 0) {
    const discountedPrice = (actualPrice * (1 - discountPercentage / 100)).toFixed(2);
    document.getElementById('discountedPrice').value = discountedPrice;
  }
}

// Add event listeners for price calculations
document.addEventListener('DOMContentLoaded', function() {
  const actualPriceInput = document.getElementById('actualPrice');
  const discountedPriceInput = document.getElementById('discountedPrice');
  const discountPercentageInput = document.getElementById('discountPercentage');
  
  if (actualPriceInput && discountedPriceInput) {
    actualPriceInput.addEventListener('input', function() {
      if (discountedPriceInput.value) {
        calculateDiscountPercentage();
      } else if (discountPercentageInput.value) {
        calculateDiscountedPrice();
      }
    });
    
    discountedPriceInput.addEventListener('input', calculateDiscountPercentage);
    discountPercentageInput.addEventListener('input', calculateDiscountedPrice);
  }
});

// Add these functions that are called but not defined
function showLoadingIndicator() {
  const loadingIndicator = document.getElementById('loadingIndicator') || createLoadingIndicator();
  loadingIndicator.style.display = 'flex';
}

function hideLoadingIndicator() {
  const loadingIndicator = document.getElementById('loadingIndicator');
  if (loadingIndicator) {
    loadingIndicator.style.display = 'none';
  }
}

function createLoadingIndicator() {
  const indicator = document.createElement('div');
  indicator.id = 'loadingIndicator';
  indicator.innerHTML = '<div class="spinner"></div><p>Processing...</p>';
  indicator.style.display = 'none';
  indicator.style.position = 'fixed';
  indicator.style.top = '0';
  indicator.style.left = '0';
  indicator.style.width = '100%';
  indicator.style.height = '100%';
  indicator.style.backgroundColor = 'rgba(0,0,0,0.5)';
  indicator.style.zIndex = '1000';
  indicator.style.justifyContent = 'center';
  indicator.style.alignItems = 'center';
  indicator.style.color = 'white';
  document.body.appendChild(indicator);
  return indicator;
}

function createRecommendationSection() {
  const section = document.createElement('div');
  section.id = 'recommendationResult';
  section.className = 'recommendation-result';
  section.style.display = 'none';
  
  // Add the section to the page
  const pricingTab = document.getElementById('pricing-tab');
  pricingTab.appendChild(section);
  
  return section;
}


// Modified getPriceRecommendation function to call Python script
async function getPriceRecommendation() {
  // Show loading indicator
  showLoadingIndicator();
  
  try {
    // Get form values
    const productType = document.getElementById('productType').value;
    const productGroup = document.getElementById('productGroup').value;
    const actualPrice = parseFloat(document.getElementById('actualPrice').value);
    const discountedPrice = parseFloat(document.getElementById('discountedPrice').value);
    const discountPercentage = parseFloat(document.getElementById('discountPercentage').value);
    const rating = parseFloat(document.getElementById('rating').value) || 0;
    const averagePrice = parseFloat(document.getElementById('averagePrice').value) || 0;
    const averageShippingValue = parseFloat(document.getElementById('averageShippingValue').value) || 0;
    const numberOfOrders = parseInt(document.getElementById('numberOfOrders').value) || 0;
    const competitorPrice = parseFloat(document.getElementById('competitorPrice').value) || 0;
    
    // Validate inputs
    if (!productType || !productGroup) {
      alert('Please select both product type and product group.');
      hideLoadingIndicator();
      return;
    }
    
    if (isNaN(actualPrice) || isNaN(discountedPrice) || isNaN(discountPercentage)) {
      alert('Please enter valid numbers for price information.');
      hideLoadingIndicator();
      return;
    }
    
    // Prepare data for Python script
    const pythonData = {
      productType,
      productGroup,
      discountedPrice,
      actualPrice,
      discountPercentage,
      rating,
      averagePrice,
      averageShippingValue,
      numberOfOrders,
      competitorPrice
    };
    
    // Call Python script via API endpoint
    const response = await fetch('/api/predict-price', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(pythonData)
    });
    
    if (!response.ok) {
      throw new Error('Failed to get prediction from Python model');
    }
    
    // Get the recommendation from Python
    const recommendation = await response.json();
    
    // Display the recommendation
    displayRecommendation(recommendation);
  } catch (error) {
    console.error('Error during prediction:', error);
    alert('An error occurred while generating the price recommendation.');
  } finally {
    hideLoadingIndicator();
  }
}

// Function to display the recommendation results
function displayRecommendation(recommendation) {
  // Create or update the recommendation result section
  const resultSection = document.getElementById('recommendationResult') || 
                       createRecommendationSection();
  
  // Format currency values
  const formatCurrency = new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD'
  });
  
  // Get original price for comparison
  const originalPrice = parseFloat(document.getElementById('actualPrice').value);
  const recommendedPrice = recommendation.predicted_price;
  const priceDifference = recommendedPrice - originalPrice;
  const percentChange = (priceDifference / originalPrice) * 100;
  
  // Build HTML for the result
  resultSection.innerHTML = `
    <h3>Price Recommendation</h3>
    <div class="result-item">
      <strong>Recommended Price:</strong> ${formatCurrency.format(recommendedPrice)}
    </div>
    <div class="result-item">
      <strong>Suggested Price Range:</strong> 
      ${formatCurrency.format(recommendedPrice * 0.95)} - ${formatCurrency.format(recommendedPrice * 1.05)}
    </div>
    <div class="result-item">
      <strong>Price Change:</strong> 
      <span class="${priceDifference >= 0 ? 'positive' : 'negative'}">
        ${priceDifference >= 0 ? '+' : ''}${formatCurrency.format(priceDifference)} 
        (${percentChange >= 0 ? '+' : ''}${percentChange.toFixed(2)}%)
      </span>
    </div>
    <div class="result-item">
      <strong>Model Used:</strong> ${recommendation.model_info || "DQN"}
    </div>
    <div class="charts">
      <div id="priceChart" class="chart-container"></div>
    </div>
  `;
  
  // Show the result section
  resultSection.style.display = 'block';
  
  // Render a simple chart if Chart.js is available
  if (typeof Chart !== 'undefined') {
    renderPriceComparisonChart(recommendedPrice);
  }
}

// Modified function to render a price comparison chart
function renderPriceComparisonChart(recommendedPrice) {
  const chartContainer = document.getElementById('priceChart');
  if (!chartContainer) return;
  
  // Clear any existing chart
  chartContainer.innerHTML = '<canvas id="priceComparisonCanvas"></canvas>';
  
  // Get original price and competitor price
  const originalPrice = parseFloat(document.getElementById('actualPrice').value);
  const competitorPrice = parseFloat(document.getElementById('competitorPrice').value) || 0;
  
  // Create the chart
  const ctx = document.getElementById('priceComparisonCanvas').getContext('2d');
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Current Price', 'Recommended Price', 'Competitor Price'],
      datasets: [{
        label: 'Price Comparison',
        data: [
          originalPrice, 
          recommendedPrice, 
          competitorPrice || null
        ],
        backgroundColor: [
          'rgba(54, 162, 235, 0.5)',
          'rgba(75, 192, 192, 0.5)',
          'rgba(255, 99, 132, 0.5)'
        ],
        borderColor: [
          'rgba(54, 162, 235, 1)',
          'rgba(75, 192, 192, 1)',
          'rgba(255, 99, 132, 1)'
        ],
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      scales: {
        y: {
          beginAtZero: false
        }
      }
    }
  });
}



