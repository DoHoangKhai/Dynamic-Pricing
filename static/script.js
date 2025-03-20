const productGroupMap = {
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
  // Remove active class from all tab buttons
  const tabButtons = document.querySelectorAll('.tab-button');
  tabButtons.forEach(button => button.classList.remove('active'));
  
  // Hide all tab contents
  const tabContents = document.querySelectorAll('.tab-content');
  tabContents.forEach(content => content.classList.remove('active'));
  
  // Add active class to selected tab button and show corresponding content
  const selectedButton = document.querySelector(`.tab-button[onclick="switchTab('${tabName}')"]`);
  if (selectedButton) {
    selectedButton.classList.add('active');
  }
  
  // Show the selected tab content
  const selectedContent = document.getElementById(tabName);
  if (selectedContent) {
    selectedContent.classList.add('active');
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
  // Get the result card
  const resultCard = document.getElementById('resultCard');
  resultCard.style.display = 'block';
  
  // Format currency values
  const formatCurrency = new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD'
  });
  
  // Get original price for comparison
  const originalPrice = parseFloat(document.getElementById('actualPrice').value);
  
  // Extract values from API response with proper key names
  const recommendedPrice = recommendation.recommended_price;
  const minPrice = recommendation.min_price;
  const maxPrice = recommendation.max_price;
  const elasticity = recommendation.elasticity;
  const explanation = recommendation.explanation;
  
  // Calculate price difference
  const priceDifference = originalPrice ? (recommendedPrice - originalPrice) : 0;
  const percentChange = originalPrice ? ((priceDifference / originalPrice) * 100) : 0;
  
  // Update the display elements
  document.getElementById('recommendedPrice').textContent = formatCurrency.format(recommendedPrice);
  document.getElementById('priceRange').textContent = `Competitive price range: ${formatCurrency.format(minPrice)} - ${formatCurrency.format(maxPrice)}`;
  
  // Create detailed explanation text
  let detailedExplanation = `
    <p><strong>Elasticity:</strong> ${elasticity}</p>
  `;
  
  // Only add price change if originalPrice exists
  if (originalPrice) {
    detailedExplanation += `
      <p><strong>Price Change:</strong> 
      <span class="${priceDifference >= 0 ? 'positive' : 'negative'}">
        ${priceDifference >= 0 ? '+' : ''}${formatCurrency.format(priceDifference)} 
        (${percentChange >= 0 ? '+' : ''}${percentChange.toFixed(2)}%)
      </span>
      </p>
    `;
  }
  
  // Add the API explanation
  detailedExplanation += `<p>${explanation}</p>`;
  
  // Add competitor price comparison if available
  const competitorPrice = parseFloat(document.getElementById('competitorPrice').value);
  if (!isNaN(competitorPrice) && competitorPrice > 0) {
    const competitiveDiff = recommendedPrice - competitorPrice;
    const competitivePercent = (competitiveDiff / competitorPrice) * 100;
    
    detailedExplanation += `
      <p><strong>Competitor Comparison:</strong> 
        <span class="${competitiveDiff >= 0 ? 'positive' : 'negative'}">
          ${competitiveDiff >= 0 ? '+' : ''}${formatCurrency.format(competitiveDiff)} 
          (${competitivePercent >= 0 ? '+' : ''}${competitivePercent.toFixed(2)}%)
        </span>
      </p>
    `;
  }
  
  // Update the explanation text
  document.getElementById('priceExplanation').innerHTML = detailedExplanation;
  
  // Update styling based on elasticity
  if (elasticity === "high") {
    resultCard.className = 'result-card high-elasticity';
  } else if (elasticity === "low") {
    resultCard.className = 'result-card low-elasticity';
  } else {
    resultCard.className = 'result-card medium-elasticity';
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

// Function to calculate optimal price
function calculatePrice() {
    // Get input values
    const productType = document.getElementById('productType').value;
    const productGroup = document.getElementById('productGroup').value;
    const asin = document.getElementById('asin').value;
    const actualPrice = parseFloat(document.getElementById('actualPrice').value);
    const competitorPrice = parseFloat(document.getElementById('competitorPrice').value);
    const rating = parseFloat(document.getElementById('rating').value);
    const numberOfOrders = parseInt(document.getElementById('numberOfOrders').value);
    
    console.log("Preparing to calculate price with:", {
        productType, productGroup, asin, actualPrice, 
        competitorPrice, rating, numberOfOrders
    });
    
    // Validate inputs
    if (isNaN(actualPrice) || isNaN(competitorPrice) || isNaN(rating) || isNaN(numberOfOrders)) {
        // Show error in the results area instead of an alert
        document.getElementById('noResults').innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-circle me-2"></i>
                <strong>Error:</strong> Please enter valid numbers for price, rating, and orders.
            </div>
        `;
        document.getElementById('noResults').style.display = 'block';
        document.getElementById('resultsContent').classList.add('hidden');
        return;
    }
    
    // Show loading state
    document.getElementById('calculateButton').textContent = 'Calculating...';
    document.getElementById('calculateButton').disabled = true;
    document.getElementById('noResults').innerHTML = `
        <div class="text-center py-4">
            <div class="spinner-border text-primary mb-3" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mb-0 text-muted">Calculating optimal price...</p>
        </div>
    `;
    document.getElementById('noResults').style.display = 'block';
    document.getElementById('resultsContent').classList.add('hidden');
    
    // Prepare request payload
    const payload = {
        productType: productType,
        productGroup: productGroup,
        asin: asin,
        actualPrice: actualPrice,
        competitorPrice: competitorPrice,
        rating: rating,
        numberOfOrders: numberOfOrders
    };
    
    console.log("Sending API request with payload:", payload);
    
    // Make API request
    fetch('/api/predict-price', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
    })
    .then(response => {
        console.log("Received API response status:", response.status);
        if (!response.ok) {
            throw new Error(`Server returned ${response.status}: ${response.statusText}`);
        }
        return response.json();
    })
    .then(data => {
        console.log("API response data:", data);
        // Clear error display
        document.getElementById('noResults').style.display = 'none';
        
        // Check if data contains the expected properties
        if (!data.recommendedPrice && !data.minPrice && !data.maxPrice) {
            throw new Error('Invalid response from server: missing price data');
        }
        
        // Update UI with results - safely handle potentially undefined values
        document.getElementById('recommendedPrice').textContent = data.recommendedPrice !== undefined ? 
            `$${Number(data.recommendedPrice).toFixed(2)}` : '$0.00';
            
        document.getElementById('minPrice').textContent = data.minPrice !== undefined ? 
            `$${Number(data.minPrice).toFixed(2)}` : '$0.00';
            
        document.getElementById('maxPrice').textContent = data.maxPrice !== undefined ? 
            `$${Number(data.maxPrice).toFixed(2)}` : '$0.00';
            
        document.getElementById('elasticityCategory').textContent = data.elasticityCategory || 'Unknown';
        document.getElementById('explanation').textContent = data.explanation || 'No explanation available';
        document.getElementById('ratingImpact').textContent = data.ratingImpact !== undefined ? 
            `${Number(data.ratingImpact).toFixed(1)}%` : '0%';
            
        document.getElementById('orderImpact').textContent = data.orderImpact !== undefined ? 
            `${Number(data.orderImpact).toFixed(1)}%` : '0%';
            
        document.getElementById('marketImpact').textContent = data.marketImpact !== undefined ? 
            `${Number(data.marketImpact).toFixed(1)}%` : '0%';
        
        // Show market insights if available
        if (data.marketInsights) {
            document.getElementById('marketInsightsSection').classList.remove('hidden');
            document.getElementById('priceTrend').textContent = data.marketInsights.priceTrend || 'stable';
            document.getElementById('priceVolatility').textContent = data.marketInsights.priceVolatility !== undefined ? 
                `${Number(data.marketInsights.priceVolatility).toFixed(1)}%` : '0%';
                
            document.getElementById('marketPosition').textContent = data.marketInsights.marketPosition || 'average';
            document.getElementById('sentimentScore').textContent = data.marketInsights.sentimentScore !== undefined ? 
                Number(data.marketInsights.sentimentScore).toFixed(2) : '0.00';
            
            // Add classes based on price trend
            document.getElementById('priceTrend').className = `trend-${data.marketInsights.priceTrend || 'stable'}`;
        } else {
            document.getElementById('marketInsightsSection').classList.add('hidden');
        }
        
        // Show results
        document.getElementById('resultsContent').classList.remove('hidden');
    })
    .catch(error => {
        console.error('Error:', error);
        // Display error message in the results area
        document.getElementById('noResults').innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-circle me-2"></i>
                <strong>Error calculating price:</strong> ${error.message}
                <hr>
                <p class="mb-0">Please try again or contact support if the problem persists.</p>
            </div>
        `;
        document.getElementById('noResults').style.display = 'block';
        document.getElementById('resultsContent').classList.add('hidden');
    })
    .finally(() => {
        // Reset button
        document.getElementById('calculateButton').textContent = 'Calculate Optimal Price';
        document.getElementById('calculateButton').disabled = false;
    });
}

// Initialize charts when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Update product groups based on initial product type
    updateProductGroups();
    
    // Initialize revenue chart
    const revenueChartCtx = document.getElementById('revenueChart').getContext('2d');
    new Chart(revenueChartCtx, {
        type: 'line',
        data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            datasets: [{
                label: 'Monthly Revenue',
                data: [28500, 32400, 35200, 42100, 43250, 48352],
                borderColor: '#2c7be5',
                backgroundColor: 'rgba(44, 123, 229, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toLocaleString();
                        }
                    }
                }
            }
        }
    });
    
    // Initialize sales chart
    const salesChartCtx = document.getElementById('salesChart').getContext('2d');
    new Chart(salesChartCtx, {
        type: 'bar',
        data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            datasets: [{
                label: 'Total Sales',
                data: [856, 932, 978, 1054, 1150, 1245],
                backgroundColor: '#00d97e',
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
});

// Function to format currency
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(amount);
}

// Function to refresh market data
function refreshMarketData() {
    const asin = document.getElementById('marketProductAsin').value.trim();
    
    if (!asin) {
        alert('Please enter a valid product ASIN');
        return;
    }
    
    // Show loading state
    document.getElementById('refreshMarketDataBtn').textContent = 'Refreshing...';
    document.getElementById('refreshMarketDataBtn').disabled = true;
    
    // Call the API to refresh market data
    fetch(`/api/market-data/refresh?asin=${asin}`, {
        method: 'GET'
    })
    .then(response => response.json())
    .then(data => {
        // Update status indicators
        document.getElementById('bestSellersStatus').textContent = data.bestSellersStatus || 'Not Collected';
        document.getElementById('productDetailsStatus').textContent = data.productDetailsStatus || 'Not Collected';
        document.getElementById('reviewsStatus').textContent = data.reviewsStatus || 'Not Collected';
        document.getElementById('dealsStatus').textContent = data.dealsStatus || 'Not Collected';
        
        // Show success message
        alert('Market data refreshed successfully');
    })
    .catch(error => {
        console.error('Error refreshing market data:', error);
        alert('Error refreshing market data. See console for details.');
    })
    .finally(() => {
        // Reset button
        document.getElementById('refreshMarketDataBtn').textContent = 'Refresh Market Data';
        document.getElementById('refreshMarketDataBtn').disabled = false;
    });
}

// Function to analyze product
function analyzeProduct() {
    const asin = document.getElementById('marketProductAsin').value.trim();
    
    if (!asin) {
        alert('Please enter a valid product ASIN');
        return;
    }
    
    // Show loading state
    document.getElementById('analyzeProductBtn').textContent = 'Analyzing...';
    document.getElementById('analyzeProductBtn').disabled = true;
    
    // Call the API to analyze the product
    fetch(`/api/market-data/analyze?asin=${asin}`, {
        method: 'GET'
    })
    .then(response => response.json())
    .then(data => {
        // Update price trend chart
        updatePriceTrendChart(data.priceTrend);
        
        // Update sentiment chart
        updateSentimentChart(data.sentimentData);
        
        // Update market position
        document.getElementById('pricePosition').textContent = data.pricePosition || 'Average';
        document.getElementById('pricePosition').className = `value status-${(data.pricePosition || 'average').toLowerCase().replace(' ', '-')}`;
        document.getElementById('competitiveIndex').textContent = `${data.competitiveIndex || '65'}/100`;
        document.getElementById('pricePercentile').textContent = `${data.pricePercentile || '50'}%`;
        document.getElementById('priceVolatilityValue').textContent = data.priceVolatility || 'Medium';
        
        // Update report date
        document.getElementById('reportDate').textContent = `Generated: ${new Date().toLocaleString()}`;
        
        // Update market summary content
        const summaryContent = document.getElementById('marketSummaryContent');
        if (data.summary) {
            summaryContent.innerHTML = `
                <div class="market-summary">
                    <h4>Market Overview</h4>
                    <p>${data.summary.overview || 'No overview available'}</p>
                    
                    <div class="summary-stats">
                        <div class="stat">
                            <span>Average Price</span>
                            <span class="value">${formatCurrency(data.summary.averagePrice || 0)}</span>
                        </div>
                        <div class="stat">
                            <span>Price Range</span>
                            <span class="value">${formatCurrency(data.summary.minPrice || 0)} - ${formatCurrency(data.summary.maxPrice || 0)}</span>
                        </div>
                        <div class="stat">
                            <span>Your Position</span>
                            <span class="value">${data.summary.pricePosition || 'Average'}</span>
                        </div>
                        <div class="stat">
                            <span>Market Size</span>
                            <span class="value">${data.summary.marketSize || 'Medium'}</span>
                        </div>
                    </div>
                </div>
                
                <div class="deals-summary">
                    <h4>Deals Activity</h4>
                    <p>${data.summary.dealsOverview || 'No deals data available'}</p>
                    
                    <div class="trend-stats">
                        <div class="trend-stat">
                            <span>Price Trend</span>
                            <span class="value trend-${data.summary.priceTrend || 'stable'}">${data.summary.priceTrend || 'Stable'}</span>
                        </div>
                        <div class="trend-stat">
                            <span>Deal Frequency</span>
                            <span class="value">${data.summary.dealFrequency || 'Low'}</span>
                        </div>
                        <div class="trend-stat">
                            <span>Average Discount</span>
                            <span class="value">${data.summary.averageDiscount || '0'}%</span>
                        </div>
                    </div>
                </div>
            `;
        } else {
            summaryContent.innerHTML = `<p class="no-data-message">No market summary data available.</p>`;
        }
    })
    .catch(error => {
        console.error('Error analyzing product:', error);
        alert('Error analyzing product. See console for details.');
    })
    .finally(() => {
        // Reset button
        document.getElementById('analyzeProductBtn').textContent = 'Analyze Product';
        document.getElementById('analyzeProductBtn').disabled = false;
    });
}

// Function to update price trend chart
function updatePriceTrendChart(trendData) {
    try {
        // If trendData is not provided, use dummy data
        const data = trendData || {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            values: [149.99, 159.99, 154.99, 149.99, 144.99, 139.99]
        };
        
        const chartElement = document.getElementById('priceTrendChart');
        if (!chartElement) {
            console.error("Price trend chart element not found");
            return;
        }
        
        const ctx = chartElement.getContext('2d');
        
        // Clear any existing chart
        if (window.priceTrendChart && typeof window.priceTrendChart.destroy === 'function') {
            window.priceTrendChart.destroy();
        } else {
            // If chart exists but destroy method is not available, create a fresh canvas
            console.warn("Could not properly destroy previous chart. Creating new canvas.");
            const parent = chartElement.parentNode;
            if (parent) {
                // Remove the old canvas
                parent.removeChild(chartElement);
                
                // Create a new canvas with the same ID
                const newCanvas = document.createElement('canvas');
                newCanvas.id = 'priceTrendChart';
                parent.appendChild(newCanvas);
                
                // Get the new context
                const newCtx = newCanvas.getContext('2d');
                ctx = newCtx;
            }
        }
        
        // Create new chart
        window.priceTrendChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Product Price',
                    data: data.values,
                    borderColor: '#2c7be5',
                    backgroundColor: 'rgba(44, 123, 229, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        ticks: {
                            callback: function(value) {
                                return '$' + value.toLocaleString();
                            }
                        }
                    }
                }
            }
        });
    } catch (err) {
        console.error("Error updating price trend chart:", err);
        // Create error message in the chart container
        const chartContainer = document.getElementById('priceTrendChart').parentNode;
        if (chartContainer) {
            chartContainer.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-circle me-2"></i>
                    <strong>Chart Error:</strong> Could not display price trend chart.
                </div>
                <canvas id="priceTrendChart"></canvas>
            `;
        }
    }
}

// Function to update sentiment chart
function updateSentimentChart(sentimentData) {
    // If sentimentData is not provided, use dummy data
    const data = sentimentData || {
        labels: ['Positive', 'Neutral', 'Negative'],
        values: [65, 25, 10]
    };
    
    const ctx = document.getElementById('sentimentChart').getContext('2d');
    
    // Clear any existing chart
    if (window.sentimentChart) {
        window.sentimentChart.destroy();
    }
    
    // Create new chart
    window.sentimentChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: data.labels,
            datasets: [{
                data: data.values,
                backgroundColor: [
                    '#00d97e',  // Positive - green
                    '#95aac9',  // Neutral - gray
                    '#e63757'   // Negative - red
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right'
                }
            }
        }
    });
}

// Initialize charts with dummy data on page load
document.addEventListener('DOMContentLoaded', function() {
    // Initialize the existing charts in the Overview tab
    
    // Initialize market analysis charts with dummy data
    // These will be updated with real data when the user clicks Analyze Product
    setTimeout(() => {
        if (document.getElementById('priceTrendChart')) {
            updatePriceTrendChart();
        }
        if (document.getElementById('sentimentChart')) {
            updateSentimentChart();
        }
    }, 1000);
});



