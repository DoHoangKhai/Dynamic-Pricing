// price_visualizations.js - Handles all pricing tool visualizations

// Create namespace for pricing visualizations
window.pricingViz = window.pricingViz || {};

// Global chart objects for reference
window.pricingViz.elasticityChart = null;
window.pricingViz.profitOptimizationChart = null;

// Initialize visualizations when document is loaded
document.addEventListener('DOMContentLoaded', function() {
  // Create empty charts initially if the elements exist
  if (document.getElementById('profitOptimizationChart')) {
    createProfitOptimizationChart();
  }
  
  // Listen for the pricing calculation result
  document.addEventListener('pricingCalculated', function(e) {
    const data = e.detail;
    updateAllVisualizations(data);
  });

  // Listen for theme toggle changes
  document.getElementById('theme-switch').addEventListener('change', function() {
    // Update charts after a small delay to ensure the theme has changed
    setTimeout(updateChartTheme, 50);
  });
});

/**
 * Update all visualizations with the pricing data
 * @param {Object} data - The pricing calculation result data
 */
function updateAllVisualizations(data) {
  // Update elasticity visualization - display only category, no chart
  updateElasticityVisualization(data.elasticityCategory, data.recommendedPrice, data.actualPrice);
  
  // Update price position visualization
  updatePricePositionVisualization(data.actualPrice, data.competitorPrice, data.recommendedPrice);
  
  // Update profit optimization chart
  updateProfitOptimizationChart(data);
  
  // Update impact factor bars
  updateImpactFactorBars(data);
  
  // Update price impact analysis
  updatePriceImpactAnalysis(data);
}

/**
 * Safely destroy a chart if it exists
 * @param {string} chartInstance - Chart instance or object containing the chart
 */
function safelyDestroyChart(chartInstance) {
  try {
    if (chartInstance && typeof chartInstance.destroy === 'function') {
      chartInstance.destroy();
    }
  } catch (e) {
    console.warn(`Error destroying chart: ${e.message}`);
  }
}

/**
 * Updates elasticity visualization with category data
 */
function updateElasticityVisualization(elasticityCategory, recommendedPrice, actualPrice) {
  console.log("Updating elasticity visualization with category:", elasticityCategory);
  
  // Initialize pricingViz namespace if it doesn't exist
  if (!window.pricingViz) {
    window.pricingViz = {
      elasticityChart: null,
      profitOptimizationChart: null
    };
  }

  // Get the elasticity category container
  const elasticitySection = document.getElementById('elasticity-section');
  
  // Show the section if previously hidden
  if (elasticitySection) {
    elasticitySection.style.display = 'block';
    
    // Update elasticity category display
    const elasticityValue = document.getElementById('elasticity-value');
    if (elasticityValue) {
      // Make sure we have a valid category, default to 'medium' if not defined
      if (!elasticityCategory) {
        elasticityCategory = 'medium';
        console.warn("No elasticity category provided, defaulting to 'medium'");
      }
      
      // Map category to display value
      const displayCategory = elasticityCategory.charAt(0).toUpperCase() + elasticityCategory.slice(1);
      elasticityValue.textContent = displayCategory;
      
      // Update class for styling - ensure default class exists if category is invalid
      const validCategories = ['high', 'medium', 'low'];
      const safeCategory = validCategories.includes(elasticityCategory) ? elasticityCategory : 'medium';
      elasticityValue.className = `elasticity-value ${safeCategory}-elasticity`;
    }
    
    // Update elasticity explanation
    updateElasticityExplanation(elasticityCategory || 'medium');
    
    // Calculate and update price change impact
    if (recommendedPrice && actualPrice) {
    const priceChangePercent = ((recommendedPrice - actualPrice) / actualPrice) * 100;
      updateElasticityImpact(elasticityCategory || 'medium', priceChangePercent);
    }
  }
}

/**
 * Updates the elasticity explanation based on the category
 * @param {string} elasticityCategory - high, medium, or low
 */
function updateElasticityExplanation(elasticityCategory) {
  const explanationElement = document.getElementById('elasticityExplanation');
  if (!explanationElement) return;
  
  let explanationText = '';
  
  switch (elasticityCategory) {
    case 'high':
      explanationText = 'Your product is <strong>elastic</strong>, meaning small price changes will have a significant impact on demand. Customers are price-sensitive.';
      break;
    case 'low':
      explanationText = 'Your product is <strong>inelastic</strong>, meaning demand is less affected by price changes. Customers value the product regardless of small price variations.';
      break;
    default: // medium
      explanationText = 'Your product has <strong>unit elasticity</strong>, meaning price changes and demand changes are proportional.';
  }
  
  explanationElement.innerHTML = explanationText;
}

/**
 * Updates the elasticity impact description
 * @param {string} elasticityCategory - high, medium, or low
 * @param {number} priceChangePercent - percentage change in price
 */
function updateElasticityImpact(elasticityCategory, priceChangePercent) {
  const impactElement = document.getElementById('elasticityImpactValue');
  if (!impactElement) return;
  
  // Get elasticity factor based on category
  let elasticityFactor;
  switch (elasticityCategory) {
    case 'high':
      elasticityFactor = -1.5; // Highly elastic (more responsive to price)
      break;
    case 'low':
      elasticityFactor = -0.5; // Inelastic (less responsive to price)
      break;
    default: // medium
      elasticityFactor = -1.0; // Unit elasticity
  }
  
  // Calculate demand impact
  const demandChangePercent = elasticityFactor * priceChangePercent;
  
  // Display the impact with sign
  const formattedImpact = demandChangePercent.toFixed(1);
  
  if (demandChangePercent > 0) {
    impactElement.textContent = `+${formattedImpact}%`;
    impactElement.className = 'impact-value positive-impact';
  } else if (demandChangePercent < 0) {
    impactElement.textContent = `${formattedImpact}%`;
    impactElement.className = 'impact-value negative-impact';
  } else {
    impactElement.textContent = `${formattedImpact}%`;
    impactElement.className = 'impact-value';
  }
}

/**
 * Creates the initial profit optimization chart
 * @returns {Chart} The created chart instance
 */
function createProfitOptimizationChart() {
  try {
    console.log("Creating profit optimization chart");
    
    // Check if the chart element exists
    const chartCanvas = document.getElementById('profitOptimizationChart');
    if (!chartCanvas) {
      console.error("Cannot find profitOptimizationChart canvas element");
      // Try to create the container if it doesn't exist
      ensureProfitOptimizationContainer();
      // Check again
      const newCanvas = document.getElementById('profitOptimizationChart');
      if (!newCanvas) {
        console.error("Still cannot find chart canvas after creating container");
        return null;
      }
    }
    
    // Destroy existing chart if it exists
    if (window.pricingViz.profitOptimizationChart) {
      console.log("Destroying existing chart instance");
      try {
        window.pricingViz.profitOptimizationChart.destroy();
      } catch (e) {
        console.warn("Error destroying existing chart:", e);
      }
      window.pricingViz.profitOptimizationChart = null;
    }

    // Get canvas context
    const ctx = document.getElementById('profitOptimizationChart').getContext('2d');
    if (!ctx) {
      console.error("Failed to get canvas context");
      return null;
    }

    // Check if Chart.js is available
    if (typeof Chart === 'undefined') {
      console.error("Chart.js is not loaded");
      // Try to dynamically load Chart.js
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js';
      document.head.appendChild(script);
      // Cannot proceed without Chart.js
      return null;
    }
    
    console.log("Chart.js is available, creating new chart");
    
    // Detect dark mode
    const isDarkMode = document.documentElement.getAttribute('data-theme') === 'dark';
    const axisColor = isDarkMode ? '#e2e8f0' : '#12263f';
    const gridColor = isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
    
    try {
      // Create chart with initial sample data
      const chart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: ['600', '700', '800', '900', '1000', '1100', '1200'],
          datasets: [
            {
              label: 'Revenue',
              data: [60000, 70000, 75000, 78000, 80000, 75000, 70000],
              borderColor: '#4c6ef5',
              backgroundColor: 'rgba(76, 110, 245, 0.1)',
              borderWidth: 3,
              fill: false,
              tension: 0.1
            },
            {
              label: 'Profit',
              data: [20000, 30000, 35000, 38000, 40000, 35000, 30000],
              borderColor: '#28a745',
              backgroundColor: 'rgba(40, 167, 69, 0.1)',
              borderWidth: 3,
              fill: false,
              tension: 0.1
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: {
            duration: 1000
          },
          plugins: {
            legend: {
              position: 'top',
              labels: {
                color: axisColor,
                font: {
                  family: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
                  size: 12
                },
                padding: 20
              }
            },
            tooltip: {
              backgroundColor: isDarkMode ? '#1e293b' : 'rgba(255, 255, 255, 0.9)',
              titleColor: axisColor,
              bodyColor: axisColor,
              borderColor: isDarkMode ? '#2c5282' : '#e2e8f0',
              borderWidth: 1,
              padding: 10,
              displayColors: true,
              callbacks: {
                label: function(context) {
                  let label = context.dataset.label || '';
                  let value = context.parsed.y;
                  return `${label}: $${value.toLocaleString()}`;
                }
              }
            }
          },
          scales: {
            x: {
              title: {
                display: true,
                text: 'Price ($)',
                color: axisColor,
                font: {
                  size: 14,
                  weight: 'bold'
                },
                padding: {
                  top: 10
                }
              },
              ticks: {
                color: axisColor,
                font: {
                  family: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
                }
              },
              grid: {
                color: gridColor,
                display: true
              }
            },
            y: {
              title: {
                display: true,
                text: 'Amount ($)',
                color: axisColor,
                font: {
                  size: 14,
                  weight: 'bold'
                }
              },
              ticks: {
                color: axisColor,
                font: {
                  family: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
                },
                callback: function(value) {
                  if (value >= 1000) {
                    return '$' + (value / 1000).toFixed(0) + 'k';
                  }
                  return '$' + value;
                }
              },
              grid: {
                color: gridColor,
                display: true
              },
              beginAtZero: true
            }
          }
        }
      });
      
      console.log("Chart created successfully, ID:", chart.id);
      window.pricingViz.profitOptimizationChart = chart;
      return chart;
    } catch (chartError) {
      console.error("Error creating chart:", chartError);
      return null;
    }
  } catch (error) {
    console.error("Failed to create profit optimization chart:", error);
    return null;
  }
}

/**
 * Ensures the profit optimization container exists in the DOM
 */
function ensureProfitOptimizationContainer() {
  // Check if container already exists
  if (document.getElementById('profitOptimizationChart')) {
    return; // Container exists, nothing to do
  }
  
  console.debug('Creating profit optimization chart container');
  
  // Find the results content element
  const resultsContent = document.querySelector('.results-content') || document.getElementById('results-content');
  if (!resultsContent) {
    console.error('Results content element not found');
    return;
  }
  
  // Find the visualization row or create it
  let visualizationRow = resultsContent.querySelector('.visualization-row');
  if (!visualizationRow) {
    visualizationRow = document.createElement('div');
    visualizationRow.className = 'visualization-row';
    resultsContent.insertBefore(visualizationRow, resultsContent.firstChild);
  }
  
  // Create the chart container
  const chartContainer = document.createElement('div');
  chartContainer.className = 'visualization-card';
  chartContainer.innerHTML = `
    <h3>Price-Profit Optimization</h3>
    <div class="profit-chart-container">
      <canvas id="profitOptimizationChart"></canvas>
    </div>
  `;
  
  // Add to visualization row
  visualizationRow.appendChild(chartContainer);
}

/**
 * Update profit optimization chart with data
 * Only create the chart if data is available
 */
function updateProfitOptimizationChart(data) {
  try {
    console.log('Updating profit optimization chart with data:', data);
    
    // Initialize namespace if needed
    if (!window.pricingViz) {
      window.pricingViz = {
        profitOptimizationChart: null,
        elasticityChart: null
      };
    }
    
    // Only proceed if we have data
    if (!data) {
      console.log('No data provided for profit optimization chart, skipping update');
      return;
    }
    
    // Get the canvas element
    const canvas = document.getElementById('profitOptimizationChart');
    if (!canvas) {
      console.error('Profit optimization chart canvas not found');
      return;
    }
    
    // Safely destroy existing chart if it exists
    if (window.pricingViz.profitOptimizationChart) {
      try {
        window.pricingViz.profitOptimizationChart.destroy();
      } catch (e) {
        console.warn('Error destroying existing profit chart:', e);
      }
      window.pricingViz.profitOptimizationChart = null;
    }

    // Extract price values from data
    const actualPrice = getPriceValue(data, 'actualPrice', 'actual_price') || 0;
    const minPrice = getPriceValue(data, 'minPrice', 'min_price') || (actualPrice * 0.8);
    const maxPrice = getPriceValue(data, 'maxPrice', 'max_price') || (actualPrice * 1.2);
    const recommendedPrice = getPriceValue(data, 'recommendedPrice', 'recommended_price') || actualPrice;
    
    console.log(`Using prices: actual=${actualPrice}, min=${minPrice}, max=${maxPrice}, recommended=${recommendedPrice}`);
    
    // Calculate price range points for the chart
    const steps = 10;
    const pricePoints = generatePriceRange(actualPrice, 0.8, 1.2, steps);
    
    // Generate profit and demand data for each price point
    const profitData = simulateProfitData(pricePoints, recommendedPrice);
    const demandData = simulateDemandData(pricePoints, recommendedPrice);
    
    // Determine chart theme colors based on current theme
    const isDarkTheme = document.documentElement.getAttribute('data-theme') === 'dark';
    const textColor = isDarkTheme ? '#e2e8f0' : '#212529';
    const gridColor = isDarkTheme ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
    const axisColor = isDarkTheme ? '#a0aec0' : '#6c757d';
    
    // Create chart configuration
    const ctx = canvas.getContext('2d');
    window.pricingViz.profitOptimizationChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: pricePoints.map(p => formatPrice(p)),
        datasets: [
          {
            label: 'Profit',
            data: profitData,
            borderColor: 'rgba(72, 187, 120, 1)',
            backgroundColor: 'rgba(72, 187, 120, 0.1)',
            yAxisID: 'y',
            fill: true,
            tension: 0.4,
            pointRadius: 3,
            pointHoverRadius: 5
          },
          {
            label: 'Demand',
            data: demandData,
            borderColor: 'rgba(237, 137, 54, 1)',
            backgroundColor: 'rgba(237, 137, 54, 0)',
            yAxisID: 'y1',
            fill: false,
            tension: 0.4,
            borderDash: [5, 5],
            pointRadius: 2,
            pointHoverRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        plugins: {
          legend: {
            position: 'top',
            labels: {
              color: textColor
            }
          },
          tooltip: {
            backgroundColor: isDarkTheme ? '#2d3748' : '#ffffff',
            titleColor: textColor,
            bodyColor: textColor,
            borderColor: isDarkTheme ? '#4a5568' : '#dee2e6',
            borderWidth: 1,
            callbacks: {
              label: function(context) {
                let label = context.dataset.label || '';
                if (label) {
                  label += ': ';
                }
                  if (context.datasetIndex === 0) {
                  label += '$' + context.raw.toFixed(2);
                  } else {
                  label += context.raw.toFixed(1) + ' units';
                }
                return label;
              }
            }
          },
          annotation: {
            annotations: {
              recommendedLine: {
                type: 'line',
                xMin: pricePoints.findIndex(p => p >= recommendedPrice),
                xMax: pricePoints.findIndex(p => p >= recommendedPrice),
                borderColor: 'rgba(66, 153, 225, 0.8)',
                borderWidth: 2,
                borderDash: [5, 5],
                label: {
                  backgroundColor: 'rgba(66, 153, 225, 0.8)',
                  content: 'Recommended',
                  enabled: true,
                  position: 'top'
                }
              }
            }
          }
        },
        scales: {
          x: {
            grid: {
              color: gridColor
            },
            ticks: {
              color: axisColor
            },
            title: {
              display: true,
              text: 'Price',
              color: axisColor
            }
            },
          y: {
            position: 'left',
            grid: {
              color: gridColor
            },
            ticks: {
              color: axisColor,
              callback: function(value) {
                return '$' + value;
            }
          },
            title: {
              display: true,
              text: 'Profit Per Unit',
              color: axisColor
            }
            },
          y1: {
            position: 'right',
            grid: {
              drawOnChartArea: false
            },
            ticks: {
              color: axisColor
            },
            title: {
              display: true,
              text: 'Estimated Demand',
              color: axisColor
            }
          }
        }
      }
    });
    
    console.log("Profit optimization chart updated successfully");
  } catch (error) {
    console.error('Error updating profit optimization chart:', error);
  }
}

/**
 * Format price for display
 */
const formatPrice = (price) => {
  return '$' + price.toFixed(2);
};

/**
 * Simulates profit data based on price points and recommended price
 * Creates a realistic profit curve that peaks at the recommended price
 * @param {Array} pricePoints - Array of price points
 * @param {number} recommendedPrice - The recommended price
 * @returns {Array} - Array of profit values
 */
function simulateProfitData(pricePoints, recommendedPrice) {
  // Find the index of the price closest to the recommended price
  const recommendedIndex = pricePoints.findIndex(price => 
    Math.abs(price - recommendedPrice) === Math.min(...pricePoints.map(p => Math.abs(p - recommendedPrice)))
  );
  
  // Create profit data that peaks at the recommended price
  return pricePoints.map((price, index) => {
    // Distance from recommended price (normalized)
    const distanceFromOptimal = Math.abs(index - recommendedIndex) / pricePoints.length;
    
    // Base profit is higher near recommended price, lower farther away
    const baseProfitFactor = 1 - (distanceFromOptimal * 1.5);
    
    // Higher prices have higher profit potential but diminish after optimal point
    const priceFactor = price / recommendedPrice;
    
    // Combine factors and scale to reasonable profit value (adjusted to price scale)
    const profitValue = (100 * baseProfitFactor * (priceFactor - 0.3)) * (recommendedPrice / 100);
    
    // Ensure profit is positive for visualization purposes
    return Math.max(0, profitValue);
  });
}

/**
 * Simulates demand data based on price points and recommended price
 * Creates a realistic demand curve that decreases as price increases
 * @param {Array} pricePoints - Array of price points
 * @param {number} recommendedPrice - The recommended price
 * @returns {Array} - Array of demand values
 */
function simulateDemandData(pricePoints, recommendedPrice) {
  // Base demand at recommended price
  const baseDemand = 100;
  
  // Elasticity factor (simulating price elasticity of demand)
  const elasticity = -1.2;
  
  return pricePoints.map(price => {
    // Calculate percentage change from recommended price
    const pricePctChange = (price - recommendedPrice) / recommendedPrice;
    
    // Apply elasticity formula: %change in demand = elasticity * %change in price
    const demandPctChange = elasticity * pricePctChange;
    
    // Calculate demand with elasticity effect and add some randomness
    const demand = baseDemand * (1 + demandPctChange) * (1 + (Math.random() * 0.1 - 0.05));
    
    // Ensure demand is positive
    return Math.max(0, demand);
  });
}

/**
 * Shows an error message in the chart container
 * @param {string} message - The error message to display
 */
function showChartError(message) {
  const container = document.getElementById("profit-optimization-container");
  if (!container) return;
  
  // Remove existing error messages
  const existingError = container.querySelector('.chart-error');
  if (existingError) {
    existingError.remove();
  }
  
  // Create error message element
  const errorElement = document.createElement('div');
  errorElement.className = 'chart-error';
  errorElement.innerHTML = `
    <div class="alert alert-danger">
      <i class="fas fa-exclamation-triangle"></i>
      <p>${message}</p>
      <button onclick="retryChart()" class="retry-btn">
        <i class="fas fa-redo"></i> Try Again
      </button>
    </div>
  `;
  
  // Add to container
  container.appendChild(errorElement);
}

/**
 * Retries creating the profit optimization chart
 */
function retryChart() {
  console.log("Retrying chart creation");
  const recommendationData = window.lastRecommendationData;
  
  if (recommendationData) {
    updateProfitOptimizationChart(recommendationData);
  } else {
    console.error("No recommendation data available for retry");
    showChartError("No data available for chart. Please recalculate price.");
  }
}

/**
 * Gets a price value from data handling different property names
 * @param {Object} data - The data object
 * @param {string} camelCaseProp - The camelCase property name
 * @param {string} snakeCaseProp - The snake_case property name
 * @returns {number} The price value
 */
function getPriceValue(data, camelCaseProp, snakeCaseProp) {
  // Try to extract the value looking for both camelCase and snake_case versions
  let value = data[camelCaseProp] || data[snakeCaseProp];
  
  // If the value is a string, convert it to a number
  if (typeof value === 'string') {
    value = parseFloat(value.replace(/[^0-9.-]+/g, ''));
  }
  
  // Validate and return the value
  return isValidNumber(value) ? value : null;
}

/**
 * Checks if a value is a valid number
 * @param {any} value - The value to check
 * @returns {boolean} Whether the value is a valid number
 */
function isValidNumber(value) {
  return value !== null && 
         value !== undefined && 
         !isNaN(value) && 
         isFinite(value) && 
         value > 0;
}

/**
 * Updates the price position visualization
 * @param {number} actualPrice - The current price
 * @param {number} competitorPrice - The competitor price
 * @param {number} recommendedPrice - The recommended price
 */
function updatePricePositionVisualization(actualPrice, competitorPrice, recommendedPrice) {
  // Clean any $ and parse as float
  actualPrice = parseFloat(typeof actualPrice === 'string' ? actualPrice.replace('$', '') : actualPrice);
  competitorPrice = parseFloat(typeof competitorPrice === 'string' ? competitorPrice.replace('$', '') : competitorPrice);
  recommendedPrice = parseFloat(typeof recommendedPrice === 'string' ? recommendedPrice.replace('$', '') : recommendedPrice);
  
  // Get the position scale element
  const positionScale = document.querySelector('.position-scale');
  if (!positionScale) return;
  
  // Calculate relative positions (0% = low end, 100% = high end)
  const priceRange = Math.max(actualPrice, competitorPrice, recommendedPrice) * 1.3 - 
                    Math.min(actualPrice, competitorPrice, recommendedPrice) * 0.7;
  
  const minPrice = Math.min(actualPrice, competitorPrice, recommendedPrice) * 0.7;
  const yourPosition = ((actualPrice - minPrice) / priceRange) * 100;
  const competitorPosition = ((competitorPrice - minPrice) / priceRange) * 100;
  const recommendedPosition = ((recommendedPrice - minPrice) / priceRange) * 100;
  
  // Clamp positions between 5% and 95% for display purposes
  const clampPosition = pos => Math.min(95, Math.max(5, pos));
  
  // Update marker positions
  const yourPriceMarker = document.getElementById('yourPriceMarker');
  const competitorPriceMarker = document.getElementById('competitorPriceMarker');
  const recommendedPriceMarker = document.getElementById('recommendedPriceMarker');
  
  if (yourPriceMarker) {
    yourPriceMarker.style.left = `${clampPosition(yourPosition)}%`;
    yourPriceMarker.textContent = `Your: $${actualPrice.toFixed(2)}`;
  }
  
  if (competitorPriceMarker) {
    competitorPriceMarker.style.left = `${clampPosition(competitorPosition)}%`;
    competitorPriceMarker.textContent = `Comp: $${competitorPrice.toFixed(2)}`;
  }
  
  if (recommendedPriceMarker) {
    recommendedPriceMarker.style.left = `${clampPosition(recommendedPosition)}%`;
    recommendedPriceMarker.textContent = `Optimal: $${recommendedPrice.toFixed(2)}`;
  }
}

/**
 * Updates impact factor bars with data from pricing model
 * @param {Object} data - The pricing model data
 */
function updateImpactFactorBars(data) {
  try {
    console.log("Updating impact factor bars with data:", data);
    
    // Extract impact values
    const ratingImpact = data.ratingImpact || 0;
    const orderImpact = data.orderImpact || 0;
    const marketImpact = data.marketImpact || 0;
    const competitorImpact = data.competitorImpact || 0; // Added competitor impact
    
    // Update each factor bar
  updateFactorBar('ratingImpact', 'ratingImpactBar', ratingImpact);
  updateFactorBar('orderImpact', 'orderImpactBar', orderImpact);
  updateFactorBar('marketImpact', 'marketImpactBar', marketImpact);
    updateFactorBar('competitorImpact', 'competitorImpactBar', competitorImpact); // Added competitor impact
    
    // Highlight the most significant factor
    highlightSignificantFactors(ratingImpact, orderImpact, marketImpact, competitorImpact);
    
  } catch (error) {
    console.error("Error updating impact factor bars:", error);
  }
}

/**
 * Updates a single impact factor bar
 * @param {string} valueId - The ID of the value element
 * @param {string} barId - The ID of the bar element
 * @param {number} value - The impact value
 */
function updateFactorBar(valueId, barId, value) {
  try {
    // Get elements
    const valueElement = document.getElementById(valueId);
    const barElement = document.getElementById(barId);
    
    if (!valueElement || !barElement) {
      console.warn(`Element not found: ${valueId} or ${barId}`);
      return;
    }
    
    // Format value with sign and percentage
    const formattedValue = (value > 0 ? '+' : '') + value.toFixed(1) + '%';
    valueElement.textContent = formattedValue;
    
    // Set bar width with maximum of 100%
    const barWidth = Math.min(Math.abs(value), 100);
    barElement.style.width = barWidth + '%';
    
    // Set color based on impact (positive or negative)
    if (value > 0) {
      barElement.classList.remove('negative-impact', 'neutral-impact');
      barElement.classList.add('positive-impact');
      
      // Add data-content attribute for tooltip
      barElement.setAttribute('data-content', `+${value.toFixed(1)}% price increase`);
      
    } else if (value < 0) {
      barElement.classList.remove('positive-impact', 'neutral-impact');
      barElement.classList.add('negative-impact');
      
      // Add data-content attribute for tooltip
      barElement.setAttribute('data-content', `${value.toFixed(1)}% price decrease`);
      
    } else {
      barElement.classList.remove('positive-impact', 'negative-impact');
      barElement.classList.add('neutral-impact');
      
      // Add data-content attribute for tooltip
      barElement.setAttribute('data-content', 'No impact on price');
    }
    
  } catch (error) {
    console.error("Error updating factor bar:", error);
  }
}

/**
 * Highlights the most significant pricing factors
 * @param {number} ratingImpact - Rating impact value
 * @param {number} orderImpact - Order impact value
 * @param {number} marketImpact - Market impact value
 * @param {number} competitorImpact - Competitor impact value
 */
function highlightSignificantFactors(ratingImpact, orderImpact, marketImpact, competitorImpact) {
  try {
    // Get the absolute values
    const impacts = [
      { name: 'Rating', value: Math.abs(ratingImpact), element: 'ratingImpact' },
      { name: 'Order Volume', value: Math.abs(orderImpact), element: 'orderImpact' },
      { name: 'Market', value: Math.abs(marketImpact), element: 'marketImpact' },
      { name: 'Competitor', value: Math.abs(competitorImpact), element: 'competitorImpact' }
    ];
    
    // Sort by impact value (highest first)
    impacts.sort((a, b) => b.value - a.value);
    
    // Highlight the top factor
    if (impacts.length > 0 && impacts[0].value > 0) {
      const topFactor = impacts[0];
      const element = document.getElementById(topFactor.element);
      
      if (element) {
        // Add a special class to highlight
        element.classList.add('highlighted-impact');
        
        // Find all factor container elements
        const allFactors = document.querySelectorAll('[id$="Impact"]');
        allFactors.forEach(el => {
          if (el.id !== topFactor.element) {
            el.classList.remove('highlighted-impact');
          }
        });
        
        // Set explanation text if the element exists
        const explanationElement = document.getElementById('pricingFactorsExplanation');
        if (explanationElement) {
          const sign = impacts[0].value > 0 ? 'positive' : 'negative';
          explanationElement.textContent = `${topFactor.name} has the most significant ${sign} impact on the recommended price.`;
        }
      }
    }
    
  } catch (error) {
    console.error("Error highlighting significant factors:", error);
  }
}

// Utility Functions

/**
 * Capitalizes the first letter of a string
 * @param {string} string - The input string
 * @return {string} The capitalized string
 */
function capitalizeFirstLetter(string) {
  return string.charAt(0).toUpperCase() + string.slice(1);
}

/**
 * Generates a price range around a base price
 * @param {number} basePrice - The base price
 * @param {number} minFactor - Minimum multiplier (e.g., 0.7 for 70% of base)
 * @param {number} maxFactor - Maximum multiplier (e.g., 1.3 for 130% of base)
 * @param {number} steps - Number of price points to generate
 * @return {Array} Array of price points
 */
function generatePriceRange(basePrice, minFactor, maxFactor, steps) {
  const priceRange = [];
  const min = basePrice * minFactor;
  const max = basePrice * maxFactor;
  const step = (max - min) / (steps - 1);
  
  for (let i = 0; i < steps; i++) {
    priceRange.push(min + step * i);
  }
  
  return priceRange;
}

/**
 * Generates demand data based on price and elasticity
 * @param {Array} priceRange - Array of price points
 * @param {string} elasticityCategory - Elasticity category (high, medium, low)
 * @return {Array} Array of demand values
 */
function generateDemandData(priceRange, elasticityCategory) {
  // Get elasticity factor based on category
  let elasticityFactor;
  switch (elasticityCategory) {
    case 'high':
      elasticityFactor = -1.5; // Highly elastic (more responsive to price)
      break;
    case 'low':
      elasticityFactor = -0.5; // Inelastic (less responsive to price)
      break;
    default: // medium
      elasticityFactor = -1.0; // Unit elasticity
  }
  
  // Find middle price to use as reference price
  const basePrice = priceRange[Math.floor(priceRange.length / 2)];
  // Set a realistic baseline demand
  const baseDemand = 1.0; // Starting at 100% for relative demand
  
  // Calculate demand using constant elasticity formula
  return priceRange.map(price => {
    // Using the constant elasticity demand formula: Q = A * P^elasticity
    // Where A is a scaling factor, P is price, elasticity is the factor
    const priceRatio = price / basePrice;
    const demand = baseDemand * Math.pow(priceRatio, elasticityFactor);
    
    // Scale to 0-1 range
    return demand;
  });
}

// Additional function to update chart colors when theme changes
function updateChartTheme() {
  const isDarkMode = document.documentElement.getAttribute('data-theme') === 'dark';
  const axisColor = isDarkMode ? '#e2e8f0' : '#12263f';
  
  // Update profit optimization chart
  if (window.pricingViz.profitOptimizationChart) {
    window.pricingViz.profitOptimizationChart.options.scales.x.title.color = axisColor;
    window.pricingViz.profitOptimizationChart.options.scales.y.title.color = axisColor;
    window.pricingViz.profitOptimizationChart.options.scales.x.ticks.color = axisColor;
    window.pricingViz.profitOptimizationChart.options.scales.y.ticks.color = axisColor;
    window.pricingViz.profitOptimizationChart.options.scales.x.grid.color = isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
    window.pricingViz.profitOptimizationChart.options.scales.y.grid.color = isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
    window.pricingViz.profitOptimizationChart.options.plugins.legend.labels.color = axisColor;
    window.pricingViz.profitOptimizationChart.update();
  }
}

/**
 * Update price impact analysis visualization
 */
function updatePriceImpactAnalysis(data) {
  try {
    console.log("Updating price impact analysis with data:", data);
    
    // Initialize namespace if needed
    if (!window.pricingViz) {
      window.pricingViz = {
        priceImpactChart: null
      };
    }
    
    // Get the canvas element
    const canvas = document.getElementById('priceImpactChart');
    if (!canvas) {
      console.error("Price impact chart element not found");
      return;
    }
    
    // Safely destroy the existing chart if it exists
    if (window.pricingViz.priceImpactChart && typeof window.pricingViz.priceImpactChart.destroy === 'function') {
      try {
        window.pricingViz.priceImpactChart.destroy();
      } catch (e) {
        console.warn("Error destroying price impact chart:", e);
      }
    }
    window.pricingViz.priceImpactChart = null;
    
    // Extract data
    const elasticity = data.price_elasticity || -1.0;
    const recommendedPrice = parseFloat(data.recommended_price || 0);
    const actualPrice = parseFloat(data.actual_price || data.actualPrice || 0);
    
    // Create price range
    const priceRange = generatePriceRange(recommendedPrice, 0.8, 1.2, 10);
    
    // Generate demand impact data
    const demandImpact = [];
    const revenueImpact = [];
    
    for (let price of priceRange) {
      const percentChange = (price - recommendedPrice) / recommendedPrice;
      const demandChange = elasticity * percentChange;
      const demand = 100 * (1 + demandChange);
      demandImpact.push(demand);
      revenueImpact.push(price * demand / 100);
    }
    
    // Format labels as price
    const labels = priceRange.map(price => '$' + price.toFixed(2));
    
    // Get chart theme colors
    const isDarkTheme = document.documentElement.getAttribute('data-theme') === 'dark';
    const textColor = isDarkTheme ? '#e2e8f0' : '#212529';
    const gridColor = isDarkTheme ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
    
    // Create chart
    const ctx = canvas.getContext('2d');
    window.pricingViz.priceImpactChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Demand',
            data: demandImpact,
            borderColor: 'rgba(237, 137, 54, 1)',
            backgroundColor: 'rgba(237, 137, 54, 0.1)',
            yAxisID: 'y',
            fill: true,
            tension: 0.4
          },
          {
            label: 'Revenue',
            data: revenueImpact,
            borderColor: 'rgba(72, 187, 120, 1)',
            backgroundColor: 'rgba(72, 187, 120, 0.1)',
            yAxisID: 'y1',
            fill: true,
            tension: 0.4
          }
        ]
      },
      options: {
        responsive: true,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        plugins: {
          legend: {
            position: 'top',
            labels: {
              color: textColor
            }
          },
          tooltip: {
            backgroundColor: isDarkTheme ? '#2d3748' : '#ffffff',
            titleColor: textColor,
            bodyColor: textColor
          },
          annotation: {
            annotations: {
              recommendedLine: {
                type: 'line',
                xMin: priceRange.findIndex(p => p >= recommendedPrice),
                xMax: priceRange.findIndex(p => p >= recommendedPrice),
                borderColor: 'rgba(66, 153, 225, 0.8)',
                borderWidth: 2,
                label: {
                  backgroundColor: 'rgba(66, 153, 225, 0.8)',
                  content: 'Recommended',
                  enabled: true,
                  position: 'top'
                }
              }
            }
          }
        },
        scales: {
          x: {
            grid: {
              color: gridColor
            },
            ticks: {
              color: textColor
            },
            title: {
              display: true,
              text: 'Price',
              color: textColor
            }
          },
          y: {
            position: 'left',
            grid: {
              color: gridColor
            },
            ticks: {
              color: textColor
            },
            title: {
              display: true,
              text: 'Demand (%)',
              color: textColor
            }
          },
          y1: {
            position: 'right',
            grid: {
              drawOnChartArea: false
            },
            ticks: {
              color: textColor
            },
            title: {
              display: true,
              text: 'Revenue',
              color: textColor
            }
          }
        }
      }
    });
  } catch (error) {
    console.error("Error updating price impact analysis:", error);
    // Try to recover by creating a placeholder
    const container = document.getElementById('price-impact-container');
    if (container) {
      container.innerHTML = '<div class="chart-error">Unable to create price impact visualization</div>';
    }
  }
}

/**
 * Creates the Review Sentiment Analysis chart
 * Shows the relationship between review rating and price perception
 */
function createReviewSentimentChart() {
    const ctx = document.getElementById('reviewSentimentChart').getContext('2d');
    
    // Initial empty chart
    const chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['5★', '4★', '3★', '2★', '1★'],
            datasets: [{
                label: 'Price Perception',
                data: [0, 0, 0, 0, 0],
                backgroundColor: [
                    'rgba(72, 187, 120, 0.7)',
                    'rgba(56, 161, 105, 0.7)',
                    'rgba(246, 173, 85, 0.7)',
                    'rgba(237, 100, 166, 0.7)',
                    'rgba(229, 62, 62, 0.7)'
                ],
                borderColor: [
                    'rgba(72, 187, 120, 1)',
                    'rgba(56, 161, 105, 1)',
                    'rgba(246, 173, 85, 1)',
                    'rgba(237, 100, 166, 1)',
                    'rgba(229, 62, 62, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Price mentions: ${context.raw}`;
                        }
                    }
                },
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Price Mentions in Reviews'
                    }
                }
            }
        }
    });
    
    // Store the chart instance for later updates
    window.reviewSentimentChart = chart;
}

/**
 * Creates the Competitor Price Comparison chart
 * Compares your product price to competitors in the same category
 */
function createCompetitorPriceChart() {
    const ctx = document.getElementById('competitorPriceChart').getContext('2d');
    
    // Initial empty chart
    const chart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Competitors',
                data: [],
                backgroundColor: 'rgba(66, 153, 225, 0.5)',
                borderColor: 'rgba(66, 153, 225, 0.8)',
                pointRadius: 5,
                pointHoverRadius: 7
            }, {
                label: 'Your Product',
                data: [],
                backgroundColor: 'rgba(237, 100, 166, 0.8)',
                borderColor: 'rgba(237, 100, 166, 1)',
                pointRadius: 8,
                pointHoverRadius: 10
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: {
                        display: true,
                        text: 'Price ($)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Rating (1-5)'
                    },
                    min: 1,
                    max: 5
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const point = context.raw;
                            return [
                                `Product: ${point.title || 'Unknown'}`,
                                `Price: $${point.x}`,
                                `Rating: ${point.y}`
                            ];
                        }
                    }
                }
            }
        }
    });
    
    // Store the chart instance for later updates
    window.competitorPriceChart = chart;
}

/**
 * Creates the Historical Price Trend chart
 * Shows how the product price has changed over time
 */
function createPriceTrendChart() {
    const ctx = document.getElementById('priceTrendChart').getContext('2d');
    
    // Initial empty chart
    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Price History',
                data: [],
                borderColor: 'rgba(72, 187, 120, 1)',
                backgroundColor: 'rgba(72, 187, 120, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }, {
                label: 'Recommended Price',
                data: [],
                borderColor: 'rgba(237, 137, 54, 1)',
                backgroundColor: 'rgba(237, 137, 54, 0.5)',
                borderWidth: 2,
                borderDash: [5, 5],
                pointRadius: 0,
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'Price ($)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Date'
                    }
                }
            }
        }
    });
    
    // Store the chart instance for later updates
    window.priceTrendChart = chart;
}

/**
 * Creates the Sales Rank vs Price Correlation chart
 * Analyzes how price affects BSR in the category
 */
function createRankPriceChart() {
    const ctx = document.getElementById('rankPriceChart').getContext('2d');
    
    // Initial empty chart
    const chart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Category Products',
                data: [],
                backgroundColor: 'rgba(102, 126, 234, 0.5)',
                borderColor: 'rgba(102, 126, 234, 0.8)',
                pointRadius: 5,
                pointHoverRadius: 7
            }, {
                label: 'Your Product',
                data: [],
                backgroundColor: 'rgba(237, 100, 166, 0.8)',
                borderColor: 'rgba(237, 100, 166, 1)',
                pointRadius: 8,
                pointHoverRadius: 10
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: {
                        display: true,
                        text: 'Price ($)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Sales Rank (lower is better)'
                    },
                    reverse: true
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const point = context.raw;
                            return [
                                `Product: ${point.title || 'Unknown'}`,
                                `Price: $${point.x}`,
                                `Rank: ${point.y}`
                            ];
                        }
                    }
                }
            }
        }
    });
    
    // Store the chart instance for later updates
    window.rankPriceChart = chart;
}

/**
 * Fetch data from Amazon API and update all charts
 */
function fetchAndUpdateCharts() {
    // Get the ASIN from the UI
    const asinInput = document.querySelector('input[placeholder="e.g. B08N5KWB9H"]');
    const asin = asinInput ? asinInput.value : '';
    
    if (!asin) {
        console.log('No ASIN provided, using demo data');
        updateChartsWithDemoData();
        return;
    }
    
    console.log(`Fetching data for ASIN: ${asin}`);
    
    // Simulate API loading state
    showLoadingState();
    
    // In a real implementation, you would make API calls here
    // For demonstration, we'll simulate the API responses with a timeout
    setTimeout(() => {
        // Fetch and update review sentiment data
        fetchReviewSentiment(asin)
            .then(data => updateReviewSentimentChart(data))
            .catch(error => {
                console.error('Error fetching review sentiment:', error);
                updateReviewSentimentChart(null);
            });
        
        // Fetch and update competitor price data
        fetchCompetitorData(asin)
            .then(data => updateCompetitorPriceChart(data))
            .catch(error => {
                console.error('Error fetching competitor data:', error);
                updateCompetitorPriceChart(null);
            });
        
        // Fetch and update price history data
        fetchPriceHistory(asin)
            .then(data => updatePriceTrendChart(data))
            .catch(error => {
                console.error('Error fetching price history:', error);
                updatePriceTrendChart(null);
            });
        
        // Fetch and update rank correlation data
        fetchRankData(asin)
            .then(data => updateRankPriceChart(data))
            .catch(error => {
                console.error('Error fetching rank data:', error);
                updateRankPriceChart(null);
            });
    }, 500);
}

/**
 * Show loading state for all charts
 */
function showLoadingState() {
    const containers = [
        'reviewSentimentChart',
        'competitorPriceChart',
        'priceTrendChart',
        'rankPriceChart'
    ];
    
    containers.forEach(id => {
        const container = document.getElementById(id);
        if (container) {
            container.style.opacity = '0.5';
        }
    });
}

/**
 * Simulate fetching review sentiment data from API
 * @param {string} asin - Product ASIN
 * @returns {Promise} - Promise resolving to sentiment data
 */
function fetchReviewSentiment(asin) {
    // Simulate API call - in production would call the Reviews API
    return new Promise((resolve) => {
        setTimeout(() => {
            // Generate sample data based on the ASIN
            const hash = hashCode(asin);
            const data = {
                fiveStar: 15 + (hash % 10),
                fourStar: 8 + (hash % 5),
                threeStar: 5 + (hash % 3),
                twoStar: 3 + (hash % 3),
                oneStar: 2 + (hash % 2)
            };
            resolve(data);
        }, 300);
    });
}

/**
 * Simulate fetching competitor pricing data
 * @param {string} asin - Product ASIN
 * @returns {Promise} - Promise resolving to competitor data
 */
function fetchCompetitorData(asin) {
    // Simulate API call - in production would call the SearchProduct and Offers APIs
    return new Promise((resolve) => {
        setTimeout(() => {
            const hash = hashCode(asin);
            const yourPrice = 100 + (hash % 50);
            const yourRating = 3.5 + (hash % 15) / 10;
            
            // Generate competitor data
            const competitors = [];
            const numCompetitors = 8 + (hash % 8);
            
            for (let i = 0; i < numCompetitors; i++) {
                competitors.push({
                    title: `Competitor ${i+1}`,
                    price: yourPrice * (0.8 + (Math.random() * 0.4)),
                    rating: Math.max(1, Math.min(5, yourRating * (0.7 + (Math.random() * 0.6))))
                });
            }
            
            resolve({
                yourProduct: {
                    title: 'Your Product',
                    price: yourPrice,
                    rating: yourRating
                },
                competitors: competitors
            });
        }, 300);
    });
}

/**
 * Simulate fetching price history data
 * @param {string} asin - Product ASIN
 * @returns {Promise} - Promise resolving to price history data
 */
function fetchPriceHistory(asin) {
    // Simulate API call - in production would call the ProductDetails API
    return new Promise((resolve) => {
        setTimeout(() => {
            const hash = hashCode(asin);
            const currentPrice = 100 + (hash % 50);
            const recommendedPrice = currentPrice * (0.9 + (hash % 20) / 100);
            
            // Generate price history for the last 12 months
            const months = [];
            const prices = [];
            const recommendedPrices = [];
            
            const today = new Date();
            for (let i = 11; i >= 0; i--) {
                const date = new Date(today.getFullYear(), today.getMonth() - i, 1);
                months.push(date.toLocaleString('default', { month: 'short' }));
                
                // Create a realistic price trend with some fluctuation
                const trendFactor = 1 + (Math.sin(i / 3) * 0.1);
                const volatilityFactor = 1 + ((Math.random() - 0.5) * 0.1);
                prices.push(currentPrice * trendFactor * volatilityFactor);
                
                // Add a flat recommended price line
                recommendedPrices.push(recommendedPrice);
            }
            
            resolve({
                months: months,
                prices: prices,
                recommendedPrice: recommendedPrices
            });
        }, 300);
    });
}

/**
 * Simulate fetching sales rank correlation data
 * @param {string} asin - Product ASIN
 * @returns {Promise} - Promise resolving to rank correlation data
 */
function fetchRankData(asin) {
    // Simulate API call - in production would call the BestSellers API
    return new Promise((resolve) => {
        setTimeout(() => {
            const hash = hashCode(asin);
            const yourPrice = 100 + (hash % 50);
            const yourRank = 1000 + (hash % 5000);
            
            // Generate category data
            const categoryProducts = [];
            const numProducts = 15 + (hash % 10);
            
            for (let i = 0; i < numProducts; i++) {
                // Create a general trend where higher price = higher rank (worse)
                // but with enough variation to be realistic
                const price = 50 + Math.random() * 150;
                const baseRank = 1000 + (price * 10);
                const rank = baseRank * (0.5 + Math.random());
                
                categoryProducts.push({
                    title: `Product ${i+1}`,
                    price: price,
                    rank: rank
                });
            }
            
            resolve({
                yourProduct: {
                    title: 'Your Product',
                    price: yourPrice,
                    rank: yourRank
                },
                categoryProducts: categoryProducts
            });
        }, 300);
    });
}

/**
 * Update the Review Sentiment chart with data
 * @param {Object} data - Review sentiment data
 */
function updateReviewSentimentChart(data) {
    const chart = window.reviewSentimentChart;
    if (!chart) return;
    
    if (!data) {
        // No data available, show demo data
        data = {
            fiveStar: 18,
            fourStar: 12,
            threeStar: 6,
            twoStar: 4,
            oneStar: 3
        };
    }
    
    chart.data.datasets[0].data = [
        data.fiveStar,
        data.fourStar,
        data.threeStar,
        data.twoStar,
        data.oneStar
    ];
    
    chart.update();
}

/**
 * Update the Competitor Price chart with data
 * @param {Object} data - Competitor price data
 */
function updateCompetitorPriceChart(data) {
    const chart = window.competitorPriceChart;
    if (!chart) return;
    
    if (!data) {
        // Demo data
        data = {
            yourProduct: {
                title: 'Your Product',
                price: 119.99,
                rating: 4.2
            },
            competitors: [
                { title: 'Competitor 1', price: 99.99, rating: 3.8 },
                { title: 'Competitor 2', price: 129.99, rating: 4.5 },
                { title: 'Competitor 3', price: 89.99, rating: 3.5 },
                { title: 'Competitor 4', price: 139.99, rating: 4.7 },
                { title: 'Competitor 5', price: 109.99, rating: 4.0 }
            ]
        };
    }
    
    // Format competitor data for chart
    chart.data.datasets[0].data = data.competitors.map(c => ({
        x: c.price,
        y: c.rating,
        title: c.title
    }));
    
    // Format your product data for chart
    chart.data.datasets[1].data = [{
        x: data.yourProduct.price,
        y: data.yourProduct.rating,
        title: data.yourProduct.title
    }];
    
    chart.update();
}

/**
 * Update the Price Trend chart with data
 * @param {Object} data - Price trend data
 */
function updatePriceTrendChart(data) {
    const chart = window.priceTrendChart;
    if (!chart) return;
    
    if (!data) {
        // Demo data
        const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
        const prices = [99.99, 109.99, 119.99, 129.99, 119.99, 109.99, 104.99, 99.99, 89.99, 99.99, 119.99, 129.99];
        const recommendedPrice = Array(12).fill(105.99);
        
        data = {
            months: months,
            prices: prices,
            recommendedPrice: recommendedPrice
        };
    }
    
    chart.data.labels = data.months;
    chart.data.datasets[0].data = data.prices;
    chart.data.datasets[1].data = data.recommendedPrice;
    
    chart.update();
}

/**
 * Update the Rank Price chart with data
 * @param {Object} data - Rank price correlation data
 */
function updateRankPriceChart(data) {
    const chart = window.rankPriceChart;
    if (!chart) return;
    
    if (!data) {
        // Demo data
        data = {
            yourProduct: {
                title: 'Your Product',
                price: 119.99,
                rank: 2500
            },
            categoryProducts: [
                { title: 'Product 1', price: 79.99, rank: 1200 },
                { title: 'Product 2', price: 99.99, rank: 1800 },
                { title: 'Product 3', price: 149.99, rank: 3200 },
                { title: 'Product 4', price: 199.99, rank: 4500 },
                { title: 'Product 5', price: 59.99, rank: 900 },
                { title: 'Product 6', price: 129.99, rank: 2800 },
                { title: 'Product 7', price: 89.99, rank: 1500 },
                { title: 'Product 8', price: 169.99, rank: 3800 }
            ]
        };
    }
    
    // Format category products data for chart
    chart.data.datasets[0].data = data.categoryProducts.map(p => ({
        x: p.price,
        y: p.rank,
        title: p.title
    }));
    
    // Format your product data for chart
    chart.data.datasets[1].data = [{
        x: data.yourProduct.price,
        y: data.yourProduct.rank,
        title: data.yourProduct.title
    }];
    
    chart.update();
}

/**
 * Update all charts with demo data
 */
function updateChartsWithDemoData() {
    updateReviewSentimentChart(null);
    updateCompetitorPriceChart(null);
    updatePriceTrendChart(null);
    updateRankPriceChart(null);
}

/**
 * Simple string hash function for generating consistent demo data
 * @param {string} str - Input string
 * @returns {number} - Hash code
 */
function hashCode(str) {
    let hash = 0;
    if (!str || str.length === 0) return hash;
    
    for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash; // Convert to 32bit integer
    }
    
    return Math.abs(hash);
} 