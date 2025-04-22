/**
 * Demand Forecast Visualization Module
 * 
 * This module provides visualizations for demand forecasting data
 * and integrates with the pricing tool.
 */

// Colors for demand forecast themes
const forecastColors = {
  light: {
    forecast: '#3E92CC',
    feature: '#2a628f',
    time_series: '#16324f',
    actual: '#1F618D',
    trend: '#5DADE2',
    seasonality: '#13293d',
    grid: '#e1e8ed',
    text: '#333333'
  },
  dark: {
    forecast: '#4EA8E8',
    feature: '#2d7db9',
    time_series: '#206290',
    actual: '#3498DB',
    trend: '#6FB7E9',
    seasonality: '#2980B9',
    grid: '#1d3c54',
    text: '#d1e3f7'
  }
};

// Get current theme colors
function getThemeColors() {
  const isDarkMode = document.body.getAttribute('data-theme') === 'dark';
  return isDarkMode ? forecastColors.dark : forecastColors.light;
}

/**
 * Initialize demand forecast visualization
 */
function initDemandForecastVisualization() {
  // Add forecast container to results
  if (!document.getElementById('demandForecastContainer')) {
    const container = document.createElement('div');
    container.id = 'demandForecastContainer';
    container.className = 'visualization-card forecast-viz hidden';
    container.innerHTML = `
      <h3>Demand Forecast</h3>
      <div class="forecast-visualization">
        <div class="forecast-chart-container">
          <canvas id="demandForecastChart"></canvas>
        </div>
        <div class="forecast-stats-container" id="forecastStats">
          <div class="no-forecast-data">No forecast data available</div>
        </div>
      </div>
      <div class="forecast-info">
        <div class="forecast-price-impact">
          <h4>Price Impact on Demand</h4>
          <div class="price-impact-container">
            <div class="price-impact-value" id="priceImpactValue">-</div>
            <div class="price-impact-description" id="priceImpactDescription">-</div>
          </div>
        </div>
        <div class="forecast-models-used">
          <h4>Forecast Models</h4>
          <div id="forecastModelsUsed" class="models-list">-</div>
        </div>
      </div>
    `;
    
    // Find where to insert - after segment visualization
    const segmentViz = document.getElementById('segmentVisualizationContainer');
    if (segmentViz) {
      segmentViz.parentNode.insertBefore(container, segmentViz.nextSibling);
    } else {
      // Fallback - append to results content
      document.getElementById('resultsContent').appendChild(container);
    }
  }
}

/**
 * Render demand forecast chart
 * 
 * @param {Object} forecastData - Forecast data from API
 */
function renderDemandForecastChart(forecastData) {
  const canvas = document.getElementById('demandForecastChart');
  if (!canvas) return;
  
  // Check if chart already exists
  if (window.demandForecastChart) {
    try {
      window.demandForecastChart.destroy();
    } catch (error) {
      console.warn('Error destroying existing forecast chart:', error);
      // Reset the chart object if destruction fails
      window.demandForecastChart = null;
    }
  }
  
  // Prepare data
  const dates = forecastData.dates || [];
  const forecast = forecastData.forecast || [];
  
  // Get theme colors
  const colors = getThemeColors();
  
  // Create datasets
  const datasets = [
    {
      label: 'Forecast',
      data: forecast,
      borderColor: colors.forecast,
      backgroundColor: `${colors.forecast}33`, // 20% opacity
      borderWidth: 2,
      fill: true,
      tension: 0.3
    }
  ];
  
  // Add feature-based forecast if available
  if (forecastData.feature_forecast) {
    datasets.push({
      label: 'Feature Model',
      data: forecastData.feature_forecast,
      borderColor: colors.feature,
      borderWidth: 1,
      borderDash: [5, 5],
      fill: false,
      pointRadius: 0,
      tension: 0.3
    });
  }
  
  // Add time series forecast if available
  if (forecastData.time_series_forecast) {
    datasets.push({
      label: 'Time Series Model',
      data: forecastData.time_series_forecast,
      borderColor: colors.time_series,
      borderWidth: 1,
      borderDash: [2, 2],
      fill: false,
      pointRadius: 0,
      tension: 0.3
    });
  }
  
  // Create chart with theme-appropriate options
  try {
    window.demandForecastChart = new Chart(canvas, {
      type: 'line',
      data: {
        labels: dates,
        datasets: datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          tooltip: {
            mode: 'index',
            intersect: false,
            backgroundColor: document.body.getAttribute('data-theme') === 'dark' ? '#132736' : 'rgba(255, 255, 255, 0.8)',
            titleColor: colors.text,
            bodyColor: colors.text,
            borderColor: colors.grid,
            borderWidth: 1
          },
          legend: {
            position: 'top',
            labels: {
              color: colors.text,
              font: {
                family: "'Source Sans Pro', Arial, sans-serif"
              }
            }
          }
        },
        scales: {
          x: {
            display: true,
            title: {
              display: true,
              text: 'Date',
              color: colors.text
            },
            ticks: {
              maxRotation: 45,
              minRotation: 45,
              color: colors.text
            },
            grid: {
              color: colors.grid
            }
          },
          y: {
            display: true,
            title: {
              display: true,
              text: 'Demand',
              color: colors.text
            },
            beginAtZero: true,
            ticks: {
              color: colors.text
            },
            grid: {
              color: colors.grid
            }
          }
        }
      }
    });
  } catch (error) {
    console.error('Error creating demand forecast chart:', error);
  }
}

/**
 * Render forecast statistics
 * 
 * @param {Object} forecastData - Forecast data from API
 */
function renderForecastStats(forecastData) {
  const container = document.getElementById('forecastStats');
  if (!container) return;
  
  // Ensure statistics are available
  const stats = forecastData.statistics || null;
  
  if (!stats) {
    container.innerHTML = '<div class="no-forecast-data">No forecast statistics available</div>';
    return;
  }
  
  // Create stats table
  let html = '<div class="forecast-stats">';
  
  // Add statistics
  html += `
    <div class="stat-item">
      <div class="stat-label">Average Daily Demand</div>
      <div class="stat-value">${Math.round(stats.mean * 10) / 10}</div>
    </div>
    <div class="stat-item">
      <div class="stat-label">Peak Demand</div>
      <div class="stat-value">${Math.round(stats.max * 10) / 10}</div>
    </div>
    <div class="stat-item">
      <div class="stat-label">Total Forecast Period</div>
      <div class="stat-value">${Math.round(stats.total * 10) / 10}</div>
    </div>
  `;
  
  html += '</div>';
  container.innerHTML = html;
}

/**
 * Render price impact information
 * 
 * @param {Object} priceImpact - Price impact data from API
 */
function renderPriceImpact(priceImpact) {
  const valueElement = document.getElementById('priceImpactValue');
  const descriptionElement = document.getElementById('priceImpactDescription');
  
  if (!valueElement || !descriptionElement) return;
  
  if (!priceImpact) {
    valueElement.textContent = '-';
    descriptionElement.textContent = 'No price impact data available';
    return;
  }
  
  // Calculate display values
  const changePercent = Math.round(priceImpact.demand_change_pct * 10) / 10;
  const isPositive = changePercent > 0;
  const priceChangePercent = Math.round((priceImpact.price_ratio - 1) * 1000) / 10;
  
  // Set impact value with appropriate class
  valueElement.textContent = `${isPositive ? '+' : ''}${changePercent}%`;
  valueElement.className = `price-impact-value ${isPositive ? 'positive' : 'negative'}`;
  
  // Set description
  if (Math.abs(changePercent) < 1) {
    descriptionElement.textContent = `Minimal impact on demand with a ${Math.abs(priceChangePercent)}% ${priceChangePercent > 0 ? 'increase' : 'decrease'} in price.`;
  } else {
    descriptionElement.textContent = `A ${Math.abs(priceChangePercent)}% ${priceChangePercent > 0 ? 'increase' : 'decrease'} in price is estimated to ${isPositive ? 'increase' : 'decrease'} demand by ${Math.abs(changePercent)}%.`;
  }
}

/**
 * Render models information
 * 
 * @param {Array} modelsUsed - List of models used for forecasting
 */
function renderModelsUsed(modelsUsed) {
  const container = document.getElementById('forecastModelsUsed');
  if (!container) return;
  
  if (!modelsUsed || modelsUsed.length === 0) {
    container.textContent = 'No models information available';
    return;
  }
  
  // Format model names
  const modelNames = modelsUsed.map(model => {
    if (model === 'feature_based') return 'ML-based model';
    if (model === 'time_series') return 'Time series model';
    if (model === 'combined') return 'Ensemble model';
    return model;
  });
  
  // Create list
  container.textContent = modelNames.join(', ');
}

/**
 * Update demand forecast visualization with new data
 * 
 * @param {Object} forecastData - Forecast data from API
 */
function updateDemandForecastVisualization(forecastData) {
  try {
    // Show visualization container
    let container = document.getElementById('demandForecastContainer');
    if (!container) {
      // Container doesn't exist, initialize it
      initDemandForecastVisualization();
      container = document.getElementById('demandForecastContainer');
      if (!container) {
        console.error('Failed to initialize demand forecast container');
        return; // Still doesn't exist, exit
      }
    }
    
    if (!forecastData || Object.keys(forecastData).length === 0) {
      container.classList.add('hidden');
      return;
    }
    
    container.classList.remove('hidden');
    
    // Check if Chart.js is available
    if (typeof Chart === 'undefined') {
      console.error('Chart.js library not loaded');
      return;
    }
    
    // Render visualizations
    renderDemandForecastChart(forecastData);
    renderForecastStats(forecastData);
    
    // Render price impact if available
    if (forecastData.price_impact) {
      renderPriceImpact(forecastData.price_impact);
    }
    
    // Render models used if available
    if (forecastData.models_used) {
      renderModelsUsed(forecastData.models_used);
    }
  } catch (error) {
    console.error('Error updating demand forecast visualization:', error);
  }
}

// Update chart when theme changes
function updateChartTheme() {
  if (window.demandForecastChart && window.lastForecastData) {
    renderDemandForecastChart(window.lastForecastData);
  }
}

// Save last forecast data for theme updates
window.updateDemandForecastVisualization = function(forecastData) {
  window.lastForecastData = forecastData;
  updateDemandForecastVisualization(forecastData);
};

// Listen for theme changes
document.addEventListener('themeChanged', function() {
  updateChartTheme();
});

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', function() {
  initDemandForecastVisualization();
}); 