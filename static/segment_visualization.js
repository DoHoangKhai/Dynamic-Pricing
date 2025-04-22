/**
 * Customer Segment Visualization Module
 * 
 * This module provides visualizations for customer segment data
 * and integrates with the pricing tool.
 */

// Create a namespace for segment visualization
window.segmentViz = window.segmentViz || {};

// Customer segment colors - update with blue theme colors
window.segmentViz.segmentColors = {
  'Price Sensitive': '#3E92CC',
  'Value Seekers': '#2a628f',
  'Quality Focused': '#16324f',
  'Premium Buyers': '#13293d',
  'Bargain Hunters': '#5DADE2',
  'Trend Followers': '#1F618D'
};

// Default segment if no color defined
window.segmentViz.defaultColor = '#7C7C7C';

/**
 * Initialize segment visualization
 */
function initSegmentVisualization() {
  // Use the existing segment visualization container
  let container = document.getElementById('segmentVisualizationContainer');
  
  // We will only work with the container defined in the HTML
  // and not create a duplicate one
  if (!container) {
    console.log('Segment visualization container not found in DOM. Using the one defined in HTML.');
    container = document.querySelector('.customer-segments');
  }
  
  // Apply theme styling
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  if (container && isDark) {
    container.style.backgroundColor = '#132736';
    container.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.2)';
  } else if (container) {
    container.style.backgroundColor = '';
    container.style.boxShadow = '';
  }
  
  return container;
}

/**
 * Render segment pie chart
 * 
 * @param {Object} segmentData - Segment data from API
 */
function renderSegmentPieChart(segmentData) {
  try {
    console.log('Rendering segment pie chart with data:', segmentData);
    
    // Get the canvas element
    const canvas = document.getElementById('segmentPieChart');
    if (!canvas) {
      console.error('Segment pie chart canvas not found');
      return;
    }
    
    // Check if chart already exists
    if (window.segmentPieChart) {
      try {
        window.segmentPieChart.destroy();
        console.log('Destroyed existing segment chart');
      } catch (error) {
        console.warn('Error destroying existing chart:', error);
        // Reset the chart object if destruction fails
        window.segmentPieChart = null;
      }
    }
    
    // Prepare data
    const segments = segmentData.segments || [];
    if (segments.length === 0) {
      console.warn('No segments data found');
      return;
    }
    
    const labels = segments.map(s => s.name);
    const data = segments.map(s => s.weight);
    const colors = segments.map(s => window.segmentViz.segmentColors[s.name] || window.segmentViz.defaultColor);
    
    console.log('Preparing chart with segments:', labels);
    
    // Get theme styling
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const textColor = isDark ? '#e2e8f0' : '#12263f';
    
    // Create chart
    const ctx = canvas.getContext('2d');
    window.segmentPieChart = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: labels,
        datasets: [{
          data: data,
          backgroundColor: colors,
          hoverOffset: 4,
          borderWidth: 1,
          borderColor: isDark ? '#1a3a5f' : '#ffffff'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '60%',
        plugins: {
          legend: {
            position: 'right',
            labels: {
              color: textColor,
              boxWidth: 15,
              padding: 15,
              font: {
                size: 12
              }
            }
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                const label = context.label || '';
                const value = context.raw;
                const percentage = Math.round(value * 1000) / 10;
                return `${label}: ${percentage}%`;
              }
            },
            backgroundColor: isDark ? '#1a3a5f' : 'rgba(255, 255, 255, 0.9)',
            titleColor: textColor,
            bodyColor: textColor,
            borderColor: isDark ? '#2c5282' : '#e2e8f0',
            borderWidth: 1
          }
        }
      }
    });
    
    console.log('Segment pie chart created successfully');
  } catch (error) {
    console.error('Error creating segment chart:', error);
  }
}

/**
 * Render segment details table
 * 
 * @param {Object} segmentData - Segment data from API
 */
function renderSegmentDetails(segmentData) {
  const container = document.getElementById('segmentDetails');
  if (!container) return;
  
  // Prepare data
  const segments = segmentData.segments || [];
  
  // Clear previous content
  container.innerHTML = '';
  
  // If no segments, show message
  if (segments.length === 0) {
    container.innerHTML = '<div class="no-segment-data">No segment data available</div>';
    return;
  }
  
  // Create segment table
  const table = document.createElement('table');
  table.className = 'segment-table';
  
  // Create header row
  const thead = document.createElement('thead');
  thead.innerHTML = `
    <tr>
      <th>Segment</th>
      <th>Size</th>
      <th>Price Sensitivity</th>
      <th>Base Conv.</th>
    </tr>
  `;
  table.appendChild(thead);
  
  // Create table body
  const tbody = document.createElement('tbody');
  
  // Add rows for each segment
  segments.forEach(segment => {
    const tr = document.createElement('tr');
    
    // Determine price sensitivity display
    let sensitivityDisplay = 'Medium';
    const priceSensitivity = segment.price_sensitivity || segment.priceSensitivity || 1;
    
    if (priceSensitivity >= 2) {
      sensitivityDisplay = 'Very High';
    } else if (priceSensitivity >= 1.5) {
      sensitivityDisplay = 'High';
    } else if (priceSensitivity <= 0.5) {
      sensitivityDisplay = 'Very Low';
    } else if (priceSensitivity <= 0.8) {
      sensitivityDisplay = 'Low';
    }
    
    // Determine conversion rate
    const conversionRate = segment.conversion_rate || segment.conversionRate || segment.baseConversion || 0;
    const conversionDisplay = conversionRate <= 1 ? 
      `${(conversionRate * 100).toFixed(0)}%` : 
      `${conversionRate.toFixed(0)}%`;
    
    // Add a colored dot based on segment
    const segmentNameCell = document.createElement('td');
    segmentNameCell.className = 'segment-name';
    
    const colorDot = document.createElement('span');
    colorDot.className = 'color-dot';
    colorDot.style.backgroundColor = window.segmentViz.segmentColors[segment.name] || window.segmentViz.defaultColor;
    
    segmentNameCell.appendChild(colorDot);
    segmentNameCell.appendChild(document.createTextNode(segment.name));
    
    // Add segment data to row
    tr.innerHTML = `
      <td class="segment-name">${segmentNameCell.outerHTML}</td>
      <td class="segment-size">${segment.displayWeight || 'N/A'}</td>
      <td class="segment-sensitivity">${sensitivityDisplay}</td>
      <td class="segment-conversion">${conversionDisplay}</td>
    `;
    
    tbody.appendChild(tr);
  });
  
  table.appendChild(tbody);
  container.appendChild(table);
}

/**
 * Render segment impact chart - Now disabled
 * 
 * @param {Object} impactData - Segment impact data from API
 */
function renderSegmentImpactChart(impactData) {
  console.log('renderSegmentImpactChart has been disabled');
  return; // Function disabled to remove redundant section
}

/**
 * Get price sensitivity label
 * 
 * @param {number} sensitivity - Price sensitivity value
 * @returns {string} Sensitivity label
 */
function getPriceSensitivityLabel(sensitivity) {
  if (sensitivity >= 2.0) return 'Very High';
  if (sensitivity >= 1.5) return 'High';
  if (sensitivity >= 1.0) return 'Medium';
  if (sensitivity >= 0.5) return 'Low';
  return 'Very Low';
}

/**
 * Update segment visualization with data from API
 * 
 * @param {Object} segmentData - Segment data from API
 */
function updateSegmentVisualization(segmentData) {
  try {
    console.log('Updating segment visualization with data:', segmentData);
    
    // Validate segment data
    if (!segmentData || !segmentData.segments) {
      console.error("No segment data available");
      // Don't create anything if no data
      const segmentContainer = document.querySelector('.segment-visualization');
      if (segmentContainer) {
        segmentContainer.innerHTML = '<div class="no-data-message">No segment data available</div>';
      }
      return;
    }
    
    // Initialize segment visualization (uses existing container)
    const container = initSegmentVisualization();
    if (!container) {
      // If no container exists and none could be found, exit
      console.warn('No container for segment visualization');
      return;
    }
    
    // Get segment chart container
    const segmentChartContainer = document.getElementById('segmentChart');
    if (!segmentChartContainer) {
      console.warn('Segment chart container not found');
      return;
    }
    
    // Get segment table container
    const segmentTableBody = document.getElementById('segmentTableBody');
    if (!segmentTableBody) {
      console.warn('Segment table body not found');
      return;
    }
    
    // Process segment data
    let segments = segmentData.segments || [];
    
    // Normalize weights if they don't add up to 1
    const totalWeight = segments.reduce((sum, segment) => sum + (segment.weight || 0), 0);
    if (totalWeight > 0 && Math.abs(totalWeight - 1) > 0.01) {
      segments = segments.map(segment => ({
        ...segment,
        normalizedWeight: segment.weight / totalWeight
      }));
    } else {
      segments = segments.map(segment => ({
        ...segment,
        normalizedWeight: segment.weight
      }));
    }
    
    // Clear existing table
    segmentTableBody.innerHTML = '';
    
    // Create pie chart
    createPieChart(segments);
    
    // Create segment table rows
    segments.forEach((segment, index) => {
      const row = document.createElement('tr');
      
      // Create cells
      const segmentCell = document.createElement('td');
      segmentCell.textContent = segment.name;
      
      const sizeCell = document.createElement('td');
      sizeCell.textContent = `${(segment.normalizedWeight * 100).toFixed(1)}%`;
      
      const sensitivityCell = document.createElement('td');
      sensitivityCell.textContent = getSensitivityLabel(segment.price_sensitivity);
      sensitivityCell.className = getSensitivityClass(segment.price_sensitivity);
      
      const convRateCell = document.createElement('td');
      const convRate = segment.conversion_rate || segment.conversionRate || 0;
      convRateCell.textContent = `${(convRate * 100).toFixed(1)}%`;
      
      // Add cells to row
      row.appendChild(segmentCell);
      row.appendChild(sizeCell);
      row.appendChild(sensitivityCell);
      row.appendChild(convRateCell);
      
      // Add row to table
      segmentTableBody.appendChild(row);
    });
    
    console.log('Segment visualization updated successfully');
  } catch (error) {
    console.error('Error updating segment visualization:', error);
  }
}

/**
 * Creates a pie chart for segments
 * @param {Array} segments - Segment data
 */
function createPieChart(segments) {
  try {
    // Get canvas
    const canvas = document.getElementById('segmentChart');
    if (!canvas) return;
    
    // Destroy existing chart
    if (window.segmentChart instanceof Chart) {
      window.segmentChart.destroy();
    }
    
    // Segment colors
    const colors = [
      '#4a6cf7', // Blue
      '#ef4444', // Red
      '#10b981', // Green
      '#f59e0b', // Orange
      '#8b5cf6'  // Purple
    ];
    
    // Format data
    const labels = segments.map(segment => segment.name);
    const data = segments.map(segment => segment.normalizedWeight * 100);
    
    // Create chart
    const ctx = canvas.getContext('2d');
    window.segmentChart = new Chart(ctx, {
      type: 'pie',
      data: {
        labels: labels,
        datasets: [{
          data: data,
          backgroundColor: segments.map((_, i) => colors[i % colors.length]),
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              padding: 15,
              boxWidth: 12
            }
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                const label = context.label || '';
                const value = context.raw || 0;
                return `${label}: ${value.toFixed(1)}%`;
              }
            }
          }
        }
      }
    });
  } catch (error) {
    console.error('Error creating pie chart:', error);
  }
}

/**
 * Get sensitivity label based on value
 * @param {number} value - Sensitivity value
 * @returns {string} Sensitivity label
 */
function getSensitivityLabel(value) {
  if (!value) return 'Medium';
  if (value >= 1.5) return 'High';
  if (value <= 0.7) return 'Low';
  return 'Medium';
}

/**
 * Get CSS class for sensitivity
 * @param {number} value - Sensitivity value
 * @returns {string} CSS class
 */
function getSensitivityClass(value) {
  if (!value) return 'medium-sensitivity';
  if (value >= 1.5) return 'high-sensitivity';
  if (value <= 0.7) return 'low-sensitivity';
  return 'medium-sensitivity';
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', function() {
  initSegmentVisualization();
});

// Add CSS styles for segment visualization
document.addEventListener('DOMContentLoaded', function() {
  const styleEl = document.createElement('style');
  styleEl.id = 'segmentVisualizationStyles';
  styleEl.textContent = `
    .customer-segments {
      margin-top: 1.5rem;
      padding: 1.5rem;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
      background-color: #fff;
    }
    
    [data-theme="dark"] .customer-segments {
      background-color: #132736;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    .segment-visualization {
      display: flex;
      flex-wrap: wrap;
      gap: 1.5rem;
      margin-top: 1rem;
    }
    
    .segment-pie-container {
      flex: 1;
      min-width: 200px;
      height: 300px;
      position: relative;
    }
    
    .segment-details-container {
      flex: 1;
      min-width: 280px;
    }
    
    .no-segment-data {
      color: #6c757d;
      font-style: italic;
      text-align: center;
      padding: 2rem 0;
    }
    
    .segment-table {
      width: 100%;
      border-collapse: collapse;
    }
    
    .segment-table-header {
      display: grid;
      grid-template-columns: 2fr 1fr 1.5fr 1fr;
      gap: 0.5rem;
      padding: 0.75rem 0.5rem;
      border-bottom: 2px solid #dee2e6;
      font-weight: bold;
      font-size: 0.9rem;
    }
    
    [data-theme="dark"] .segment-table-header {
      border-bottom-color: #355675;
      color: #a8c7e5;
    }
    
    .segment-table-row {
      display: grid;
      grid-template-columns: 2fr 1fr 1.5fr 1fr;
      gap: 0.5rem;
      padding: 0.75rem 0.5rem;
      border-bottom: 1px solid #dee2e6;
      align-items: center;
    }
    
    [data-theme="dark"] .segment-table-row {
      border-bottom-color: #355675;
      color: #d1e3f7;
    }
    
    .segment-table-row:last-child {
      border-bottom: none;
    }
    
    .segment-color-dot {
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 50%;
      margin-right: 8px;
    }
    
    .segment-name {
      display: flex;
      align-items: center;
      font-weight: 500;
    }
    
    .segment-size {
      text-align: center;
    }
    
    .segment-sensitivity {
      text-align: center;
    }
    
    .segment-conversion {
      text-align: center;
    }
  `;
  document.head.appendChild(styleEl);
});

// Expose functions to global scope for external access
window.updateSegmentVisualization = updateSegmentVisualization;
window.initSegmentVisualization = initSegmentVisualization; 