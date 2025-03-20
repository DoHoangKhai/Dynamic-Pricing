// Category Revenue Chart Functionality
let categoryRevenueChart = null;

// Fetch revenue data for a category from our API
async function fetchCategoryRevenueData(category = 'Electronics') {
  try {
    const response = await fetch(`/api/category-revenue?category=${encodeURIComponent(category)}`);
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching category revenue data:', error);
    return [];
  }
}

// Initialize or update the category revenue chart
async function updateCategoryRevenueChart(category = 'Electronics') {
  const data = await fetchCategoryRevenueData(category);
  
  const ctx = document.getElementById('categoryChart').getContext('2d');
  
  // Destroy existing chart if it exists
  if (categoryRevenueChart) {
    categoryRevenueChart.destroy();
  }
  
  // Format data for chart
  const months = data.map(item => item.month);
  const revenues = data.map(item => item.revenue);

  // Create chart
  categoryRevenueChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: months,
      datasets: [{
        label: `${category.replace(/&/g, ' & ')} Revenue`,
        data: revenues,
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 2,
        tension: 0.4,
        fill: true
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: false,
          ticks: {
            callback: function(value) {
              return '$' + value.toLocaleString();
            }
          }
        }
      },
      plugins: {
        legend: {
          display: true,
          position: 'top'
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              return '$' + context.raw.toLocaleString();
            }
          }
        }
      }
    }
  });
}

// Create category selector for the revenue chart
function createCategorySelector() {
  const chartContainer = document.querySelector('.chart-container');
  if (!chartContainer) return;
  
  // Create dropdown container
  const selectorContainer = document.createElement('div');
  selectorContainer.className = 'category-selector';
  selectorContainer.style.marginBottom = '10px';
  selectorContainer.style.textAlign = 'right';
  
  // Create label
  const label = document.createElement('label');
  label.setAttribute('for', 'revenueCategorySelect');
  label.textContent = 'Category: ';
  label.style.marginRight = '5px';
  
  // Create select element
  const select = document.createElement('select');
  select.id = 'revenueCategorySelect';
  select.className = 'form-select-sm';
  select.style.width = 'auto';
  select.style.display = 'inline-block';
  
  // Add options based on available categories in productGroupMap
  const categories = ['Electronics', 'Computers&Accessories', 'MusicalInstruments', 'OfficeProducts', 
                      'Home&Kitchen', 'HomeImprovement', 'Toys&Games', 'Car&Motorbike', 'Health&PersonalCare'];
  
  categories.forEach(category => {
    const option = document.createElement('option');
    option.value = category;
    option.textContent = category.replace(/&/g, ' & ');
    select.appendChild(option);
  });
  
  // Add event listener to update chart when selection changes
  select.addEventListener('change', function() {
    updateCategoryRevenueChart(this.value);
  });
  
  // Append elements to container
  selectorContainer.appendChild(label);
  selectorContainer.appendChild(select);
  
  // Insert before the chart
  chartContainer.insertBefore(selectorContainer, chartContainer.firstChild);
}

// Initialize everything when the DOM loads
document.addEventListener('DOMContentLoaded', function() {
  const categoryChartElement = document.getElementById('categoryChart');
  if (categoryChartElement) {
    createCategorySelector();
    updateCategoryRevenueChart();
  }
}); 