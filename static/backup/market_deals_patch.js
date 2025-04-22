// SIMPLE DIRECT FIX - NO BULLSHIT
(function() {
  // ======================================================================
  // PART 1: FIX MARKET DEALS DATA STORAGE
  // ======================================================================
  // Find the original function
  const originalFetchDeals = window.fetchDeals;
  
  if (!originalFetchDeals) {
    console.error("fetchDeals not found - can't apply fix");
    return;
  }
  
  // Replace with fixed version
  window.fetchDeals = async function() {
    // Call the original to do its work
    await originalFetchDeals.apply(this, arguments);
    
    // Wait just enough time for DOM to update
    await new Promise(r => setTimeout(r, 100));
    
    // Get data from UI 
    const dealsCount = document.getElementById('totalDealsValue')?.textContent;
    const avgDiscount = document.getElementById('avgDiscountValue')?.textContent;
    const marketActivity = document.getElementById('marketActivityValue')?.textContent;
    
    // MANUALLY add data to localStorage - the DIRECT way
    try {
      // Get current data
      const storageData = JSON.parse(localStorage.getItem('userUsageData') || '{}');
      
      // Make sure features object exists
      if (!storageData.features) storageData.features = {};
      
      // Create fetchMarketDeals entry if it doesn't exist
      if (!storageData.features.fetchMarketDeals) {
        storageData.features.fetchMarketDeals = {
          count: 0,
          lastUsed: null,
          details: []
        };
      }
      
      // Update usage stats
      storageData.features.fetchMarketDeals.count++;
      storageData.features.fetchMarketDeals.lastUsed = Date.now();
      
      // Add detailed entry
      storageData.features.fetchMarketDeals.details.push({
        timestamp: Date.now(),
        dealsCount: dealsCount || '0',
        avgDiscount: (avgDiscount || '0%').replace('%', ''),
        marketActivity: marketActivity || 'Low',
        category: document.getElementById('dealProductType')?.value || 'All Categories'
      });
      
      // Keep only last 10 detailed entries
      if (storageData.features.fetchMarketDeals.details.length > 10) {
        storageData.features.fetchMarketDeals.details.shift();
      }
      
      // Save back to localStorage
      localStorage.setItem('userUsageData', JSON.stringify(storageData));
      
      console.log("✅ Deal data saved to localStorage:", storageData.features.fetchMarketDeals.details[storageData.features.fetchMarketDeals.details.length-1]);
    } catch (e) {
      console.error("Error saving deal data:", e);
    }
  };
  
  // ======================================================================
  // PART 2: FIX FOR DUPLICATE PRICE OPTIMIZATION RECORDS
  // ======================================================================
  
  // Find the original calculatePrice function
  const originalCalculatePrice = window.calculatePrice;
  
  if (originalCalculatePrice) {
    // Override with our fixed version that removes the first tracking call
    window.calculatePrice = function() {
      try {
        console.log("Using fixed calculatePrice function (removed duplicate tracking)");
        
        // Original setup code
        document.getElementById('resultsLoading').style.display = 'block';
        document.getElementById('elasticity-section').style.display = 'none';
        document.getElementById('resultsContent').classList.add('hidden');
        document.getElementById('apiErrorMessage').classList.add('hidden');
        
        // Get form values
        const productType = document.getElementById('productType').value;
        const productGroup = document.getElementById('productGroup').value;
        const actualPrice = parseFloat(document.getElementById('actualPrice').value);
        const competitorPrice = parseFloat(document.getElementById('competitorPrice').value);
        const rating = parseFloat(document.getElementById('rating').value);
        const numberOfOrders = parseFloat(document.getElementById('numberOfOrders').value);
        
        // Validate inputs
        if (isNaN(actualPrice) || isNaN(competitorPrice) || isNaN(rating) || isNaN(numberOfOrders)) {
          showError("Please enter valid numeric values for all fields.");
          return;
        }
        
        // Create request payload
        const requestData = {
          productType: productType,
          productGroup: productGroup,
          actualPrice: actualPrice,
          competitorPrice: competitorPrice,
          starRating: rating,
          ordersPerMonth: numberOfOrders,
          productCost: actualPrice * 0.6
        };
        
        console.log("Submitting price calculation with values:", requestData);
        
        // REMOVED: First tracking call that causes the duplicate
        // Original code was:
        // UsageTracker.trackFeature('priceCalculation', {
        //   productType, productGroup, actualPrice, competitorPrice, rating, numberOfOrders
        // });
        
        // Continue with API request
        fetch('/calculate-price', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(requestData)
        })
        .then(response => {
          if (!response.ok) {
            throw new Error('API request failed');
          }
          return response.json();
        })
        .then(data => {
          console.log("Price calculation response:", data);
          
          // Process data and continue with the original function
          window.lastPriceCalculationData = data;
          document.getElementById('resultsLoading').style.display = 'none';
          document.getElementById('resultsContent').classList.remove('hidden');
          updatePriceElements(data);
          
          // Handle elasticity category
          if (!data.elasticityCategory && data.price_elasticity !== undefined) {
            const absElasticity = Math.abs(data.price_elasticity);
            let elasticityCategory = 'medium';
            if (absElasticity < 0.5) {
              elasticityCategory = 'low';
            } else if (absElasticity >= 1.0) {
              elasticityCategory = 'high';
            }
            data.elasticityCategory = elasticityCategory;
          } else if (!data.elasticityCategory) {
            data.elasticityCategory = 'medium';
          }
          
          updateElasticityVisualization(data.elasticityCategory, data.recommendedPrice, actualPrice);
          updateVisualizationsWithApiData(data);
          
          // Only keep this one tracking call with the full results data
          const calculationResults = {
            productType, productGroup, actualPrice, competitorPrice, rating, numberOfOrders,
            asin: document.getElementById('asin')?.value || 'N/A',
            recommendedPrice: data.recommended_price || data.recommendedPrice,
            minPrice: data.min_price || (data.recommended_price ? data.recommended_price * 0.95 : null),
            maxPrice: data.max_price || (data.recommended_price ? data.recommended_price * 1.05 : null),
            priceElasticity: data.price_elasticity || -1.2,
            volumeImpact: data.volume_impact || data.segment_impact?.total_impact || 0,
            revenueImpact: data.revenue_impact || data.margin_change_pct || 0,
            segmentImpact: data.segment_impact || null,
            pricingFactorsPct: data.price_factors_pct || null,
            timestamp: new Date().toISOString()
          };
          
          // Only this ONE tracking call remains
          UsageTracker.trackFeature('priceCalculation', calculationResults);
          
          // Complete the UI updates
          document.getElementById('elasticity-section').style.display = 'block';
          const resultsContent = document.getElementById('resultsContent');
          if (resultsContent.classList.contains('hidden')) {
            resultsContent.classList.remove('hidden');
          }
        })
        .catch(error => {
          console.error("Error calculating price:", error);
          document.getElementById('resultsLoading').style.display = 'none';
          document.getElementById('apiErrorMessage').classList.remove('hidden');
          document.getElementById('apiErrorMessage').querySelector('span').textContent = 
            'Error: ' + (error.message || 'Failed to calculate optimal price');
        });
      } catch (error) {
        console.error("Error in calculatePrice function:", error);
        showError("An unexpected error occurred. Please try again.");
      }
    };
    
    console.log("✅ Fixed calculate price function to remove duplicate records");
  }
  
  // ======================================================================
  // PART 3: FIX DUPLICATE MARKET DEALS ENTRIES
  // ======================================================================
  
  // Fix the updateRecentActivities function to prevent generic "Market deals analyzed" message
  if (window.updateRecentActivities) {
    const originalUpdateRecentActivities = window.updateRecentActivities;
    
    window.updateRecentActivities = function() {
      // Call original function to ensure other functionality works
      originalUpdateRecentActivities.apply(this, arguments);
      
      // Now modify the DOM to fix any "Market deals analyzed" entries
      const activitiesList = document.getElementById('recentActivitiesList');
      if (!activitiesList) return;
      
      // Find and remove activity items with generic "Market deals analyzed" text
      const activityItems = activitiesList.querySelectorAll('.usage-history-item');
      activityItems.forEach(item => {
        const detailsElement = item.querySelector('.usage-history-info');
        if (detailsElement && detailsElement.textContent === 'Market deals analyzed') {
          console.log("✅ Removing generic market deals entry");
          item.remove();
        }
      });
    };
    
    console.log("✅ Fixed updateRecentActivities to remove generic market deals entries");
  }
  
  // Also fix setupUsageTracking to remove the fetchDealsBtn event listener
  if (typeof window.setupUsageTracking === 'function') {
    // Store original setup function
    const originalSetupUsageTracking = window.setupUsageTracking;
    
    // Override it to remove both price and market deals duplicate records
    window.setupUsageTracking = function() {
      // Track tab switching (keep this)
      const tabButtons = document.querySelectorAll('.tab-button');
      tabButtons.forEach(button => {
        button.addEventListener('click', function() {
          const tabName = this.textContent.trim();
          UsageTracker.trackFeature('tabSwitch', { tab: tabName });
        });
      });
      
      // REMOVED: Price calculation event listener (already fixed)
      
      // Track market deals analysis
      const categorySelect = document.getElementById('dealsCategorySelect');
      if (categorySelect) {
        categorySelect.addEventListener('change', function() {
          UsageTracker.trackFeature('marketDealsFilter', {
            category: this.value
          });
        });
      }
      
      // REMOVED: fetchDealsBtn event listener that causes duplicate market deals entries
      // Original code was:
      // const fetchDealsBtn = document.getElementById('fetchDealsBtn');
      // if (fetchDealsBtn) {
      //   fetchDealsBtn.addEventListener('click', function() {
      //     UsageTracker.trackFeature('fetchMarketDeals');
      //   });
      // }
      
      // Track theme changes
      const themeSwitch = document.getElementById('theme-switch');
      if (themeSwitch) {
        themeSwitch.addEventListener('change', function() {
          const newTheme = this.checked ? 'light' : 'dark';
          UsageTracker.trackFeature('themeChange', { theme: newTheme });
        });
      }
      
      // Track sliders usage
      const priceSlider = document.getElementById('actualPriceSlider');
      if (priceSlider) {
        priceSlider.addEventListener('change', function() {
          UsageTracker.trackFeature('priceSliderAdjustment', { value: this.value });
        });
      }
      
      // Track competitor price slider usage
      const competitorSlider = document.getElementById('competitorPriceSlider');
      if (competitorSlider) {
        competitorSlider.addEventListener('change', function() {
          UsageTracker.trackFeature('competitorSliderAdjustment', { value: this.value });
        });
      }
      
      // Track ASIN entry
      const asinInput = document.getElementById('asin');
      if (asinInput) {
        asinInput.addEventListener('change', function() {
          if (this.value && this.value.length > 0) {
            UsageTracker.trackFeature('asinLookup', { asin: this.value });
            UsageTracker.trackProductSearch(this.value, document.getElementById('productType')?.value);
          }
        });
      }
      
      // Track chart interactions
      document.querySelectorAll('.chart-container').forEach(container => {
        container.addEventListener('click', function() {
          const containerId = this.id || 'unknown-chart';
          UsageTracker.trackFeature('chartInteraction', { chart: containerId });
        });
      });
    };
    
    console.log("✅ Fixed duplicate price and market deals event listeners");
    
    // Re-apply the setup if the DOM is already loaded
    if (document.readyState === 'complete' || document.readyState === 'interactive') {
      window.setupUsageTracking();
    }
  }
  
  console.log("✅ All fixes applied successfully");
})(); 