// Fix for removing duplicate price optimization records
(function() {
  console.log("Applying fix for duplicate price optimization records");
  
  // Remove the calculateButton event listener that creates the duplicate record
  // This is in setupUsageTracking() function, around line 6590
  const originalSetupUsageTracking = window.setupUsageTracking;
  
  if (typeof originalSetupUsageTracking === 'function') {
    // Override the function
    window.setupUsageTracking = function() {
      // Track tab switching
      const tabButtons = document.querySelectorAll('.tab-button');
      tabButtons.forEach(button => {
        button.addEventListener('click', function() {
          const tabName = this.textContent.trim();
          UsageTracker.trackFeature('tabSwitch', { tab: tabName });
        });
      });
      
      // REMOVED: Track price calculations - this is the duplicate one
      // The original calculation tracking will still work inside calculatePrice()
      // at lines 4468 and 4553
      
      // Track market deals analysis
      const categorySelect = document.getElementById('dealsCategorySelect');
      if (categorySelect) {
        categorySelect.addEventListener('change', function() {
          UsageTracker.trackFeature('marketDealsFilter', {
            category: this.value
          });
        });
      }
      
      // Track API fetches
      const fetchDealsBtn = document.getElementById('fetchDealsBtn');
      if (fetchDealsBtn) {
        fetchDealsBtn.addEventListener('click', function() {
          UsageTracker.trackFeature('fetchMarketDeals');
        });
      }
      
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
    
    console.log("Successfully overrode setupUsageTracking to remove duplicate price records");
    
    // Re-apply the setup if the DOM is already loaded
    if (document.readyState === 'complete' || document.readyState === 'interactive') {
      window.setupUsageTracking();
      console.log("Re-applied fixed setupUsageTracking");
    }
  } else {
    console.error("Could not find setupUsageTracking function");
  }
})(); 