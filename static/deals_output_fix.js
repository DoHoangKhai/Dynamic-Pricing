// Direct fix for market deals data storage - exact copy of pricing tool approach
(function() {
  console.log("DIRECT FIX: Applying output data fix for market deals");

  // Direct patch to override the fetchDeals function
  if (typeof window.fetchDeals === 'function') {
    console.log("Found fetchDeals function, overriding it");
    
    // Store original function
    const originalFetchDeals = window.fetchDeals;
    
    // Replace with our enhanced version
    window.fetchDeals = async function() {
      console.log("DIRECT FIX: Enhanced fetchDeals called");
      
      // Get selected category before calling original
      const categorySelect = document.getElementById('dealProductType');
      const selectedCategory = categorySelect ? categorySelect.value : 'All Categories';
      
      try {
        // Call original function
        await originalFetchDeals.apply(this, arguments);
        
        // IMPORTANT: Directly access the data that's shown on screen
        // This is the most reliable way to capture what the user sees
        const dealsCount = document.getElementById('totalDealsValue')?.textContent || '0';
        const avgDiscount = document.getElementById('avgDiscountValue')?.textContent || '0%';
        const marketActivity = document.getElementById('marketActivityValue')?.textContent || 'Low';
        
        console.log("CAPTURED DEAL DATA:", {
          dealsCount,
          avgDiscount,
          marketActivity
        });
        
        // Create data object that EXACTLY follows how other features store data
        const dealData = {
          dealsCount: dealsCount, // Keep as string like the price tool does
          avgDiscount: avgDiscount.replace('%', ''),
          marketActivity: marketActivity,
          category: selectedCategory,
          timestamp: new Date().toISOString()
        };
        
        // Save to UsageTracker in exactly the same format as the pricing tool
        if (window.UsageTracker) {
          console.log("DIRECT FIX: Saving market deals data to UsageTracker", dealData);
          window.UsageTracker.trackFeature('fetchMarketDeals', dealData);
        } else {
          console.error("DIRECT FIX: UsageTracker not found!");
        }
      } catch (error) {
        console.error("Error in enhanced fetchDeals", error);
      }
    };
    
    console.log("DIRECT FIX: Successfully overrode fetchDeals");
  } else {
    console.error("DIRECT FIX: fetchDeals function not found!");
  }
  
  // Also override the output HTML generation to match the expected format
  if (typeof window.generateOutputDetailsHTML === 'function') {
    console.log("DIRECT FIX: Overriding generateOutputDetailsHTML");
    
    const originalGenerateOutputDetailsHTML = window.generateOutputDetailsHTML;
    
    window.generateOutputDetailsHTML = function(activity) {
      // Only modify our specific feature, leave others untouched
      if (activity.featureName === 'fetchMarketDeals') {
        console.log("DIRECT FIX: Generating HTML for market deals", activity);
        
        // Extract data - this is the critical part
        const data = activity.activityData || {};
        
        // Return the HTML in the exact same format as other features
        return `
          <div class="deals-analysis-card">
            <div class="deals-header">
              <h5>Market Deals Analysis</h5>
            </div>
            <div class="deals-body">
              <div class="deals-metrics">
                <div class="metric">
                  <span class="metric-label">Total Deals:</span>
                  <span class="metric-value">${data.dealsCount || 'N/A'}</span>
                </div>
                <div class="metric">
                  <span class="metric-label">Average Discount:</span>
                  <span class="metric-value">${data.avgDiscount ? data.avgDiscount + '%' : 'N/A'}</span>
                </div>
                <div class="metric">
                  <span class="metric-label">Market Activity:</span>
                  <span class="metric-value">${data.marketActivity || 'N/A'}</span>
                </div>
              </div>
              <p class="deals-insight">Deals analysis completed successfully.</p>
            </div>
          </div>
        `;
      }
      
      // Use original function for other activities
      return originalGenerateOutputDetailsHTML.apply(this, arguments);
    };
    
    console.log("DIRECT FIX: Successfully overrode generateOutputDetailsHTML");
  } else {
    console.error("DIRECT FIX: generateOutputDetailsHTML function not found!");
  }
  
  console.log("DIRECT FIX: Market deals output data fix complete");
})(); 