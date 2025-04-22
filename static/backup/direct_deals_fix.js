// Direct modification patch for the fetchDeals function
(function() {
  console.log("Applying direct fix to fetchDeals...");
  
  // Find the fetchDeals function in the original script
  const originalScript = document.querySelector('script[src*="script.js"]');
  if (!originalScript) {
    console.error("Could not find main script!");
    return;
  }
  
  // Create an injection point right after the original script
  const injectionScript = document.createElement('script');
  injectionScript.textContent = `
    // Direct patch for fetchDeals
    (function() {
      if (!window.originalFetchDeals && window.fetchDeals) {
        console.log("Applying direct fix to fetchDeals function");
        
        // Save original function
        window.originalFetchDeals = window.fetchDeals;
        
        // Replace with enhanced version
        window.fetchDeals = async function() {
          console.log('Direct enhanced fetchDeals called');
          
          try {
            // Get selected category
            const categorySelect = document.getElementById('dealProductType');
            const selectedCategory = categorySelect ? categorySelect.value : 'All Categories';
            
            // Call original function but intercept it to capture data
            const originalResult = await window.originalFetchDeals.apply(this, arguments);
            
            // Give time for DOM updates
            await new Promise(resolve => setTimeout(resolve, 200));
            
            // Get data from UI
            const dealsCount = document.getElementById('totalDealsValue')?.textContent || '0';
            const avgDiscount = document.getElementById('avgDiscountValue')?.textContent || '0%';
            const marketActivity = document.getElementById('marketActivityValue')?.textContent || 'Low';
            
            // Process values
            const dealsCountNum = parseInt(dealsCount) || 0;
            const avgDiscountValue = avgDiscount.replace('%', '') || '0';
            
            // Build insight message
            let dealInsight = '';
            if (dealsCountNum === 0) {
              dealInsight = 'No active deals found. Current pricing can be maintained.';
            } else if (parseFloat(avgDiscountValue) > 25) {
              dealInsight = 'High discount activity detected. Consider competitive pricing to maintain market share.';
            } else if (parseFloat(avgDiscountValue) > 15) {
              dealInsight = 'Moderate discount activity. Consider slight price adjustments to remain competitive.';
            } else {
              dealInsight = 'Low discount activity. Current pricing strategy appears sustainable.';
            }
            
            // Create tracking data object
            const dealData = {
              category: selectedCategory,
              dealsCount: dealsCountNum,
              avgDiscount: avgDiscountValue,
              marketActivity: marketActivity,
              dealInsight: dealInsight,
              timestamp: Date.now()
            };
            
            console.log('Saving deal analysis data:', dealData);
            
            // Track this activity in UsageTracker
            if (window.UsageTracker) {
              window.UsageTracker.trackFeature('fetchMarketDeals', dealData);
              
              // Force update Recent Activities
              if (window.updateRecentActivities) {
                setTimeout(() => window.updateRecentActivities(), 300);
              }
            } else {
              console.error("UsageTracker not available for deal tracking!");
            }
            
            return originalResult;
          } catch (error) {
            console.error('Error in direct enhanced fetchDeals:', error);
            return window.originalFetchDeals.apply(this, arguments);
          }
        };
        
        console.log("Direct fix applied to fetchDeals");
      } else {
        console.log("Direct fix already applied or fetchDeals not available");
      }
    })();
  `;
  
  // Insert after original script
  originalScript.parentNode.insertBefore(injectionScript, originalScript.nextSibling);
  console.log("Direct fix injection added");
})(); 