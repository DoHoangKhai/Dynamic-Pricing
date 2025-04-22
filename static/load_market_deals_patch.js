// This script ensures the market deals patch is loaded after all other scripts
(function() {
  console.log("Loading market deals patch loader");
  
  function attemptToApplyPatch() {
    // Check if required globals are available
    if (window.fetchDeals && window.UsageTracker && window.generateOutputDetailsHTML) {
      // Load the patch script
      const script = document.createElement('script');
      script.src = 'market_deals_patch.js';
      script.onload = function() {
        console.log("Market deals patch successfully loaded");
      };
      script.onerror = function() {
        console.error("Failed to load market deals patch");
      };
      document.head.appendChild(script);
    } else {
      console.log("Required globals not yet available, retrying in 500ms...");
      setTimeout(attemptToApplyPatch, 500);
    }
  }
  
  // Wait for DOM to be fully loaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
      console.log("DOM loaded, attempting to apply patch");
      // Allow a brief delay for other scripts to initialize
      setTimeout(attemptToApplyPatch, 500);
    });
  } else {
    // DOM already loaded, try to apply patch now
    console.log("DOM already loaded, attempting to apply patch");
    setTimeout(attemptToApplyPatch, 500);
  }
})(); 