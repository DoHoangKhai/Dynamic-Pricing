/**
 * This script can be run in the browser console to test Market Analysis tracking
 * It simulates different types of market analysis activities in the Recent Activities panel
 */

// Function to simulate tracking a Market Analysis activity
function simulateMarketAnalysisActivity(activityType, asin, details = {}) {
  console.log(`Simulating ${activityType} activity for ASIN ${asin}`);
  
  // Format details based on activity type
  const trackingDetails = {
    asin: asin,
    timestamp: new Date().toISOString(),
    activityType: activityType,
    productType: details.productType || 'Amazon Product',
    action: activityType
  };
  
  // Add specific details based on activity type
  if (activityType === 'Price History Analysis') {
    trackingDetails.pricePoints = details.pricePoints || 12;
  } else if (activityType === 'Competitor Analysis') {
    trackingDetails.competitorCount = details.competitorCount || 8;
  } else if (activityType === 'Reviews Analysis') {
    trackingDetails.reviewCount = details.reviewCount || 42;
  }
  
  // Track the activity using the UsageTracker
  window.UsageTracker.trackFeature('marketAnalysis', trackingDetails);
  
  // Refresh the Recent Activities display
  window.updateRecentActivities();
  
  console.log('Activity tracked and Recent Activities refreshed');
}

// Run test simulations for different types of Market Analysis activities
function runMarketAnalysisTests() {
  // First clear all usage data if requested
  if (confirm('Clear existing usage data before running tests?')) {
    window.UsageTracker.clearAllData();
    console.log('All usage data cleared');
  }
  
  // Simulate Price History Analysis
  simulateMarketAnalysisActivity('Price History Analysis', 'B08N5KWB9H', {
    pricePoints: 24,
    productType: 'Electronics'
  });
  
  // Wait 1 second before next simulation
  setTimeout(() => {
    // Simulate Competitor Analysis
    simulateMarketAnalysisActivity('Competitor Analysis', 'B08N5KWB9H', {
      competitorCount: 15,
      productType: 'Electronics'
    });
    
    // Wait 1 second before next simulation
    setTimeout(() => {
      // Simulate Reviews Analysis
      simulateMarketAnalysisActivity('Reviews Analysis', 'B08N5KWB9H', {
        reviewCount: 89,
        productType: 'Electronics'
      });
      
      console.log('All Market Analysis tests completed');
      
      // Switch to Overview tab to see the Recent Activities
      setTimeout(() => {
        window.switchTab('overview');
        console.log('Switched to Overview tab to view Recent Activities');
      }, 500);
    }, 1000);
  }, 1000);
}

// Execute the test function
console.log('Market Analysis Test Script loaded');
console.log('Call runMarketAnalysisTests() to simulate Market Analysis activities');

// Optional: Automatically run tests if enabled
// runMarketAnalysisTests(); 