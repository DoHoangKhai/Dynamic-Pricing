/**
 * Market Analysis Activity Tracker
 * This script enhances the Recent Activities panel to properly display Market Analysis activities
 */

document.addEventListener('DOMContentLoaded', function() {
  console.log('Market Analysis Tracker initialized');
  
  // Create a few sample Market Analysis activities if none exist
  setTimeout(() => {
    createSampleMarketAnalysisActivities();
  }, 2000);
  
  // Override the updateRecentActivities function to handle Market Analysis activities better
  if (window.updateRecentActivities) {
    console.log('Patching updateRecentActivities function');
    
    // Store the original function
    const originalUpdateRecentActivities = window.updateRecentActivities;
    
    // Replace with enhanced version
    window.updateRecentActivities = function() {
      // Call the original function first
      originalUpdateRecentActivities.apply(this, arguments);
      
      // Then enhance Market Analysis activities display
      enhanceMarketAnalysisActivities();
    };
  }
});

/**
 * Create sample Market Analysis activities in the usage tracker
 */
function createSampleMarketAnalysisActivities() {
  if (!window.UsageTracker) return;
  
  // Check if there are already market analysis activities
  const usageData = window.UsageTracker.getData();
  const hasMarketAnalysis = usageData.features && 
                          usageData.features.marketAnalysis && 
                          usageData.features.marketAnalysis.details && 
                          usageData.features.marketAnalysis.details.length > 0;
  
  // Only create sample activities if none exist
  if (!hasMarketAnalysis) {
    console.log('Creating sample Market Analysis activities');
    
    // Populate with sample data
    const asin = 'B08N5KWB9H';
    
    // Sample Price History Analysis
    window.UsageTracker.trackFeature('marketAnalysis', {
      asin: asin,
      timestamp: new Date(Date.now() - 30 * 60000).toISOString(), // 30 minutes ago
      activityType: 'Price History Analysis',
      productType: 'Electronics',
      pricePoints: 24
    });
    
    // Sample Competitor Analysis
    window.UsageTracker.trackFeature('marketAnalysis', {
      asin: asin,
      timestamp: new Date(Date.now() - 20 * 60000).toISOString(), // 20 minutes ago
      activityType: 'Competitor Analysis',
      productType: 'Electronics',
      competitorCount: 15
    });
    
    // Sample Reviews Analysis
    window.UsageTracker.trackFeature('marketAnalysis', {
      asin: asin,
      timestamp: new Date(Date.now() - 10 * 60000).toISOString(), // 10 minutes ago
      activityType: 'Reviews Analysis',
      productType: 'Electronics',
      reviewCount: 89
    });
    
    console.log('Sample Market Analysis activities created');
    
    // Refresh the display
    if (window.updateRecentActivities) {
      window.updateRecentActivities();
    }
  }
}

/**
 * Enhance the display of Market Analysis activities in the Recent Activities panel
 */
function enhanceMarketAnalysisActivities() {
  const activitiesList = document.getElementById('recentActivitiesList');
  if (!activitiesList) return;
  
  // Find all Market Analysis activities in the list
  const activityItems = activitiesList.querySelectorAll('.usage-history-item');
  
  activityItems.forEach(item => {
    const titleElement = item.querySelector('.usage-history-title');
    if (!titleElement) return;
    
    const title = titleElement.textContent;
    
    // Apply specific styling based on activity type
    if (title === 'Price History Analysis') {
      titleElement.innerHTML = '<i class="fas fa-chart-line"></i> ' + title;
      titleElement.style.color = '#4299e1'; // Blue
    } else if (title === 'Competitor Analysis') {
      titleElement.innerHTML = '<i class="fas fa-store"></i> ' + title;
      titleElement.style.color = '#48bb78'; // Green
    } else if (title === 'Reviews Analysis') {
      titleElement.innerHTML = '<i class="fas fa-comment-dots"></i> ' + title;
      titleElement.style.color = '#ed8936'; // Orange
    } else if (title === 'Market Analysis') {
      titleElement.innerHTML = '<i class="fas fa-chart-bar"></i> ' + title;
      titleElement.style.color = '#805ad5'; // Purple
    }
  });
} 