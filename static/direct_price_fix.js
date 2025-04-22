// Direct fix for duplicate price calculation records
(function() {
  // Get the original calculatePrice function
  const originalCalculatePrice = window.calculatePrice;
  
  if (!originalCalculatePrice) {
    console.error("calculatePrice function not found");
    return;
  }
  
  // Replace with our fixed version that removes the first tracking call
  window.calculatePrice = function() {
    try {
      console.log("Using fixed calculatePrice function (removed duplicate tracking)");
      
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
      
      // Create request payload with correct parameter names
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
      
      // REMOVED: First tracking call that creates duplicate record
      
      // Send API request
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
     
        // Store the data for potential reuse
        window.lastPriceCalculationData = data;
        
        // Hide loading indicators
        document.getElementById('resultsLoading').style.display = 'none';
        document.getElementById('resultsContent').classList.remove('hidden');
        
        // Update UI with recommendation
        updatePriceElements(data);
        
        // Update elasticity metrics
        if (!data.elasticityCategory && data.price_elasticity !== undefined) {
          // Determine category based on price elasticity value
          const absElasticity = Math.abs(data.price_elasticity);
          let elasticityCategory = 'medium';
          if (absElasticity < 0.5) {
            elasticityCategory = 'low';
          } else if (absElasticity >= 1.0) {
            elasticityCategory = 'high';
          }
          data.elasticityCategory = elasticityCategory;
        } else if (!data.elasticityCategory) {
          // Default to medium if no elasticity data is available
          data.elasticityCategory = 'medium';
        }
        updateElasticityVisualization(data.elasticityCategory, data.recommendedPrice, actualPrice);
        
        // Update all visualizations with API data
        updateVisualizationsWithApiData(data);
        
        // Prepare tracking data with complete results
        const calculationResults = {
          // Input parameters
          productType: productType,
          productGroup: productGroup,
          actualPrice: actualPrice,
          competitorPrice: competitorPrice,
          rating: rating,
          numberOfOrders: numberOfOrders,
          asin: document.getElementById('asin')?.value || 'N/A',
          
          // Calculation results from API
          recommendedPrice: data.recommended_price || data.recommendedPrice,
          minPrice: data.min_price || (data.recommended_price ? data.recommended_price * 0.95 : null),
          maxPrice: data.max_price || (data.recommended_price ? data.recommended_price * 1.05 : null),
          priceElasticity: data.price_elasticity || -1.2,
          volumeImpact: data.volume_impact || data.segment_impact?.total_impact || 0,
          revenueImpact: data.revenue_impact || data.margin_change_pct || 0,
          
          // Additional data for visualization
          segmentImpact: data.segment_impact || null,
          pricingFactorsPct: data.price_factors_pct || null,
          
          // Timestamp
          timestamp: new Date().toISOString()
        };
        
        // THIS IS THE ONLY TRACKING CALL WE KEEP:
        // Track this calculation with full result data
        UsageTracker.trackFeature('priceCalculation', calculationResults);
        
        // Show the results section
        document.getElementById('elasticity-section').style.display = 'block';
        
        // Remove any hidden class from resultsContent just in case
        const resultsContent = document.getElementById('resultsContent');
        if (resultsContent.classList.contains('hidden')) {
          resultsContent.classList.remove('hidden');
        }
      })
      .catch(error => {
        console.error("Error calculating price:", error);
        
        // Show error message
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
  
  console.log("Successfully replaced calculatePrice function to remove duplicate tracking");
})(); 