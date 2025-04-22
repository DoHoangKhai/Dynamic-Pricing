(function() {
  // Original calculatePrice function
  const originalCalculatePrice = window.calculatePrice;
  
  // Override it with our fixed version
  window.calculatePrice = function() {
    try {
      // Show loading state
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
        starRating: rating,  // Changed from rating to starRating to match server expectation
        ordersPerMonth: numberOfOrders,
        productCost: actualPrice * 0.6  // Add productCost since server requires it
      };
      
      console.log("Submitting price calculation with values:", requestData);
      
      // REMOVED: First tracking call - This is the fix!
      // Original code was:
      // UsageTracker.trackFeature('priceCalculation', {
      //   productType: productType,
      //   productGroup: productGroup,
      //   actualPrice: actualPrice,
      //   competitorPrice: competitorPrice,
      //   rating: rating,
      //   numberOfOrders: numberOfOrders
      // });
      
      // Send API request - everything else is unchanged
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
        
        // Rest of the function remains the same as original
        // Process the data and update UI
        window.lastPriceCalculationData = data;
        document.getElementById('resultsLoading').style.display = 'none';
        document.getElementById('resultsContent').classList.remove('hidden');
        updatePriceElements(data);
        
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
        
        // Prepare tracking data with complete results
        const calculationResults = {
          productType: productType,
          productGroup: productGroup,
          actualPrice: actualPrice,
          competitorPrice: competitorPrice,
          rating: rating,
          numberOfOrders: numberOfOrders,
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
        
        // This is the only tracking call we keep
        UsageTracker.trackFeature('priceCalculation', calculationResults);
        
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
  
  console.log("Price calculation function fixed to remove duplicate tracking");
})(); 