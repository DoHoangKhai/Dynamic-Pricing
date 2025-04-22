/**
 * Market Analysis Visualizations
 * This file contains functions for visualizing market data in the Market Analysis tab.
 */

// Initialize global objects for storing market analysis data
window.amazonMarketData = {
    priceHistory: null,
    competitor: null,
    reviews: null
};

// Initialize chart objects in a namespace
window.marketCharts = window.marketCharts || {};
window.marketCharts.priceHistory = null;
window.marketCharts.competitor = null;
window.marketCharts.reviews = null;

/**
 * Initialize market analysis visualizations
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing market analysis visualizations');
    
    // Initialize the market charts object
    window.marketCharts = {
        competitor: null,
        priceHistory: null,
        reviews: null
    };
    
    // Add event listener for fetch data button - fixed selector to match the button in UI
    const fetchDataBtn = document.querySelector('button.btn-primary[data-action="fetch"]') || document.getElementById('fetchDataBtn');
    if (fetchDataBtn) {
        console.log('Found fetch data button:', fetchDataBtn);
        fetchDataBtn.addEventListener('click', fetchMarketData);
    } else {
        console.error('Fetch data button not found. Looking for button.btn-primary[data-action="fetch"] or #fetchDataBtn');
    }
    
    // Add event listener for clear data button
    const clearDataBtn = document.querySelector('.btn-outline-secondary') || document.querySelector('button[data-action="clear"]');
    if (clearDataBtn) {
        console.log('Found clear data button:', clearDataBtn);
        clearDataBtn.addEventListener('click', clearMarketData);
    } else {
        console.error('Clear data button not found. Looking for .btn-outline-secondary or button[data-action="clear"]');
    }
    
    // Add event listener for refresh status button
    const refreshStatusBtn = document.querySelector('.btn-outline-info') || document.querySelector('button[data-action="refresh"]');
    if (refreshStatusBtn) {
        console.log('Found refresh status button:', refreshStatusBtn);
        refreshStatusBtn.addEventListener('click', refreshMarketDataStatus);
    } else {
        console.error('Refresh status button not found. Looking for .btn-outline-info or button[data-action="refresh"]');
    }
});

/**
 * Fetch market data from the API
 */
function fetchMarketData() {
    // Initialize global data store
    window.amazonMarketData = {
        priceHistory: null,
        competitor: null,
        reviews: null
    };
    
    // Try multiple strategies to find the ASIN input field with a value
    let asinInput, asin;
    
    // First, look for the input element by ID - this is the most reliable way
    asinInput = document.getElementById('asinMarketAnalysis');
    
    // If not found by ID, look for the input in the market analysis tab
    if (!asinInput) {
        const marketAnalysisTab = document.getElementById('marketAnalysis');
        if (marketAnalysisTab) {
            asinInput = marketAnalysisTab.querySelector('input[placeholder="Enter Amazon ASIN"]') ||
                      marketAnalysisTab.querySelector('input[type="text"]');
        }
    }
    
    // Get the ASIN value, falling back to a default for testing
    if (asinInput && asinInput.value && asinInput.value.trim() !== '') {
        asin = asinInput.value.trim();
        console.log(`Found ASIN input value: "${asin}"`);
    } else {
        // This is a fallback - we really should have an ASIN by now
        console.warn('Could not find ASIN input, using default value "B0BYS2D9CJ"');
        asin = 'B0BYS2D9CJ';
        
        // If we found the input but it's empty, populate it with the default
        if (asinInput) {
            asinInput.value = asin;
        }
    }
    
    console.log(`Fetching market data for ASIN: ${asin}`);
    
    // Track market analysis feature usage
    if (window.UsageTracker) {
        window.UsageTracker.trackFeature('marketAnalysis', {
            asin: asin,
            timestamp: new Date().toISOString(),
            competitorCount: 0, // Will be updated if data fetching succeeds
            productType: 'Amazon Product',
            action: 'Full Market Analysis',
            details: `Analyzing market data for ${asin}`
        });
    } else {
        console.warn('UsageTracker not found, market analysis activity will not be tracked');
    }
    
    // Also track this as a product search for cross-referencing
    if (window.UsageTracker) {
        window.UsageTracker.trackProductSearch(asin, 'Amazon Product', 'Market Analysis');
    }
    
    // Update UI to show loading state
    setButtonLoading(true);
    
    // Update status indicators to "loading"
    updateStatusIndicator('priceHistoryStatus', 'loading');
    updateStatusIndicator('competitorStatus', 'loading');
    updateStatusIndicator('reviewsStatus', 'loading');
    
    // Clear any existing charts
    clearCharts();
    
    // Debug output
    console.log(`Starting sequential DIRECT API requests for ASIN: ${asin}`);
    
    // Check for amazonApi availability 
    if (!window.amazonApi) {
        console.error('window.amazonApi is not available. This is required for competitor analysis.');
        console.error('Make sure amazon_api_direct_new.js is properly loaded in the HTML file before market_analysis.js');
        console.error('Falling back to server API...');
        fallbackToServerAPI(asin);
        return;
    }
    
    // DIRECT AMAZON API CALLS - EXACT STRUCTURE FROM api_am.js
    
    // Step 1: First fetch product details
    console.log(`[STEP 1] Fetching product details for ${asin}...`);
    
    // Use the amazonApi direct client if it exists
    window.amazonApi.getProductDetails(asin)
        .then(productDetails => {
            console.log(`[PRODUCT DETAILS SUCCESS] Got data:`, productDetails);
            
            // Extract product name, price, and other details
            const productName = productDetails.title || 'Unknown Product';
            let currentPrice = 0;
            
            try {
                if (productDetails.price) {
                    // Handle price formatting variations
                    const priceStr = String(productDetails.price).replace('$', '').replace(',', '');
                    currentPrice = parseFloat(priceStr);
                }
            } catch (error) {
                console.error(`[PRICE PARSING ERROR] Could not parse price:`, error);
            }
            
            // Convert product details into price history format
            const priceHistoryData = generatePriceHistoryFromProductDetails(productDetails);
            
            // Update UI with product details
            updateStatusIndicator('priceHistoryStatus', 'collected');
            visualizePriceHistory(priceHistoryData);
            
            // STEP 2: Search for competitor products using product title
            const searchQuery = productDetails.title || productDetails.productTitle || asin;
            console.log(`[STEP 2] Searching for competitor products with query: "${searchQuery}"`);
            
            return window.amazonApi.searchProducts(searchQuery)
                .then(searchResults => {
                    console.log(`[SEARCH RESULTS SUCCESS] Got ${searchResults.searchProducts?.length || 0} products`);
                    
                    // Process competitor data
                    const competitorData = processCompetitorData(searchResults, asin, currentPrice);
                    
                    // Update UI with competitor data
                    updateStatusIndicator('competitorStatus', 'collected');
                    visualizeCompetitorData(competitorData);
                    
                    // Update usage tracking with actual competitor count
                    if (window.UsageTracker && competitorData && competitorData.competitors) {
                        window.UsageTracker.trackFeature('marketAnalysis', {
                            asin: asin,
                            timestamp: new Date().toISOString(),
                            competitorCount: competitorData.competitors.length,
                            productType: 'Amazon Product',
                            action: 'Full Market Analysis',
                            details: `Analyzed ${competitorData.competitors.length} competitors for ${asin}`
                        });
                    }
                    
                    return window.amazonApi.getProductReviews(asin);
                })
                .catch(error => {
                    console.error(`[SEARCH PRODUCTS ERROR] Failed to get search results:`, error);
                    updateStatusIndicator('competitorStatus', 'error', 'Failed to fetch competitor data');
                    
                    // No fallback - continue to reviews
                    console.log(`[NO FALLBACK] Proceeding to reviews without competitor data`);
                    return window.amazonApi.getProductReviews(asin);
                });
        })
        .then(reviewsData => {
            console.log(`[PRODUCT REVIEWS SUCCESS] Got data:`, reviewsData);
            
            // Update UI with reviews data - directly use the raw API response
            updateStatusIndicator('reviewsStatus', 'collected');
            visualizeReviewsData(reviewsData);
            
            // All data fetching complete
            console.log('[API SEQUENCE COMPLETE] All API calls finished');
            setButtonLoading(false);
        })
        .catch(error => {
            console.error('[API SEQUENCE ERROR] Error in API sequence:', error);
            
            // Set appropriate error indicator based on where the error occurred
            if (!window.amazonMarketData.priceHistory) {
                updateStatusIndicator('priceHistoryStatus', 'error', error.message);
            }
            if (!window.amazonMarketData.competitor) {
                updateStatusIndicator('competitorStatus', 'error', error.message);
            }
            if (!window.amazonMarketData.reviews) {
                updateStatusIndicator('reviewsStatus', 'error', error.message);
            }
            
            setButtonLoading(false);
        });
}

/**
 * Fallback to server API if direct client not available
 */
function fallbackToServerAPI(asin) {
    console.log(`Using server API fallback for ASIN: ${asin}`);
    
    // Step 1: First fetch price history
    console.log(`[API CALL 1] Fetching price history for ${asin}`);
    fetch(`http://localhost:5050/api/market/price-history/${asin}`, {
        method: 'GET',
        headers: { 'Accept': 'application/json' }
    })
    .then(response => {
        console.log(`[API RESPONSE 1] Price history status: ${response.status}`);
        return response.json();
    })
    .then(data => {
        console.log('[API DATA 1] Price history data:', data);
        
        if (data.success) {
            window.amazonMarketData.priceHistory = data;
            updateStatusIndicator('priceHistoryStatus', 'collected');
            visualizePriceHistory(data);
        } else {
            updateStatusIndicator('priceHistoryStatus', 'error', data.message);
            console.error('[API ERROR 1] Price history error:', data.message);
        }
        
        // Step 2: After price history completes, fetch competitive position
        console.log(`[API CALL 2] Fetching competitive position for ${asin}`);
        return fetch(`http://localhost:5050/api/market/competitive-position/${asin}`, {
            method: 'GET',
            headers: { 'Accept': 'application/json' }
        });
    })
    .then(response => {
        console.log(`[API RESPONSE 2] Competitive position status: ${response.status}`);
        return response.json();
    })
    .then(data => {
        console.log('[API DATA 2] Competitive position data:', data);
        
        if (data.success) {
            window.amazonMarketData.competitor = data;
            updateStatusIndicator('competitorStatus', 'collected');
            visualizeCompetitorData(data);
        } else {
            updateStatusIndicator('competitorStatus', 'error', data.message);
            console.error('[API ERROR 2] Competitive position error:', data.message);
        }
        
        // Step 3: After competitive position completes, fetch reviews
        console.log(`[API CALL 3] Fetching reviews for ${asin}`);
        return fetch(`http://localhost:5050/api/reviews?asin=${asin}`, {
            method: 'GET',
            headers: { 'Accept': 'application/json' }
        });
    })
    .then(response => {
        console.log(`[API RESPONSE 3] Reviews status: ${response.status}`);
        return response.json();
    })
    .then(data => {
        console.log('[API DATA 3] Reviews data:', data);
        
        if (data.success) {
            window.amazonMarketData.reviews = data;
            updateStatusIndicator('reviewsStatus', 'collected');
            visualizeReviewsData(data);
        } else {
            updateStatusIndicator('reviewsStatus', 'error', data.message);
            console.error('[API ERROR 3] Reviews error:', data.message);
        }
        
        // All data fetching complete
        console.log('[API SEQUENCE COMPLETE] All API calls finished');
        setButtonLoading(false);
    })
    .catch(error => {
        console.error('[API SEQUENCE ERROR] Error in API sequence:', error);
        
        // Set appropriate error indicator based on where the error occurred
        if (!window.amazonMarketData.priceHistory) {
            updateStatusIndicator('priceHistoryStatus', 'error', error.message);
        }
        if (!window.amazonMarketData.competitor) {
            updateStatusIndicator('competitorStatus', 'error', error.message);
        }
        if (!window.amazonMarketData.reviews) {
            updateStatusIndicator('reviewsStatus', 'error', error.message);
        }
        
        setButtonLoading(false);
    });
}

/**
 * Generate price history from product details
 * - Eliminating all synthetic data generation
 */
function generatePriceHistoryFromProductDetails(productDetails) {
    console.log('Processing product details for price history');
    console.log('Raw product details:', productDetails);
    
    // Extract current price
    let currentPrice = 0;
    try {
        if (productDetails.price) {
            // Handle different price format possibilities
            if (typeof productDetails.price === 'string') {
                currentPrice = parseFloat(productDetails.price.replace(/[^0-9.]/g, ''));
            } else {
                currentPrice = parseFloat(productDetails.price);
            }
        }
    } catch (error) {
        console.error('Error parsing current price:', error);
    }
    
    // Return data we actually have from the API, preserving the productDetails array
    return {
        success: true,
        asin: productDetails.asin || 'Unknown',
        title: productDetails.title || productDetails.productTitle || 'Unknown Product',
        current_price: currentPrice,
        price_history: {
            dates: [],
            prices: [],
            avg_prices: []
        },
        price_insights: {
            messages: ["Insufficient historical price data available."],
            trend: "unknown",
            metrics: {
                volatility: 0,
                min_price: currentPrice,
                max_price: currentPrice,
                avg_price: currentPrice
            }
        },
        // Preserve important direct API fields for demand forecasting
        productDetails: productDetails.productDetails || [],
        prime: productDetails.prime || false,
        pastSales: productDetails.pastSales || null,
        categories: productDetails.categories || [],
        categoryTree: productDetails.categoryTree || [],
        bestSellerRank: productDetails.bestSellerRank || null,
        salesRank: productDetails.salesRank || null
    };
}

/**
 * Generate competitive position data from search results
 * This new function extracts competitor data from search results instead of offers
 */
function generateCompetitivePositionFromSearch(searchData, currentPrice) {
    console.log(`Parsing competitive position from search product data`);
    console.log('Raw search data:', searchData);
    
    // Ensure searchData is an object
    if (!searchData || typeof searchData !== 'object') {
        console.error(`Invalid searchData parameter:`, searchData);
        return { success: false, message: "Invalid search data parameter" };
    }
    
    // Extract product details from search results
    let competitorPrices = [];
    let competitorProducts = [];
    let competitorCount = 0;
    let searchProductDetails = searchData.searchProductDetails || [];
    
    // Ensure searchProductDetails is an array
    if (!Array.isArray(searchProductDetails)) {
        console.error(`searchProductDetails is not an array:`, searchProductDetails);
        return { success: false, message: "Search product details not in expected format" };
    }
    
    try {
        // Get all products with valid prices
        searchProductDetails.forEach((product, index) => {
            // Skip if it's the current product
            if (product.asin === searchData.asin) {
                console.log(`Skipping current product at index ${index}`);
                return;
            }
            
            // Try to extract price - handle different formats
            let price = null;
            if (product.price) {
                // Direct price field
                price = extractPrice(product.price);
            } else if (product.priceDto && product.priceDto.priceValue) {
                // Price from DTO
                price = extractPrice(product.priceDto.priceValue);
            } else if (product.displayPrice) {
                // Display price
                price = extractPrice(product.displayPrice);
            } else if (product.priceRange && product.priceRange.min) {
                // Price range minimum
                price = extractPrice(product.priceRange.min);
            }
            
            // Only add products with valid prices
            if (price && !isNaN(price) && price > 0) {
                competitorPrices.push(price);
                competitorProducts.push({
                    asin: product.asin || 'unknown',
                    title: product.title || product.productTitle || 'Unknown Product',
                    price: price
                });
            }
        });
        
        competitorCount = competitorPrices.length;
        console.log(`Found ${competitorCount} competitor products with valid prices`);
    } catch (error) {
        console.error(`[SEARCH PARSING ERROR] Could not parse search results:`, error);
        return { success: false, message: "Error parsing search data" };
    }
    
    // If no competitor prices found, return error
    if (competitorPrices.length === 0) {
        console.error("No competitor prices found in the search results");
        return { success: false, message: "No competitor prices found" };
    }
    
    // Calculate market average price
    const avgMarketPrice = competitorPrices.reduce((sum, price) => sum + price, 0) / competitorPrices.length;
    
    // Calculate position percentile
    // If currentPrice isn't passed, try to get it from the search data
    if (!currentPrice || isNaN(currentPrice) || currentPrice <= 0) {
        if (searchData.price) {
            currentPrice = extractPrice(searchData.price);
        } else {
            // Set a default price if we can't extract it
            currentPrice = avgMarketPrice;
            console.warn(`No valid current price found, using average market price: ${currentPrice}`);
        }
    }
    
    const lowerPrices = competitorPrices.filter(p => p < currentPrice).length;
    const percentile = Math.round((lowerPrices / competitorPrices.length) * 100);
    
    // Generate price distribution brackets
    const minPrice = Math.min(...competitorPrices);
    const maxPrice = Math.max(...competitorPrices);
    const priceDiff = maxPrice - minPrice;
    const bracketSize = priceDiff > 0 ? priceDiff / 5 : 1;
    
    const brackets = [];
    for (let i = 0; i < 5; i++) {
        const bracketMin = minPrice + (i * bracketSize);
        const bracketMax = minPrice + ((i + 1) * bracketSize);
        
        // Count competitors in this range
        const count = competitorPrices.filter(p => p >= bracketMin && (i === 4 ? p <= bracketMax : p < bracketMax)).length;
        
        brackets.push({
            range: `$${bracketMin.toFixed(2)} - $${bracketMax.toFixed(2)}`,
            count: count
        });
    }
    
    // Generate insights based on percentile
    let insights = [];
    if (percentile < 25) {
        insights = [
            "Your price is lower than most competitors.",
            "Consider testing slight price increases.",
            "Highlight unique value propositions beyond price."
        ];
    } else if (percentile < 50) {
        insights = [
            "Your price is in the lower mid-range of the market.",
            "Good value positioning relative to competitors.",
            "Monitor competitor pricing strategies."
        ];
    } else if (percentile < 75) {
        insights = [
            "Your price is in the upper mid-range of the market.",
            "Emphasize product quality and features.",
            "Consider promotional offers to increase competitiveness."
        ];
    } else {
        insights = [
            "Your price is higher than most competitors.",
            "Ensure product quality justifies premium pricing.",
            "Highlight premium features and benefits."
        ];
    }
    
    // Add top competitor products
    const topCompetitors = competitorProducts
        .sort((a, b) => a.price - b.price)
        .slice(0, 5);
    
    return {
        success: true,
        asin: searchData.asin || 'Unknown',
        current_price: currentPrice,
        keyword: searchData.keyword || 'Unknown',
        competitive_position: {
            percentile: percentile,
            avg_market_price: parseFloat(avgMarketPrice.toFixed(2)),
            competitor_count: competitorCount
        },
        price_distribution: {
            brackets: brackets
        },
        pricing_context: {
            insights: insights,
            top_competitors: topCompetitors
        }
    };
}

/**
 * Helper function to extract price from various formats
 */
function extractPrice(priceStr) {
    if (!priceStr) return null;
    
    // If already a number, return it
    if (typeof priceStr === 'number') return priceStr;
    
    try {
        // Convert string price to number
        const price = parseFloat(String(priceStr).replace(/[^0-9.]/g, ''));
        return !isNaN(price) ? price : null;
    } catch (error) {
        console.error(`Error extracting price from ${priceStr}:`, error);
        return null;
    }
}

/**
 * Visualize demand forecast data instead of price history
 */
function visualizePriceHistory(data) {
    console.log('Visualizing price history data:', data);
    console.log('Raw product details:', data);
    
    // Find the price history container (we're repurposing it for demand data)
    const demandContainer = document.getElementById('priceHistoryChart');
    
    if (!demandContainer) {
        console.error('Demand forecast container not found');
        return;
    }
    
    // Clear container
    demandContainer.innerHTML = '';
    
    // Extract product details
    const productTitle = data.title || 'Unknown Product';
    const currentPrice = data.current_price || 0;
    const asin = data.asin || 'Unknown';
    
    // Create section for demand insights
    const demandInsightsContainer = document.createElement('div');
    demandInsightsContainer.className = 'demand-forecast-container';
    
    // Set basic styles
    demandInsightsContainer.style.padding = '15px';
    demandInsightsContainer.style.borderRadius = '8px';
    demandInsightsContainer.style.backgroundColor = 'rgba(0,0,0,0.1)';
    
    // Extract demand-related data from the product details
    // Log important information for debugging
    console.log('Extracting best seller rank from:', data);
    if (data.productDetails) {
        console.log('productDetails available:', data.productDetails.length, 'items');
    }
    
    const bestSellerRank = extractBestSellerRank(data);
    console.log('Extracted best seller rank:', bestSellerRank);
    
    const categories = extractCategories(data);
    console.log('Extracted categories:', categories);
    
    const dateFirstAvailable = extractDateFirstAvailable(data);
    console.log('Extracted date first available:', dateFirstAvailable);
    
    const pastSales = extractPastSales(data);
    console.log('Extracted past sales:', pastSales);
    
    const prime = data.prime || false;
    
    // Calculate product age in days
    const productAgeDays = calculateProductAge(dateFirstAvailable);
    
    // Calculate category seasonality score (mock implementation)
    const seasonalityData = calculateCategorySeasonality(categories);
    
    // Create HTML for the demand insights
    let demandInsightsHTML = `
        <h3 style="margin-top: 0;">Demand Forecast Insights</h3>
        
        <div class="demand-metrics-grid" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px;">
            <div class="metric-card" style="background-color: rgba(0,0,0,0.15); padding: 15px; border-radius: 8px;">
                <div class="metric-value" style="font-size: 1.5rem; font-weight: bold; color: #4299e1;">${bestSellerRank.mainRank || 'N/A'}</div>
                <div class="metric-label" style="font-size: 0.9rem; opacity: 0.8;">Best Seller Rank</div>
                <div class="metric-context" style="font-size: 0.8rem; margin-top: 5px;">${bestSellerRank.category || ''}</div>
            </div>
            
            <div class="metric-card" style="background-color: rgba(0,0,0,0.15); padding: 15px; border-radius: 8px;">
                <div class="metric-value" style="font-size: 1.5rem; font-weight: bold; color: #48bb78;">${pastSales || 'Unknown'}</div>
                <div class="metric-label" style="font-size: 0.9rem; opacity: 0.8;">Recent Sales</div>
                <div class="metric-context" style="font-size: 0.8rem; margin-top: 5px;">Past 30 days estimate</div>
            </div>
            
            <div class="metric-card" style="background-color: rgba(0,0,0,0.15); padding: 15px; border-radius: 8px;">
                <div class="metric-value" style="font-size: 1.5rem; font-weight: bold; color: #ed8936;">${productAgeDays} days</div>
                <div class="metric-label" style="font-size: 0.9rem; opacity: 0.8;">Product Age</div>
                <div class="metric-context" style="font-size: 0.8rem; margin-top: 5px;">Since ${dateFirstAvailable || 'Unknown'}</div>
            </div>
            
            <div class="metric-card" style="background-color: rgba(0,0,0,0.15); padding: 15px; border-radius: 8px;">
                <div class="metric-value" style="font-size: 1.5rem; font-weight: bold; color: ${prime ? '#4299e1' : '#a0aec0'};">${prime ? 'Yes' : 'No'}</div>
                <div class="metric-label" style="font-size: 0.9rem; opacity: 0.8;">Prime Eligible</div>
                <div class="metric-context" style="font-size: 0.8rem; margin-top: 5px;">${prime ? 'Faster shipping available' : 'Standard shipping only'}</div>
            </div>
        </div>
        
        <div class="category-seasonality" style="margin-bottom: 20px;">
            <h4 style="margin-bottom: 10px;">Category Seasonality</h4>
            <div class="seasonality-chart" style="height: 80px; display: flex; align-items: center; margin-bottom: 10px;">
                ${generateSeasonalityChart(seasonalityData)}
            </div>
            <div class="seasonality-legend" style="display: flex; justify-content: space-between; font-size: 0.8rem;">
                <span>Jan</span>
                <span>Feb</span>
                <span>Mar</span>
                <span>Apr</span>
                <span>May</span>
                <span>Jun</span>
                <span>Jul</span>
                <span>Aug</span>
                <span>Sep</span>
                <span>Oct</span>
                <span>Nov</span>
                <span>Dec</span>
            </div>
        </div>
        
        <div class="rank-details" style="margin-bottom: 20px;">
            <h4 style="margin-bottom: 10px;">Sales Rank Details</h4>
            ${generateRankDetails(bestSellerRank)}
        </div>
        
        <div class="demand-forecast-insights">
            <h4 style="margin-bottom: 10px;">Demand Insights</h4>
            <ul style="padding-left: 20px; margin-top: 5px;">
                ${generateDemandInsights(bestSellerRank, categories, productAgeDays, seasonalityData)}
            </ul>
        </div>
    `;
    
    // Set the HTML content
    demandInsightsContainer.innerHTML = demandInsightsHTML;
    
    // Append to container
    demandContainer.appendChild(demandInsightsContainer);
    
    // Track price history analysis activity
    if (data && data.asin) {
        trackMarketAnalysisActivity('Price History Analysis', data.asin, {
            pricePoints: data.history ? data.history.length : 0
        });
    }
}

/**
 * Extract Best Seller Rank from product details
 */
function extractBestSellerRank(data) {
    try {
        // Try to find best seller rank in different possible locations
        let rankText = '';
        let rankObj = { mainRank: 'N/A', category: 'N/A', subRank: null };
        
        // Check for productDetails array (from the response we can see this is the correct structure)
        if (data && data.productDetails && Array.isArray(data.productDetails)) {
            // Check for the "Best Sellers Rank" field in productDetails array
            const bsrDetail = data.productDetails.find(detail => 
                detail.name && detail.name.includes('Best Sellers Rank'));
            
            if (bsrDetail && bsrDetail.value) {
                rankText = bsrDetail.value;
                console.log("Found Best Sellers Rank:", rankText);
            }
        }
        
        // Legacy checks for other API formats
        if (!rankText) {
            if (data.bestSellerRank) {
                rankText = data.bestSellerRank;
            } else if (data.salesRank) {
                rankText = data.salesRank;
            }
        }
        
        // Parse the rank text
        if (rankText) {
            // Extract main rank number (e.g., "#18 in Arts, Crafts & Sewing")
            const mainRankMatch = rankText.match(/#(\d+)\s+in\s+([^(]+)/);
            if (mainRankMatch) {
                rankObj.mainRank = `#${mainRankMatch[1]}`;
                rankObj.category = mainRankMatch[2].trim();
            }
            
            // Extract sub-rank if available (e.g., "#1 in Drawing Pencils")
            const subRankMatch = rankText.match(/#(\d+)\s+in\s+([^(]+)(?!\()/g);
            if (subRankMatch && subRankMatch.length > 1) {
                const subMatch = subRankMatch[1].match(/#(\d+)\s+in\s+([^(]+)/);
                if (subMatch) {
                    rankObj.subRank = {
                        rank: `#${subMatch[1]}`,
                        category: subMatch[2].trim()
                    };
                }
            }
        }
        
        return rankObj;
    } catch (error) {
        console.error('Error extracting Best Seller Rank:', error);
        return { mainRank: 'N/A', category: 'N/A', subRank: null };
    }
}

/**
 * Extract categories from product details
 */
function extractCategories(data) {
    try {
        if (data.categories && Array.isArray(data.categories)) {
            return data.categories;
        } else if (data.categoryTree && Array.isArray(data.categoryTree)) {
            return data.categoryTree.map(cat => cat.name || cat);
        }
        return ['Unknown Category'];
    } catch (error) {
        console.error('Error extracting categories:', error);
        return ['Unknown Category'];
    }
}

/**
 * Extract date first available from product details
 */
function extractDateFirstAvailable(data) {
    try {
        // Primary check: Look in the productDetails array
        if (data && data.productDetails && Array.isArray(data.productDetails)) {
            // Find date first available in productDetails array
            const dateDetail = data.productDetails.find(detail => 
                detail.name && (
                    detail.name.includes('Date First Available') || 
                    detail.name.includes('Release Date')
                )
            );
            
            if (dateDetail && dateDetail.value) {
                console.log("Found Date First Available:", dateDetail.value);
                return dateDetail.value;
            }
        }
        
        // Fallback checks
        if (data.dateFirstAvailable) {
            return data.dateFirstAvailable;
        } else if (data.releaseDate) {
            return data.releaseDate;
        }
        
        return 'Unknown';
    } catch (error) {
        console.error('Error extracting date first available:', error);
        return 'Unknown';
    }
}

/**
 * Extract past sales information
 */
function extractPastSales(data) {
    try {
        // Check for sales data in the API response
        if (data && data.pastSales) {
            console.log("Found pastSales:", data.pastSales);
            return data.pastSales;
        }
        
        // Directly check for "600+ bought in past month" field
        // This appears to be available in the console output
        if (data && typeof data === 'object') {
            for (const key in data) {
                if (typeof data[key] === 'string' && 
                    data[key].includes('bought in past month')) {
                    console.log(`Found sales in ${key}:`, data[key]);
                    return data[key];
                }
            }
        }
        
        return 'Unknown';
    } catch (error) {
        console.error('Error extracting past sales:', error);
        return 'Unknown';
    }
}

/**
 * Calculate product age in days
 */
function calculateProductAge(dateString) {
    try {
        if (dateString === 'Unknown') return 'Unknown';
        
        const date = new Date(dateString);
        if (isNaN(date.getTime())) return 'Unknown';
        
        const today = new Date();
        const diffTime = Math.abs(today - date);
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
        
        return diffDays;
    } catch (error) {
        console.error('Error calculating product age:', error);
        return 'Unknown';
    }
}

/**
 * Calculate category seasonality
 */
function calculateCategorySeasonality(categories) {
    // This is a simplified mock implementation
    // In a real implementation, this would use historical data or industry benchmarks
    
    // Default seasonality (flat)
    let seasonality = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50];
    
    // Apply different seasonality patterns based on category
    if (categories.some(cat => cat.includes('Art') || cat.includes('Craft'))) {
        // Art supplies peak during back-to-school and holiday seasons
        seasonality = [40, 45, 50, 55, 60, 55, 50, 80, 85, 70, 90, 95];
    } else if (categories.some(cat => cat.includes('Electronics'))) {
        // Electronics peak during holiday season and summer
        seasonality = [50, 40, 35, 40, 50, 70, 75, 70, 60, 70, 90, 100];
    } else if (categories.some(cat => cat.includes('Kitchen'))) {
        // Kitchen items peak during holiday season
        seasonality = [60, 50, 45, 50, 60, 65, 70, 70, 65, 70, 85, 100];
    } else if (categories.some(cat => cat.includes('Toy'))) {
        // Toys peak during holiday season
        seasonality = [35, 30, 35, 40, 45, 50, 55, 60, 65, 70, 90, 100];
    } else if (categories.some(cat => cat.includes('Garden'))) {
        // Garden items peak during spring and summer
        seasonality = [30, 40, 70, 90, 100, 95, 80, 70, 60, 50, 40, 30];
    }
    
    // Get current month for highlighting
    const currentMonth = new Date().getMonth(); // 0-11
    
    return {
        values: seasonality,
        currentMonth: currentMonth
    };
}

/**
 * Generate HTML for seasonality chart
 */
function generateSeasonalityChart(seasonalityData) {
    const { values, currentMonth } = seasonalityData;
    
    // Create the bars
    let chartHTML = '';
    
    values.forEach((value, index) => {
        const height = Math.max(10, value * 0.7); // Scale value to reasonable height (max 70px)
        const isCurrentMonth = index === currentMonth;
        
        // Determine color based on value height
        let color;
        if (value >= 80) color = 'rgba(72, 187, 120, 0.7)'; // Green for high demand
        else if (value >= 50) color = 'rgba(237, 137, 54, 0.7)'; // Orange for medium
        else color = 'rgba(229, 62, 62, 0.5)'; // Red for low
        
        // Highlight current month
        const border = isCurrentMonth ? '2px solid white' : 'none';
        const borderRadius = '4px 4px 0 0';
        
        chartHTML += `<div style="flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: flex-end; height: 100%;">
            <div style="height: ${height}px; width: 80%; background-color: ${color}; border: ${border}; border-radius: ${borderRadius};"></div>
        </div>`;
    });
    
    return chartHTML;
}

/**
 * Generate HTML for rank details
 */
function generateRankDetails(bestSellerRank) {
    let html = '';
    
    if (bestSellerRank.mainRank !== 'N/A') {
        html += `<div class="rank-item" style="margin-bottom: 8px;">
            <span style="font-weight: bold; color: #4299e1;">${bestSellerRank.mainRank}</span> in 
            <span style="font-style: italic;">${bestSellerRank.category}</span>
        </div>`;
    }
    
    if (bestSellerRank.subRank) {
        html += `<div class="rank-item" style="margin-bottom: 8px;">
            <span style="font-weight: bold; color: #48bb78;">${bestSellerRank.subRank.rank}</span> in 
            <span style="font-style: italic;">${bestSellerRank.subRank.category}</span>
        </div>`;
    }
    
    if (html === '') {
        html = '<p>No sales rank data available</p>';
    }
    
    return html;
}

/**
 * Generate demand insights based on available data
 */
function generateDemandInsights(bestSellerRank, categories, productAge, seasonalityData) {
    const insights = [];
    
    // Sales rank insights
    if (bestSellerRank.mainRank !== 'N/A') {
        const rankNum = parseInt(bestSellerRank.mainRank.replace('#', ''));
        if (rankNum <= 20) {
            insights.push('Very high demand product based on top 20 sales rank.');
        } else if (rankNum <= 100) {
            insights.push('Strong demand product based on top 100 sales rank.');
        } else if (rankNum <= 1000) {
            insights.push('Moderate demand product based on sales rank.');
        } else {
            insights.push('Lower demand product based on sales rank.');
        }
    }
    
    // Category insights
    if (categories.length > 0) {
        const mainCategory = categories[0];
        if (mainCategory.includes('Art') || mainCategory.includes('Craft')) {
            insights.push('Art supplies typically show increased demand during back-to-school season (August-September) and holiday season (November-December).');
        } else if (mainCategory.includes('Electronics')) {
            insights.push('Electronics typically peak during holiday season (November-December) and new product launch windows.');
        }
    }
    
    // Product age insights
    if (productAge !== 'Unknown') {
        if (productAge < 30) {
            insights.push('New product (less than 30 days old) - typically experiences higher interest and sales velocity.');
        } else if (productAge < 90) {
            insights.push('Recently launched product (less than 3 months old) - still in early adoption phase.');
        } else if (productAge > 365) {
            insights.push('Mature product (more than 1 year old) - likely has stable demand pattern.');
        }
    }
    
    // Seasonality insights
    const currentMonth = new Date().getMonth();
    const nextMonth = (currentMonth + 1) % 12;
    const currentSeasonality = seasonalityData.values[currentMonth];
    const nextMonthSeasonality = seasonalityData.values[nextMonth];
    
    const monthNames = ['January', 'February', 'March', 'April', 'May', 'June', 
                        'July', 'August', 'September', 'October', 'November', 'December'];
    
    if (nextMonthSeasonality > currentSeasonality + 10) {
        insights.push(`Demand expected to increase in ${monthNames[nextMonth]} based on seasonal trends.`);
    } else if (nextMonthSeasonality < currentSeasonality - 10) {
        insights.push(`Demand expected to decrease in ${monthNames[nextMonth]} based on seasonal trends.`);
    } else {
        insights.push(`Demand expected to remain stable through ${monthNames[nextMonth]}.`);
    }
    
    // Return insights as HTML list items
    return insights.map(insight => `<li>${insight}</li>`).join('');
}

/**
 * Visualize competitor data
 */
function visualizeCompetitorData(data) {
    console.log('Visualizing competitor data:', data);
    console.log('Raw competitor data:', data);
    
    // Initialize the marketCharts object if it doesn't exist
    if (!window.marketCharts) {
        window.marketCharts = {};
    }
    
    // Find the competitor container
    const competitorContainer = document.getElementById('competitorChart');
    
    if (!competitorContainer) {
        console.error('Competitor container not found');
        return;
    }
    
    // Clear container
    competitorContainer.innerHTML = '';
    
    // Get competitive position and price distribution data directly from API
    const competitivePosition = data.competitive_position || {};
    const priceDistribution = data.price_distribution || {};
    const pricingContext = data.pricing_context || {};
    
    // Create enhanced position meter
    const positionMeter = document.createElement('div');
    positionMeter.className = 'position-meter-container';
    positionMeter.style.marginBottom = '30px';
    
    const percentile = competitivePosition.percentile || 0;
    const avgMarketPrice = competitivePosition.avg_market_price || 0;
    const competitorCount = competitivePosition.competitor_count || 0;
    const currentPrice = data.current_price || 0;
    
    // Calculate price position stats
    const priceDiff = currentPrice - avgMarketPrice;
    const priceDiffPercent = avgMarketPrice > 0 ? (priceDiff / avgMarketPrice) * 100 : 0;
    const priceDiffFormatted = priceDiff >= 0 ? `+$${priceDiff.toFixed(2)}` : `-$${Math.abs(priceDiff).toFixed(2)}`;
    const priceDiffPercentFormatted = priceDiffPercent >= 0 ? `+${priceDiffPercent.toFixed(1)}%` : `-${Math.abs(priceDiffPercent).toFixed(1)}%`;
    
    // Create classes based on percentile
    let positionClass = 'medium';
    let positionMessage = 'mid-range price';
    
    if (percentile < 25) {
        positionClass = 'low';
        positionMessage = 'lower than most competitors';
    } else if (percentile >= 75) {
        positionClass = 'high';
        positionMessage = 'higher than most competitors';
    }
    
    positionMeter.innerHTML = `
        <div class="price-position-header">
            <h4>Price Position</h4>
            <div class="price-badge price-badge-${positionClass}">${percentile}th percentile</div>
        </div>
        
        <div class="price-position-summary">
            <div class="your-price-container">
                <span class="your-price-value">$${currentPrice.toFixed(2)}</span>
                <span class="your-price-label">Your Price</span>
                <span class="price-difference ${priceDiffPercent >= 0 ? 'text-danger' : 'text-success'}">
                    ${priceDiffFormatted} (${priceDiffPercentFormatted}) vs. market average
                </span>
            </div>
        </div>
        
        <div class="price-position-meter ${positionClass}">
            <div class="meter-sections">
                <div class="meter-section low"></div>
                <div class="meter-section medium"></div>
                <div class="meter-section high"></div>
            </div>
            <div class="position-indicator" style="left: ${percentile}%">
                <div class="indicator-dot"></div>
                <div class="indicator-line"></div>
            </div>
            <div class="position-labels">
                <span>Lower Priced</span>
                <span>Higher Priced</span>
            </div>
        </div>
        
        <p class="position-description">Your price is <strong>${positionMessage}</strong> (${percentile}% of competitors have lower prices)</p>
        
        <div class="competitor-stats">
            <div class="stat-item">
                <span class="stat-value">$${avgMarketPrice.toFixed(2)}</span>
                <span class="stat-label">Market Average</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">${competitorCount}</span>
                <span class="stat-label">Competitors</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">${data.keyword ? data.keyword.substring(0, 25) + (data.keyword.length > 25 ? '...' : '') : "N/A"}</span>
                <span class="stat-label">Search Term</span>
            </div>
        </div>
    `;
    
    competitorContainer.appendChild(positionMeter);
    
    // Create price distribution chart if data is available
    if (priceDistribution.brackets && priceDistribution.brackets.length > 0) {
        const chartContainer = document.createElement('div');
        chartContainer.className = 'price-distribution-container';
        chartContainer.style.height = '180px';
        chartContainer.innerHTML = '<h4>Price Distribution</h4><canvas id="priceDistributionCanvas"></canvas>';
        competitorContainer.appendChild(chartContainer);
        
        const canvas = document.getElementById('priceDistributionCanvas');
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        
        const brackets = priceDistribution.brackets;
        const ranges = brackets.map(b => b.range);
        const counts = brackets.map(b => b.count);
        
        // Highlight the bracket containing the current price
        const backgroundColors = brackets.map((bracket, index) => {
            // Parse the min and max from the range string
            const rangeStr = bracket.range;
            const [minStr, maxStr] = rangeStr.split(' - ');
            const min = parseFloat(minStr.replace('$', ''));
            const max = parseFloat(maxStr.replace('$', ''));
            
            // Check if current price is in this bracket
            if (currentPrice >= min && currentPrice <= max) {
                return 'rgba(66, 153, 225, 0.8)'; // Highlighted
            }
            return 'rgba(66, 153, 225, 0.5)'; // Normal
        });
        
        const ctx = canvas.getContext('2d');
        
        // Check if Chart.js is available
        if (typeof Chart === 'undefined') {
            console.error('Chart.js is not available. Cannot create price distribution chart.');
            chartContainer.innerHTML += '<p class="error-message">Chart library not loaded. Cannot display price distribution.</p>';
            return;
        }
        
        window.marketCharts.competitor = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ranges,
                datasets: [{
                    label: 'Number of Products',
                    data: counts,
                    backgroundColor: backgroundColors,
                    borderColor: 'rgba(66, 153, 225, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                layout: {
                    padding: {
                        top: 5,
                        right: 20,
                        bottom: 5,
                        left: 10
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Number of Products'
                        },
                        ticks: {
                            maxTicksLimit: 5
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Price Range'
                        },
                        ticks: {
                            maxRotation: 45,
                            minRotation: 45
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Price Distribution in Market',
                        font: {
                            size: 14
                        }
                    },
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            title: function(context) {
                                return context[0].label;
                            },
                            label: function(context) {
                                return `${context.raw} products in this range`;
                            },
                            afterLabel: function(context) {
                                const rangeStr = context.label;
                                const [minStr, maxStr] = rangeStr.split(' - ');
                                const min = parseFloat(minStr.replace('$', ''));
                                const max = parseFloat(maxStr.replace('$', ''));
                                
                                if (currentPrice >= min && currentPrice <= max) {
                                    return `Your price: $${currentPrice.toFixed(2)} (in this range)`;
                                }
                                return '';
                            }
                        }
                    }
                }
            }
        });
    }
    
    // Add top competitors table if available
    if (pricingContext.top_competitors && pricingContext.top_competitors.length > 0) {
        const topCompetitorsContainer = document.createElement('div');
        topCompetitorsContainer.className = 'top-competitors-container';
        topCompetitorsContainer.style.marginTop = '20px';
        
        let tableHtml = `
            <h4>Top Competitors by Price</h4>
            <div class="table-container" style="max-height: 180px; overflow-y: auto;">
                <table class="competitors-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Product</th>
                            <th>Price</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        // Add top competitors to the table
        pricingContext.top_competitors.forEach((competitor, index) => {
            // Truncate title to 50 characters
            const title = competitor.title.length > 50 ? 
                competitor.title.substring(0, 50) + '...' : 
                competitor.title;
                
            tableHtml += `
                <tr>
                    <td>${index + 1}</td>
                    <td title="${competitor.title}">${title}</td>
                    <td>$${competitor.price.toFixed(2)}</td>
                </tr>
            `;
        });
        
        // Close the table
        tableHtml += `
                    </tbody>
                </table>
            </div>
        `;
        
        topCompetitorsContainer.innerHTML = tableHtml;
        competitorContainer.appendChild(topCompetitorsContainer);
    }
    
    // Add pricing insights if available
    if (pricingContext.insights && pricingContext.insights.length > 0) {
        const insightsContainer = document.createElement('div');
        insightsContainer.className = 'competitor-insights';
        insightsContainer.style.marginTop = '20px';
        
        let insightsHtml = '<h4>Market Insights</h4><ul>';
        pricingContext.insights.forEach(insight => {
            insightsHtml += `<li>${insight}</li>`;
        });
        insightsHtml += '</ul>';
        
        insightsContainer.innerHTML = insightsHtml;
        competitorContainer.appendChild(insightsContainer);
    }
    
    // Track competitor analysis activity
    if (data && data.asin) {
        trackMarketAnalysisActivity('Competitor Analysis', data.asin, {
            competitorCount: data.competitors ? data.competitors.length : 0
        });
    }
}

/**
 * Visualize reviews data
 */
function visualizeReviewsData(data) {
    console.log('Visualizing reviews data:', data);
    console.log('Raw reviews data received:', data);
    
    // Find the reviews container
    const reviewsContainer = document.getElementById('reviewsChart');
    
    if (!reviewsContainer) {
        console.error('Reviews container not found');
        return;
    }
    
    // Clear container
    reviewsContainer.innerHTML = '';
    
    // DIRECT USE OF RAW API RESPONSE DATA - NO FALLBACKS
    let rating = 0;
    let reviewCount = 0;
    
    // Extract rating from API response - try different formats
    if (data.average_rating && !isNaN(parseFloat(data.average_rating))) {
        rating = parseFloat(data.average_rating);
    } else if (data.productRating) {
        const ratingMatch = data.productRating.match(/([0-9.]+)/);
        if (ratingMatch) {
            rating = parseFloat(ratingMatch[1]);
        }
    } else if (data.reviewSummary && data.reviewSummary.averageRating) {
        rating = parseFloat(data.reviewSummary.averageRating);
    }
    
    // Extract review count from API response - try different formats
    if (data.total_reviews && !isNaN(parseInt(data.total_reviews))) {
        reviewCount = parseInt(data.total_reviews);
    } else if (data.countReviews && !isNaN(parseInt(data.countReviews))) {
        reviewCount = parseInt(data.countReviews);
    } else if (data.countRatings && !isNaN(parseInt(data.countRatings))) {
        reviewCount = parseInt(data.countRatings);
    }
    
    console.log('Using actual API data:', { rating, reviewCount });
    
    // Calculate sentiment directly from ratings breakdown
    let sentimentData = { positive: 0, neutral: 0, negative: 0 };
    let hasValidSentimentData = false;
    
    if (data.ratings_breakdown) {
        console.log("Using actual ratings breakdown from API");
        // Get the numbers directly from the API's ratings breakdown
        const fiveStar = parseInt(data.ratings_breakdown.five_star || 0);
        const fourStar = parseInt(data.ratings_breakdown.four_star || 0);
        const threeStar = parseInt(data.ratings_breakdown.three_star || 0);
        const twoStar = parseInt(data.ratings_breakdown.two_star || 0);
        const oneStar = parseInt(data.ratings_breakdown.one_star || 0);
        
        sentimentData = {
            positive: fiveStar + fourStar,
            neutral: threeStar,
            negative: twoStar + oneStar
        };
        
        hasValidSentimentData = (sentimentData.positive > 0 || sentimentData.neutral > 0 || sentimentData.negative > 0);
    }
    else if (data.reviewSummary) {
        console.log("Using reviewSummary stars breakdown from API");
        // Calculate based on percentages and total reviews
        const fiveStarPct = parseInt(data.reviewSummary.fiveStar?.percentage || 0);
        const fourStarPct = parseInt(data.reviewSummary.fourStar?.percentage || 0);
        const threeStarPct = parseInt(data.reviewSummary.threeStar?.percentage || 0);
        const twoStarPct = parseInt(data.reviewSummary.twoStar?.percentage || 0);
        const oneStarPct = parseInt(data.reviewSummary.oneStar?.percentage || 0);
        
        sentimentData = {
            positive: Math.round(((fiveStarPct + fourStarPct) / 100) * reviewCount),
            neutral: Math.round((threeStarPct / 100) * reviewCount),
            negative: Math.round(((twoStarPct + oneStarPct) / 100) * reviewCount)
        };
        
        hasValidSentimentData = (sentimentData.positive > 0 || sentimentData.neutral > 0 || sentimentData.negative > 0);
    }
    
    console.log('Final sentiment data:', sentimentData);
    
    // Create review summary
    const summaryContainer = document.createElement('div');
    summaryContainer.className = 'review-summary';
    summaryContainer.style.height = '70px';
    summaryContainer.style.marginBottom = '15px';
    
    summaryContainer.innerHTML = `
        <h4>Review Summary</h4>
        <div class="rating-container">
            <div class="rating-value">${rating.toFixed(1)}</div>
            <div class="rating-stars">
                ${generateStarRating(rating)}
            </div>
            <div class="review-count">${reviewCount} reviews</div>
        </div>
    `;
    
    reviewsContainer.appendChild(summaryContainer);
    
    // Only create sentiment distribution if we have valid data
    if (hasValidSentimentData) {
        const sentimentContainer = document.createElement('div');
        sentimentContainer.className = 'sentiment-container';
        sentimentContainer.style.height = '220px';
        sentimentContainer.innerHTML = '<h4>Sentiment Distribution</h4><canvas id="sentimentCanvas"></canvas>';
        reviewsContainer.appendChild(sentimentContainer);
        
        const canvas = document.getElementById('sentimentCanvas');
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        
        // Create sentiment chart
        const ctx = document.getElementById('sentimentCanvas').getContext('2d');
        window.marketCharts.reviews = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Positive', 'Neutral', 'Negative'],
                datasets: [{
                    data: [
                        sentimentData.positive || 0,
                        sentimentData.neutral || 0,
                        sentimentData.negative || 0
                    ],
                    backgroundColor: [
                        'rgba(72, 187, 120, 0.7)',
                        'rgba(237, 137, 54, 0.7)',
                        'rgba(229, 62, 62, 0.7)'
                    ],
                    borderColor: [
                        'rgba(72, 187, 120, 1)',
                        'rgba(237, 137, 54, 1)',
                        'rgba(229, 62, 62, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                layout: {
                    padding: 20
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Review Sentiment Distribution',
                        font: {
                            size: 14
                        },
                        padding: {
                            bottom: 10
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.raw || 0;
                                const percentage = ((value / (sentimentData.positive + sentimentData.neutral + sentimentData.negative || 1)) * 100).toFixed(1);
                                return `${label}: ${value} (${percentage}%)`;
                            }
                        }
                    },
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 15,
                            boxWidth: 12
                        }
                    }
                }
            }
        });
    } else {
        // Display message when we don't have sentiment data
        const noDataContainer = document.createElement('div');
        noDataContainer.className = 'no-data-message';
        noDataContainer.style.height = '100px';
        noDataContainer.style.display = 'flex';
        noDataContainer.style.justifyContent = 'center';
        noDataContainer.style.alignItems = 'center';
        noDataContainer.style.backgroundColor = 'rgba(0,0,0,0.1)';
        noDataContainer.style.borderRadius = '8px';
        noDataContainer.style.margin = '10px 0';
        noDataContainer.innerHTML = '<p>No sentiment distribution data available from API</p>';
        reviewsContainer.appendChild(noDataContainer);
    }
    
    // Track reviews analysis activity
    if (data && data.asin) {
        trackMarketAnalysisActivity('Reviews Analysis', data.asin, {
            reviewCount: data.reviews ? data.reviews.length : 0
        });
    }
}

/**
 * Clear market data and visualizations
 */
function clearMarketData() {
    console.log('Clearing all market data and visualizations');
    
    // Clear data storage
    window.amazonMarketData.priceHistory = null;
    window.amazonMarketData.competitor = null;
    window.amazonMarketData.reviews = null;
    
    // Clear any existing charts
    clearCharts();
    
    // Reset status indicators
    updateStatusIndicator('priceHistoryStatus', 'not-collected');
    updateStatusIndicator('competitorStatus', 'not-collected');
    updateStatusIndicator('reviewsStatus', 'not-collected');
    
    // Try to find and keep the ASIN value for user convenience
    let asinInput = document.getElementById('asin');
    
    // If not found by ID, try by placeholder or any other input with ASIN value
    if (!asinInput) {
        asinInput = document.querySelector('input[placeholder="Enter Amazon ASIN"]') || 
                   document.querySelector('input[placeholder="B0BYS2D9CJ"]');
    }
    
    // If still not found, try looking for any visible input in the market analysis tab
    if (!asinInput) {
        const marketTab = document.getElementById('marketAnalysis');
        if (marketTab) {
            asinInput = marketTab.querySelector('input[type="text"]');
        }
    }
    
    // If found, don't clear the ASIN to make it easier for the user to try again
    if (asinInput) {
        console.log(`Keeping ASIN value: "${asinInput.value}" for user convenience`);
    }
    
    // Reset the fetch button if it was in loading state
    setButtonLoading(false);
    
    console.log('Market data cleared successfully');
}

/**
 * Refresh market data status indicators from API
 */
function refreshMarketDataStatus() {
    // Use the same ASIN resolution strategy as fetchMarketData
    let asin = '';
    
    // 1. First try the displayed value in the Market Analysis tab
    const visibleAsinElement = document.querySelector('#marketAnalysis input[type="text"]');
    if (visibleAsinElement && visibleAsinElement.value && visibleAsinElement.value.trim()) {
        asin = visibleAsinElement.value.trim();
        console.log(`[STRATEGY 1] Found ASIN from visible input: "${asin}"`);
    }
    
    // 2. Look for any input with ID 'asin' that has a value
    if (!asin) {
        const allAsinInputs = document.querySelectorAll('#asin');
        for (const input of allAsinInputs) {
            if (input.value && input.value.trim()) {
                asin = input.value.trim();
                console.log(`[STRATEGY 2] Found ASIN from input with ID 'asin': "${asin}"`);
                break;
            }
        }
    }
    
    // 3. LAST RESORT: Just use the default ASIN directly
    if (!asin) {
        asin = 'B0BYS2D9CJ';
        console.log(`[STRATEGY 3] Using default hardcoded ASIN: "${asin}"`);
    }
    
    console.log(`[STATUS CHECK] Checking market data status for ASIN: "${asin}"`);
    
    // Show checking status
    updateStatusIndicator('priceHistoryStatus', 'loading');
    updateStatusIndicator('competitorStatus', 'loading');
    updateStatusIndicator('reviewsStatus', 'loading');
    
    // Fetch status from API
    console.log(`[STATUS API CALL] Fetching data status for ${asin}`);
    fetch(`http://localhost:5050/api/market/data-status?asin=${asin}`, {
        method: 'GET',
        headers: { 'Accept': 'application/json' }
    })
    .then(response => {
        console.log(`[STATUS API RESPONSE] Data status response: ${response.status}`);
        return response.json();
    })
    .then(data => {
        console.log('[STATUS API DATA] Market data status:', data);
        
        if (data.success) {
            // Update status indicators
            updateStatusIndicator('priceHistoryStatus', 
                data.price_history ? 'available' : 'unavailable');
                
            updateStatusIndicator('competitorStatus', 
                data.competitor_data ? 'available' : 'unavailable');
                
            updateStatusIndicator('reviewsStatus', 
                data.reviews_data ? 'available' : 'unavailable');
                
            // Enable/disable Analyze button based on data availability
            const analyzeButton = document.getElementById('analyzeButton');
            if (analyzeButton) {
                if (data.product_available) {
                    analyzeButton.disabled = false;
                    analyzeButton.querySelector('.button-text').textContent = 'Analyze Product';
                    console.log('[STATUS] Product data is available for analysis');
                } else {
                    analyzeButton.disabled = true;
                    analyzeButton.querySelector('.button-text').textContent = 'Data Unavailable';
                    console.log('[STATUS] Product data is NOT available for analysis');
                }
            }
        } else {
            // Handle error
            console.error('[STATUS API ERROR] Error checking data status:', data.message);
            
            // Reset status indicators
            updateStatusIndicator('priceHistoryStatus', 'error', data.message);
            updateStatusIndicator('competitorStatus', 'error', data.message);
            updateStatusIndicator('reviewsStatus', 'error', data.message);
        }
    })
    .catch(error => {
        console.error('[STATUS API SEQUENCE ERROR] Error fetching market data status:', error);
        
        // Reset status indicators on error
        updateStatusIndicator('priceHistoryStatus', 'error', error.message);
        updateStatusIndicator('competitorStatus', 'error', error.message);
        updateStatusIndicator('reviewsStatus', 'error', error.message);
    });
}

/**
 * Update status indicator for data collections
 * @param {string} id - ID of the status indicator element
 * @param {string} status - Status type (loading, available, unavailable, error, collected)
 * @param {string} message - Optional error message
 */
function updateStatusIndicator(id, status, message = '') {
    console.log(`Updating status indicator '${id}' to '${status}'`);
    
    // Find the indicator element
    const indicator = document.getElementById(id);
    if (!indicator) {
        console.error(`Status indicator element not found: ${id}`);
        return;
    }
    
    // Update the indicator based on status
    switch (status) {
        case 'loading':
            indicator.innerHTML = '<i class="fas fa-spinner fa-spin text-info"></i> Loading...';
            indicator.className = 'status loading';
            break;
            
        case 'available':
            indicator.innerHTML = '<i class="fas fa-check-circle text-success"></i> Available';
            indicator.className = 'status available';
            break;
            
        case 'unavailable':
            indicator.innerHTML = '<i class="fas fa-times-circle text-secondary"></i> Unavailable';
            indicator.className = 'status unavailable';
            break;
            
        case 'collected':
            indicator.innerHTML = '<i class="fas fa-check-circle text-success"></i> Collected';
            indicator.className = 'status collected';
            break;
            
        case 'not-collected':
            indicator.innerHTML = '<i class="fas fa-times-circle text-danger"></i> Not Collected';
            indicator.className = 'status not-collected';
            break;
            
        case 'error':
            const errorText = message ? `: ${message}` : '';
            indicator.innerHTML = `<i class="fas fa-exclamation-circle text-danger"></i> Error${errorText}`;
            indicator.className = 'status error';
            break;
            
        default:
            indicator.innerHTML = '<i class="fas fa-question-circle text-warning"></i> Unknown';
            indicator.className = 'status unknown';
    }
}

/**
 * Helper function to show alert
 */
function showAlert(message, type = 'info') {
    console.log(`Alert: ${message} (${type})`);
    
    // Create alert element
    const alertElement = document.createElement('div');
    alertElement.className = `alert alert-${type}`;
    alertElement.role = 'alert';
    alertElement.innerHTML = message;
    
    // Add alert to DOM
    const alertContainer = document.createElement('div');
    alertContainer.className = 'alert-container';
    alertContainer.appendChild(alertElement);
    document.body.appendChild(alertContainer);
    
    // Remove alert after 3 seconds
    setTimeout(() => {
        alertContainer.classList.add('fade-out');
        setTimeout(() => {
            document.body.removeChild(alertContainer);
        }, 300);
    }, 3000);
}

/**
 * Helper function to set button loading state
 */
function setButtonLoading(isLoading) {
    // Find the button using multiple possible selectors
    const fetchDataBtn = document.getElementById('fetchDataBtn') || 
                         document.querySelector('button.btn-primary[data-action="fetch"]') ||
                         document.querySelector('button.btn-primary');
                         
    if (!fetchDataBtn) {
        console.error('Fetch data button not found for setting loading state');
        return;
    }
    
    const btnContent = fetchDataBtn.querySelector('.btn-content') || fetchDataBtn;
    const btnLoading = fetchDataBtn.querySelector('.btn-loading');
    
    if (isLoading) {
        fetchDataBtn.disabled = true;
        if (btnContent) btnContent.style.display = 'none';
        if (btnLoading) btnLoading.style.display = 'inline-block';
        
        // If no loading spinner exists, create one
        if (!btnLoading) {
            const spinner = document.createElement('span');
            spinner.className = 'btn-loading spinner-border spinner-border-sm';
            spinner.setAttribute('role', 'status');
            spinner.setAttribute('aria-hidden', 'true');
            fetchDataBtn.appendChild(spinner);
        }
    } else {
        fetchDataBtn.disabled = false;
        if (btnContent) btnContent.style.display = 'inline-block';
        if (btnLoading) btnLoading.style.display = 'none';
    }
}

/**
 * Clear all charts
 */
function clearCharts() {
    try {
        // Initialize chart namespace if it doesn't exist
        window.marketCharts = window.marketCharts || {};
        
        // Safely destroy price history chart
        if (window.marketCharts.priceHistory) {
            if (typeof window.marketCharts.priceHistory.destroy === 'function') {
                window.marketCharts.priceHistory.destroy();
            }
            window.marketCharts.priceHistory = null;
        }
        
        // Safely destroy competitor chart
        if (window.marketCharts.competitor) {
            if (typeof window.marketCharts.competitor.destroy === 'function') {
                window.marketCharts.competitor.destroy();
            }
            window.marketCharts.competitor = null;
        }
        
        // Safely destroy reviews chart
        if (window.marketCharts.reviews) {
            if (typeof window.marketCharts.reviews.destroy === 'function') {
                window.marketCharts.reviews.destroy();
            }
            window.marketCharts.reviews = null;
        }
        
        console.log('All charts cleared successfully');
    } catch (error) {
        console.error('Error clearing charts:', error);
    }
}

/**
 * Helper function to generate star rating HTML
 */
function generateStarRating(rating) {
    const fullStars = Math.floor(rating);
    const halfStar = rating % 1 >= 0.5;
    const emptyStars = 5 - fullStars - (halfStar ? 1 : 0);
    
    let starsHtml = '';
    
    // Add full stars
    for (let i = 0; i < fullStars; i++) {
        starsHtml += '<i class="fas fa-star text-warning"></i>';
    }
    
    // Add half star if needed
    if (halfStar) {
        starsHtml += '<i class="fas fa-star-half-alt text-warning"></i>';
    }
    
    // Add empty stars
    for (let i = 0; i < emptyStars; i++) {
        starsHtml += '<i class="far fa-star text-warning"></i>';
    }
    
    return starsHtml;
}

/**
 * Helper function to calculate average
 */
function calculateAverage(values) {
    if (!values || values.length === 0) return 0;
    const sum = values.reduce((acc, val) => acc + val, 0);
    return sum / values.length;
}

/**
 * Helper function to calculate volatility
 */
function calculateVolatility(values) {
    if (!values || values.length <= 1) return 0;
    
    const avg = calculateAverage(values);
    const squaredDiffs = values.map(val => Math.pow(val - avg, 2));
    const variance = calculateAverage(squaredDiffs);
    const stdDev = Math.sqrt(variance);
    
    // Return as percentage of mean
    return (stdDev / avg) * 100;
}

/**
 * Process competitor data from search results
 * @param {Object} searchResults - The results from product search
 * @param {string} currentAsin - The ASIN of the current product 
 * @param {number} currentPrice - The current price of the product
 * @returns {Object} - Processed competitor data
 */
function processCompetitorData(searchResults, currentAsin, currentPrice) {
    console.log(`Processing competitor data for ${currentAsin}`);
    console.log('Search results:', searchResults);
    
    // Handle different possible API response formats
    let searchProducts = null;
    
    // Check for various possible structures in the API response
    if (searchResults && searchResults.searchProducts) {
        // Original expected format
        searchProducts = searchResults.searchProducts;
        console.log('Found searchProducts in expected location');
    } else if (searchResults && searchResults.searchProductDetails) {
        // Alternative format directly in the results
        searchProducts = searchResults.searchProductDetails;
        console.log('Found searchProductDetails in results');
    } else if (searchResults && searchResults.products) {
        // Another possible format
        searchProducts = searchResults.products;
        console.log('Found products array in results');
    } else if (searchResults && searchResults.searchResult && searchResults.searchResult.products) {
        // Yet another possible nested format
        searchProducts = searchResults.searchResult.products;
        console.log('Found products in searchResult');
    } else if (searchResults && Array.isArray(searchResults)) {
        // Handle case where the response is directly an array of products
        searchProducts = searchResults;
        console.log('Search results is directly an array');
    }
    
    // If we couldn't find any product array, check if we have a single product object
    if (!searchProducts && searchResults && searchResults.asin && !currentAsin.includes(searchResults.asin)) {
        // If the response is a single product that isn't the current one, wrap it in an array
        searchProducts = [searchResults];
        console.log('Search results contains a single product, using it');
    }
    
    if (!searchProducts || !Array.isArray(searchProducts) || searchProducts.length === 0) {
        console.error('[COMPETITOR DATA ERROR] Invalid search results format');
        return { success: false, message: "Invalid search results" };
    }
    
    // Use the existing function to generate competitive position from search
    return generateCompetitivePositionFromSearch(
        {
            asin: currentAsin,
            searchProductDetails: searchProducts,
            keyword: searchResults.keyword || searchResults.searchTerm || 'Unknown'
        }, 
        currentPrice
    );
}

/**
 * Track a Market Analysis activity for Recent Activities display
 */
function trackMarketAnalysisActivity(activityType, asin, details = {}) {
  if (window.UsageTracker) {
    window.UsageTracker.trackFeature('marketAnalysis', {
      asin: asin,
      timestamp: new Date().toISOString(),
      activityType: activityType,
      productType: details.productType || 'Amazon Product',
      competitorCount: details.competitorCount,
      reviewCount: details.reviewCount,
      pricePoints: details.pricePoints,
      action: activityType
    });
    console.log(`Tracked Market Analysis activity: ${activityType}`);
  }
}

/**
 * Generate price history from product details
 * - Eliminating all synthetic data generation
 */
function generatePriceHistoryFromProductDetails(productDetails) {
    console.log('Processing product details for price history');
    console.log('Raw product details:', productDetails);
    
    // Extract current price
    let currentPrice = 0;
    try {
        if (productDetails.price) {
            // Handle different price format possibilities
            if (typeof productDetails.price === 'string') {
                currentPrice = parseFloat(productDetails.price.replace(/[^0-9.]/g, ''));
            } else {
                currentPrice = parseFloat(productDetails.price);
            }
        }
    } catch (error) {
        console.error('Error parsing current price:', error);
    }
    
    // Return data we actually have from the API, preserving the productDetails array
    return {
        success: true,
        asin: productDetails.asin || 'Unknown',
        title: productDetails.title || productDetails.productTitle || 'Unknown Product',
        current_price: currentPrice,
        price_history: {
            dates: [],
            prices: [],
            avg_prices: []
        },
        price_insights: {
            messages: ["Insufficient historical price data available."],
            trend: "unknown",
            metrics: {
                volatility: 0,
                min_price: currentPrice,
                max_price: currentPrice,
                avg_price: currentPrice
            }
        },
        // Preserve important direct API fields for demand forecasting
        productDetails: productDetails.productDetails || [],
        prime: productDetails.prime || false,
        pastSales: productDetails.pastSales || null,
        categories: productDetails.categories || [],
        categoryTree: productDetails.categoryTree || [],
        bestSellerRank: productDetails.bestSellerRank || null,
        salesRank: productDetails.salesRank || null
    };
}

/**
 * Generate competitive position data from search results
 * This new function extracts competitor data from search results instead of offers
 */
function generateCompetitivePositionFromSearch(searchData, currentPrice) {
    console.log(`Parsing competitive position from search product data`);
    console.log('Raw search data:', searchData);
    
    // Ensure searchData is an object
    if (!searchData || typeof searchData !== 'object') {
        console.error(`Invalid searchData parameter:`, searchData);
        return { success: false, message: "Invalid search data parameter" };
    }
    
    // Extract product details from search results
    let competitorPrices = [];
    let competitorProducts = [];
    let competitorCount = 0;
    let searchProductDetails = searchData.searchProductDetails || [];
    
    // Ensure searchProductDetails is an array
    if (!Array.isArray(searchProductDetails)) {
        console.error(`searchProductDetails is not an array:`, searchProductDetails);
        return { success: false, message: "Search product details not in expected format" };
    }
    
    try {
        // Get all products with valid prices
        searchProductDetails.forEach((product, index) => {
            // Skip if it's the current product
            if (product.asin === searchData.asin) {
                console.log(`Skipping current product at index ${index}`);
                return;
            }
            
            // Try to extract price - handle different formats
            let price = null;
            if (product.price) {
                // Direct price field
                price = extractPrice(product.price);
            } else if (product.priceDto && product.priceDto.priceValue) {
                // Price from DTO
                price = extractPrice(product.priceDto.priceValue);
            } else if (product.displayPrice) {
                // Display price
                price = extractPrice(product.displayPrice);
            } else if (product.priceRange && product.priceRange.min) {
                // Price range minimum
                price = extractPrice(product.priceRange.min);
            }
            
            // Only add products with valid prices
            if (price && !isNaN(price) && price > 0) {
                competitorPrices.push(price);
                competitorProducts.push({
                    asin: product.asin || 'unknown',
                    title: product.title || product.productTitle || 'Unknown Product',
                    price: price
                });
            }
        });
        
        competitorCount = competitorPrices.length;
        console.log(`Found ${competitorCount} competitor products with valid prices`);
    } catch (error) {
        console.error(`[SEARCH PARSING ERROR] Could not parse search results:`, error);
        return { success: false, message: "Error parsing search data" };
    }
    
    // If no competitor prices found, return error
    if (competitorPrices.length === 0) {
        console.error("No competitor prices found in the search results");
        return { success: false, message: "No competitor prices found" };
    }
    
    // Calculate market average price
    const avgMarketPrice = competitorPrices.reduce((sum, price) => sum + price, 0) / competitorPrices.length;
    
    // Calculate position percentile
    // If currentPrice isn't passed, try to get it from the search data
    if (!currentPrice || isNaN(currentPrice) || currentPrice <= 0) {
        if (searchData.price) {
            currentPrice = extractPrice(searchData.price);
        } else {
            // Set a default price if we can't extract it
            currentPrice = avgMarketPrice;
            console.warn(`No valid current price found, using average market price: ${currentPrice}`);
        }
    }
    
    const lowerPrices = competitorPrices.filter(p => p < currentPrice).length;
    const percentile = Math.round((lowerPrices / competitorPrices.length) * 100);
    
    // Generate price distribution brackets
    const minPrice = Math.min(...competitorPrices);
    const maxPrice = Math.max(...competitorPrices);
    const priceDiff = maxPrice - minPrice;
    const bracketSize = priceDiff > 0 ? priceDiff / 5 : 1;
    
    const brackets = [];
    for (let i = 0; i < 5; i++) {
        const bracketMin = minPrice + (i * bracketSize);
        const bracketMax = minPrice + ((i + 1) * bracketSize);
        
        // Count competitors in this range
        const count = competitorPrices.filter(p => p >= bracketMin && (i === 4 ? p <= bracketMax : p < bracketMax)).length;
        
        brackets.push({
            range: `$${bracketMin.toFixed(2)} - $${bracketMax.toFixed(2)}`,
            count: count
        });
    }
    
    // Generate insights based on percentile
    let insights = [];
    if (percentile < 25) {
        insights = [
            "Your price is lower than most competitors.",
            "Consider testing slight price increases.",
            "Highlight unique value propositions beyond price."
        ];
    } else if (percentile < 50) {
        insights = [
            "Your price is in the lower mid-range of the market.",
            "Good value positioning relative to competitors.",
            "Monitor competitor pricing strategies."
        ];
    } else if (percentile < 75) {
        insights = [
            "Your price is in the upper mid-range of the market.",
            "Emphasize product quality and features.",
            "Consider promotional offers to increase competitiveness."
        ];
    } else {
        insights = [
            "Your price is higher than most competitors.",
            "Ensure product quality justifies premium pricing.",
            "Highlight premium features and benefits."
        ];
    }
    
    // Add top competitor products
    const topCompetitors = competitorProducts
        .sort((a, b) => a.price - b.price)
        .slice(0, 5);
    
    return {
        success: true,
        asin: searchData.asin || 'Unknown',
        current_price: currentPrice,
        keyword: searchData.keyword || 'Unknown',
        competitive_position: {
            percentile: percentile,
            avg_market_price: parseFloat(avgMarketPrice.toFixed(2)),
            competitor_count: competitorCount
        },
        price_distribution: {
            brackets: brackets
        },
        pricing_context: {
            insights: insights,
            top_competitors: topCompetitors
        }
    };
}

/**
 * Helper function to extract price from various formats
 */
function extractPrice(priceStr) {
    if (!priceStr) return null;
    
    // If already a number, return it
    if (typeof priceStr === 'number') return priceStr;
    
    try {
        // Convert string price to number
        const price = parseFloat(String(priceStr).replace(/[^0-9.]/g, ''));
        return !isNaN(price) ? price : null;
    } catch (error) {
        console.error(`Error extracting price from ${priceStr}:`, error);
        return null;
    }
}

/**
 * Visualize demand forecast data instead of price history
 */
function visualizePriceHistory(data) {
    console.log('Visualizing demand forecast data');
    console.log('Raw product details:', data);
    
    // Find the price history container (we're repurposing it for demand data)
    const demandContainer = document.getElementById('priceHistoryChart');
    
    if (!demandContainer) {
        console.error('Demand forecast container not found');
        return;
    }
    
    // Clear container
    demandContainer.innerHTML = '';
    
    // Extract product details
    const productTitle = data.title || 'Unknown Product';
    const currentPrice = data.current_price || 0;
    const asin = data.asin || 'Unknown';
    
    // Create section for demand insights
    const demandInsightsContainer = document.createElement('div');
    demandInsightsContainer.className = 'demand-forecast-container';
    
    // Set basic styles
    demandInsightsContainer.style.padding = '15px';
    demandInsightsContainer.style.borderRadius = '8px';
    demandInsightsContainer.style.backgroundColor = 'rgba(0,0,0,0.1)';
    
    // Extract demand-related data from the product details
    // Log important information for debugging
    console.log('Extracting best seller rank from:', data);
    if (data.productDetails) {
        console.log('productDetails available:', data.productDetails.length, 'items');
    }
    
    const bestSellerRank = extractBestSellerRank(data);
    console.log('Extracted best seller rank:', bestSellerRank);
    
    const categories = extractCategories(data);
    console.log('Extracted categories:', categories);
    
    const dateFirstAvailable = extractDateFirstAvailable(data);
    console.log('Extracted date first available:', dateFirstAvailable);
    
    const pastSales = extractPastSales(data);
    console.log('Extracted past sales:', pastSales);
    
    const prime = data.prime || false;
    
    // Calculate product age in days
    const productAgeDays = calculateProductAge(dateFirstAvailable);
    
    // Calculate category seasonality score (mock implementation)
    const seasonalityData = calculateCategorySeasonality(categories);
    
    // Create HTML for the demand insights
    let demandInsightsHTML = `
        <h3 style="margin-top: 0;">Demand Forecast Insights</h3>
        
        <div class="demand-metrics-grid" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px;">
            <div class="metric-card" style="background-color: rgba(0,0,0,0.15); padding: 15px; border-radius: 8px;">
                <div class="metric-value" style="font-size: 1.5rem; font-weight: bold; color: #4299e1;">${bestSellerRank.mainRank || 'N/A'}</div>
                <div class="metric-label" style="font-size: 0.9rem; opacity: 0.8;">Best Seller Rank</div>
                <div class="metric-context" style="font-size: 0.8rem; margin-top: 5px;">${bestSellerRank.category || ''}</div>
            </div>
            
            <div class="metric-card" style="background-color: rgba(0,0,0,0.15); padding: 15px; border-radius: 8px;">
                <div class="metric-value" style="font-size: 1.5rem; font-weight: bold; color: #48bb78;">${pastSales || 'Unknown'}</div>
                <div class="metric-label" style="font-size: 0.9rem; opacity: 0.8;">Recent Sales</div>
                <div class="metric-context" style="font-size: 0.8rem; margin-top: 5px;">Past 30 days estimate</div>
            </div>
            
            <div class="metric-card" style="background-color: rgba(0,0,0,0.15); padding: 15px; border-radius: 8px;">
                <div class="metric-value" style="font-size: 1.5rem; font-weight: bold; color: #ed8936;">${productAgeDays} days</div>
                <div class="metric-label" style="font-size: 0.9rem; opacity: 0.8;">Product Age</div>
                <div class="metric-context" style="font-size: 0.8rem; margin-top: 5px;">Since ${dateFirstAvailable || 'Unknown'}</div>
            </div>
            
            <div class="metric-card" style="background-color: rgba(0,0,0,0.15); padding: 15px; border-radius: 8px;">
                <div class="metric-value" style="font-size: 1.5rem; font-weight: bold; color: ${prime ? '#4299e1' : '#a0aec0'};">${prime ? 'Yes' : 'No'}</div>
                <div class="metric-label" style="font-size: 0.9rem; opacity: 0.8;">Prime Eligible</div>
                <div class="metric-context" style="font-size: 0.8rem; margin-top: 5px;">${prime ? 'Faster shipping available' : 'Standard shipping only'}</div>
            </div>
        </div>
        
        <div class="category-seasonality" style="margin-bottom: 20px;">
            <h4 style="margin-bottom: 10px;">Category Seasonality</h4>
            <div class="seasonality-chart" style="height: 80px; display: flex; align-items: center; margin-bottom: 10px;">
                ${generateSeasonalityChart(seasonalityData)}
            </div>
            <div class="seasonality-legend" style="display: flex; justify-content: space-between; font-size: 0.8rem;">
                <span>Jan</span>
                <span>Feb</span>
                <span>Mar</span>
                <span>Apr</span>
                <span>May</span>
                <span>Jun</span>
                <span>Jul</span>
                <span>Aug</span>
                <span>Sep</span>
                <span>Oct</span>
                <span>Nov</span>
                <span>Dec</span>
            </div>
        </div>
        
        <div class="rank-details" style="margin-bottom: 20px;">
            <h4 style="margin-bottom: 10px;">Sales Rank Details</h4>
            ${generateRankDetails(bestSellerRank)}
        </div>
        
        <div class="demand-forecast-insights">
            <h4 style="margin-bottom: 10px;">Demand Insights</h4>
            <ul style="padding-left: 20px; margin-top: 5px;">
                ${generateDemandInsights(bestSellerRank, categories, productAgeDays, seasonalityData)}
            </ul>
        </div>
    `;
    
    // Set the HTML content
    demandInsightsContainer.innerHTML = demandInsightsHTML;
    
    // Append to container
    demandContainer.appendChild(demandInsightsContainer);
    
    // Track price history analysis activity
    if (data && data.asin) {
        trackMarketAnalysisActivity('Price History Analysis', data.asin, {
            pricePoints: data.history ? data.history.length : 0
        });
    }
}

/**
 * Extract Best Seller Rank from product details
 */
function extractBestSellerRank(data) {
    try {
        // Try to find best seller rank in different possible locations
        let rankText = '';
        let rankObj = { mainRank: 'N/A', category: 'N/A', subRank: null };
        
        // Check for productDetails array (from the response we can see this is the correct structure)
        if (data && data.productDetails && Array.isArray(data.productDetails)) {
            // Check for the "Best Sellers Rank" field in productDetails array
            const bsrDetail = data.productDetails.find(detail => 
                detail.name && detail.name.includes('Best Sellers Rank'));
            
            if (bsrDetail && bsrDetail.value) {
                rankText = bsrDetail.value;
                console.log("Found Best Sellers Rank:", rankText);
            }
        }
        
        // Legacy checks for other API formats
        if (!rankText) {
            if (data.bestSellerRank) {
                rankText = data.bestSellerRank;
            } else if (data.salesRank) {
                rankText = data.salesRank;
            }
        }
        
        // Parse the rank text
        if (rankText) {
            // Extract main rank number (e.g., "#18 in Arts, Crafts & Sewing")
            const mainRankMatch = rankText.match(/#(\d+)\s+in\s+([^(]+)/);
            if (mainRankMatch) {
                rankObj.mainRank = `#${mainRankMatch[1]}`;
                rankObj.category = mainRankMatch[2].trim();
            }
            
            // Extract sub-rank if available (e.g., "#1 in Drawing Pencils")
            const subRankMatch = rankText.match(/#(\d+)\s+in\s+([^(]+)(?!\()/g);
            if (subRankMatch && subRankMatch.length > 1) {
                const subMatch = subRankMatch[1].match(/#(\d+)\s+in\s+([^(]+)/);
                if (subMatch) {
                    rankObj.subRank = {
                        rank: `#${subMatch[1]}`,
                        category: subMatch[2].trim()
                    };
                }
            }
        }
        
        return rankObj;
    } catch (error) {
        console.error('Error extracting Best Seller Rank:', error);
        return { mainRank: 'N/A', category: 'N/A', subRank: null };
    }
}

/**
 * Extract categories from product details
 */
function extractCategories(data) {
    try {
        if (data.categories && Array.isArray(data.categories)) {
            return data.categories;
        } else if (data.categoryTree && Array.isArray(data.categoryTree)) {
            return data.categoryTree.map(cat => cat.name || cat);
        }
        return ['Unknown Category'];
    } catch (error) {
        console.error('Error extracting categories:', error);
        return ['Unknown Category'];
    }
}

/**
 * Extract date first available from product details
 */
function extractDateFirstAvailable(data) {
    try {
        // Primary check: Look in the productDetails array
        if (data && data.productDetails && Array.isArray(data.productDetails)) {
            // Find date first available in productDetails array
            const dateDetail = data.productDetails.find(detail => 
                detail.name && (
                    detail.name.includes('Date First Available') || 
                    detail.name.includes('Release Date')
                )
            );
            
            if (dateDetail && dateDetail.value) {
                console.log("Found Date First Available:", dateDetail.value);
                return dateDetail.value;
            }
        }
        
        // Fallback checks
        if (data.dateFirstAvailable) {
            return data.dateFirstAvailable;
        } else if (data.releaseDate) {
            return data.releaseDate;
        }
        
        return 'Unknown';
    } catch (error) {
        console.error('Error extracting date first available:', error);
        return 'Unknown';
    }
}

/**
 * Extract past sales information
 */
function extractPastSales(data) {
    try {
        // Check for sales data in the API response
        if (data && data.pastSales) {
            console.log("Found pastSales:", data.pastSales);
            return data.pastSales;
        }
        
        // Directly check for "600+ bought in past month" field
        // This appears to be available in the console output
        if (data && typeof data === 'object') {
            for (const key in data) {
                if (typeof data[key] === 'string' && 
                    data[key].includes('bought in past month')) {
                    console.log(`Found sales in ${key}:`, data[key]);
                    return data[key];
                }
            }
        }
        
        return 'Unknown';
    } catch (error) {
        console.error('Error extracting past sales:', error);
        return 'Unknown';
    }
}

/**
 * Calculate product age in days
 */
function calculateProductAge(dateString) {
    try {
        if (dateString === 'Unknown') return 'Unknown';
        
        const date = new Date(dateString);
        if (isNaN(date.getTime())) return 'Unknown';
        
        const today = new Date();
        const diffTime = Math.abs(today - date);
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
        
        return diffDays;
    } catch (error) {
        console.error('Error calculating product age:', error);
        return 'Unknown';
    }
}

/**
 * Calculate category seasonality
 */
function calculateCategorySeasonality(categories) {
    // This is a simplified mock implementation
    // In a real implementation, this would use historical data or industry benchmarks
    
    // Default seasonality (flat)
    let seasonality = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50];
    
    // Apply different seasonality patterns based on category
    if (categories.some(cat => cat.includes('Art') || cat.includes('Craft'))) {
        // Art supplies peak during back-to-school and holiday seasons
        seasonality = [40, 45, 50, 55, 60, 55, 50, 80, 85, 70, 90, 95];
    } else if (categories.some(cat => cat.includes('Electronics'))) {
        // Electronics peak during holiday season and summer
        seasonality = [50, 40, 35, 40, 50, 70, 75, 70, 60, 70, 90, 100];
    } else if (categories.some(cat => cat.includes('Kitchen'))) {
        // Kitchen items peak during holiday season
        seasonality = [60, 50, 45, 50, 60, 65, 70, 70, 65, 70, 85, 100];
    } else if (categories.some(cat => cat.includes('Toy'))) {
        // Toys peak during holiday season
        seasonality = [35, 30, 35, 40, 45, 50, 55, 60, 65, 70, 90, 100];
    } else if (categories.some(cat => cat.includes('Garden'))) {
        // Garden items peak during spring and summer
        seasonality = [30, 40, 70, 90, 100, 95, 80, 70, 60, 50, 40, 30];
    }
    
    // Get current month for highlighting
    const currentMonth = new Date().getMonth(); // 0-11
    
    return {
        values: seasonality,
        currentMonth: currentMonth
    };
}

/**
 * Generate HTML for seasonality chart
 */
function generateSeasonalityChart(seasonalityData) {
    const { values, currentMonth } = seasonalityData;
    
    // Create the bars
    let chartHTML = '';
    
    values.forEach((value, index) => {
        const height = Math.max(10, value * 0.7); // Scale value to reasonable height (max 70px)
        const isCurrentMonth = index === currentMonth;
        
        // Determine color based on value height
        let color;
        if (value >= 80) color = 'rgba(72, 187, 120, 0.7)'; // Green for high demand
        else if (value >= 50) color = 'rgba(237, 137, 54, 0.7)'; // Orange for medium
        else color = 'rgba(229, 62, 62, 0.5)'; // Red for low
        
        // Highlight current month
        const border = isCurrentMonth ? '2px solid white' : 'none';
        const borderRadius = '4px 4px 0 0';
        
        chartHTML += `<div style="flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: flex-end; height: 100%;">
            <div style="height: ${height}px; width: 80%; background-color: ${color}; border: ${border}; border-radius: ${borderRadius};"></div>
        </div>`;
    });
    
    return chartHTML;
}

/**
 * Generate HTML for rank details
 */
function generateRankDetails(bestSellerRank) {
    let html = '';
    
    if (bestSellerRank.mainRank !== 'N/A') {
        html += `<div class="rank-item" style="margin-bottom: 8px;">
            <span style="font-weight: bold; color: #4299e1;">${bestSellerRank.mainRank}</span> in 
            <span style="font-style: italic;">${bestSellerRank.category}</span>
        </div>`;
    }
    
    if (bestSellerRank.subRank) {
        html += `<div class="rank-item" style="margin-bottom: 8px;">
            <span style="font-weight: bold; color: #48bb78;">${bestSellerRank.subRank.rank}</span> in 
            <span style="font-style: italic;">${bestSellerRank.subRank.category}</span>
        </div>`;
    }
    
    if (html === '') {
        html = '<p>No sales rank data available</p>';
    }
    
    return html;
}

/**
 * Generate demand insights based on available data
 */
function generateDemandInsights(bestSellerRank, categories, productAge, seasonalityData) {
    const insights = [];
    
    // Sales rank insights
    if (bestSellerRank.mainRank !== 'N/A') {
        const rankNum = parseInt(bestSellerRank.mainRank.replace('#', ''));
        if (rankNum <= 20) {
            insights.push('Very high demand product based on top 20 sales rank.');
        } else if (rankNum <= 100) {
            insights.push('Strong demand product based on top 100 sales rank.');
        } else if (rankNum <= 1000) {
            insights.push('Moderate demand product based on sales rank.');
        } else {
            insights.push('Lower demand product based on sales rank.');
        }
    }
    
    // Category insights
    if (categories.length > 0) {
        const mainCategory = categories[0];
        if (mainCategory.includes('Art') || mainCategory.includes('Craft')) {
            insights.push('Art supplies typically show increased demand during back-to-school season (August-September) and holiday season (November-December).');
        } else if (mainCategory.includes('Electronics')) {
            insights.push('Electronics typically peak during holiday season (November-December) and new product launch windows.');
        }
    }
    
    // Product age insights
    if (productAge !== 'Unknown') {
        if (productAge < 30) {
            insights.push('New product (less than 30 days old) - typically experiences higher interest and sales velocity.');
        } else if (productAge < 90) {
            insights.push('Recently launched product (less than 3 months old) - still in early adoption phase.');
        } else if (productAge > 365) {
            insights.push('Mature product (more than 1 year old) - likely has stable demand pattern.');
        }
    }
    
    // Seasonality insights
    const currentMonth = new Date().getMonth();
    const nextMonth = (currentMonth + 1) % 12;
    const currentSeasonality = seasonalityData.values[currentMonth];
    const nextMonthSeasonality = seasonalityData.values[nextMonth];
    
    const monthNames = ['January', 'February', 'March', 'April', 'May', 'June', 
                        'July', 'August', 'September', 'October', 'November', 'December'];
    
    if (nextMonthSeasonality > currentSeasonality + 10) {
        insights.push(`Demand expected to increase in ${monthNames[nextMonth]} based on seasonal trends.`);
    } else if (nextMonthSeasonality < currentSeasonality - 10) {
        insights.push(`Demand expected to decrease in ${monthNames[nextMonth]} based on seasonal trends.`);
    } else {
        insights.push(`Demand expected to remain stable through ${monthNames[nextMonth]}.`);
    }
    
    // Return insights as HTML list items
    return insights.map(insight => `<li>${insight}</li>`).join('');
}

/**
 * Visualize competitor data
 */
function visualizeCompetitorData(data) {
    console.log('Visualizing competitor data:', data);
    console.log('Raw competitor data:', data);
    
    // Initialize the marketCharts object if it doesn't exist
    if (!window.marketCharts) {
        window.marketCharts = {};
    }
    
    // Find the competitor container
    const competitorContainer = document.getElementById('competitorChart');
    
    if (!competitorContainer) {
        console.error('Competitor container not found');
        return;
    }
    
    // Clear container
    competitorContainer.innerHTML = '';
    
    // Get competitive position and price distribution data directly from API
    const competitivePosition = data.competitive_position || {};
    const priceDistribution = data.price_distribution || {};
    const pricingContext = data.pricing_context || {};
    
    // Create enhanced position meter
    const positionMeter = document.createElement('div');
    positionMeter.className = 'position-meter-container';
    positionMeter.style.marginBottom = '30px';
    
    const percentile = competitivePosition.percentile || 0;
    const avgMarketPrice = competitivePosition.avg_market_price || 0;
    const competitorCount = competitivePosition.competitor_count || 0;
    const currentPrice = data.current_price || 0;
    
    // Calculate price position stats
    const priceDiff = currentPrice - avgMarketPrice;
    const priceDiffPercent = avgMarketPrice > 0 ? (priceDiff / avgMarketPrice) * 100 : 0;
    const priceDiffFormatted = priceDiff >= 0 ? `+$${priceDiff.toFixed(2)}` : `-$${Math.abs(priceDiff).toFixed(2)}`;
    const priceDiffPercentFormatted = priceDiffPercent >= 0 ? `+${priceDiffPercent.toFixed(1)}%` : `-${Math.abs(priceDiffPercent).toFixed(1)}%`;
    
    // Create classes based on percentile
    let positionClass = 'medium';
    let positionMessage = 'mid-range price';
    
    if (percentile < 25) {
        positionClass = 'low';
        positionMessage = 'lower than most competitors';
    } else if (percentile >= 75) {
        positionClass = 'high';
        positionMessage = 'higher than most competitors';
    }
    
    positionMeter.innerHTML = `
        <div class="price-position-header">
            <h4>Price Position</h4>
            <div class="price-badge price-badge-${positionClass}">${percentile}th percentile</div>
        </div>
        
        <div class="price-position-summary">
            <div class="your-price-container">
                <span class="your-price-value">$${currentPrice.toFixed(2)}</span>
                <span class="your-price-label">Your Price</span>
                <span class="price-difference ${priceDiffPercent >= 0 ? 'text-danger' : 'text-success'}">
                    ${priceDiffFormatted} (${priceDiffPercentFormatted}) vs. market average
                </span>
            </div>
        </div>
        
        <div class="price-position-meter ${positionClass}">
            <div class="meter-sections">
                <div class="meter-section low"></div>
                <div class="meter-section medium"></div>
                <div class="meter-section high"></div>
            </div>
            <div class="position-indicator" style="left: ${percentile}%">
                <div class="indicator-dot"></div>
                <div class="indicator-line"></div>
            </div>
            <div class="position-labels">
                <span>Lower Priced</span>
                <span>Higher Priced</span>
            </div>
        </div>
        
        <p class="position-description">Your price is <strong>${positionMessage}</strong> (${percentile}% of competitors have lower prices)</p>
        
        <div class="competitor-stats">
            <div class="stat-item">
                <span class="stat-value">$${avgMarketPrice.toFixed(2)}</span>
                <span class="stat-label">Market Average</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">${competitorCount}</span>
                <span class="stat-label">Competitors</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">${data.keyword ? data.keyword.substring(0, 25) + (data.keyword.length > 25 ? '...' : '') : "N/A"}</span>
                <span class="stat-label">Search Term</span>
            </div>
        </div>
    `;
    
    competitorContainer.appendChild(positionMeter);
    
    // Create price distribution chart if data is available
    if (priceDistribution.brackets && priceDistribution.brackets.length > 0) {
        const chartContainer = document.createElement('div');
        chartContainer.className = 'price-distribution-container';
        chartContainer.style.height = '180px';
        chartContainer.innerHTML = '<h4>Price Distribution</h4><canvas id="priceDistributionCanvas"></canvas>';
        competitorContainer.appendChild(chartContainer);
        
        const canvas = document.getElementById('priceDistributionCanvas');
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        
        const brackets = priceDistribution.brackets;
        const ranges = brackets.map(b => b.range);
        const counts = brackets.map(b => b.count);
        
        // Highlight the bracket containing the current price
        const backgroundColors = brackets.map((bracket, index) => {
            // Parse the min and max from the range string
            const rangeStr = bracket.range;
            const [minStr, maxStr] = rangeStr.split(' - ');
            const min = parseFloat(minStr.replace('$', ''));
            const max = parseFloat(maxStr.replace('$', ''));
            
            // Check if current price is in this bracket
            if (currentPrice >= min && currentPrice <= max) {
                return 'rgba(66, 153, 225, 0.8)'; // Highlighted
            }
            return 'rgba(66, 153, 225, 0.5)'; // Normal
        });
        
        const ctx = canvas.getContext('2d');
        
        // Check if Chart.js is available
        if (typeof Chart === 'undefined') {
            console.error('Chart.js is not available. Cannot create price distribution chart.');
            chartContainer.innerHTML += '<p class="error-message">Chart library not loaded. Cannot display price distribution.</p>';
            return;
        }
        
        window.marketCharts.competitor = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ranges,
                datasets: [{
                    label: 'Number of Products',
                    data: counts,
                    backgroundColor: backgroundColors,
                    borderColor: 'rgba(66, 153, 225, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                layout: {
                    padding: {
                        top: 5,
                        right: 20,
                        bottom: 5,
                        left: 10
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Number of Products'
                        },
                        ticks: {
                            maxTicksLimit: 5
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Price Range'
                        },
                        ticks: {
                            maxRotation: 45,
                            minRotation: 45
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Price Distribution in Market',
                        font: {
                            size: 14
                        }
                    },
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            title: function(context) {
                                return context[0].label;
                            },
                            label: function(context) {
                                return `${context.raw} products in this range`;
                            },
                            afterLabel: function(context) {
                                const rangeStr = context.label;
                                const [minStr, maxStr] = rangeStr.split(' - ');
                                const min = parseFloat(minStr.replace('$', ''));
                                const max = parseFloat(maxStr.replace('$', ''));
                                
                                if (currentPrice >= min && currentPrice <= max) {
                                    return `Your price: $${currentPrice.toFixed(2)} (in this range)`;
                                }
                                return '';
                            }
                        }
                    }
                }
            }
        });
    }
    
    // Add top competitors table if available
    if (pricingContext.top_competitors && pricingContext.top_competitors.length > 0) {
        const topCompetitorsContainer = document.createElement('div');
        topCompetitorsContainer.className = 'top-competitors-container';
        topCompetitorsContainer.style.marginTop = '20px';
        
        let tableHtml = `
            <h4>Top Competitors by Price</h4>
            <div class="table-container" style="max-height: 180px; overflow-y: auto;">
                <table class="competitors-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Product</th>
                            <th>Price</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        // Add top competitors to the table
        pricingContext.top_competitors.forEach((competitor, index) => {
            // Truncate title to 50 characters
            const title = competitor.title.length > 50 ? 
                competitor.title.substring(0, 50) + '...' : 
                competitor.title;
                
            tableHtml += `
                <tr>
                    <td>${index + 1}</td>
                    <td title="${competitor.title}">${title}</td>
                    <td>$${competitor.price.toFixed(2)}</td>
                </tr>
            `;
        });
        
        // Close the table
        tableHtml += `
                    </tbody>
                </table>
            </div>
        `;
        
        topCompetitorsContainer.innerHTML = tableHtml;
        competitorContainer.appendChild(topCompetitorsContainer);
    }
    
    // Add pricing insights if available
    if (pricingContext.insights && pricingContext.insights.length > 0) {
        const insightsContainer = document.createElement('div');
        insightsContainer.className = 'competitor-insights';
        insightsContainer.style.marginTop = '20px';
        
        let insightsHtml = '<h4>Market Insights</h4><ul>';
        pricingContext.insights.forEach(insight => {
            insightsHtml += `<li>${insight}</li>`;
        });
        insightsHtml += '</ul>';
        
        insightsContainer.innerHTML = insightsHtml;
        competitorContainer.appendChild(insightsContainer);
    }
    
    // Track competitor analysis activity
    if (data && data.asin) {
        trackMarketAnalysisActivity('Competitor Analysis', data.asin, {
            competitorCount: data.competitors ? data.competitors.length : 0
        });
    }
}

/**
 * Visualize reviews data
 */
function visualizeReviewsData(data) {
    console.log('Visualizing reviews data:', data);
    console.log('Raw reviews data received:', data);
    
    // Find the reviews container
    const reviewsContainer = document.getElementById('reviewsChart');
    
    if (!reviewsContainer) {
        console.error('Reviews container not found');
        return;
    }
    
    // Clear container
    reviewsContainer.innerHTML = '';
    
    // DIRECT USE OF RAW API RESPONSE DATA - NO FALLBACKS
    let rating = 0;
    let reviewCount = 0;
    
    // Extract rating from API response - try different formats
    if (data.average_rating && !isNaN(parseFloat(data.average_rating))) {
        rating = parseFloat(data.average_rating);
    } else if (data.productRating) {
        const ratingMatch = data.productRating.match(/([0-9.]+)/);
        if (ratingMatch) {
            rating = parseFloat(ratingMatch[1]);
        }
    } else if (data.reviewSummary && data.reviewSummary.averageRating) {
        rating = parseFloat(data.reviewSummary.averageRating);
    }
    
    // Extract review count from API response - try different formats
    if (data.total_reviews && !isNaN(parseInt(data.total_reviews))) {
        reviewCount = parseInt(data.total_reviews);
    } else if (data.countReviews && !isNaN(parseInt(data.countReviews))) {
        reviewCount = parseInt(data.countReviews);
    } else if (data.countRatings && !isNaN(parseInt(data.countRatings))) {
        reviewCount = parseInt(data.countRatings);
    }
    
    console.log('Using actual API data:', { rating, reviewCount });
    
    // Calculate sentiment directly from ratings breakdown
    let sentimentData = { positive: 0, neutral: 0, negative: 0 };
    let hasValidSentimentData = false;
    
    if (data.ratings_breakdown) {
        console.log("Using actual ratings breakdown from API");
        // Get the numbers directly from the API's ratings breakdown
        const fiveStar = parseInt(data.ratings_breakdown.five_star || 0);
        const fourStar = parseInt(data.ratings_breakdown.four_star || 0);
        const threeStar = parseInt(data.ratings_breakdown.three_star || 0);
        const twoStar = parseInt(data.ratings_breakdown.two_star || 0);
        const oneStar = parseInt(data.ratings_breakdown.one_star || 0);
        
        sentimentData = {
            positive: fiveStar + fourStar,
            neutral: threeStar,
            negative: twoStar + oneStar
        };
        
        hasValidSentimentData = (sentimentData.positive > 0 || sentimentData.neutral > 0 || sentimentData.negative > 0);
    }
    else if (data.reviewSummary) {
        console.log("Using reviewSummary stars breakdown from API");
        // Calculate based on percentages and total reviews
        const fiveStarPct = parseInt(data.reviewSummary.fiveStar?.percentage || 0);
        const fourStarPct = parseInt(data.reviewSummary.fourStar?.percentage || 0);
        const threeStarPct = parseInt(data.reviewSummary.threeStar?.percentage || 0);
        const twoStarPct = parseInt(data.reviewSummary.twoStar?.percentage || 0);
        const oneStarPct = parseInt(data.reviewSummary.oneStar?.percentage || 0);
        
        sentimentData = {
            positive: Math.round(((fiveStarPct + fourStarPct) / 100) * reviewCount),
            neutral: Math.round((threeStarPct / 100) * reviewCount),
            negative: Math.round(((twoStarPct + oneStarPct) / 100) * reviewCount)
        };
        
        hasValidSentimentData = (sentimentData.positive > 0 || sentimentData.neutral > 0 || sentimentData.negative > 0);
    }
    
    console.log('Final sentiment data:', sentimentData);
    
    // Create review summary
    const summaryContainer = document.createElement('div');
    summaryContainer.className = 'review-summary';
    summaryContainer.style.height = '70px';
    summaryContainer.style.marginBottom = '15px';
    
    summaryContainer.innerHTML = `
        <h4>Review Summary</h4>
        <div class="rating-container">
            <div class="rating-value">${rating.toFixed(1)}</div>
            <div class="rating-stars">
                ${generateStarRating(rating)}
            </div>
            <div class="review-count">${reviewCount} reviews</div>
        </div>
    `;
    
    reviewsContainer.appendChild(summaryContainer);
    
    // Only create sentiment distribution if we have valid data
    if (hasValidSentimentData) {
        const sentimentContainer = document.createElement('div');
        sentimentContainer.className = 'sentiment-container';
        sentimentContainer.style.height = '220px';
        sentimentContainer.innerHTML = '<h4>Sentiment Distribution</h4><canvas id="sentimentCanvas"></canvas>';
        reviewsContainer.appendChild(sentimentContainer);
        
        const canvas = document.getElementById('sentimentCanvas');
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        
        // Create sentiment chart
        const ctx = document.getElementById('sentimentCanvas').getContext('2d');
        window.marketCharts.reviews = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Positive', 'Neutral', 'Negative'],
                datasets: [{
                    data: [
                        sentimentData.positive || 0,
                        sentimentData.neutral || 0,
                        sentimentData.negative || 0
                    ],
                    backgroundColor: [
                        'rgba(72, 187, 120, 0.7)',
                        'rgba(237, 137, 54, 0.7)',
                        'rgba(229, 62, 62, 0.7)'
                    ],
                    borderColor: [
                        'rgba(72, 187, 120, 1)',
                        'rgba(237, 137, 54, 1)',
                        'rgba(229, 62, 62, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                layout: {
                    padding: 20
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Review Sentiment Distribution',
                        font: {
                            size: 14
                        },
                        padding: {
                            bottom: 10
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.raw || 0;
                                const percentage = ((value / (sentimentData.positive + sentimentData.neutral + sentimentData.negative || 1)) * 100).toFixed(1);
                                return `${label}: ${value} (${percentage}%)`;
                            }
                        }
                    },
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 15,
                            boxWidth: 12
                        }
                    }
                }
            }
        });
    } else {
        // Display message when we don't have sentiment data
        const noDataContainer = document.createElement('div');
        noDataContainer.className = 'no-data-message';
        noDataContainer.style.height = '100px';
        noDataContainer.style.display = 'flex';
        noDataContainer.style.justifyContent = 'center';
        noDataContainer.style.alignItems = 'center';
        noDataContainer.style.backgroundColor = 'rgba(0,0,0,0.1)';
        noDataContainer.style.borderRadius = '8px';
        noDataContainer.style.margin = '10px 0';
        noDataContainer.innerHTML = '<p>No sentiment distribution data available from API</p>';
        reviewsContainer.appendChild(noDataContainer);
    }
    
    // Track reviews analysis activity
    if (data && data.asin) {
        trackMarketAnalysisActivity('Reviews Analysis', data.asin, {
            reviewCount: data.reviews ? data.reviews.length : 0
        });
    }
}

/**
 * Clear market data and visualizations
 */
function clearMarketData() {
    console.log('Clearing all market data and visualizations');
    
    // Clear data storage
    window.amazonMarketData.priceHistory = null;
    window.amazonMarketData.competitor = null;
    window.amazonMarketData.reviews = null;
    
    // Clear any existing charts
    clearCharts();
    
    // Reset status indicators
    updateStatusIndicator('priceHistoryStatus', 'not-collected');
    updateStatusIndicator('competitorStatus', 'not-collected');
    updateStatusIndicator('reviewsStatus', 'not-collected');
    
    // Try to find and keep the ASIN value for user convenience
    let asinInput = document.getElementById('asin');
    
    // If not found by ID, try by placeholder or any other input with ASIN value
    if (!asinInput) {
        asinInput = document.querySelector('input[placeholder="Enter Amazon ASIN"]') || 
                   document.querySelector('input[placeholder="B0BYS2D9CJ"]');
    }
    
    // If still not found, try looking for any visible input in the market analysis tab
    if (!asinInput) {
        const marketTab = document.getElementById('marketAnalysis');
        if (marketTab) {
            asinInput = marketTab.querySelector('input[type="text"]');
        }
    }
    
    // If found, don't clear the ASIN to make it easier for the user to try again
    if (asinInput) {
        console.log(`Keeping ASIN value: "${asinInput.value}" for user convenience`);
    }
    
    // Reset the fetch button if it was in loading state
    setButtonLoading(false);
    
    console.log('Market data cleared successfully');
}

/**
 * Refresh market data status indicators from API
 */
function refreshMarketDataStatus() {
    // Use the same ASIN resolution strategy as fetchMarketData
    let asin = '';
    
    // 1. First try the displayed value in the Market Analysis tab
    const visibleAsinElement = document.querySelector('#marketAnalysis input[type="text"]');
    if (visibleAsinElement && visibleAsinElement.value && visibleAsinElement.value.trim()) {
        asin = visibleAsinElement.value.trim();
        console.log(`[STRATEGY 1] Found ASIN from visible input: "${asin}"`);
    }
    
    // 2. Look for any input with ID 'asin' that has a value
    if (!asin) {
        const allAsinInputs = document.querySelectorAll('#asin');
        for (const input of allAsinInputs) {
            if (input.value && input.value.trim()) {
                asin = input.value.trim();
                console.log(`[STRATEGY 2] Found ASIN from input with ID 'asin': "${asin}"`);
                break;
            }
        }
    }
    
    // 3. LAST RESORT: Just use the default ASIN directly
    if (!asin) {
        asin = 'B0BYS2D9CJ';
        console.log(`[STRATEGY 3] Using default hardcoded ASIN: "${asin}"`);
    }
    
    console.log(`[STATUS CHECK] Checking market data status for ASIN: "${asin}"`);
    
    // Show checking status
    updateStatusIndicator('priceHistoryStatus', 'loading');
    updateStatusIndicator('competitorStatus', 'loading');
    updateStatusIndicator('reviewsStatus', 'loading');
    
    // Fetch status from API
    console.log(`[STATUS API CALL] Fetching data status for ${asin}`);
    fetch(`http://localhost:5050/api/market/data-status?asin=${asin}`, {
        method: 'GET',
        headers: { 'Accept': 'application/json' }
    })
    .then(response => {
        console.log(`[STATUS API RESPONSE] Data status response: ${response.status}`);
        return response.json();
    })
    .then(data => {
        console.log('[STATUS API DATA] Market data status:', data);
        
        if (data.success) {
            // Update status indicators
            updateStatusIndicator('priceHistoryStatus', 
                data.price_history ? 'available' : 'unavailable');
                
            updateStatusIndicator('competitorStatus', 
                data.competitor_data ? 'available' : 'unavailable');
                
            updateStatusIndicator('reviewsStatus', 
                data.reviews_data ? 'available' : 'unavailable');
                
            // Enable/disable Analyze button based on data availability
            const analyzeButton = document.getElementById('analyzeButton');
            if (analyzeButton) {
                if (data.product_available) {
                    analyzeButton.disabled = false;
                    analyzeButton.querySelector('.button-text').textContent = 'Analyze Product';
                    console.log('[STATUS] Product data is available for analysis');
                } else {
                    analyzeButton.disabled = true;
                    analyzeButton.querySelector('.button-text').textContent = 'Data Unavailable';
                    console.log('[STATUS] Product data is NOT available for analysis');
                }
            }
        } else {
            // Handle error
            console.error('[STATUS API ERROR] Error checking data status:', data.message);
            
            // Reset status indicators
            updateStatusIndicator('priceHistoryStatus', 'error', data.message);
            updateStatusIndicator('competitorStatus', 'error', data.message);
            updateStatusIndicator('reviewsStatus', 'error', data.message);
        }
    })
    .catch(error => {
        console.error('[STATUS API SEQUENCE ERROR] Error fetching market data status:', error);
        
        // Reset status indicators on error
        updateStatusIndicator('priceHistoryStatus', 'error', error.message);
        updateStatusIndicator('competitorStatus', 'error', error.message);
        updateStatusIndicator('reviewsStatus', 'error', error.message);
    });
}

/**
 * Update status indicator for data collections
 * @param {string} id - ID of the status indicator element
 * @param {string} status - Status type (loading, available, unavailable, error, collected)
 * @param {string} message - Optional error message
 */
function updateStatusIndicator(id, status, message = '') {
    console.log(`Updating status indicator '${id}' to '${status}'`);
    
    // Find the indicator element
    const indicator = document.getElementById(id);
    if (!indicator) {
        console.error(`Status indicator element not found: ${id}`);
        return;
    }
    
    // Update the indicator based on status
    switch (status) {
        case 'loading':
            indicator.innerHTML = '<i class="fas fa-spinner fa-spin text-info"></i> Loading...';
            indicator.className = 'status loading';
            break;
            
        case 'available':
            indicator.innerHTML = '<i class="fas fa-check-circle text-success"></i> Available';
            indicator.className = 'status available';
            break;
            
        case 'unavailable':
            indicator.innerHTML = '<i class="fas fa-times-circle text-secondary"></i> Unavailable';
            indicator.className = 'status unavailable';
            break;
            
        case 'collected':
            indicator.innerHTML = '<i class="fas fa-check-circle text-success"></i> Collected';
            indicator.className = 'status collected';
            break;
            
        case 'not-collected':
            indicator.innerHTML = '<i class="fas fa-times-circle text-danger"></i> Not Collected';
            indicator.className = 'status not-collected';
            break;
            
        case 'error':
            const errorText = message ? `: ${message}` : '';
            indicator.innerHTML = `<i class="fas fa-exclamation-circle text-danger"></i> Error${errorText}`;
            indicator.className = 'status error';
            break;
            
        default:
            indicator.innerHTML = '<i class="fas fa-question-circle text-warning"></i> Unknown';
            indicator.className = 'status unknown';
    }
}

/**
 * Helper function to show alert
 */
function showAlert(message, type = 'info') {
    console.log(`Alert: ${message} (${type})`);
    
    // Create alert element
    const alertElement = document.createElement('div');
    alertElement.className = `alert alert-${type}`;
    alertElement.role = 'alert';
    alertElement.innerHTML = message;
    
    // Add alert to DOM
    const alertContainer = document.createElement('div');
    alertContainer.className = 'alert-container';
    alertContainer.appendChild(alertElement);
    document.body.appendChild(alertContainer);
    
    // Remove alert after 3 seconds
    setTimeout(() => {
        alertContainer.classList.add('fade-out');
        setTimeout(() => {
            document.body.removeChild(alertContainer);
        }, 300);
    }, 3000);
}

/**
 * Helper function to set button loading state
 */
function setButtonLoading(isLoading) {
    // Find the button using multiple possible selectors
    const fetchDataBtn = document.getElementById('fetchDataBtn') || 
                         document.querySelector('button.btn-primary[data-action="fetch"]') ||
                         document.querySelector('button.btn-primary');
                         
    if (!fetchDataBtn) {
        console.error('Fetch data button not found for setting loading state');
        return;
    }
    
    const btnContent = fetchDataBtn.querySelector('.btn-content') || fetchDataBtn;
    const btnLoading = fetchDataBtn.querySelector('.btn-loading');
    
    if (isLoading) {
        fetchDataBtn.disabled = true;
        if (btnContent) btnContent.style.display = 'none';
        if (btnLoading) btnLoading.style.display = 'inline-block';
        
        // If no loading spinner exists, create one
        if (!btnLoading) {
            const spinner = document.createElement('span');
            spinner.className = 'btn-loading spinner-border spinner-border-sm';
            spinner.setAttribute('role', 'status');
            spinner.setAttribute('aria-hidden', 'true');
            fetchDataBtn.appendChild(spinner);
        }
    } else {
        fetchDataBtn.disabled = false;
        if (btnContent) btnContent.style.display = 'inline-block';
        if (btnLoading) btnLoading.style.display = 'none';
    }
}

/**
 * Clear all charts
 */
function clearCharts() {
    try {
        // Initialize chart namespace if it doesn't exist
        window.marketCharts = window.marketCharts || {};
        
        // Safely destroy price history chart
        if (window.marketCharts.priceHistory) {
            if (typeof window.marketCharts.priceHistory.destroy === 'function') {
                window.marketCharts.priceHistory.destroy();
            }
            window.marketCharts.priceHistory = null;
        }
        
        // Safely destroy competitor chart
        if (window.marketCharts.competitor) {
            if (typeof window.marketCharts.competitor.destroy === 'function') {
                window.marketCharts.competitor.destroy();
            }
            window.marketCharts.competitor = null;
        }
        
        // Safely destroy reviews chart
        if (window.marketCharts.reviews) {
            if (typeof window.marketCharts.reviews.destroy === 'function') {
                window.marketCharts.reviews.destroy();
            }
            window.marketCharts.reviews = null;
        }
        
        console.log('All charts cleared successfully');
    } catch (error) {
        console.error('Error clearing charts:', error);
    }
}

/**
 * Helper function to generate star rating HTML
 */
function generateStarRating(rating) {
    const fullStars = Math.floor(rating);
    const halfStar = rating % 1 >= 0.5;
    const emptyStars = 5 - fullStars - (halfStar ? 1 : 0);
    
    let starsHtml = '';
    
    // Add full stars
    for (let i = 0; i < fullStars; i++) {
        starsHtml += '<i class="fas fa-star text-warning"></i>';
    }
    
    // Add half star if needed
    if (halfStar) {
        starsHtml += '<i class="fas fa-star-half-alt text-warning"></i>';
    }
    
    // Add empty stars
    for (let i = 0; i < emptyStars; i++) {
        starsHtml += '<i class="far fa-star text-warning"></i>';
    }
    
    return starsHtml;
}

/**
 * Helper function to calculate average
 */
function calculateAverage(values) {
    if (!values || values.length === 0) return 0;
    const sum = values.reduce((acc, val) => acc + val, 0);
    return sum / values.length;
}

/**
 * Helper function to calculate volatility
 */
function calculateVolatility(values) {
    if (!values || values.length <= 1) return 0;
    
    const avg = calculateAverage(values);
    const squaredDiffs = values.map(val => Math.pow(val - avg, 2));
    const variance = calculateAverage(squaredDiffs);
    const stdDev = Math.sqrt(variance);
    
    // Return as percentage of mean
    return (stdDev / avg) * 100;
}

/**
 * Process competitor data from search results
 * @param {Object} searchResults - The results from product search
 * @param {string} currentAsin - The ASIN of the current product 
 * @param {number} currentPrice - The current price of the product
 * @returns {Object} - Processed competitor data
 */
function processCompetitorData(searchResults, currentAsin, currentPrice) {
    console.log(`Processing competitor data for ${currentAsin}`);
    console.log('Search results:', searchResults);
    
    // Handle different possible API response formats
    let searchProducts = null;
    
    // Check for various possible structures in the API response
    if (searchResults && searchResults.searchProducts) {
        // Original expected format
        searchProducts = searchResults.searchProducts;
        console.log('Found searchProducts in expected location');
    } else if (searchResults && searchResults.searchProductDetails) {
        // Alternative format directly in the results
        searchProducts = searchResults.searchProductDetails;
        console.log('Found searchProductDetails in results');
    } else if (searchResults && searchResults.products) {
        // Another possible format
        searchProducts = searchResults.products;
        console.log('Found products array in results');
    } else if (searchResults && searchResults.searchResult && searchResults.searchResult.products) {
        // Yet another possible nested format
        searchProducts = searchResults.searchResult.products;
        console.log('Found products in searchResult');
    } else if (searchResults && Array.isArray(searchResults)) {
        // Handle case where the response is directly an array of products
        searchProducts = searchResults;
        console.log('Search results is directly an array');
    }
    
    // If we couldn't find any product array, check if we have a single product object
    if (!searchProducts && searchResults && searchResults.asin && !currentAsin.includes(searchResults.asin)) {
        // If the response is a single product that isn't the current one, wrap it in an array
        searchProducts = [searchResults];
        console.log('Search results contains a single product, using it');
    }
    
    if (!searchProducts || !Array.isArray(searchProducts) || searchProducts.length === 0) {
        console.error('[COMPETITOR DATA ERROR] Invalid search results format');
        return { success: false, message: "Invalid search results" };
    }
    
    // Use the existing function to generate competitive position from search
    return generateCompetitivePositionFromSearch(
        {
            asin: currentAsin,
            searchProductDetails: searchProducts,
            keyword: searchResults.keyword || searchResults.searchTerm || 'Unknown'
        }, 
        currentPrice
    );
} 