// Market analysis functions for real-time Amazon data

// Store analyzed product data
let analyzedProductData = null;

// Function to analyze a product using our web scraper
async function analyzeProductWithScraper() {
    const asin = document.getElementById('marketProductAsin').value.trim();
    
    if (!asin) {
        alert('Please enter a valid Amazon ASIN');
        return;
    }
    
    try {
        // Update UI
        const analyzeBtn = document.getElementById('analyzeProductBtn');
        const originalBtnText = analyzeBtn.textContent;
        analyzeBtn.textContent = 'Analyzing...';
        analyzeBtn.disabled = true;
        
        updateStatus('bestSellersStatus', 'Fetching...');
        updateStatus('productDetailsStatus', 'Fetching...');
        updateStatus('reviewsStatus', 'Fetching...');
        updateStatus('dealsStatus', 'Fetching...');
        
        // Call our new scraper endpoint
        const response = await fetch(`/market/analyze?asin=${encodeURIComponent(asin)}`);
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.message || 'Failed to analyze product');
        }
        
        // Store the data
        analyzedProductData = data;
        
        // Update status indicators
        updateStatus('productDetailsStatus', 'Complete', 'success');
        
        // If we have price trends, update that status too
        if (data.price_trends && data.price_trends.success) {
            updateStatus('bestSellersStatus', 'Complete', 'success');
        } else {
            updateStatus('bestSellersStatus', 'Not Available', 'warning');
        }
        
        // Update other statuses
        updateStatus('reviewsStatus', 'Not Available', 'warning');
        updateStatus('dealsStatus', 'Not Available', 'warning');
        
        // Show product title in the report
        const productTitle = data.product_details.title || 'Unknown Product';
        document.getElementById('reportDate').textContent = `Generated: ${new Date().toLocaleString()}`;
        
        // Visualize the data - use real data if available, otherwise use our synthetic data functions
        visualizePriceTrends(data.price_trends);
        visualizeMarketPosition(data.market_position);
        
        // Create sentiment chart with synthetic data based on product rating
        updateSentimentChart();
        
        // Generate market summary report
        generateMarketSummary(data);
        
        // Success message
        console.log("Product analysis complete with available data. Some visualizations may use synthetic data.");
        
    } catch (error) {
        console.error('Error analyzing product:', error);
        alert(`Error analyzing product: ${error.message}`);
        
        // Even if main analysis fails, try to get product details from a direct API call
        try {
            console.log("Attempting to get basic product details via direct API...");
            const backupResponse = await fetch(`/api/product-info?asin=${encodeURIComponent(asin)}`);
            const backupData = await backupResponse.json();
            
            if (backupData && backupData.title) {
                // We at least have basic product info, create synthetic data
                analyzedProductData = {
                    product_details: backupData,
                    price_trends: null,
                    market_position: {
                        position: "average",
                        percentile: 50,
                        competitive_index: 5
                    }
                };
                
                // Update status indicators for partial success
                updateStatus('productDetailsStatus', 'Partial', 'warning');
                updateStatus('bestSellersStatus', 'Not Available', 'error');
                
                // Create charts with synthetic data
                visualizePriceTrends(null);
                updateSentimentChart();
                
                console.log("Created fallback visualization with synthetic data");
            } else {
                // Update status indicators to show complete failure
                updateStatus('bestSellersStatus', 'Failed', 'error');
                updateStatus('productDetailsStatus', 'Failed', 'error');
                updateStatus('reviewsStatus', 'Failed', 'error');
                updateStatus('dealsStatus', 'Failed', 'error');
            }
        } catch (backupError) {
            console.error("Backup data fetching also failed:", backupError);
            // Update status indicators to show failure
            updateStatus('bestSellersStatus', 'Failed', 'error');
            updateStatus('productDetailsStatus', 'Failed', 'error');
            updateStatus('reviewsStatus', 'Failed', 'error');
            updateStatus('dealsStatus', 'Failed', 'error');
        }
    } finally {
        // Restore button
        const analyzeBtn = document.getElementById('analyzeProductBtn');
        analyzeBtn.textContent = 'Analyze Product';
        analyzeBtn.disabled = false;
    }
}

// Update status indicator
function updateStatus(elementId, status, type = 'default') {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    element.textContent = status;
    
    // Remove existing classes
    element.classList.remove('status-success', 'status-error', 'status-warning', 'status-default');
    
    // Add appropriate class
    switch (type) {
        case 'success':
            element.classList.add('status-success');
            break;
        case 'error':
            element.classList.add('status-error');
            break;
        case 'warning':
            element.classList.add('status-warning');
            break;
        default:
            element.classList.add('status-default');
    }
}

/**
 * Visualizes price trends for a product
 * If price history isn't available, creates synthetic data based on current price
 */
function visualizePriceTrends(priceData) {
    try {
        // Find the canvas element
        const chartElement = document.getElementById('priceTrendChart');
        if (!chartElement) {
            console.error('Price trend chart element not found');
            return;
        }
        
        // Get the chart's parent container
        const chartContainer = chartElement.parentNode;
        if (!chartContainer) {
            console.error('Price trend chart container not found');
            return;
        }
        
        // Clear previous chart if exists
        chartContainer.innerHTML = '';
        const canvas = document.createElement('canvas');
        canvas.id = 'priceTrendChart';
        chartContainer.appendChild(canvas);
        
        let dates = [];
        let prices = [];
        let currentPrice = 0;
        let synthetic = false;
        
        // Check if we have real price history data
        if (priceData && priceData.success && priceData.price_history && priceData.price_history.length > 0) {
            // Use real data
            const history = priceData.price_history;
            dates = history.map(h => h.date);
            prices = history.map(h => h.price);
            currentPrice = prices[prices.length - 1];
            console.log('Using real price history data for chart');
        } else {
            // Generate synthetic data based on current product price
            synthetic = true;
            
            // Try to get current price from analyzed product data
            if (analyzedProductData && analyzedProductData.product_details) {
                const productDetails = analyzedProductData.product_details;
                
                // Parse price from string to number, handling various formats
                if (productDetails.price) {
                    if (typeof productDetails.price === 'number') {
                        currentPrice = productDetails.price;
                    } else {
                        // Remove currency symbols and convert to number
                        const priceStr = productDetails.price.toString().replace(/[^0-9.]/g, '');
                        currentPrice = parseFloat(priceStr);
                    }
                }
            }
            
            // If we couldn't get a valid price, use a default
            if (!currentPrice || isNaN(currentPrice)) {
                currentPrice = 99.99;
            }
            
            // Generate 6 months of synthetic price data
            const today = new Date();
            for (let i = 5; i >= 0; i--) {
                let date = new Date();
                date.setMonth(today.getMonth() - i);
                dates.push(date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' }));
                
                // Create random price fluctuations around current price (±15%)
                const fluctuation = (Math.random() * 0.3) - 0.15;
                const randomPrice = currentPrice * (1 + fluctuation);
                prices.push(parseFloat(randomPrice.toFixed(2)));
            }
            
            console.log('Using synthetic price history data for chart');
        }
        
        // Create chart with real or synthetic data
        const ctx = document.getElementById('priceTrendChart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [{
                    label: 'Price (USD)',
                    data: prices,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.3,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    title: {
                        display: true,
                        text: synthetic ? 'Price Trend (Estimated)' : 'Price Trend History'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `$${context.parsed.y.toFixed(2)}`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        ticks: {
                            callback: function(value) {
                                return '$' + value;
                            }
                        }
                    }
                }
            }
        });
        
        // Always add note about data source
        const noteDiv = document.createElement('div');
        noteDiv.className = 'chart-note';
        noteDiv.style.position = 'absolute';
        noteDiv.style.right = '10px';
        noteDiv.style.top = '30px';
        noteDiv.style.fontSize = '0.75rem';
        noteDiv.style.fontStyle = 'italic';
        noteDiv.style.color = '#6c757d';
        noteDiv.style.maxWidth = '300px';
        noteDiv.style.textAlign = 'right';
        
        if (synthetic) {
            noteDiv.innerHTML = 'Note: Displaying estimated price trends based on current price. Historical data not available.';
        } else {
            noteDiv.innerHTML = 'Note: Displaying actual price history data from market analysis.';
        }
        
        chartContainer.style.position = 'relative';
        chartContainer.appendChild(noteDiv);
        
    } catch (error) {
        console.error('Error creating price trend chart:', error);
        
        // Show error message in the chart container
        const chartElement = document.getElementById('priceTrendChart');
        if (chartElement && chartElement.parentNode) {
            const chartContainer = chartElement.parentNode;
            chartContainer.innerHTML = `
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle"></i> 
                    Chart Error: Could not display price trend chart
                </div>
                <canvas id="priceTrendChart" style="display:none"></canvas>
            `;
        }
    }
}

/**
 * Visualizes the market position of a product
 * Creates a visual representation even when position data is unavailable
 */
function visualizeMarketPosition(positionData) {
    try {
        // Since we'll be updating existing elements in the position-card rather than
        // creating a new one, let's find these elements
        const pricePosition = document.getElementById('pricePosition');
        const competitiveIndex = document.getElementById('competitiveIndex');
        const pricePercentile = document.getElementById('pricePercentile');
        
        if (!pricePosition || !competitiveIndex || !pricePercentile) {
            console.error('Market position elements not found');
            return;
        }
        
        let position = 'average';
        let percentile = 50;
        let competitiveValue = 65;  // Competitive index out of 100
        let synthetic = false;
        
        // Check if we have real position data
        if (positionData && positionData.position) {
            position = positionData.position;
            
            if (positionData.percentile !== undefined) {
                percentile = positionData.percentile;
            }
            
            if (positionData.competitive_index !== undefined) {
                // Convert from 1-10 to 1-100 scale
                competitiveValue = Math.min(positionData.competitive_index * 10, 100);
            }
            
            console.log('Using real market position data for visualization');
        } else {
            // Generate synthetic position data
            synthetic = true;
            
            // Try to get price from analyzed product data to determine position
            if (analyzedProductData && analyzedProductData.product_details) {
                const productDetails = analyzedProductData.product_details;
                
                // Parse price from string to number
                let price = 0;
                if (productDetails.price) {
                    if (typeof productDetails.price === 'number') {
                        price = productDetails.price;
                    } else {
                        // Remove currency symbols and convert to number
                        const priceStr = productDetails.price.toString().replace(/[^0-9.]/g, '');
                        price = parseFloat(priceStr);
                    }
                }
                
                // Assign position based on price ranges
                if (price > 0) {
                    if (price < 50) {
                        position = 'budget';
                        percentile = 25;
                        competitiveValue = 80; // Higher competition at budget level
                    } else if (price > 200) {
                        position = 'premium';
                        percentile = 85;
                        competitiveValue = 30; // Lower competition at premium level
                    } else {
                        position = 'average';
                        percentile = 50;
                        competitiveValue = 60; // Medium competition at average level
                    }
                }
            }
            
            console.log('Using synthetic market position data for visualization');
        }
        
        // Remove existing position classes
        pricePosition.classList.remove('status-budget', 'status-competitive', 
                                      'status-average', 'status-premium', 'status-luxury');
        
        // Add class and text based on position
        pricePosition.textContent = position.charAt(0).toUpperCase() + position.slice(1);
        switch (position.toLowerCase()) {
            case 'budget':
            case 'low':
            case 'value':
                pricePosition.classList.add('status-budget');
                break;
            case 'competitive':
                pricePosition.classList.add('status-competitive');
                break;
            case 'premium':
            case 'high':
            case 'luxury':
                pricePosition.classList.add('status-premium');
                break;
            case 'mid-range':
            case 'average':
            case 'middle':
            default:
                pricePosition.classList.add('status-average');
                break;
        }
        
        // Update percentile and competitive index values
        pricePercentile.textContent = `${percentile}%`;
        competitiveIndex.textContent = `${competitiveValue}/100`;
        
        // Add a note about data source (either synthetic or real)
        const positionCard = document.querySelector('.position-card');
        if (positionCard) {
            // Remove any existing note
            const existingNote = document.getElementById('positionSyntheticNote');
            if (existingNote) {
                existingNote.remove();
            }
            
            // Create new note
            const noteDiv = document.createElement('div');
            noteDiv.id = 'positionSyntheticNote';
            noteDiv.className = 'data-source-note';
            
            if (synthetic) {
                noteDiv.innerHTML = 'Note: Market position is estimated based on product information. Actual competitive data not available.';
            } else {
                noteDiv.innerHTML = 'Note: Displaying actual market position data from market analysis.';
            }
            
            positionCard.appendChild(noteDiv);
            
            // Add style for this note if not already added
            if (!document.getElementById('dataSourceNoteStyle')) {
                const style = document.createElement('style');
                style.id = 'dataSourceNoteStyle';
                style.textContent = `
                    .data-source-note {
                        margin-top: 15px;
                        color: #6c757d;
                        font-size: 0.75rem;
                        font-style: italic;
                        border-top: 1px solid #eee;
                        padding-top: 10px;
                        text-align: right;
                    }
                `;
                document.head.appendChild(style);
            }
        }
        
    } catch (error) {
        console.error('Error visualizing market position:', error);
        
        // Reset to default values on error
        const elements = [
            { id: 'pricePosition', value: 'Unknown' },
            { id: 'competitiveIndex', value: '0/100' },
            { id: 'pricePercentile', value: '0%' }
        ];
        
        elements.forEach(elem => {
            const element = document.getElementById(elem.id);
            if (element) {
                element.textContent = elem.value;
                if (elem.id === 'pricePosition') {
                    element.classList.remove('status-budget', 'status-competitive', 
                                           'status-average', 'status-premium', 'status-luxury');
                }
            }
        });
    }
}

// Generate market summary report
function generateMarketSummary(data) {
    if (!data || !data.product_details) {
        console.error('No product data available for market summary');
        return;
    }
    
    const product = data.product_details;
    const priceTrends = data.price_trends || {};
    const marketPosition = data.market_position || {};
    
    const summaryContainer = document.getElementById('marketSummaryContent');
    if (!summaryContainer) return;
    
    // Create summary HTML
    let summaryHTML = `
        <div class="market-summary">
            <h4>${product.title || 'Unknown Product'}</h4>
            <div class="summary-item">
                <span class="label">Current Price:</span>
                <span class="value">$${product.current_price?.toFixed(2) || 'N/A'}</span>
            </div>
            <div class="summary-item">
                <span class="label">Price Range (30d):</span>
                <span class="value">$${priceTrends.min_price?.toFixed(2) || 'N/A'} - $${priceTrends.max_price?.toFixed(2) || 'N/A'}</span>
            </div>
            <div class="summary-item">
                <span class="label">Price Trend:</span>
                <span class="value ${getTrendClass(priceTrends.trend)}">${priceTrends.trend || 'Unknown'}</span>
            </div>
            <div class="summary-item">
                <span class="label">Price Volatility:</span>
                <span class="value">${priceTrends.volatility?.toFixed(1) || 'N/A'}%</span>
            </div>
            <div class="summary-item">
                <span class="label">Market Position:</span>
                <span class="value ${getPositionClass(marketPosition.position)}">${marketPosition.position || 'Unknown'}</span>
            </div>
            <div class="summary-item">
                <span class="label">Rating:</span>
                <span class="value">${product.rating || 'N/A'} ★</span>
            </div>
            
            <h4 class="mt-4">Market Insights</h4>
            <p>${generateMarketInsights(data)}</p>
            
            <h4 class="mt-4">Pricing Recommendations</h4>
            <p>${generatePricingRecommendations(data)}</p>
        </div>
    `;
    
    summaryContainer.innerHTML = summaryHTML;
}

// Helper function to get CSS class based on trend
function getTrendClass(trend) {
    if (!trend) return '';
    
    switch (trend.toLowerCase()) {
        case 'increasing':
            return 'trend-up';
        case 'decreasing':
            return 'trend-down';
        default:
            return 'trend-stable';
    }
}

// Helper function to get CSS class based on position
function getPositionClass(position) {
    if (!position) return '';
    
    switch (position.toLowerCase()) {
        case 'budget':
            return 'position-budget';
        case 'competitive':
            return 'position-competitive';
        case 'average':
            return 'position-average';
        case 'premium':
            return 'position-premium';
        case 'luxury':
            return 'position-luxury';
        default:
            return '';
    }
}

// Generate market insights text
function generateMarketInsights(data) {
    if (!data) return 'No data available for market insights.';
    
    const product = data.product_details || {};
    const priceTrends = data.price_trends || {};
    const marketPosition = data.market_position || {};
    
    const insights = [];
    
    // Price trend insights
    if (priceTrends.trend) {
        switch (priceTrends.trend.toLowerCase()) {
            case 'increasing':
                insights.push('Prices in this category are trending upward, suggesting increasing demand or reduced supply.');
                break;
            case 'decreasing':
                insights.push('Prices in this category are trending downward, possibly due to increased competition or seasonal factors.');
                break;
            default:
                insights.push('Prices in this category have remained relatively stable over the past 30 days.');
                break;
        }
    }
    
    // Volatility insights
    if (priceTrends.volatility !== undefined) {
        if (priceTrends.volatility > 10) {
            insights.push('Price volatility is high, indicating an unstable market with frequent price changes.');
        } else if (priceTrends.volatility > 5) {
            insights.push('Price volatility is moderate, suggesting some market competition or seasonal adjustments.');
        } else {
            insights.push('Price volatility is low, indicating a stable market with consistent pricing.');
        }
    }
    
    // Market position insights
    if (marketPosition.position) {
        switch (marketPosition.position.toLowerCase()) {
            case 'budget':
                insights.push('This product is positioned at the lower end of the market, appealing to price-sensitive customers.');
                break;
            case 'competitive':
                insights.push('This product has competitive pricing, positioned slightly below the market average.');
                break;
            case 'average':
                insights.push('This product is priced around the market average for similar items.');
                break;
            case 'premium':
                insights.push('This product is positioned above the market average, suggesting premium features or brand value.');
                break;
            case 'luxury':
                insights.push('This product is positioned at the high end of the market, targeting premium customers.');
                break;
        }
    }
    
    // Competitive landscape insights
    const competitiveIndex = marketPosition.competitive_index || 0;
    if (competitiveIndex > 6) {
        insights.push('The market has many competing products, indicating high competition.');
    } else if (competitiveIndex > 3) {
        insights.push('The market has a moderate number of competing products.');
    } else {
        insights.push('The market has relatively few competing products, suggesting a potential niche.');
    }
    
    return insights.join(' ');
}

// Generate pricing recommendations text
function generatePricingRecommendations(data) {
    if (!data) return 'No data available for pricing recommendations.';
    
    const product = data.product_details || {};
    const priceTrends = data.price_trends || {};
    const marketPosition = data.market_position || {};
    
    const recommendations = [];
    
    // Current price
    const currentPrice = product.current_price || 0;
    
    // Market position based recommendations
    if (marketPosition.position) {
        switch (marketPosition.position.toLowerCase()) {
            case 'budget':
                if (priceTrends.trend === 'increasing') {
                    recommendations.push(`Consider a modest price increase to $${(currentPrice * 1.05).toFixed(2)} to capitalize on the rising market trend while maintaining budget positioning.`);
                } else {
                    recommendations.push(`Maintain competitive low pricing around $${currentPrice.toFixed(2)} to appeal to price-sensitive customers.`);
                }
                break;
            case 'competitive':
                if (priceTrends.trend === 'decreasing') {
                    recommendations.push(`Consider a slight price reduction to $${(currentPrice * 0.95).toFixed(2)} to maintain competitive advantage in a declining price environment.`);
                } else {
                    recommendations.push(`Optimal price point appears to be around $${(currentPrice * 1.02).toFixed(2)} to balance competitiveness with profitability.`);
                }
                break;
            case 'average':
                if (priceTrends.volatility > 8) {
                    recommendations.push(`In this volatile market, consider dynamic pricing between $${(currentPrice * 0.95).toFixed(2)} and $${(currentPrice * 1.05).toFixed(2)} based on demand fluctuations.`);
                } else {
                    recommendations.push(`Maintain pricing around $${currentPrice.toFixed(2)} to keep market position stable, with potential for slight increases based on product differentiation.`);
                }
                break;
            case 'premium':
                if (priceTrends.trend === 'increasing') {
                    recommendations.push(`The market supports a price increase to $${(currentPrice * 1.08).toFixed(2)} based on the upward trend and premium positioning.`);
                } else {
                    recommendations.push(`Maintain premium pricing at $${currentPrice.toFixed(2)} with focus on highlighting value proposition and quality differentiators.`);
                }
                break;
            case 'luxury':
                recommendations.push(`Continue luxury positioning with price point at $${(currentPrice * 1.05).toFixed(2)} or higher, emphasizing exclusivity and premium features.`);
                break;
        }
    }
    
    // Price trend based recommendations
    if (priceTrends.trend) {
        switch (priceTrends.trend.toLowerCase()) {
            case 'increasing':
                recommendations.push(`With the current upward market trend, consider gradual price increases of 2-5% over the next 30 days.`);
                break;
            case 'decreasing':
                recommendations.push(`In a declining price market, focus on value-added services or bundle offers rather than competing solely on price.`);
                break;
            default:
                recommendations.push(`In this stable price environment, differentiate through product features, customer service, or targeted marketing.`);
                break;
        }
    }
    
    return recommendations.join(' ');
}

// Replace the standard analyzeProduct function with our new implementation
function analyzeProduct() {
    analyzeProductWithScraper();
}

// Initialize with dummy data
document.addEventListener('DOMContentLoaded', function() {
    // Add placeholder ASIN suggestion to the input field
    const asinInput = document.getElementById('marketProductAsin');
    if (asinInput) {
        asinInput.placeholder = 'e.g., B07ZPKBL9V (Echo Dot 4th Gen)';
    }
});

/**
 * Updates the sentiment chart with synthetic data if review data is not available
 * Uses the product rating to estimate sentiment distribution
 */
function updateSentimentChart() {
    try {
        // Find the sentiment chart element
        const chartElement = document.getElementById('sentimentChart');
        if (!chartElement) {
            console.error('Sentiment chart element not found');
            return;
        }
        
        // Get the chart's parent container
        const chartContainer = chartElement.parentNode;
        if (!chartContainer) {
            console.error('Sentiment chart container not found');
            return;
        }
        
        // Clear previous chart if exists
        chartContainer.innerHTML = '';
        const canvas = document.createElement('canvas');
        canvas.id = 'sentimentChart';
        chartContainer.appendChild(canvas);
        
        // Default sentiment values (if no rating available)
        let positive = 65;
        let neutral = 25;
        let negative = 10;
        
        // Check if we have product rating from product details
        if (analyzedProductData && analyzedProductData.product_details) {
            const productDetails = analyzedProductData.product_details;
            
            if (productDetails.rating) {
                let rating = 0;
                
                // Parse rating from string to number, handling various formats
                if (typeof productDetails.rating === 'number') {
                    rating = productDetails.rating;
                } else {
                    // Try to extract numerical rating
                    const ratingStr = productDetails.rating.toString();
                    const ratingMatch = ratingStr.match(/([0-9]\.[0-9])|([0-9])/);
                    if (ratingMatch) {
                        rating = parseFloat(ratingMatch[0]);
                    }
                }
                
                // If we have a valid rating, convert it to sentiment percentages
                if (rating > 0 && rating <= 5) {
                    // Scale rating from 1-5 to sentiment distribution
                    // Higher rating = more positive sentiment
                    const ratingRatio = rating / 5;
                    
                    // Calculate sentiment percentages based on rating
                    positive = Math.round(ratingRatio * 100 * 0.9); // Max positive is 90%
                    negative = Math.round((1 - ratingRatio) * 100 * 0.5); // Max negative is 50%
                    neutral = 100 - positive - negative;
                    
                    // Ensure percentages are reasonable
                    if (neutral < 5) neutral = 5;
                    if (positive > 90) positive = 90;
                    if (negative > 50) negative = 50;
                    
                    // Adjust to ensure sum is 100%
                    const total = positive + neutral + negative;
                    if (total !== 100) {
                        neutral += (100 - total);
                    }
                }
            }
        }
        
        // Create sentiment chart
        const ctx = document.getElementById('sentimentChart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Positive', 'Neutral', 'Negative'],
                datasets: [{
                    data: [positive, neutral, negative],
                    backgroundColor: [
                        'rgba(75, 192, 120, 0.8)',
                        'rgba(54, 162, 235, 0.8)',
                        'rgba(255, 99, 132, 0.8)'
                    ],
                    borderColor: [
                        'rgba(75, 192, 120, 1)',
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 99, 132, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'right',
                    },
                    title: {
                        display: true,
                        text: 'Estimated Sentiment Distribution'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.label}: ${context.parsed}%`;
                            }
                        }
                    }
                }
            }
        });
        
        // Add note that this is synthetic data
        const noteDiv = document.createElement('div');
        noteDiv.className = 'chart-note';
        noteDiv.innerHTML = '<small><i>Note: Displaying estimated sentiment based on product rating. Actual review sentiment data not available.</i></small>';
        chartContainer.appendChild(noteDiv);
        
    } catch (error) {
        console.error('Error creating sentiment chart:', error);
        
        // Show error message in the chart container
        const chartElement = document.getElementById('sentimentChart');
        if (chartElement && chartElement.parentNode) {
            const chartContainer = chartElement.parentNode;
            chartContainer.innerHTML = `
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle"></i> 
                    Chart Error: Could not display sentiment chart
                </div>
                <canvas id="sentimentChart" style="display:none"></canvas>
            `;
        }
    }
} 