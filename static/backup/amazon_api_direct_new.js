/**
 * Direct Amazon API client that exactly follows the structure of api_am.js
 * Uses the same RapidAPI endpoint with proper error handling and console logging
 */

// Create our namespace
window.amazonApiClient = window.amazonApiClient || {};

// API configuration within our namespace
window.amazonApiClient.options = {
    method: 'GET',
    headers: {
        'x-rapidapi-key': 'd128131b33msh2d7517b075673cfp176ac2jsn7c6a97484ac2',
        'x-rapidapi-host': 'axesso-axesso-amazon-data-service-v1.p.rapidapi.com'
    }
};

/**
 * Get product details by ASIN
 * @param {string} asin - Amazon ASIN
 */
async function getProductDetails(asin) {
    console.log(`Fetching product details for ASIN ${asin}...`);
    const url = `https://axesso-axesso-amazon-data-service-v1.p.rapidapi.com/amz/amazon-lookup-product?url=https%3A%2F%2Fwww.amazon.com%2Fdp%2F${asin}%2F`;

    try {
        console.log(`[API REQUEST] Product details: ${url}`);
        const response = await fetch(url, window.amazonApiClient.options);
        const result = await response.json();
        console.log(`[API RESPONSE] Product details:`, result);
        return result;
    } catch (error) {
        console.error(`[API ERROR] Product details for ${asin}:`, error);
        throw error;
    }
}

/**
 * Search products
 * @param {string} keyword - Search keyword
 */
async function searchProducts(keyword) {
    console.log(`Searching products for keyword ${keyword}...`);
    const url = `https://axesso-axesso-amazon-data-service-v1.p.rapidapi.com/amz/amazon-search-by-keyword-asin?domainCode=com&keyword=${encodeURIComponent(keyword)}&page=1&excludeSponsored=false&sortBy=relevanceblender&withCache=true`;

    try {
        console.log(`[API REQUEST] Search products: ${url}`);
        const response = await fetch(url, window.amazonApiClient.options);
        const result = await response.json();
        console.log(`[API RESPONSE] Search products:`, result);
        return result;
    } catch (error) {
        console.error(`[API ERROR] Search products for ${keyword}:`, error);
        throw error;
    }
}

/**
 * Get product reviews
 * @param {string} asin - Amazon ASIN
 */
async function getProductReviews(asin) {
    console.log(`Fetching reviews for ASIN ${asin}...`);
    const url = `https://axesso-axesso-amazon-data-service-v1.p.rapidapi.com/amz/amazon-lookup-reviews?page=1&domainCode=com&asin=${asin}&sortBy=recent&filters=reviewerType=avp_only_reviews;filterByStar=five_star`;

    try {
        console.log(`[API REQUEST] Product reviews: ${url}`);
        const response = await fetch(url, window.amazonApiClient.options);
        const result = await response.json();
        console.log(`[API RESPONSE] Product reviews:`, result);
        return result;
    } catch (error) {
        console.error(`[API ERROR] Product reviews for ${asin}:`, error);
        throw error;
    }
}

/**
 * Get deals
 */
async function getDeals() {
    console.log(`Fetching Amazon deals...`);
    const url = 'https://axesso-axesso-amazon-data-service-v1.p.rapidapi.com/amz/v2/amazon-search-deals?domainCode=com&page=1';

    try {
        console.log(`[API REQUEST] Deals: ${url}`);
        const response = await fetch(url, window.amazonApiClient.options);
        const result = await response.json();
        console.log(`[API RESPONSE] Deals:`, result);
        return result;
    } catch (error) {
        console.error(`[API ERROR] Deals:`, error);
        throw error;
    }
}

/**
 * Get deals filter
 */
async function getDealsFilter() {
    console.log(`Fetching Amazon deals filter...`);
    const url = 'https://axesso-axesso-amazon-data-service-v1.p.rapidapi.com/amz/v2/amazon-deal-filter?domainCode=com';

    try {
        console.log(`[API REQUEST] Deals filter: ${url}`);
        const response = await fetch(url, window.amazonApiClient.options);
        const result = await response.json();
        console.log(`[API RESPONSE] Deals filter:`, result);
        return result;
    } catch (error) {
        console.error(`[API ERROR] Deals filter:`, error);
        throw error;
    }
}

/**
 * Get product offers
 * @param {string} asin - Amazon ASIN
 */
async function getProductOffers(asin) {
    console.log(`Fetching offers for ASIN ${asin}...`);
    const url = `https://axesso-axesso-amazon-data-service-v1.p.rapidapi.com/v2/amz/amazon-lookup-prices?page=1&domainCode=com&asin=${asin}`;

    try {
        console.log(`[API REQUEST] Product offers: ${url}`);
        const response = await fetch(url, window.amazonApiClient.options);
        const result = await response.json();
        console.log(`[API RESPONSE] Product offers:`, result);
        return result;
    } catch (error) {
        console.error(`[API ERROR] Product offers for ${asin}:`, error);
        throw error;
    }
}

/**
 * Get best sellers
 * @param {string} category - Category
 */
async function getBestSellers(category = 'toys-and-games') {
    console.log(`Fetching best sellers for category ${category}...`);
    const url = `https://axesso-axesso-amazon-data-service-v1.p.rapidapi.com/amz/amazon-best-sellers-list?url=https%3A%2F%2Fwww.amazon.com%2Fgp%2Fmovers-and-shakers%2F${category}%2Fref%3Dzg_bs_pg_1_${category}_1%3Fie%3DUTF8%26pg%3D1`;

    try {
        console.log(`[API REQUEST] Best sellers: ${url}`);
        const response = await fetch(url, window.amazonApiClient.options);
        const result = await response.json();
        console.log(`[API RESPONSE] Best sellers:`, result);
        return result;
    } catch (error) {
        console.error(`[API ERROR] Best sellers for ${category}:`, error);
        throw error;
    }
}

/**
 * Get seller details
 * @param {string} sellerId - Seller ID
 */
async function getSellerDetails(sellerId) {
    console.log(`Fetching seller details for sellerId ${sellerId}...`);
    const url = `https://axesso-axesso-amazon-data-service-v1.p.rapidapi.com/amz/amazon-lookup-seller?sellerId=${sellerId}&domainCode=com`;

    try {
        console.log(`[API REQUEST] Seller details: ${url}`);
        const response = await fetch(url, window.amazonApiClient.options);
        const result = await response.json();
        console.log(`[API RESPONSE] Seller details:`, result);
        return result;
    } catch (error) {
        console.error(`[API ERROR] Seller details for ${sellerId}:`, error);
        throw error;
    }
}

/**
 * Get seller products
 * @param {string} sellerId - Seller ID
 */
async function getSellerProducts(sellerId) {
    console.log(`Fetching seller products for sellerId ${sellerId}...`);
    const url = `https://axesso-axesso-amazon-data-service-v1.p.rapidapi.com/amz/amazon-seller-products?domainCode=com&sellerId=${sellerId}&page=1`;

    try {
        console.log(`[API REQUEST] Seller products: ${url}`);
        const response = await fetch(url, window.amazonApiClient.options);
        const result = await response.json();
        console.log(`[API RESPONSE] Seller products:`, result);
        return result;
    } catch (error) {
        console.error(`[API ERROR] Seller products for ${sellerId}:`, error);
        throw error;
    }
}

/**
 * Get profile
 */
async function getProfile() {
    console.log(`Fetching profile...`);
    const url = 'https://axesso-axesso-amazon-data-service-v1.p.rapidapi.com/amz/amazon-lookup-profile?domainCode=com';

    try {
        console.log(`[API REQUEST] Profile: ${url}`);
        const response = await fetch(url, window.amazonApiClient.options);
        const result = await response.json();
        console.log(`[API RESPONSE] Profile:`, result);
        return result;
    } catch (error) {
        console.error(`[API ERROR] Profile:`, error);
        throw error;
    }
}

/**
 * Get review details
 */
async function getReviewDetails() {
    console.log(`Fetching review details...`);
    const url = 'https://axesso-axesso-amazon-data-service-v1.p.rapidapi.com/amz/amazon-review-details';

    try {
        console.log(`[API REQUEST] Review details: ${url}`);
        const response = await fetch(url, window.amazonApiClient.options);
        const result = await response.json();
        console.log(`[API RESPONSE] Review details:`, result);
        return result;
    } catch (error) {
        console.error(`[API ERROR] Review details:`, error);
        throw error;
    }
}

// Export all functions
window.amazonApi = {
    getProductDetails,
    searchProducts,
    getProductReviews,
    getDeals,
    getDealsFilter,
    getProductOffers,
    getBestSellers,
    getSellerDetails,
    getSellerProducts,
    getProfile,
    getReviewDetails
}; 