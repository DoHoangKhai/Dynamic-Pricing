import json
import requests
import time

# Test data - keep this consistent for all requests
test_data = {
    "productType": "Electronics",
    "productGroup": "Smartphones",
    "actualPrice": 499.99,
    "competitorPrice": 520.00,
    "rating": 4.2,
    "numberOfOrders": 150
}

# URL for the API
url = "http://localhost:5050/api/predict-price"

# Number of test iterations
iterations = 10

# Run test
print("Testing for price drift with identical inputs...")
print(f"Input data: {json.dumps(test_data, indent=2)}")
print("\nResults:")
print("-" * 80)
print(f"{'Iteration':<10} {'Recommended Price':<20} {'Diff from Prev':<20} {'Diff from First':<20}")
print("-" * 80)

previous_price = None
first_price = None

for i in range(iterations):
    # Make the prediction
    response = requests.post(url, json=test_data)
    result = response.json()
    
    # Get the recommended price
    current_price = result.get('recommendedPrice')
    
    # Calculate differences
    diff_from_prev = 0
    if previous_price is not None:
        diff_from_prev = current_price - previous_price
    
    diff_from_first = 0
    if first_price is not None:
        diff_from_first = current_price - first_price
    else:
        first_price = current_price
    
    # Print the results
    print(f"{i+1:<10} {current_price:<20.2f} {diff_from_prev:<20.2f} {diff_from_first:<20.2f}")
    
    # Update previous price
    previous_price = current_price
    
    # Small delay to see effects more clearly
    time.sleep(0.5)

print("-" * 80)
print("Test complete. If you see differences in recommended prices despite identical inputs,")
print("this indicates a price drift issue likely caused by maintained state between API calls.") 