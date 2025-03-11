from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict-price', methods=['POST'])
def predict_price():
    if not request.json:
        return jsonify({"error": "No data provided"}), 400
        
    try:
        data = request.json
        print("Received data:", data)
        
        # Extract features with default values to prevent NaN
        product_type = data.get('productType', '')
        
        # Handle actual_price safely
        try:
            actual_price = float(data.get('actualPrice', 0))
            if actual_price <= 0:
                actual_price = 100.0  # Default if zero or negative
        except (TypeError, ValueError):
            actual_price = 100.0  # Default if conversion fails
            
        # Handle competitor_price safely
        try:
            competitor_price = float(data.get('competitorPrice', 0))
            if competitor_price <= 0:
                competitor_price = actual_price * 0.9  # Default to 90% of actual price
        except (TypeError, ValueError):
            competitor_price = actual_price * 0.9  # Default if conversion fails
            
        print(f"Processed values: actual_price={actual_price}, competitor_price={competitor_price}")
        
        # Simple elasticity mapping
        elasticity = "medium"
        if product_type == "Electronics":
            elasticity = "high"
        elif product_type == "Computers&Accessories":
            elasticity = "low"
            
        # Calculate price recommendations
        if elasticity == "high":
            recommended_price = competitor_price * 0.95
            min_price = recommended_price * 0.9
            max_price = recommended_price * 1.05
            explanation = "This product has high price elasticity. We recommend competitive pricing."
        elif elasticity == "low":
            recommended_price = actual_price * 1.1
            min_price = actual_price * 1.05
            max_price = actual_price * 1.2
            explanation = "This product has low price elasticity. You can prioritize higher margins."
        else:
            recommended_price = (actual_price + competitor_price) / 2
            min_price = recommended_price * 0.95
            max_price = recommended_price * 1.1
            explanation = "This product has moderate price elasticity. We recommend balanced pricing."
        
        # Round values to prevent floating point issues
        recommended_price = round(recommended_price, 2)
        min_price = round(min_price, 2)
        max_price = round(max_price, 2)
        
        # Prepare response
        response = {
            "recommended_price": recommended_price,
            "min_price": min_price,
            "max_price": max_price,
            "elasticity": elasticity,
            "explanation": explanation
        }
        
        print("Sending response:", response)
        return jsonify(response)
        
    except Exception as e:
        import traceback
        print(f"Error during prediction: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5005)
