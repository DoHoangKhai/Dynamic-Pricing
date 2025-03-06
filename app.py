from flask import Flask, render_template, request, jsonify
import os
# Import the API module, not just the function
import api

app = Flask(__name__)

# Serve the main HTML page
@app.route('/')
def index():
    # Make sure this matches your actual template name (index.html or home.html)
    return render_template('index.html')

# Forward the prediction request to the API endpoint
@app.route('/api/predict-price', methods=['POST'])
def handle_predict():
    # Call the function from the imported module
    return api.predict_price()

if __name__ == "__main__":
    app.run(debug=True, port=5000)
