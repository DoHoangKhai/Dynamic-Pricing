#!/bin/bash

# Dynamic Pricing System Startup Script
# This script helps users start the dynamic pricing system and provides options for testing

echo "=== Dynamic Pricing System ==="
echo "Starting up the system..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is required but not installed."
    exit 1
fi

# Check if requirements are installed
echo "Checking dependencies..."
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found."
    exit 1
fi

# Install requirements if needed
pip3 install -r requirements.txt

# Check if the model file exists
if [ ! -f "dynamic_pricing_dqn.zip" ]; then
    echo "Warning: Model file (dynamic_pricing_dqn.zip) not found."
    echo "The system may not function correctly without the trained model."
fi

# Start the Flask server in the background
echo "Starting the server..."
python3 app.py &
SERVER_PID=$!

# Wait for the server to start
echo "Waiting for server to start..."
sleep 3

# Check if server started successfully
if ! curl -s http://localhost:5001 > /dev/null; then
    echo "Error: Server failed to start."
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

echo "Server started successfully!"
echo "You can access the web interface at: http://localhost:5001"

# Provide options to the user
echo ""
echo "Options:"
echo "1. Open web interface (if browser is available)"
echo "2. Run sample data tests"
echo "3. Exit"

read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        # Try to open the browser
        if command -v open &> /dev/null; then
            open http://localhost:5001
        elif command -v xdg-open &> /dev/null; then
            xdg-open http://localhost:5001
        elif command -v start &> /dev/null; then
            start http://localhost:5001
        else
            echo "Could not open browser automatically."
            echo "Please open http://localhost:5001 in your browser."
        fi
        ;;
    2)
        # Run the sample data tests
        echo "Running sample data tests..."
        python3 sample_data.py
        ;;
    3)
        # Exit and kill the server
        echo "Shutting down server..."
        kill $SERVER_PID
        echo "Server stopped."
        exit 0
        ;;
    *)
        echo "Invalid choice."
        ;;
esac

# Keep the script running to maintain the server
echo ""
echo "Server is running in the background."
echo "Press Ctrl+C to stop the server and exit."

# Wait for user to press Ctrl+C
trap "echo 'Shutting down server...'; kill $SERVER_PID; echo 'Server stopped.'; exit 0" INT
while true; do
    sleep 1
done 