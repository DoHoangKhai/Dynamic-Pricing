#!/bin/bash
# Simple startup script for Dynamic Pricing

# Kill any running Flask servers
pkill -f "python app.py" || echo "No running servers"

# Start the server
echo "Starting Dynamic Pricing server on port 5005..."
python app.py 