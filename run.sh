#!/bin/bash
# Run script for QuickPrice Web Application
# This script starts the Flask web server for the QuickPrice Dashboard

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Set the port (default to 5050)
PORT=${1:-5050}

# Go to the correct directory
cd "$DIR"

echo "Starting QuickPrice Dashboard on port $PORT..."

# Run the Flask application
export FLASK_APP=app.py
export FLASK_ENV=development
python app.py --port $PORT 