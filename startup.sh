#!/bin/bash

# Exit on any error
set -e

# Print commands as they are executed
set -x

# Ensure we're in the right directory
cd /home/site/wwwroot

echo "Current directory: $(pwd)"
echo "Python version: $(python3 --version)"
echo "Pip version: $(python3 -m pip --version)"

# Install requirements
pip install -r requirements.txt

# Install spaCy model
python -m spacy download en_core_web_sm

# Start the app
gunicorn --bind=0.0.0.0:$PORT app:app 