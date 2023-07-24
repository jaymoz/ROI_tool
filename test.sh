#!/bin/bash


# Change directory to 2023/client
cd /home/ubuntu/2023/client

# Install client dependencies
npm start &

# Change directory to 2023
cd /home/ubuntu/2023

# Activate the virtual environment
source venv/bin/activate

# Change directory to backend
cd backend

# Install Flask dependencies
pip install -r requirements.txt

# Run the Flask application (replace with your actual command)
python3 application.py &