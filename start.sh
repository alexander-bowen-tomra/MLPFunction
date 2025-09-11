#!/bin/bash

# Pull latest model files from GitHub (if public or using a token)
curl -O https://raw.githubusercontent.com/alexander-bowen-tomra/MLPFunction/main/feature_names.pkl
curl -O https://raw.githubusercontent.com/alexander-bowen-tomra/MLPFunction/main/scaler.pkl
curl -O https://raw.githubusercontent.com/alexander-bowen-tomra/MLPFunction/main/feature_names.pkl

# Start the FastAPI app
uvicorn main:app --host 0.0.0.0 --port $PORT
