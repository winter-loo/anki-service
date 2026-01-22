#!/bin/bash

# Change to the application directory
cd /app/anki-service

# Start LibreTranslate in the background
libretranslate --host 0.0.0.0 --port 5000 &

# Start the Anki Web API
uvicorn web_api:app --host 0.0.0.0 --port 8000
