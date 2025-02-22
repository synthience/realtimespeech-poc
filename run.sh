#!/bin/bash

# Ensure virtual environment is active
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install requirements if needed
if [ ! -f ".venv/installed" ]; then
    echo "Installing dependencies..."
    pip install vosk sounddevice soundfile webrtcvad numpy requests edge-tts
    touch .venv/installed
fi

# Ensure Vosk model is downloaded
MODEL_PATH="vosk-model-en-us-0.42-gigaspeech"
if [ ! -d "$MODEL_PATH" ]; then
    echo "Downloading Vosk model..."
    if [ ! -f "${MODEL_PATH}.zip" ]; then
        curl -O https://alphacephei.com/vosk/models/${MODEL_PATH}.zip
    fi
    unzip ${MODEL_PATH}.zip
fi

# Create output directory
mkdir -p output

# Run the speech recognizer
echo "Starting speech recognition..."
python3 speech_recognizer.py