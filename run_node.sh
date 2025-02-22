#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Setting up Real-time Speech Recognition...${NC}"

# Check for Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is not installed. Please install Node.js first.${NC}"
    exit 1
fi

# Check for sox
if ! command -v sox &> /dev/null; then
    echo -e "${RED}❌ Sox is not installed. Installing...${NC}"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install sox
    else
        sudo apt-get update && sudo apt-get install -y sox
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo -e "${GREEN}📦 Creating Python virtual environment for edge-tts...${NC}"
    python3 -m venv .venv
fi

# Activate virtual environment and install edge-tts
source .venv/bin/activate
if ! command -v edge-tts &> /dev/null; then
    echo -e "${GREEN}📦 Installing edge-tts...${NC}"
    pip install edge-tts
fi

# Install Node.js dependencies
echo -e "${GREEN}📦 Installing Node.js dependencies...${NC}"
npm install

# Ensure output directory exists
mkdir -p output

# Download Vosk model if needed
MODEL_PATH="vosk-model-en-us-0.42-gigaspeech"
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${GREEN}📥 Downloading Vosk model...${NC}"
    if [ ! -f "${MODEL_PATH}.zip" ]; then
        curl -O https://alphacephei.com/vosk/models/${MODEL_PATH}.zip
    fi
    unzip ${MODEL_PATH}.zip
fi

# Run the application
echo -e "${GREEN}✨ Starting speech recognition...${NC}"
node realtimespeech.js