#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=========================================="
echo "🚀 Starting EchoPrep Setup & Launch..."
echo "=========================================="

# 1. Verify we are in the correct directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found! Please run this script from the root of the EchoPrep project."
    exit 1
fi

# 2. Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found. Creating a template..."
    echo "GOOGLE_API_KEY=your_actual_api_key_here" > .env
    echo "👉 Please update the newly created .env file with your actual Google Gemini API key before using the app!"
fi

# 3. Create Virtual Environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment (venv)..."
    python3 -m venv venv
else
    echo "✅ Virtual environment 'venv' already exists."
fi

# 4. Activate Virtual Environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# 5. Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet

# 6. Install Dependencies
echo "📥 Installing required packages from requirements.txt (this might take a minute)..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --quiet
    echo "✅ Packages installed successfully."
else
    echo "❌ Error: requirements.txt not found!"
    exit 1
fi

# 7. Check if the trained model exists
if [ ! -f "ml/models/best_ser.pth" ]; then
    echo "⚠️  Warning: ml/models/best_ser.pth not found!"
    echo "    The vibe check will run in degraded mode (text heuristics only) until you place the model file in that folder."
else
    echo "✅ Pre-trained model (best_ser.pth) found."
fi

# 8. Start Streamlit Server
echo "=========================================="
echo "✨ Starting Streamlit server..."
echo "=========================================="
streamlit run app.py