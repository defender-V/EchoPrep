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

# 7. Check if the trained model exists (or can be restored from a zip)
MODEL_PATH="ml/models/best_ser.pth"
MODEL_ZIP_PATH="ml/models/best_ser.zip"

if [ ! -f "$MODEL_PATH" ]; then
    echo "⚠️  Warning: $MODEL_PATH not found!"

    if [ -f "$MODEL_ZIP_PATH" ]; then
        echo "📦 Found $MODEL_ZIP_PATH. Attempting to extract model..."
        python3 - <<'PY'
import os
import zipfile

zip_path = "ml/models/best_ser.zip"
target_dir = "ml/models"
target_model = os.path.join(target_dir, "best_ser.pth")

os.makedirs(target_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(target_dir)

if os.path.exists(target_model):
    print(f"✅ Extracted model to {target_model}")
else:
    print("⚠️  Zip extracted, but best_ser.pth was not found inside it.")
PY
    fi

    if [ ! -f "$MODEL_PATH" ]; then
        echo "⚠️  The vibe check will run in degraded mode (text heuristics only) until you place the model file in ml/models/."
    else
        echo "✅ Pre-trained model (best_ser.pth) restored from zip."
    fi
else
    echo "✅ Pre-trained model (best_ser.pth) found."
fi

# 8. Start Streamlit Server
echo "=========================================="
echo "✨ Starting Streamlit server..."
echo "=========================================="
streamlit run app.py