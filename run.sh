#!/bin/bash

# Define variables
VENV_DIR="venv"
REQUIREMENTS_FILE="requirements.txt"
MAIN_APP="app.py"

# Function to print colored output
print_info() {
    echo -e "\e[34m[INFO]\e[0m $1"
}

print_success() {
    echo -e "\e[32m[SUCCESS]\e[0m $1"
}

print_error() {
    echo -e "\e[31m[ERROR]\e[0m $1"
}

# 1. Create Virtual Environment
if [ ! -d "$VENV_DIR" ]; then
    print_info "Creating virtual environment in '$VENV_DIR'..."
    python -m venv $VENV_DIR
    if [ $? -ne 0 ]; then
        print_error "Failed to create virtual environment. Ensure python is installed."
        exit 1
    fi
    print_success "Virtual environment created."
else
    print_info "Virtual environment '$VENV_DIR' already exists."
fi

# 2. Activate Virtual Environment
print_info "Activating virtual environment..."
# Determine OS for activation path
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source $VENV_DIR/Scripts/activate
else
    source $VENV_DIR/bin/activate
fi

# 3. Install Dependencies
if [ -f "$REQUIREMENTS_FILE" ]; then
    print_info "Installing dependencies from '$REQUIREMENTS_FILE'..."
    pip install --upgrade pip
    pip install -r $REQUIREMENTS_FILE
    if [ $? -ne 0 ]; then
        print_error "Failed to install dependencies."
        exit 1
    fi
    print_success "Dependencies installed successfully."
else
    print_error "Requirements file '$REQUIREMENTS_FILE' not found!"
    exit 1
fi

# 4. Run the Application
print_info "Starting Streamlit application..."
streamlit run $MAIN_APP
