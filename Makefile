.PHONY: setup run test-voice clean help

VENV_DIR = venv
REQUIREMENTS = requirements.txt
MAIN_APP = app.py

# Default target
help:
	@echo "Available commands:"
	@echo "  make setup      - Create a virtual environment and install dependencies"
	@echo "  make run        - Run the Streamlit application (calls setup first if needed)"
	@echo "  make test-voice - Run the standalone test_voice.py application"
	@echo "  make clean      - Remove the virtual environment and cached files"

# Create virtual environment and install dependencies
setup: $(VENV_DIR)/touchfile

$(VENV_DIR)/touchfile: $(REQUIREMENTS)
	@echo "Setting up virtual environment and installing dependencies..."
	@if command -v python3 >/dev/null 2>&1; then \
		python3 -m venv $(VENV_DIR); \
	else \
		python -m venv $(VENV_DIR); \
	fi
	# Use different path for Windows vs. Linux/Mac
	@if [ -d "$(VENV_DIR)/Scripts" ]; then \
		$(VENV_DIR)/Scripts/pip install --upgrade pip && \
		$(VENV_DIR)/Scripts/pip install -r $(REQUIREMENTS); \
	else \
		$(VENV_DIR)/bin/pip install --upgrade pip && \
		$(VENV_DIR)/bin/pip install -r $(REQUIREMENTS); \
	fi
	@touch $(VENV_DIR)/touchfile
	@echo "Setup complete!"

# Run the application
run: setup
	@echo "Starting Streamlit application..."
	@if [ -d "$(VENV_DIR)/Scripts" ]; then \
		$(VENV_DIR)/Scripts/streamlit run $(MAIN_APP); \
	else \
		$(VENV_DIR)/bin/streamlit run $(MAIN_APP); \
	fi

# Run the standalone voice tester
test-voice: setup
	@echo "Starting Vibe Check Tester..."
	@if [ -d "$(VENV_DIR)/Scripts" ]; then \
		$(VENV_DIR)/Scripts/streamlit run test_voice.py; \
	else \
		$(VENV_DIR)/bin/streamlit run test_voice.py; \
	fi

# Clean up environment
clean:
	@echo "Removing virtual environment and cache..."
	rm -rf $(VENV_DIR)
	rm -rf __pycache__
	rm -rf ml/__pycache__
	rm -rf ml/models/__pycache__
	@echo "Clean complete!"
