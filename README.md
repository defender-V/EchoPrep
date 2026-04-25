# 🎤 EchoPrep: AI-Powered Interview Simulator

EchoPrep is an interactive, AI-driven interview practice application. It analyzes a job description to generate tailored interview questions and evaluates candidates based on two dimensions: what they say (semantic analysis via LLM) and how they say it (vocal delivery via a custom Speech Emotion Recognition model).

---

## ✨ Features

- Tailored Interview Plans: Paste any Job Description (JD) to automatically generate mixed, technical, or behavioral interview questions.
- Multimodal Input: Answer questions by speaking directly into the microphone or by typing.
- Real-Time Vibe Check (SER): Analyzes the candidate's voice using a custom PyTorch CNN-BiLSTM model to detect emotions (Neutral, Happy, Angry, Fear, Sad, Disgust), confidence, and stress levels.
- Comprehensive AI Evaluation: Combines the vocal delivery metrics with Google's Gemini 1.5 Flash LLM to provide holistic scoring and actionable feedback.
- Training Pipeline Included: Full machine learning pipeline to train the Speech Emotion Recognition (SER) model on the CREMA-D dataset from scratch.
- Automated Setup: One-click launch script to handle environments, dependencies, and server startup.

---

## 📂 Project Structure

```text
EchoPrep/
│
├── .env                        # Environment variables (Google API Key)
├── run.sh                      # One-click setup and launch script
├── requirements.txt            # Python dependencies
├── app.py                      # Main Streamlit application frontend
├── llm_service.py              # LangChain integration with Gemini 1.5 Flash
├── prompts.py                  # Prompt templates for JD parsing and evaluation
├── vibe_service.py             # Inference engine for vocal emotion analysis
├── voice_service.py            # Speech-to-Text (STT) and Text-to-Speech (TTS)
│
└── ml/                         # Machine Learning Directory
    ├── models/
    │   └── best_ser.pth        # Saved PyTorch model weights (generated after training)
    ├── ser_dataset.py          # CREMA-D dataset loader, cropping, and augmentation
    ├── ser_model.py            # EmotionCNNLSTM PyTorch architecture
    ├── train_ser.py            # Training script with early stopping and checkpoints
    ├── plot_metrics.py         # Utility to visualize training curves
    └── training_log.txt        # Training metrics log (Epoch, Loss, Accuracy)
```

---

## 🧠 Model Architecture (Speech Emotion Recognition)

The Vibe Check engine runs on a custom CNN + Bidirectional LSTM model with a Temporal Attention Mechanism.

1. Preprocessing: Audio is sliced into overlapping 3-second windows and converted into Mel-spectrograms (128 bins).
2. CNN Feature Extractor: 3 Convolutional Blocks with asymmetric pooling extract spectral features while preserving time steps.
3. Sequence Modeling: A Bidirectional LSTM (2 layers, 256 hidden units) captures temporal context across the audio chunk.
4. Attention Mechanism: Focuses on the most emotionally charged time-steps in the sequence.
5. Classification: A Fully Connected head maps the context vector to 6 emotion probabilities.

---

## 🚀 Installation and Launch (The Easy Way)

We have included a run.sh script that automatically handles creating a virtual environment, installing all dependencies from requirements.txt, setting up a placeholder .env file, and launching the app.

### 1. Clone the repository

Ensure you have Python 3.9+ installed and clone this directory to your machine.

### 2. Make the script executable

Open your terminal in the root of the project and run:

```bash
chmod +x run.sh
```

### 3. Run the app

```bash
./run.sh
```

Note: The first time you run this, it will create a .env file for you. You must open that .env file and replace the placeholder text with your actual Google Gemini API key, then run ./run.sh again to start the app properly.

---

## 🏋️ Training the ML Model (Optional)

If you do not have the pre-trained best_ser.pth file, or want to retrain the model from scratch on the CREMA-D dataset:

1. Navigate to the ml directory:

```bash
cd ml
```

2. Run the training script (it will automatically download the CREMA-D dataset):

```bash
python train_ser.py
```

3. View the training curves once complete:

```bash
python plot_metrics.py
```

---

## 🎯 Using the EchoPrep App

1. Paste a Job Description into the sidebar.
2. Select your interview preferences (Number of questions, Technical vs. Behavioral).
3. Click Start Interview.
4. Listen to the AI interviewer, record your response, and view your holistic evaluation.
