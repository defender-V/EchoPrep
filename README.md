# 🎤 EchoPrep

**EchoPrep** is an AI-powered mock interview practice application that goes beyond evaluating *what* you say and analyzes *how* you say it. 

By unifying Large Language Models (Gemini) with custom-trained Deep Learning Computer Vision/Audio models, EchoPrep acts as a holistic interview coach that provides actionable feedback on your technical accuracy, vocal delivery, and emotional state.

## 🎯 The Problem Statement

Traditional mock interview tools and LLM chatbots rely exclusively on text transcriptions. While they can tell you if your answer to "What is Python?" is technically correct, they completely miss the social and psychological cues that dictate real-world interview success. 

In a real interview, an applicant might give the perfectly correct answer while sounding terrified, using excessive filler words, or exhibiting vocal tremors. Current tools ignore this entirely. Candidates need a way to practice maintaining a confident "vibe" under pressure, receiving feedback not just on their knowledge, but their delivery.

## 💡 The Use Case

**EchoPrep** solves this by acting as a real-time, interactive HR screen:
1. **Dynamic Questioning:** The user pastes a real-world Job Description into the app. Gemini analyzes the requirements and generates tailored, domain-specific interview questions.
2. **Multimodal Input:** The user answers the questions out loud. The app captures the raw audio and transcribes it instantly.
3. **The "Vibe Check" Engine:** A custom-trained PyTorch Convolutional Neural Network (EmotionCNN2D trained on the RAVDESS dataset) analyzes the user's Mel-spectrograms. It classifies the primary vocal emotion (e.g., Calm, Happy, Fearful, Angry) and aggregates this with text heuristics to generate live **Confidence** and **Stress** scores. 
4. **Holistic Feedback:** Gemini evaluates the transcribed text *alongside* the hidden Vibe Check metrics, returning a final Score out of 10 and qualitative feedback that directly addresses both the content of the answer and the user's emotional delivery. 

## 🚀 Key Features

* **Custom PyTorch Engine:** A ~8.78 million parameter 2D CNN that runs locally to process audio feature extraction (Librosa) in real-time.
* **Streamlit Interface:** A clean, responsive frontend that tracks your scores across the interview lifecycle and maps progression charts.
* **LangChain Orchestration:** Robust LLM chaining wrapping the Gemini 2.5 Flash model for natural conversation loops.
* **Cross-Platform Setup:** Includes automated Makefiles and Shell/Batch scripts to build the virtual environment and manage dependencies on both Windows and WSL/Linux.

## 🛠️ Installation & Setup

1. Clone the repository.
2. Ensure you have Python 3.12+ installed. 
3. Create a `.env` file in the root directory and add your Google API key:
   ```env
   GOOGLE_API_KEY="your_api_key_here"
   ```
4. Build the environment and start the server:
   * **Linux / WSL / Mac:** Run `make run`
   * **Windows (PowerShell/CMD):** Run `.\run.bat`

## 🧠 Architecture Highlights

The core Vibe Service utilizes an `AdaptiveAvgPool2d` pooling layer, allowing the model to dynamically process spoken answers of anywhere from 1 to 60 seconds without breaking tensor geometries. The training loop features automatic learning-rate reduction on plateaus and validation-loss checkpointing.
