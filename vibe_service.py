import os
import torch
import librosa
import numpy as np
import tempfile
from ml.ser_model import EmotionCNNLSTM

class VibeService:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']
        self.emotion_map_reverse = {idx: label for idx, label in enumerate(self.emotion_labels)}
        
        # Determine the absolute path to the model file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, "ml", "models", "best_ser.pth")
        
        self.load_model()

    def load_model(self):
        """Loads the trained PyTorch model if it exists."""
        try:
            if os.path.exists(self.model_path):
                print(f"Loading Vibe Check model from {self.model_path}...")
                checkpoint = torch.load(self.model_path, map_location=self.device)

                # Allow loading either full training checkpoints or plain state_dict files.
                state_dict = checkpoint.get('model_state_dict') if isinstance(checkpoint, dict) else checkpoint
                arch = checkpoint.get('arch', {}) if isinstance(checkpoint, dict) else {}
                checkpoint_labels = checkpoint.get('emotion_labels') if isinstance(checkpoint, dict) else None

                if isinstance(checkpoint_labels, list) and checkpoint_labels:
                    self.emotion_labels = checkpoint_labels
                    self.emotion_map_reverse = {idx: label for idx, label in enumerate(self.emotion_labels)}

                self.model = EmotionCNNLSTM(
                    num_classes=arch.get('num_classes', len(self.emotion_labels)),
                    lstm_hidden=arch.get('lstm_hidden', 256),
                    lstm_layers=arch.get('lstm_layers', 2),
                    lstm_dropout=arch.get('lstm_dropout', 0.3),
                )
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                print("Model loaded successfully.")
            else:
                print(f"Warning: Model not found at {self.model_path}. Vibe Service will run in degraded mode.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def extract_features(self, audio_bytes: bytes):
        """Extracts Mel-spectrogram from raw audio bytes using overlapping windows."""
        try:
            # We need to save the bytes to a temporary file because librosa prefers file paths
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_wav.write(audio_bytes)
                temp_wav_path = temp_wav.name
            
            # Load audio (target 3-second chunks at 22050 Hz)
            sample_rate = 22050
            chunk_len = int(sample_rate * 3.0)
            step_len = int(sample_rate * 1.5)  # 1.5-second step (50% overlap)
            
            y, sr = librosa.load(temp_wav_path, sr=sample_rate)
            
            # Clean up temp file
            os.unlink(temp_wav_path)
            
            # Pad if shorter than 3 seconds
            if len(y) < chunk_len:
                padding = chunk_len - len(y)
                y = np.pad(y, (0, padding), 'constant')
                
            mel_tensors = []
            
            # Slide the 3-second window across the audio
            for start in range(0, len(y) - chunk_len + 1, step_len):
                chunk_y = y[start : start + chunk_len]
                
                # Extract Mel spectrogram (matching training script parameters exactly)
                mel_spec = librosa.feature.melspectrogram(
                    y=chunk_y, 
                    sr=sample_rate, 
                    n_mels=128, 
                    fmax=8000,
                    n_fft=2048,
                    hop_length=512
                )
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                # Create tensor map (1, 1, 128, ~130) -> (Batch, Channel, Height, Width)
                mel_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                mel_tensors.append(mel_tensor)
            
            # Fallback for edge cases
            if not mel_tensors:
                print("Warning: Audio too short or windowing failed, returning zero tensor.")
                return torch.zeros((1, 1, 128, 130), dtype=torch.float32)
            
            # Stack into a final batch
            return torch.cat(mel_tensors, dim=0)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def analyze_vibe(self, text: str, audio_bytes: bytes) -> str:
        """
        Analyzes the candidate's answer using the trained PyTorch ML model for audio,
        and falls back to simple text heuristics if audio/model is unavailable.
        """
        # 1. Text-based heuristic (filler words)
        filler_words = ["um", "uh", "like", "you know", "sort of", "kind of", "basically"]
        text_lower = text.lower()
        filler_count = sum(text_lower.count(fw) for fw in filler_words)
        word_count = len(text.split())
        filler_ratio = filler_count / word_count if word_count > 0 else 0
        
        # 2. ML Audio Analysis
        ml_prediction = "N/A"
        ml_confidence = 0.0
        
        if audio_bytes and self.model is not None:
            features = self.extract_features(audio_bytes)
            if features is not None:
                features = features.to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(features)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    
                    # Compute average probabilities across all 3-second chunks
                    avg_probabilities = probabilities.mean(dim=0, keepdim=True)
                    
                    # Get top prediction from the holistic average
                    top_p, top_class = avg_probabilities.topk(1, dim=1)
                    ml_prediction = self.emotion_map_reverse.get(top_class.item(), 'Unknown')
                    ml_confidence = top_p.item() * 100
        
        # 3. Aggregate Metrics
        # Map ML prediction to Confidence/Stress scores
        base_confidence = 80 - (filler_ratio * 150)
        base_stress = 20 + (filler_ratio * 100)
        
        # Strict matching to CREMA-D dataset labels
        if ml_prediction in ['Neutral', 'Happy']:
            base_confidence += 15
            base_stress -= 15
        elif ml_prediction in ['Angry', 'Fear', 'Sad']:
            base_confidence -= 20
            base_stress += 25
        elif ml_prediction in ['Disgust']:
            base_confidence -= 5
            base_stress += 10
            
        confidence = max(0, min(100, int(base_confidence)))
        stress = max(0, min(100, int(base_stress)))
        tremor_detected = stress > 75 or ml_prediction in ['Fear', 'Sad']
        
        metrics = [
            f"- Primary Vocal Emotion: {ml_prediction} (Model Confidence: {ml_confidence:.1f}%)",
            f"- Overall Confidence Score: {confidence}/100",
            f"- Stress Level: {stress}/100",
            f"- Filler Word Frequency: {'High' if filler_ratio > 0.05 else 'Low'} ({filler_count} detected)",
            f"- Voice Tremor Detected: {'Yes' if tremor_detected else 'No'}"
        ]
        
        return "\n".join(metrics)