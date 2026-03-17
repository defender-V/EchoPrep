import os
import glob
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np

class RAVDESSDataset(Dataset):
    def __init__(self, root_dir, transform=None, sample_rate=22050, max_duration=3.0):
        """
        Args:
            root_dir (string): Directory with all the Actor folders.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            sample_rate (int): Target sample rate for audio.
            max_duration (float): Maximum duration of audio in seconds.
                                  Shorter audio is padded, longer is truncated.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.sample_rate = sample_rate
        self.max_len = int(sample_rate * max_duration)
        
        # Mapping RAVDESS emotions to our integer labels (0-indexed)
        # 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
        self.emotion_map = {
            '01': 0, # Neutral
            '02': 1, # Calm
            '03': 2, # Happy
            '04': 3, # Sad
            '05': 4, # Angry
            '06': 5, # Fearful
            '07': 6, # Disgust
            '08': 7  # Surprised
        }
        
        self.file_paths = []
        self.labels = []
        
        print(f"Loading RAVDESS dataset from {self.root_dir}...")
        
        # Iterate over all Actor_* folders and .wav files
        search_pattern = os.path.join(self.root_dir, '**', '*.wav')
        wav_files = glob.glob(search_pattern, recursive=True)
        
        for file_path in wav_files:
            filename = os.path.basename(file_path)
            # Filename example: 03-01-05-01-01-01-01.wav
            parts = filename.split('-')
            
            if len(parts) >= 7:
                emotion_id = parts[2] # 3rd part is the emotion
                if emotion_id in self.emotion_map:
                    self.file_paths.append(file_path)
                    self.labels.append(self.emotion_map[emotion_id])
                    
        print(f"Found {len(self.file_paths)} audio files.")


    def __len__(self):
        return len(self.file_paths)

    def extract_mel_spectrogram(self, file_path):
        """Loads audio and extracts Mel-spectrogram."""
        try:
            # Load audio
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Pad or truncate to max_len
            if len(y) > self.max_len:
                y = y[:self.max_len]
            elif len(y) < self.max_len:
                padding = self.max_len - len(y)
                # Pad equally on both sides (or just at the end)
                y = np.pad(y, (0, padding), 'constant')
                
            # Extract Mel spectrogram
            # n_mels determines the height of the image
            # max_len // hop_length determines the width of the image
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=self.sample_rate, 
                n_mels=128, 
                fmax=8000
            ) # Default n_fft=2048, hop_length=512
            
            # Convert to decibels (log scale)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # mel_spec_db shape should be (128, 130) roughly for 3 sec at 22050hz
            # We want to return a tensor of shape (1, H, W) for a 2D CNN
            mel_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0)
            
            return mel_tensor
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Return a zero tensor if file is corrupt so training doesn't crash completely
            # Assuming expected shape based on 3.0 sec, sr=22050, hop_length=512 -> max_len=66150 -> 66150/512 = ~130 frames
            expected_width = (self.max_len // 512) + 1
            return torch.zeros((1, 128, expected_width), dtype=torch.float32)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        feature = self.extract_mel_spectrogram(file_path)
        
        if self.transform:
            feature = self.transform(feature)
            
        return feature, torch.tensor(label, dtype=torch.long)

# Quick test if run directly
if __name__ == "__main__":
    test_dir = r"C:\Users\navod\Desktop\RAVDESS"
    dataset = RAVDESSDataset(root_dir=test_dir)
    
    if len(dataset) > 0:
        sample_feature, sample_label = dataset[0]
        print(f"Sample Feature Shape: {sample_feature.shape}")
        print(f"Sample Label: {sample_label}")
    else:
        print("No files found. Please check the dataset path.")
