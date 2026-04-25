import os
import glob
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np


class CREMADDataset(Dataset):
    """
    CREMA-D dataset loader that returns Mel-spectrograms suitable for
    the CNN-LSTM model.

    CREMA-D filename format
    -----------------------
        [ActorID]_[SentenceKey]_[Emotion]_[Intensity].wav
        e.g.  1001_DFA_ANG_XX.wav

    Emotions (6 classes)
    --------------------
        ANG = Angry    → 0
        DIS = Disgust  → 1
        FEA = Fear     → 2
        HAP = Happy    → 3
        NEU = Neutral  → 4
        SAD = Sad      → 5

    Intensities (ignored for label; available for optional filtering)
        LO = Low | MD = Medium | HI = High | XX = Unspecified

    The time axis of the spectrogram is intentionally preserved so the
    downstream LSTM can learn temporal patterns across the utterance.
    """

    EMOTION_MAP: dict = {
        'ANG': 0,
        'DIS': 1,
        'FEA': 2,
        'HAP': 3,
        'NEU': 4,
        'SAD': 5,
    }
    EMOTION_LABELS: list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']
    NUM_CLASSES: int = 6

    def __init__(
        self,
        root_dir: str,
        transform=None,
        sample_rate: int = 22050,
        max_duration: float = 3.0,
        intensity_filter=None,
    ):
        """
        Parameters
        ----------
        root_dir         : Folder containing the .wav files.
                           Pass the path returned by:
                               kagglehub.dataset_download("ejlok1/cremad")
                           The loader searches recursively, so the top-level
                           path or the AudioWAV sub-folder both work.
        transform        : Optional callable applied to each spectrogram tensor.
        sample_rate      : Target sample rate for loading audio.
        max_duration     : Clips are padded / truncated to this length (seconds).
        intensity_filter : Optional list of intensity codes to include,
                           e.g. ['MD', 'HI'] to skip low-intensity recordings.
                           Pass None (default) to keep all intensities.
        """
        self.root_dir         = root_dir
        self.transform        = transform
        self.sample_rate      = sample_rate
        self.max_len          = int(sample_rate * max_duration)
        self.intensity_filter = set(intensity_filter) if intensity_filter else None

        self.file_paths: list = []
        self.labels:     list = []

        print(f"Loading CREMA-D dataset from: {self.root_dir}")
        self._scan_files()
        print(f"Found {len(self.file_paths)} valid audio files "
              f"across {self.NUM_CLASSES} emotion classes.")
        self._print_class_distribution()

    # ---------------------------------------------------------------------- #
    #  Internal helpers                                                        #
    # ---------------------------------------------------------------------- #
    def _scan_files(self) -> None:
        wav_files = glob.glob(
            os.path.join(self.root_dir, '**', '*.wav'), recursive=True
        )

        skipped_format    = 0
        skipped_emotion   = 0
        skipped_intensity = 0

        for file_path in sorted(wav_files):
            stem  = os.path.splitext(os.path.basename(file_path))[0]
            parts = stem.split('_')

            # Expected exactly 4 parts: ActorID, SentenceKey, Emotion, Intensity
            if len(parts) != 4:
                skipped_format += 1
                continue

            _, _, emotion_code, intensity_code = parts

            if emotion_code not in self.EMOTION_MAP:
                skipped_emotion += 1
                continue

            if self.intensity_filter and intensity_code not in self.intensity_filter:
                skipped_intensity += 1
                continue

            self.file_paths.append(file_path)
            self.labels.append(self.EMOTION_MAP[emotion_code])

        if skipped_format:
            print(f"  [info] Skipped {skipped_format} file(s): unexpected filename format.")
        if skipped_emotion:
            print(f"  [info] Skipped {skipped_emotion} file(s): unknown emotion code.")
        if skipped_intensity:
            print(f"  [info] Skipped {skipped_intensity} file(s): filtered by intensity.")

    def _print_class_distribution(self) -> None:
        counts = [0] * self.NUM_CLASSES
        for lbl in self.labels:
            counts[lbl] += 1
        print("  Class distribution:")
        for idx, name in enumerate(self.EMOTION_LABELS):
            bar = '█' * (counts[idx] // 50)
            print(f"    {name:10s} ({idx}): {counts[idx]:>5d}  {bar}")

    # ---------------------------------------------------------------------- #
    #  Dataset interface                                                       #
    # ---------------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.file_paths)

    def extract_mel_spectrogram(self, file_path: str) -> torch.Tensor:
        try:
            y, _ = librosa.load(file_path, sr=self.sample_rate)

            # --- IMPROVEMENT 1: Random Cropping for Training ---
            if len(y) > self.max_len:
                # Pick a random starting point instead of just truncating the end
                max_start = len(y) - self.max_len
                start = np.random.randint(0, max_start)
                y = y[start : start + self.max_len]
            else:
                y = np.pad(y, (0, self.max_len - len(y)), mode='constant')

            # --- IMPROVEMENT 2: Add Data Augmentation (Noise) ---
            # Randomly add slight background noise 50% of the time to simulate real mics
            if np.random.rand() < 0.5:
                noise_amp = 0.005 * np.random.uniform() * np.amax(y)
                y = y + noise_amp * np.random.normal(size=y.shape[0])

            # Mel-spectrogram
            mel = librosa.feature.melspectrogram(
                y=y, sr=self.sample_rate, n_mels=128, fmax=8000, n_fft=2048, hop_length=512
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)

            return torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)

        except Exception as exc:
            print(f"[WARN] Error processing {file_path}: {exc}")
            expected_T = (self.max_len // 512) + 1
            return torch.zeros((1, 128, expected_T), dtype=torch.float32)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        feature = self.extract_mel_spectrogram(self.file_paths[idx])

        if self.transform:
            feature = self.transform(feature)

        return feature, torch.tensor(self.labels[idx], dtype=torch.long)


# -------------------------------------------------------------------------- #
#  Quick sanity-check                                                          #
# -------------------------------------------------------------------------- #
if __name__ == "__main__":
    import kagglehub

    path = kagglehub.dataset_download("ejlok1/cremad")
    print("Dataset path:", path)

    ds = CREMADDataset(root_dir=path)

    if len(ds) > 0:
        feat, lbl = ds[0]
        print(f"\nFeature shape : {feat.shape}")   # expect (1, 128, ~130)
        print(f"Label         : {lbl.item()}  ({CREMADDataset.EMOTION_LABELS[lbl.item()]})")
    else:
        print("No files found — check the dataset path.")