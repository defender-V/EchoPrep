import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionCNNLSTM(nn.Module):
    """
    CNN + Bidirectional LSTM hybrid for Speech Emotion Recognition.
    Designed for CREMA-D (6 emotion classes).

    Architecture
    ------------
    1. CNN Blocks  — extract local spectral & short-term temporal features
                     from the Mel-spectrogram treated as a 2-D image.
    2. Reshape     — convert CNN feature maps into a time-step sequence
                     that the LSTM can consume.
    3. BiLSTM      — capture long-range temporal dependencies in both
                     forward and backward directions.
    4. FC Head     — classify into num_classes emotion categories.

    Input shape  : (N, 1, 128, T)
        N   = batch size
        1   = single channel (grayscale Mel-spectrogram)
        128 = n_mels (frequency bins)
        T   = time frames (~130 for 3 s at sr=22050, hop_length=512)

    Output shape : (N, num_classes)  — raw logits (use CrossEntropyLoss)
    """

    def __init__(
        self,
        num_classes: int = 6,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
    ):
        super(EmotionCNNLSTM, self).__init__()

        # ------------------------------------------------------------------ #
        #  CNN Feature Extractor                                               #
        #                                                                      #
        #  Pool strategy (preserving the time axis for the LSTM):             #
        #    Block 1 → MaxPool(2,2): H 128→64,  W T→T/2                       #
        #    Block 2 → MaxPool(2,2): H  64→32,  W T/2→T/4                     #
        #    Block 3 → MaxPool(2,1): H  32→16,  W unchanged (T/4)             #
        #                                                                      #
        #  After CNN: feature map is (N, 128, 16, T/4)                        #
        #  Each time step has feature size = 128 * 16 = 2048                  #
        # ------------------------------------------------------------------ #

        # Block 1
        self.conv1 = nn.Conv2d(1,  32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)        # H/2, W/2
        self.drop1 = nn.Dropout2d(0.2)

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)        # H/4, W/4
        self.drop2 = nn.Dropout2d(0.2)

        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # H/8, W intact
        self.drop3 = nn.Dropout2d(0.3)

        self.cnn_freq_bins  = 16           # 128 // 2 // 2 // 2
        self.cnn_channels   = 128
        self.lstm_input_dim = self.cnn_channels * self.cnn_freq_bins  # 2048

        # ------------------------------------------------------------------ #
        #  Bidirectional LSTM                                                  #
        # ------------------------------------------------------------------ #
        self.lstm = nn.LSTM(
            input_size   = self.lstm_input_dim,
            hidden_size  = lstm_hidden,
            num_layers   = lstm_layers,
            batch_first  = True,           # (N, T, features)
            bidirectional= True,
            dropout      = lstm_dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_dropout = nn.Dropout(0.5)

        # ------------------------------------------------------------------ #
        #  Classification Head                                                 #
        #  BiLSTM output = lstm_hidden * 2  (forward + backward)              #
        # ------------------------------------------------------------------ #
        lstm_out_dim = lstm_hidden * 2     # 512 with default hidden=256

        # --- Temporal Attention Layer ---
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.fc1     = nn.Linear(lstm_out_dim, 256)
        self.bn_fc1  = nn.BatchNorm1d(256)
        self.drop_fc1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, num_classes)

    # ---------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, 128, T)

        # ── CNN ──────────────────────────────────────────────────────────── #
        x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        # x: (N, 128, 16, T//4)

        # ── Reshape for LSTM ─────────────────────────────────────────────── #
        N, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2)     # (N, W, C, H)  — time-first
        x = x.reshape(N, W, C * H)    # (N, T', 2048)

        # ── BiLSTM ───────────────────────────────────────────────────────── #
        x, _ = self.lstm(x)           # (N, T', lstm_hidden * 2)

        # ---Apply Attention instead of mean() ---
        # Calculate attention weights for each time step
        attn_weights = F.softmax(self.attention(x), dim=1)  # (N, T', 1)
        # Multiply weights by LSTM outputs and sum across time
        x = torch.sum(attn_weights * x, dim=1)              # (N, lstm_hidden * 2)
        
        x = self.lstm_dropout(x)

        # ── FC Head ──────────────────────────────────────────────────────── #
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.drop_fc1(x)
        x = self.fc2(x)               # (N, num_classes)

        return x


# -------------------------------------------------------------------------- #
#  Quick sanity-check                                                          #
# -------------------------------------------------------------------------- #
if __name__ == "__main__":
    dummy = torch.randn(4, 1, 128, 130)   # batch=4, simulated 3-second clip
    model = EmotionCNNLSTM(num_classes=6) # 6 classes for CREMA-D
    out   = model(dummy)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model           : EmotionCNNLSTM (CREMA-D)")
    print(f"Input  shape    : {dummy.shape}")
    print(f"Output shape    : {out.shape}  (expected [4, 6])")
    print(f"Trainable params: {total_params:,}")