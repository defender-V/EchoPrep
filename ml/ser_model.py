import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN2D(nn.Module):
    """
    A 2D Convolutional Neural Network for processing Mel-spectrograms.
    Input shape is expected to be (N, C, H, W)
    Where:
        N = Batch Size
        C = 1 (Grayscale image representation of the spectrogram)
        H = 128 (n_mels)
        W = Audio length depending on max_duration and hop_length (e.g., ~130 for 3.0s)
    """
    def __init__(self, num_classes=8):
        super(EmotionCNN2D, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(0.2)
        
        # Block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(0.2)
        
        # Block 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(0.3)
        
        # Block 4
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout2d(0.3)
        
        # After 4 max pooling operations of 2x2:
        # H dimension (n_mels) goes from 128 -> 64 -> 32 -> 16 -> 8
        # W dimension (~130) goes from 130 -> 65 -> 32 -> 16 -> 8
        
        # AdaptiveAvgPool2d ensures the output is always fixed size (8x8)
        # before flattening, making the model robust to slightly variable audio lengths
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # Fully connected layers
        # 256 channels * 8 height * 8 width = 16384
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout5 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout6 = nn.Dropout(0.5)
        
        # Output layer
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Block 3
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Block 4
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = self.dropout4(x)
        
        # Adaptive pooling to handle varying width from different audio lengths
        x = self.adaptive_pool(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # FC Layers
        x = self.fc1(x)
        x = F.relu(self.bn_fc1(x))
        x = self.dropout5(x)
        
        x = self.fc2(x)
        x = F.relu(self.bn_fc2(x))
        x = self.dropout6(x)
        
        x = self.fc3(x)
        
        return x

# Quick test if run directly
if __name__ == "__main__":
    # Create a dummy tensor representing a batch of 4 mel-spectrograms
    # (batch_size, channels, n_mels, time_steps)
    dummy_input = torch.randn(4, 1, 128, 130)
    
    model = EmotionCNN2D(num_classes=8)
    output = model(dummy_input)
    
    print(f"Model successfully instantiated.")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape} (Expected: [4, 8])")
