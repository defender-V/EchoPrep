import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from ml.ser_dataset import RAVDESSDataset
from ml.ser_model import EmotionCNN2D

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
DATA_DIR = r"/home/navodit/my_work/RAVDESS"
MODEL_SAVE_DIR = r"ml\models"

def train():
    print("Setting up device...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading dataset...")
    full_dataset = RAVDESSDataset(root_dir=DATA_DIR)
    
    if len(full_dataset) == 0:
        print("ERROR: No files loaded. Check the dataset path.")
        return

    # Split into train and validation sets (80/20 split)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Set seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    print(f"Dataset split: {train_size} training samples, {val_size} validation samples.")

    # Create DataLoaders
    # Note: num_workers=0 on Windows is generally safer to avoid multi-processing hangs
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("Initializing Model...")
    model = EmotionCNN2D(num_classes=8).to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # Added weight decay (L2 regularization)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Ensure save directory exists
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    best_model_path = os.path.join(MODEL_SAVE_DIR, "best_ser.pth")
    
    best_val_loss = float('inf')

    print("--- Starting Training ---")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * features.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = 100 * correct / total
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Summary:")
        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")
        print(f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%")
        
        # Step the scheduler
        scheduler.step(val_epoch_loss)

        # Save the model if validation loss goes down
        if val_epoch_loss < best_val_loss:
            print(f"Validation loss decreased ({best_val_loss:.4f} --> {val_epoch_loss:.4f}). Saving model...")
            best_val_loss = val_epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, best_model_path)
            
    print("--- Training Completed ---")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best model saved to {best_model_path}")

if __name__ == "__main__":
    import multiprocessing
    # Windows-specific fix for multiprocessing if num_workers > 0
    multiprocessing.freeze_support()
    train()
