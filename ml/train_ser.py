import os
import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from ser_dataset import CREMADDataset
from ser_model   import EmotionCNNLSTM

# ─────────────────────────────────────────────────────────────────────────── #
#  Hyperparameters                                                              #
# ─────────────────────────────────────────────────────────────────────────── #
BATCH_SIZE    = 32
LEARNING_RATE = 0.001
EPOCHS        = 100
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models")
TRAINING_LOG_PATH = os.path.join(BASE_DIR, "training_log.txt")
EARLY_STOPPING_PATIENCE = 12

# CNN-LSTM architecture
LSTM_HIDDEN  = 256
LSTM_LAYERS  = 2
LSTM_DROPOUT = 0.3

# CREMA-D — 6 emotion classes
NUM_CLASSES = 6

# Optional: restrict to specific recording intensities.
# Set to None to keep all intensities (recommended to start).
# Example: ['MD', 'HI']  to use only medium & high intensity recordings.
INTENSITY_FILTER = None


# ─────────────────────────────────────────────────────────────────────────── #
def train():
    # ── Device ────────────────────────────────────────────────────────────── #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device : {device}")

    # ── Download / locate CREMA-D via kagglehub ───────────────────────────── #
    print("Fetching CREMA-D dataset via kagglehub …")
    data_dir = kagglehub.dataset_download("ejlok1/cremad")
    print(f"Dataset path : {data_dir}")

    # ── Dataset ───────────────────────────────────────────────────────────── #
    full_dataset = CREMADDataset(
        root_dir         = data_dir,
        intensity_filter = INTENSITY_FILTER,
    )

    if len(full_dataset) == 0:
        print("ERROR: No files loaded. Check the dataset path or kagglehub setup.")
        return

    train_size = int(0.8 * len(full_dataset))
    val_size   = len(full_dataset) - train_size
    generator  = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    print(f"\nSplit → {train_size} train / {val_size} val")

    # ── DataLoaders ───────────────────────────────────────────────────────── #
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ── Model ─────────────────────────────────────────────────────────────── #
    print("\nInitialising EmotionCNNLSTM …")
    model = EmotionCNNLSTM(
        num_classes  = NUM_CLASSES,
        lstm_hidden  = LSTM_HIDDEN,
        lstm_layers  = LSTM_LAYERS,
        lstm_dropout = LSTM_DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters : {total_params:,}")

    # ── Loss / Optimiser / Scheduler ─────────────────────────────────────── #
    
    # --- IMPROVEMENT 4: Class Weights ---
    # Based on CREMA-D counts: ANG:1271, DIS:1271, FEA:1271, HAP:1271, NEU:1087, SAD:1271
    class_counts = [1271.0, 1271.0, 1271.0, 1271.0, 1087.0, 1271.0]
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    weights = weights / weights.sum() * NUM_CLASSES # Normalize so sum equals num_classes
    weights = weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights) # Apply weights here
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )


    # ── Save setup & Early Stopping ───────────────────────────────────────── #
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    best_model_path = os.path.join(MODEL_SAVE_DIR, "best_ser.pth")
    
    best_val_acc = 0.0                  # For checkpointing (saving best model)
    best_val_loss_for_es = float('inf') # For early stopping
    epochs_no_improve = 0               # Counter for early stopping

    # ─────────────────────────────────────────────────────────────────────── #
    print("\n─── Starting Training ───\n")

    # --- NEW: Initialize the log file with headers ---
    with open(TRAINING_LOG_PATH, "w") as log_file:
        log_file.write("Epoch\tTrain_Loss\tTrain_Acc\tVal_Loss\tVal_Acc\n")

    for epoch in range(EPOCHS):

        # ── Train ─────────────────────────────────────────────────────────── #
        model.train()
        run_loss, correct, total = 0.0, 0, 0

        for step, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss    = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping — important for LSTM stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            run_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (step + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1:>3}/{EPOCHS}]  "
                      f"Step [{step+1:>3}/{len(train_loader)}]  "
                      f"loss: {loss.item():.4f}")

        train_loss = run_loss / train_size
        train_acc  = 100.0 * correct / total

        # ── Validation ────────────────────────────────────────────────────── #
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs  = model(features)
                loss     = criterion(outputs, labels)

                v_loss    += loss.item() * features.size(0)
                _, predicted = torch.max(outputs.data, 1)
                v_total   += labels.size(0)
                v_correct += (predicted == labels).sum().item()

        val_loss = v_loss / val_size
        val_acc  = 100.0 * v_correct / v_total

        print(f"\nEpoch [{epoch+1:>3}/{EPOCHS}] Summary")
        print(f"  Train  →  loss: {train_loss:.4f}  acc: {train_acc:.2f}%")
        print(f"  Val    →  loss: {val_loss:.4f}  acc: {val_acc:.2f}%\n")

        # --- NEW: Append the metrics with tab spaces (\t) ---
        with open(TRAINING_LOG_PATH, "a") as log_file:
            log_file.write(f"{epoch+1}\t{train_loss:.4f}\t{train_acc:.2f}%\t{val_loss:.4f}\t{val_acc:.2f}%\n")
        
        scheduler.step(val_loss)

        # ── Early Stopping (Based on Validation Loss) ─────────────────────── #
        if val_loss < best_val_loss_for_es:
            best_val_loss_for_es = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  [Info] Early stopping counter: {epochs_no_improve} / {EARLY_STOPPING_PATIENCE}")

        # ── Checkpoint (Based on Validation Accuracy) ─────────────────────── #
        if val_acc > best_val_acc:
            print(f"  ✓ Val accuracy improved ({best_val_acc:.2f}% → {val_acc:.2f}%). Saving …\n")
            best_val_acc = val_acc
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':             val_loss,
                'val_acc':              best_val_acc,
                # Stored so you can rebuild the model at inference time
                'arch': {
                    'num_classes':  NUM_CLASSES,
                    'lstm_hidden':  LSTM_HIDDEN,
                    'lstm_layers':  LSTM_LAYERS,
                    'lstm_dropout': LSTM_DROPOUT,
                },
                # CREMA-D label mapping for reference
                'emotion_labels': CREMADDataset.EMOTION_LABELS,
            }, best_model_path)
        else:
            print("\n") # Just for clean terminal formatting if it didn't save

        # ── Trigger Early Stopping ────────────────────────────────────────── #
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\n[!] Early stopping triggered after {epoch + 1} epochs.")
            print(f"    No validation loss improvement for {EARLY_STOPPING_PATIENCE} consecutive epochs.")
            break

    print("─── Training Complete ───")
    print(f"Best val accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to:    {best_model_path}")


# ─────────────────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()   # Windows safety guard
    train()