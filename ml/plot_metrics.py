import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the path to your log file
log_path = "training_log.txt"

if not os.path.exists(log_path):
    print(f"Error: Could not find {log_path}. Make sure you are in the correct directory.")
    exit()

# 1. Load the data
# We use sep='\t' because we saved it as a tab-separated file
df = pd.read_csv(log_path, sep='\t')

# 2. Clean the accuracy columns (remove the '%' sign and convert to float)
df['Train_Acc'] = df['Train_Acc'].astype(str).str.rstrip('%').astype(float)
df['Val_Acc'] = df['Val_Acc'].astype(str).str.rstrip('%').astype(float)

# 3. Find the epoch with the best validation accuracy (for our vertical marker)
best_epoch = df.loc[df['Val_Acc'].idxmax(), 'Epoch']
best_val_acc = df['Val_Acc'].max()

# 4. Set up the plotting figure (1 row, 2 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Plot 1: Accuracy ---
ax1.plot(df['Epoch'], df['Train_Acc'], label='Training Accuracy', color='#2563eb', linewidth=2)
ax1.plot(df['Epoch'], df['Val_Acc'], label='Validation Accuracy', color='#ea580c', linewidth=2)
ax1.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.7, label=f'Best Model (Epoch {best_epoch})')

ax1.set_title('Model Accuracy over Epochs', fontsize=14, pad=10)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='lower right')

# --- Plot 2: Loss ---
ax2.plot(df['Epoch'], df['Train_Loss'], label='Training Loss', color='#2563eb', linewidth=2)
ax2.plot(df['Epoch'], df['Val_Loss'], label='Validation Loss', color='#ea580c', linewidth=2)
ax2.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.7, label=f'Best Model (Epoch {best_epoch})')

ax2.set_title('Model Loss over Epochs', fontsize=14, pad=10)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Cross Entropy Loss', fontsize=12)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper right')

# 5. Final layout adjustments and save/show
plt.tight_layout()

# Save the plot as an image file in your current directory
save_path = "training_curves.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Graph successfully saved to: {save_path}")

# Display the plot in the window
plt.show()