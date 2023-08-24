import numpy as np
import matplotlib.pyplot as plt

def plot_loss_from_npy(train_loss_npy_path, val_loss_npy_path):
    train_losses = np.load(train_loss_npy_path)
    val_losses = np.load(val_loss_npy_path)

    min_val_loss_idx = np.argmin(val_losses)
    min_val_loss = val_losses[min_val_loss_idx]
    min_val_loss = min_val_loss/(205)

    plt.figure()
    plt.plot(range(len(train_losses)), train_losses, "b", label='Training Loss')
    plt.plot(range(len(val_losses)), val_losses, "r", label='Validation Loss')
    plt.scatter(min_val_loss_idx, min_val_loss, color='k', label='Minimum Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Training Loss, Min Val Loss: {min_val_loss:.4f} at Epoch {min_val_loss_idx}")
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    plot_loss_from_npy("train_models/old/cnn-lstm_30_64-128_3000/trn_loss.npy","train_models/old/cnn-lstm_30_64-128_3000/val_loss.npy")
    