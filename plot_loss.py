import numpy as np
import matplotlib.pyplot as plt

trn_loss = np.load('trn_loss.npy')
val_loss = np.load('val_loss.npy')

epochs = range(len(trn_loss))

plt.plot(epochs, trn_loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()
