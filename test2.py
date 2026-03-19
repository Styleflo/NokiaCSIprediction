import numpy as np
import matplotlib.pyplot as plt

data = np.load("csi_data_train_flattened.npy")

reals = data[:, 0]
imags = data[:, 1]
magnitudes = np.sqrt(reals**2 + imags**2)

plt.figure(figsize=(10, 4))
plt.plot(magnitudes[:50], marker='o', markersize=2, linestyle='-')
plt.title("Vérification de la continuité temporelle (Magnitude)")
plt.xlabel("Index du fichier (Temps t)")
plt.ylabel("|H|")
plt.grid(True)
plt.show()