import h5py
import numpy as np
import os


def process_augmented_data(file_path, seq_length=10, pred_step=1):
    print(f"Chargement du fichier HDF5 (v7.3) : {file_path}...")
    try:
        # On utilise h5py à la place de scipy
        with h5py.File(file_path, 'r') as f:
            # Accès à la variable H_new
            # Attention : H_new est stocké en (2, 8, 28, 100000) dans le fichier HDF5
            data_raw_hdf5 = f['H_new'][:]

            # On remet les dimensions dans l'ordre Python : (Samples, 28, 8, 2)
            # L'ordre MATLAB (100000, 28, 8, 2) devient (2, 8, 28, 100000) à la lecture
            raw_data = data_raw_hdf5.transpose(3, 2, 1, 0)

    except Exception as e:
        print(f"Erreur lors du chargement avec h5py : {e}")
        return

    num_samples = len(raw_data)
    print(f"Total des échantillons détectés : {num_samples}")
    print(f"Shape finale après transposition : {raw_data.shape}")

    # --- Le reste du code (Split et Sliding Window) reste identique ---
    train_end = int(num_samples * 0.50)
    val_end = int(num_samples * 0.80)

    splits = {
        'train': raw_data[:train_end],
        'val': raw_data[train_end:val_end],
        'test': raw_data[val_end:]
    }

    for name, data in splits.items():
        print(f"\n--- Génération des séquences : {name.upper()} ---")
        X, Y = [], []
        limit = len(data) - seq_length - pred_step + 1

        for i in range(limit):
            X.append(data[i: i + seq_length])
            Y.append(data[i + seq_length + pred_step - 1])

        X = np.array(X)
        Y = np.array(Y)

        np.save(f"X_{name}.npy", X)
        np.save(f"Y_{name}.npy", Y)
        print(f"Split {name} sauvegardé ! X: {X.shape}")


if __name__ == "__main__":
    path_to_mat = '/Users/Florian/Documents/Ecole/UQAC/Paris-Gen-IA-2026/Nokia/processedData/Data_Augmented/augmented_dataset.mat'
    process_augmented_data(path_to_mat)