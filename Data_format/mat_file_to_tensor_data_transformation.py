import scipy.io
import numpy as np
import os


def process_augmented_data(file_path, seq_length=10, pred_step=1):

    print(f"Chargement du fichier unique : {file_path}...")
    try:
        mat = scipy.io.loadmat(file_path)
        # On récupère H_new qui est (100000, 28, 8, 2)
        raw_data = np.array(mat['H_new'])
    except Exception as e:
        print(f"Erreur lors du chargement : {e}")
        return

    num_samples = len(raw_data)
    print(f"Total des échantillons détectés : {num_samples}")

    # --- 1. Définition des indices pour le split (50% / 30% / 20%) ---
    train_end = int(num_samples * 0.50)
    val_end = int(num_samples * 0.80)

    splits = {
        'train': raw_data[:train_end],
        'val': raw_data[train_end:val_end],
        'test': raw_data[val_end:]
    }

    # --- 2. Application de la fenêtre glissante sur chaque split ---
    for name, data in splits.items():
        print(f"\n--- Génération des séquences pour : {name.upper()} ---")

        X, Y = [], []
        # La boucle s'arrête pour laisser assez de place à la séquence + la cible
        limit = len(data) - seq_length - pred_step + 1

        for i in range(limit):
            # Séquence d'entrée (t à t + seq_length)
            X.append(data[i: i + seq_length])
            # Cible à prédire (t + seq_length + pred_step - 1)
            Y.append(data[i + seq_length + pred_step - 1])

        X = np.array(X)
        Y = np.array(Y)

        # --- 3. Sauvegarde au format .npy ---
        np.save(f"X_{name}.npy", X)
        np.save(f"Y_{name}.npy", Y)

        print(f"Split {name} terminé !")
        print(f"Shape X: {X.shape} | Shape Y: {Y.shape}")


if __name__ == "__main__":
    # Chemin vers ton fichier généré par MATLAB
    path_to_mat = '/Users/Florian/Documents/Ecole/UQAC/Paris-Gen-IA-2026/Nokia/processedData/Data_Augmented/augmented_dataset.mat'

    SEQUENCE_LENGTH = 10
    PREDICTION_STEP = 1

    process_augmented_data(path_to_mat, seq_length=SEQUENCE_LENGTH, pred_step=PREDICTION_STEP)