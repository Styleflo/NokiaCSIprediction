import numpy as np
import os
import glob
import scipy.io
import h5py


def load_mat_file(file_path):
    """
    Charge un fichier .mat en essayant les deux formats (v7.3 et versions antérieures).
    """
    try:
        # 1. Tentative avec scipy.io (Format MATLAB 5.0 / v7 comme ton fichier)
        data_dict = scipy.io.loadmat(file_path)
        if 'H' in data_dict:
            data = data_dict['H']
            # Vérification de la structure : scipy charge souvent (H, W, C, Samples)
            # On veut qu'à la fin la shape soit (Samples, 28, 8, 2)
            # Si le fichier est un échantillon unique de shape (28, 8, 2)
            if data.ndim == 3:
                return np.expand_dims(data, axis=0)  # Ajoute la dimension Sample
            return data

    except Exception:
        try:
            # 2. Tentative avec h5py (Format MATLAB v7.3)
            with h5py.File(file_path, 'r') as f:
                data = f['H_new'][:]
                # h5py inverse les dimensions : (Samples, C, W, H) -> (Samples, H, W, C)
                return data.transpose(3, 2, 1, 0)
        except Exception as e:
            print(f"Erreur critique sur {os.path.basename(file_path)} : {e}")
            return None
    return None


def process_multiple_mat_files(folder_path, seq_length=10, pred_step=1):
    # Liste les fichiers en ignorant les fichiers système cachés de macOS (._)
    file_list = sorted([f for f in glob.glob(os.path.join(folder_path, '*.mat'))
                        if not os.path.basename(f).startswith('._')])

    num_files = len(file_list)
    if num_files == 0:
        print(f"Aucun fichier .mat valide trouvé dans {folder_path}")
        return

    print(f"Détection de {num_files} fichiers .mat. Début du chargement...")

    all_data = []

    # Chargement des fichiers
    for i, file_path in enumerate(file_list):
        raw_sample = load_mat_file(file_path)

        if raw_sample is not None:
            all_data.append(raw_sample)

        if (i + 1) % 1000 == 0:
            print(f"Progression : {i + 1}/{num_files} fichiers chargés.")

    if not all_data:
        print(all_data)
        print("Aucune donnée n'a pu être chargée.")
        return

    # Concaténation
    print("Concaténation des données...")
    raw_data = np.concatenate(all_data, axis=0)
    print(f"Shape finale des données brutes : {raw_data.shape}")  # Doit être (Total, 28, 8, 2)

    # Splits (50% Train, 30% Val, 20% Test)
    num_samples = len(raw_data)
    train_end = int(num_samples * 0.75)
    val_end = int(num_samples * 0.85)

    splits = {
        'train': raw_data[:train_end],
        'val': raw_data[train_end:val_end],
        'test': raw_data[val_end:]
    }

    # Génération des séquences X (t...t+9) et Y (t+10)
    for name, data in splits.items():
        print(f"Génération des séquences {name.upper()}...")
        X, Y = [], []
        limit = len(data) - seq_length - pred_step + 1

        if limit <= 0:
            continue

        for i in range(limit):
            X.append(data[i: i + seq_length])
            Y.append(data[i + seq_length + pred_step - 1])

        X_arr = np.array(X)
        Y_arr = np.array(Y)

        # Sauvegarde
        np.save(f"X_{name}.npy", X_arr)
        np.save(f"Y_{name}.npy", Y_arr)
        print(f"Sauvegardé : X_{name}.npy {X_arr.shape} | Y_{name}.npy {Y_arr.shape}")


if __name__ == "__main__":
    folder = '/Users/Florian/Documents/Ecole/UQAC/Paris-Gen-IA-2026/Nokia/processedData/Data_Augmentation/Preprocessed'
    process_multiple_mat_files(folder)