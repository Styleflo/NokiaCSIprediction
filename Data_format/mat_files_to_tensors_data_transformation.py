import scipy.io
import numpy as np
import os
import glob

def extract_and_window(directory_path, split_name, seq_length=10, pred_step=1):
    """
    Extrait les matrices d'un split spécifique et applique une fenêtre glissante.
    
    :param split_name: 'train', 'val', ou 'test'
    :param seq_length: Nombre de snapshots passés à utiliser pour prédire (ex: 10)
    :param pred_step: Horizon de prédiction (ex: 1 pour prédire l'instant t+1)
    """
    # Recherche des fichiers correspondant au split
    search_path = os.path.join(directory_path, f"CDLChannelEst_{split_name}_processed_*.mat")
    file_list = sorted(glob.glob(search_path),
                       key=lambda x: int(x.split('_')[-1].split('.')[0]))

    if not file_list:
        print(f"Aucun fichier trouvé pour le split : {split_name}.")
        return

    print(f"\n--- Traitement du set '{split_name.upper()}' ({len(file_list)} fichiers) ---")

    raw_data = []

    # 1. Chargement et formatage des matrices spatiales
    for i, file_path in enumerate(file_list):
        try:
            mat = scipy.io.loadmat(file_path)
            # h_matrix est de taille (1, 32, 32, 2)
            h_matrix = mat['H']
            
            # On supprime la dimension 1 inutile pour obtenir (32, 32, 2)
            h_spatial = np.squeeze(h_matrix, axis=0) 
            raw_data.append(h_spatial)

            if (i + 1) % 2000 == 0:
                print(f"Chargement : {i + 1}/{len(file_list)}")

        except Exception as e:
            print(f"Erreur sur {file_path}: {e}")

    # Conversion en tableau NumPy (Nb_fichiers, 32, 32, 2)
    raw_data = np.array(raw_data)
    
    # 2. Création de la fenêtre glissante (Sliding Window)
    X, Y = [], []
    total_sequences = len(raw_data) - seq_length - pred_step + 1
    
    for i in range(total_sequences):
        # X prend les 'seq_length' instants consécutifs
        X.append(raw_data[i : i + seq_length])
        
        # Y prend la matrice cible à l'instant t + seq_length + pred_step - 1
        Y.append(raw_data[i + seq_length + pred_step - 1])

    X = np.array(X)
    Y = np.array(Y)

    # 3. Sauvegarde des paires X et Y
    np.save(f"X_{split_name}.npy", X)
    np.save(f"Y_{split_name}.npy", Y)

    print(f"Extraction terminée pour {split_name} !")
    print(f"Forme de X (Input)  : {X.shape} -> (Samples, Sequence_Length, Height, Width, Channels)")
    print(f"Forme de Y (Target) : {Y.shape} -> (Samples, Height, Width, Channels)")


if __name__ == "__main__":
    path = 'processedData'
    
    # Paramètres de la fenêtre temporelle
    SEQUENCE_LENGTH = 10 # Nombre d'instants t utilisés pour la prédiction
    PREDICTION_STEP = 1  # 1 = on prédit l'instant juste après la séquence
    
    # On itère sur les 3 sous-ensembles
    splits = ['train', 'val', 'test']
    
    for split in splits:
        extract_and_window(path, split, seq_length=SEQUENCE_LENGTH, pred_step=PREDICTION_STEP)