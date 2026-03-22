import scipy.io
import numpy as np
import os
import glob


def extract_for_training(directory_path, output_name="csi_data_train_flattened"):
    search_path = os.path.join(directory_path, "CDLChannelEst_train_processed_*.mat")
    file_list = sorted(glob.glob(search_path),
                       key=lambda x: int(x.split('_')[-1].split('.')[0]))

    if not file_list:
        print("Aucun fichier trouvé.")
        return

    print(f"Extraction de {len(file_list)} fichiers...")

    extracted_data = []

    for i, file_path in enumerate(file_list):
        try:
            mat = scipy.io.loadmat(file_path)
            # H shape: (1, 32, 32, 2)
            h_matrix = mat['H']

            # Aplatissement complet pour obtenir un vecteur de caractéristiques (features)
            # 32 * 32 * 2 = 2048 valeurs par snapshot
            flat_vector = h_matrix.flatten()

            extracted_data.append(flat_vector)

            if (i + 1) % 500 == 0:
                print(f"Progression : {i + 1}/10 000")

        except Exception as e:
            print(f"Erreur sur {file_path}: {e}")

    # Conversion en matrice finale (3000, 2048)
    final_matrix = np.array(extracted_data)

    np.save(f"{output_name}.npy", final_matrix)

    print(f"\nExtraction terminée !")
    print(f"Fichier sauvegardé : {output_name}.npy")
    print(f"Dimensions de la matrice : {final_matrix.shape} (Snapshots, Features)")


if __name__ == "__main__":
    path = 'processedData/processedData'
    extract_for_training(path)