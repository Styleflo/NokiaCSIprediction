import numpy as np
import os

def apply_complex_rotation(data, theta_rad):
    """Applique une rotation de phase sur une matrice CSI (Réel, Imaginaire)."""
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    real = data[..., 0]
    imag = data[..., 1]
    new_real = real * cos_t - imag * sin_t
    new_imag = real * sin_t + imag * cos_t
    return np.stack([new_real, new_imag], axis=-1)

def generate_gaussian_dataset(X_src, Y_src, num_samples, max_speed_rad=0.5):
    """
    Génère un nombre précis d'échantillons avec une distribution Gaussienne.
    """
    # Sélection aléatoire des indices sources (permet d'augmenter au-delà de la taille initiale)
    indices = np.random.choice(len(X_src), size=num_samples, replace=True)
    X_base = X_src[indices]
    Y_base = Y_src[indices]
    
    batch_size = num_samples
    seq_len = X_base.shape[1]
    
    # 1. Angle de départ aléatoire unique pour chaque nouvel échantillon
    init_phase = np.random.uniform(0, 2*np.pi, size=(batch_size, 1, 1, 1))
    
    # 2. Vitesse de rotation (Doppler) GAUSSIENNE
    std_dev = max_speed_rad / 3
    step_phase = np.random.normal(loc=0.0, scale=std_dev, size=(batch_size, 1, 1, 1))
    step_phase = np.clip(step_phase, -max_speed_rad, max_speed_rad)
    
    # 3. Application de la trajectoire temporelle
    t = np.arange(seq_len).reshape(1, seq_len, 1, 1)
    current_phases_X = init_phase + (t * step_phase)
    current_phases_Y = (init_phase + (seq_len * step_phase)).reshape(batch_size, 1, 1)
    
    X_gen = apply_complex_rotation(X_base, current_phases_X)
    Y_gen = apply_complex_rotation(Y_base, current_phases_Y)
    
    return X_gen, Y_gen

if __name__ == "__main__":
    # --- CONFIGURATION ---
    data_path = 'data_original_32x32/' 
    output_path = '.'
    
    TOTAL_SAMPLES = 20000  # Nombre total de données à générer
    TRAIN_RATIO = 0.8      # 80% Train
    VAL_RATIO = 0.1        # 10% Val
    TEST_RATIO = 0.1       # 10% Test (Assure-toi que la somme = 1.0)
    
    MAX_SPEED = 1        # Limite de vitesse en rad/snapshot

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 1. Chargement d'une source unique pour piocher dedans 
    # (On charge le train original comme base de connaissance)
    try:
        X_raw = np.load(os.path.join(data_path, 'X_train.npy'))
        Y_raw = np.load(os.path.join(data_path, 'Y_train.npy'))
        print(f"Base source chargée : {len(X_raw)} échantillons.")
    except FileNotFoundError:
        print("Erreur : Fichiers sources X_train.npy / Y_train.npy introuvables.")
        exit()

    # 2. Calcul du nombre d'échantillons par set
    n_train = int(TOTAL_SAMPLES * TRAIN_RATIO)
    n_val = int(TOTAL_SAMPLES * VAL_RATIO)
    n_test = TOTAL_SAMPLES - n_train - n_val # Pour éviter les erreurs d'arrondi

    splits = [
        ('train', n_train),
        ('val', n_val),
        ('test', n_test)
    ]

    print(f"Génération de {TOTAL_SAMPLES} données avec répartition {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}...")

    for name, count in splits:
        print(f"Génération du set {name} ({count} échantillons)...")
        
        # Génération avec Doppler Gaussien
        X_set, Y_set = generate_gaussian_dataset(X_raw, Y_raw, count, max_speed_rad=MAX_SPEED)

        X_set = X_set.astype(np.float32)
        Y_set = Y_set.astype(np.float32)
        
        # Sauvegarde
        np.save(os.path.join(output_path, f'X_{name}.npy'), X_set)
        np.save(os.path.join(output_path, f'Y_{name}.npy'), Y_set)

    print("\n" + "="*40)
    print(f"TERMINÉ : {TOTAL_SAMPLES} échantillons générés dans {output_path}")
    print("="*40)