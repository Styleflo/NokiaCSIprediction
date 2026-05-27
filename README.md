# Prédiction CSI de Nokia
### Développement d'un cadre d'IA conçu pour prédire l'état futur d'un canal radio dans les télécommunications mobiles (5G/6G) à partir d'observations historiques.

Ce référentiel contient un cadre permettant de prédire les informations d'état du canal (CSI) dans les communications sans fil (par exemple, 5G/6G) à l'aide de l'apprentissage profond et de modèles mathématiques traditionnels. L'objectif est de prévoir avec précision l'état futur d'un canal RF ($H_{t+1}$) à partir d'une séquence d'observations passées du canal.

## Caractéristiques principales

- **Génération de données synthétiques :** Génère et traite des données de canal 3GPP CDL (Clustered Delay Line) réalistes.
- **Architectures d'apprentissage profond :**
  - **ConvLSTM_V3 :** réseau LSTM convolutif équipé de mécanismes de compression et d'excitation.
  - **DiU Backbone :** une architecture inspirée de l'Unet avec une structure ConvLSTM pour le traitement spatio-temporel.
  - **Conv3D_CBAM :** un modèle hybride avancé utilisant des convolutions 3D combinées à des modules d'attention par blocs convolutifs (CBAM).
- **Références :** 
  - Référence de persistance (utilisant le dernier état connu).
  - Référence de régression de Ridge.
- **Évaluation comparative complète :** Carnets de notes unifiés pour évaluer, comparer et visualiser les prédictions. Évalue à la fois la précision (NMSE) et la cartographie de similarité visuelle des réseaux de canaux multi-antennes.

## Structure des répertoires

- `Data_format/` : contient des scripts pour la génération de données, l'augmentation des ensembles de données et les transformations de MATLAB vers Tensor (par exemple, `data_augmentation_v2.py`).
- `model_training/` : cahiers Jupyter et scripts Python pour la construction, l'entraînement et l'évaluation comparative des modèles (par exemple, `CSI_Comprehensive_Benchmark.ipynb`).
- `processedData/` : répertoire stockant les ensembles de données générés et les fichiers `.mat` transformés.

## Mode d'emploi

1. **Activer l'environnement :** assurez-vous que votre environnement est actif (par exemple, `source .venv/bin/activate`).
2. **Génération des données :** Exécutez les scripts Python dans le dossier `Data_format/` pour générer les données de séquence (généralement formatées sous forme de séquences de longueur de fenêtre `10` prédisant `1` étape future).
   ```bash
   python Data_format/data_augmentation_v2.py
   ```
3. **Entraînement et évaluation comparative des modèles :** Ouvrez le notebook `model_training/CSI_Comprehensive_Benchmark.ipynb`. Ce notebook de synthèse automatise la boucle d'entraînement des modèles d'apprentissage profond, ajuste les bases de référence de régression et génère les tableaux de comparaison graphiques finaux.

## Indicateurs d'évaluation

- **NMSE (erreur quadratique moyenne normalisée en dB) :** Mesure l'énergie/puissance continue de l'erreur de prédiction en pénalisant les erreurs importantes.
- **Matrice de similarité :** Calcule le pourcentage de valeurs matricielles absolues prédites correctement dans une tolérance d'erreur stricte (par exemple, différence $\le 0,02$).

## Configuration requise

- Python 3.x
- TensorFlow / Keras (prise en charge des GPU NVIDIA fortement recommandée)
- Scikit-learn
- NumPy, Pandas, Matplotlib
