# Nokia CSI Prediction

This repository contains a framework for predicting Channel State Information (CSI) in wireless communications (e.g., 5G/6G) using deep learning and traditional mathematical baselines. The goal is to accurately forecast the future state of an RF channel ($H_{t+1}$) based on a sequence of past channel observations.

## Key Features

- **Synthetic Data Generation:** Generates and processes realistic 3GPP CDL (Clustered Delay Line) channel data.
- **Deep Learning Architectures:**
  - **ConvLSTM_V3:** Convolutional LSTM network equipped with Squeeze-and-Excitation mechanisms.
  - **DiU Backbone:** A Unet-inspired architecture with a ConvLSTM backbone for spatial-temporal processing.
  - **Conv3D_CBAM:** An advanced hybrid model using 3D Convolutions combined with Convolutional Block Attention Modules (CBAM).
- **Baselines:** 
  - Persistence Baseline (using the last known state).
  - Ridge Regression Baseline.
- **Comprehensive Benchmarking:** Unified notebooks to evaluate, compare, and visualize predictions. Evaluates both the precision (NMSE) and the visual similarity mapping of the multi-antenna channel arrays.

## Directory Structure

- `Data_format/`: Contains scripts for data generation, dataset augmentation, and MATLAB-to-Tensor transformations (e.g., `data_augmentation_v2.py`).
- `model_training/`: Jupyter notebooks and Python scripts for model construction, training, and benchmarking evaluation (e.g., `CSI_Comprehensive_Benchmark.ipynb`).
- `processedData/`: Directory storing generated datasets and transformed `.mat` files.

## How to Use

1. **Activate Environment:** Ensure your environment is active (e.g., `source .venv/bin/activate`).
2. **Data Generation:** Execute the Python scripts inside the `Data_format/` folder to generate the sequence data (usually formatted as sequences of window length `10` predicting `1` future step).
   ```bash
   python Data_format/data_augmentation_v2.py
   ```
3. **Model Training & Benchmarking:** Open the `model_training/CSI_Comprehensive_Benchmark.ipynb` notebook. This capstone notebook automates the training loop for the deep learning models, fits the regression baselines, and generates the final graphical comparison tables.

## Evaluation Metrics

- **NMSE (Normalized Mean Square Error in dB):** Measures the continuous energy/power of the prediction error penalizing large mistakes.
- **Similarity Matrix:** Computes the percentage of absolute matrix values predicted correctly within a strict error tolerance (e.g., difference $\le 0.02$).

## Requirements

- Python 3.x
- TensorFlow / Keras (with NVIDIA GPU support heavily recommended)
- Scikit-learn
- NumPy, Pandas, Matplotlib