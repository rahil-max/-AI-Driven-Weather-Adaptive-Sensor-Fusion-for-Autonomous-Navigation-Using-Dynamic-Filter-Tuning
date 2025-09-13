# -AI-Driven-Weather-Adaptive-Sensor-Fusion-for-Autonomous-Navigation-Using-Dynamic-Filter-Tuning


This repository contains the official source code, data processing pipeline, and models for the research paper: **"AI-Driven Weather-Adaptive Sensor Fusion for Autonomous Navigation Using Dynamic Filter Tuning"**.

This project presents a novel approach for accurate, real-time weather classification (Sunny, Fog, Rain, Snow, Night) using multi-modal sensor data from autonomous driving datasets. We employ a deep learning model combining Convolutional Neural Networks (CNNs) and Gated Recurrent Units (GRUs) to effectively learn spatial and temporal features from sensor time-series data. The system is further enhanced with an Adaptive Kalman Filter to dynamically adjust to changing environmental conditions.

***

## Repository Structure

The repository is organized to ensure clarity and reproducibility, with a clear separation between source code, data, models, and results.

```
├── datasets/
│   ├──          # Stores datasets after processing and feature engineering
│  
│
├── models/
│   └── best_weather_cnn_gru_model.h5 # The final trained model
│
├── results/
│   ├   # Stores plots from EDA and feature analysis
│   ├── training_val_loss.png # Training history and Kalman Filter plots
│   └── confusion_matrix.png  # Final model evaluation plot
│
├── src/
│   ├── analytics/            # Scripts for exploratory data analysis
│   ├── 01_process_raw_data.py
│   ├── 02_combine_datasets.py
│   ├── 03_normalize_features.py
│   ├── 04_engineer_features.py
│   ├── 05_train_model.py
│   └── 06_evaluate_model.py
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

***

## Setup and Installation

Follow these steps to set up the environment and run the project.

### 1. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)[YourUsername]/AI-Driven-Weather-Adaptive-Sensor-Fusion-for-Autonomous-Navigation-Using-Dynamic-Filter-Tuning.git
cd AI-Driven-Weather-Adaptive-Sensor-Fusion-for-Autonomous-Navigation-Using-Dynamic-Filter-Tuning
```

### 2. Create a Virtual Environment (Recommended)
```bash
# For Unix/macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
All required packages are listed in `requirements.txt`.
```bash
pip install -r requirements.txt
```

### 4. Data Setup
Place your raw data folders (e.g., `city_1_0`, `rain_4_0`, etc.) containing the sensor readings into the `data/raw/` directory. The processing scripts expect this structure.

***

## Execution Workflow

The scripts in the `src/` directory are numbered to indicate the exact order of execution for reproducing the results.

### Main Pipeline
Run these scripts sequentially to process the data, train the model, and evaluate it.

1.  **`src/01_process_raw_data.py`**: This script reads the raw multi-modal sensor data from `data/raw/`. It synchronizes timestamps and uses a pretrained ResNet model to extract features from camera and radar images, fuses them with LiDAR and IMU data, and saves the output as processed `.pkl` files for each dataset.

2.  **`src/02_combine_datasets.py`**: Merges the individual processed datasets from the previous step into a single master dataset for the next stages.

3.  **`src/03_normalize_features.py`**: Normalizes high-dimensional feature vectors, such as those from the camera and radar, by calculating their L2 norm.

4.  **`src/04_engineer_features.py`**: Performs feature engineering by combining related sensor measurements (e.g., GPS, IMU, Twist) into composite features to reduce dimensionality.

5.  **`src/05_train_model.py`**: The core training script. It handles class imbalance using SMOTE, trains the final CNN-GRU model on the fully processed dataset, and saves the trained model to `models/` and the training history plot to `results/`.

6.  **`src/06_evaluate_model.py`**: Loads the trained model from `models/` and performs a comprehensive evaluation, generating and saving the classification report, confusion matrix, and ROC curves to the `results/` folder.

### Exploratory Data Analysis (Optional)
The scripts in `src/analytics/` were used for data exploration and feature analysis. They can be run after step `04` to generate the supplementary plots shown in the paper.

* **`src/analytics/generate_heatmaps.py`**: Creates correlation heatmaps of the features.
* **`src/analytics/generate_feature_plots.py`**: Generates plots showing the distribution of each feature against the weather classes.
* **`src/analytics/run_importance_analysis.py`**: Uses an XGBoost model and SHAP to determine feature importance.
