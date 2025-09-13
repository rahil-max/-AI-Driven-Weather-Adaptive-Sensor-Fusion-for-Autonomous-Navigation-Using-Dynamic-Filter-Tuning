import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# =================== Load Datasets ===================
datasets = {
    #"City":"Datasets/city.pkl",
    "Weather": "Datasets/nor_df.pkl",
    #"Rain": "Datasets/rain.pkl",
    #"Fog": "Datasets/fog.pkl",
    #"Night": "Datasets/night.pkl",
    #"Snow": "Datasets/snow.pkl"

}

# =================== Data Analysis ===================
for name, file in datasets.items():
    # Load dataset
    df = pd.read_pickle(file)

    # Drop the Timestamp column
    #df.drop(columns=["Timestamp"], inplace=True, errors="ignore")

    # Convert Camera and Radar Features (space-separated) into numeric values
    #df["Camera_Features"] = df["Camera_Features"].apply(lambda x: np.mean(np.fromstring(x.strip("[]"), sep=" ")) if isinstance(x, str) else np.nan)
    #df["Radar_Features"] = df["Radar_Features"].apply(lambda x: np.mean(np.fromstring(x.strip("[]"), sep=" ")) if isinstance(x, str) else np.nan)

    # Print basic statistics
    print(f"\n===== {name} Dataset Statistics =====")
    print(df.describe())

    # Compute correlation matrix
    corr_matrix = df.corr()

    # =================== Plot Heatmap ===================
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", linewidths=0.5)
    plt.title(f"Heatmap of {name} Dataset")
    plt.savefig(f"Data Analytics/{name}_heatmap.png")  # Save the heatmap as an image
    plt.show()
