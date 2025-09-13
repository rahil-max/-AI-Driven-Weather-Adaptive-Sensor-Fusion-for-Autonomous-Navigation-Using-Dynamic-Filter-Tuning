import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "norm_df.pkl"
data = pd.read_pickle(file_path)

# Drop covariance-related columns
excluded_columns = ["IMU_Orientation_Covariance", "IMU_AngularVelocity_Covariance", "IMU_LinearAcceleration_Covariance"]
features = [col for col in data.columns if col not in excluded_columns + ["Weather"]]

# List of numerical features (excluding 'Weather')
features = [col for col in data.columns if col != "Weather"]

# Set style for better visualization
sns.set_style("whitegrid")

# Generate separate plots for each feature
for feature in features:
    plt.figure(figsize=(8, 5))
    
    # Line plot for continuous trends
    sns.lineplot(x=data.index, y=data[feature], label=feature, color="b", alpha=0.7)
    
    # Scatter plot to show data points
    sns.scatterplot(x=data.index, y=data[feature], hue=data["Weather"], palette="viridis", alpha=0.8)

    plt.xlabel("Index")
    plt.ylabel(feature)
    plt.title(f"{feature} vs Weather")
    plt.legend()
    
    # Save each plot separately
    plt.savefig(f"Data Analytics/plots/{feature}_vs_weather.png", dpi=300)
    plt.show()
