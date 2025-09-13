import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

# Load dataset (replace with your actual data)
df = pd.read_pickle("Datasets/col_combine.pkl")

# 🔹 Drop Unnecessary Columns Based on Heatmap
#drop_columns = ['Latitude', 'Longitude', 'Altitude', 'GPS_Covariance', 'IMU_Orientation_Covariance',
                #'IMU_AngularVelocity_Covariance', 'IMU_LinearAcceleration_Covariance']
df.drop(columns="IMU_Covariance_Combined", inplace=True)

# 🔹 Handle Missing Values (if any)
print(df.isnull().sum())
df.dropna(inplace=True)
# 🔹 Detect & Handle Outliers using Z-Score (Threshold = 3)
z_scores = np.abs(df.apply(zscore))  # Compute Z-scores for each feature
df = df[(z_scores < 3).all(axis=1)]  # Keep only rows where all Z-scores are < 3

# 🔹 Separate Features and Target
X = df.drop(columns=["Weather"])  # Features
y = df["Weather"]                 # Target variable

# 🔹 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Scale Data (Recommended for models that need normalization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🔹 Train XGBoost Model
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# 🔹 Feature Importance using SHAP
explainer = shap.Explainer(xgb_model, X_train_scaled)
shap_values = explainer(X_test_scaled)

# 🔹 Plot SHAP Summary
shap.summary_plot(shap_values, X_test)

# 🔹 XGBoost Feature Importance Plot
xgb.plot_importance(xgb_model)
plt.show()
