import pandas as pd
import numpy as np

# Load dataset
df = pd.read_pickle("norm_df.pkl")

# Feature combinations
df["GPS_Combined"] = df[["Latitude", "Longitude", "Altitude", "GPS_Covariance"]].mean(axis=1)
#df["IMU_Orientation_Combined"] = np.sqrt(df["IMU_Orientation_X"]**2 + df["IMU_Orientation_Y"]**2 + df["IMU_Orientation_Z"]**2 + df["IMU_Orientation_W"]**2)
df["IMU_AngularVelocity_Combined"] = np.sqrt(df["IMU_AngularVelocity_X"]**2 + df["IMU_AngularVelocity_Y"]**2 + df["IMU_AngularVelocity_Z"]**2)
df["IMU_LinearAcceleration_Combined"] = np.sqrt(df["IMU_LinearAcceleration_X"]**2 + df["IMU_LinearAcceleration_Y"]**2 + df["IMU_LinearAcceleration_Z"]**2)
df["IMU_Covariance_Combined"] = df[["IMU_Orientation_Covariance", "IMU_AngularVelocity_Covariance", "IMU_LinearAcceleration_Covariance"]].mean(axis=1)
df["Twist_Linear_Combined"] = np.sqrt(df["Twist_Linear_X"]**2 + df["Twist_Linear_Y"]**2 + df["Twist_Linear_Z"]**2)
df["Twist_Angular_Combined"] = np.sqrt(df["Twist_Angular_X"]**2 + df["Twist_Angular_Y"]**2 + df["Twist_Angular_Z"]**2)
df["Sensor_Features_Combined"] = df[["Camera_Features", "Radar_Features", "LiDAR_Depth"]].mean(axis=1)

# Drop old columns
df.drop(columns=[
    "Latitude", "Longitude", "Altitude", "GPS_Covariance",
    "IMU_Orientation_X", "IMU_Orientation_Y", "IMU_Orientation_Z", "IMU_Orientation_W",
    "IMU_AngularVelocity_X", "IMU_AngularVelocity_Y", "IMU_AngularVelocity_Z",
    "IMU_LinearAcceleration_X", "IMU_LinearAcceleration_Y", "IMU_LinearAcceleration_Z",
    "IMU_Orientation_Covariance", "IMU_AngularVelocity_Covariance", "IMU_LinearAcceleration_Covariance",
    "Twist_Linear_X", "Twist_Linear_Y", "Twist_Linear_Z",
    "Twist_Angular_X", "Twist_Angular_Y", "Twist_Angular_Z",
    "Camera_Features", "Radar_Features", "LiDAR_Depth"
], inplace=True)

# Save new dataset
df.to_pickle("Datasets/nor_df.pkl")

print("Feature combination complete! New dataset saved.")
