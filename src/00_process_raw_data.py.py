import pandas as pd
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pickle

# =================== CONFIGURATION ===================
PARENT_FOLDERS = ["city_1_0", "city_3_1", "rain_4_0" , "rain_4_1" , "fog_8_0" , "fog_8_1" , "night_1_0" , "night_1_1" , "snow_1_0"]  # Add all your folders here
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet Model
resnet_model = models.resnet50(pretrained=True)
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])
resnet_model.to(device).eval()

# Image Transforms
transform_rgb = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_gray = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# =================== PROCESS EACH DATASET ===================
for parent_folder in PARENT_FOLDERS:
    print(f"\nProcessing dataset in: {parent_folder}")

    try:
        # File Paths
        imu_file = os.path.join(parent_folder, "imu_dataset.csv")
        lidar_folder = os.path.join(parent_folder, "velo_lidar")
        lidar_timestamps_file = os.path.join(parent_folder, "velo_lidar.txt")
        camera_folder = os.path.join(parent_folder, "zed_right")
        camera_timestamps_file = os.path.join(parent_folder, "zed_right.txt")
        radar_folder = os.path.join(parent_folder, "Navtech_Cartesian")
        radar_timestamps_file = os.path.join(parent_folder, "Navtech_Cartesian.txt")
        output_pickle_file = os.path.join(parent_folder, f"{parent_folder}_dataset.pkl")

        # Load IMU Data
        imu_df = pd.read_csv(imu_file)
        imu_df["Timestamp"] = imu_df["Timestamp"].astype(float)

        # =================== PROCESS LIDAR ===================
        lidar_timestamps = {}
        with open(lidar_timestamps_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                frame_num = int(parts[1])
                timestamp = float(parts[3])
                lidar_timestamps[timestamp] = f"{frame_num:06d}.csv"

        def calculate_lidar_depth(lidar_file):
            lidar_path = os.path.join(lidar_folder, lidar_file)
            if os.path.exists(lidar_path):
                lidar_df = pd.read_csv(lidar_path, header=None, names=["x", "y", "z", "intensity", "ring"])
                depth = np.sqrt(lidar_df["x"]**2 + lidar_df["y"]**2 + lidar_df["z"]**2)
                return depth.iloc[0]  # Arbitrary point
            return np.nan

        imu_df["LiDAR_Depth"] = [
            calculate_lidar_depth(lidar_timestamps[min(lidar_timestamps.keys(), key=lambda x: abs(x - t))]) 
            for t in imu_df["Timestamp"]
        ]

        # =================== PROCESS CAMERA ===================
        camera_timestamps = {}
        with open(camera_timestamps_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                frame_num = int(parts[1])
                timestamp = float(parts[3])
                camera_timestamps[timestamp] = f"{frame_num:06d}.png"

        def extract_image_features(image_file, folder, transform):
            image_path = os.path.join(folder, image_file)
            if os.path.exists(image_path):
                img = Image.open(image_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    features = resnet_model(img_tensor).cpu().numpy()
                return features.flatten()
            return np.nan

        # Ensure Camera_Features column is stored as an array
        imu_df["Camera_Features"] = [
            extract_image_features(camera_timestamps[min(camera_timestamps.keys(), key=lambda x: abs(x - t))], 
                                   camera_folder, transform_rgb) 
            for t in imu_df["Timestamp"]
        ]
        imu_df["Camera_Features"] = imu_df["Camera_Features"].astype(object)  # Store as full array

        # =================== PROCESS RADAR ===================
        radar_timestamps = {}
        with open(radar_timestamps_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                frame_num = int(parts[1])
                timestamp = float(parts[3])
                radar_timestamps[timestamp] = f"{frame_num:06d}.png"

        def extract_radar_features(radar_file):
            radar_path = os.path.join(radar_folder, radar_file)
            if os.path.exists(radar_path):
                img = Image.open(radar_path).convert("L")  # Convert to grayscale
                img_tensor = transform_gray(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    features = resnet_model(img_tensor).cpu().numpy()
                return features.flatten()
            return np.nan

        # Ensure Radar_Features column is stored as an array
        imu_df["Radar_Features"] = [
            extract_radar_features(radar_timestamps[min(radar_timestamps.keys(), key=lambda x: abs(x - t))]) 
            for t in imu_df["Timestamp"]
        ]
        imu_df["Radar_Features"] = imu_df["Radar_Features"].astype(object)  # Store as full array

        # =================== FINAL PROCESSING ===================
        imu_df["Weather"] = 1  # Add Weather column
        imu_df.drop(columns=["File_ID"], inplace=True, errors='ignore')  # Remove File_ID if exists

        # =================== SAVE WITHOUT TRUNCATION ===================
        with open(output_pickle_file, "wb") as f:
            pickle.dump(imu_df, f)  # Save entire dataset as a pickle file
        print(f"✅ Dataset saved: {output_pickle_file}")

        # =================== SAVE CAMERA & RADAR FEATURES SEPARATELY (Optional) ===================
        np.save(os.path.join(parent_folder, "camera_features.npy"), imu_df["Camera_Features"].to_numpy(), allow_pickle=True)
        np.save(os.path.join(parent_folder, "radar_features.npy"), imu_df["Radar_Features"].to_numpy(), allow_pickle=True)

    except Exception as e:
        print(f"❌ Error processing {parent_folder}: {e}")

# =================== LOAD DATA WITHOUT TRUNCATION ===================
# To reload the dataset without truncation:
# with open("rain_4_1_dataset.pkl", "rb") as f:
#     imu_df = pickle.load(f)
