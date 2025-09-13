import pandas as pd
import numpy as np
import ast

# Load the dataset
df = pd.read_pickle("Datasets/final1.pkl")

# Function to safely convert values to NumPy arrays
def safe_convert(x):
    if isinstance(x, np.ndarray):
        return x  # Already a NumPy array
    elif isinstance(x, object):
        return np.array(x)  # Convert list to NumPy array
    elif isinstance(x, str):
        try:
            return np.array(ast.literal_eval(x))  # Convert string to list then to NumPy array
        except (ValueError, SyntaxError):
            return np.array([])  # Return empty array if conversion fails
    else:
        return np.array([])  # Handle unexpected data types

# Convert Camera_Features and Radar_Features columns
df['Camera_Features'] = df['Camera_Features'].apply(safe_convert)
df['Radar_Features'] = df['Radar_Features'].apply(safe_convert)

# Compute L2 Norm (Euclidean Norm) for each feature array
df['Camera_Features'] = df['Camera_Features'].apply(lambda x: np.linalg.norm(x) if x.size > 0 else np.nan)
df['Radar_Features'] = df['Radar_Features'].apply(lambda x: np.linalg.norm(x) if x.size > 0 else np.nan)
df.to_pickle('norm_df')
# Print results to verify
print(df[['Camera_Features']].head())
print(f"Updated data type: {df['Camera_Features'].dtype}")
