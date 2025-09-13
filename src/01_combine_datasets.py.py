import pandas as pd
import os

# =================== CONFIGURATION ===================
PARENT_FOLDER = 'Datasets'  # Add all folders containing datasets
folders = ["city", "night","snow","rain","fog"]
#folders = ["snow_1"]
combined_output_file = "final.pkl"  # Final merged dataset

# List to store individual DataFrames
df_list = []

# =================== MERGE ALL DATASETS ===================
for folder in folders:
    dataset_file = os.path.join(PARENT_FOLDER, f"{folder}.pkl")

    if os.path.exists(dataset_file):
        df = pd.read_pickle(dataset_file)
        print(df.shape)
        #df["Weather"] = 1 # Add source folder name as a column (optional)
        df_list.append(df)
        print(f"✅ Loaded: {dataset_file}")
    else:
        print(f"❌ File not found: {dataset_file}")

# =================== CONCATENATE & SAVE ===================
if df_list:
    combined_df = pd.concat(df_list, ignore_index=True)  # Merge all datasets
    combined_df.to_pickle(combined_output_file)
    print(f"\n✅ Combined dataset saved as: {combined_output_file}")
    df_final = pd.read_pickle(combined_output_file)
    print(df_final.shape)
else:
    print("\n❌ No datasets found to merge.")
