import pandas as pd
import os
from tqdm import tqdm

# Load the original dataset
original_csv = "trainLabels.csv"  # Path to your original CSV
original_data = pd.read_csv(original_csv)

# Create a dictionary for quick lookup of labels
label_dict = dict(zip(original_data['image'], original_data['level']))

# Path to the folder with augmented images
oversampled_folder = "train_oversampled/"  # Replace with the actual folder path
oversampled_files = os.listdir(oversampled_folder)

# Create a new dataframe for the augmented dataset
oversampled_data = []

for file_name in tqdm(oversampled_files):
    # Extract the original file name by removing suffixes like '_90', '_mirrored'
    file_name = file_name.split('.')[0]
    base_name = file_name.split('_')[0] + '_' + file_name.split('_')[1]
    if base_name in label_dict:
        label = label_dict[base_name]
        oversampled_data.append({'image': file_name, 'level': label})

# Convert to DataFrame
augmented_df = pd.DataFrame(oversampled_data)

# Save to a new CSV
augmented_df.to_csv("train_oversampled.csv", index=False)
