import numpy as np

# Replace 'path_to_your_npz_file' with the actual path to your .npz file
npz_file_path = 'Datasets/pathmnist.npz'  # e.g., 'Datasets/pneumoniamnist.npz'

# Load the data
data = np.load(npz_file_path)

# Print the structure of the .npz file
print("Keys in the npz file:", list(data.keys()))
for key in data.keys():
    print(f"Shape of the data under key '{key}': {data[key].shape}")

# Closing the npz file is a good practice
data.close()
