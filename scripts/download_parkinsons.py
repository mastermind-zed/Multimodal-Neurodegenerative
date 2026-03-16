import kagglehub
import os
import shutil

# Download latest version
print("Downloading Parkinson's dataset...")
path = kagglehub.dataset_download("irfansheriff/parkinsons-brain-mri-dataset")

print("Path to dataset files:", path)

# Move data to the project's data directory
target_dir = r"d:\Machine Learning\Parkinson’s Disease\data\parkinsons"
if not os.path.exists(target_dir):
    print(f"Moving dataset to {target_dir}...")
    shutil.copytree(path, target_dir)
    print("Move complete.")
else:
    print(f"Target directory {target_dir} already exists.")
