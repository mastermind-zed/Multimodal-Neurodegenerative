import kagglehub
import os
import shutil

# Download latest version
print("Downloading OASIS dataset...")
path = kagglehub.dataset_download("ninadaithal/imagesoasis")

print("Path to dataset files:", path)

# Move data to the project's data directory
target_dir = r"d:\Machine Learning\Alzheimer’s Disease\data\oasis"
if not os.path.exists(target_dir):
    print(f"Moving dataset to {target_dir}...")
    # The downloaded path usually contains the content directly or in a subfolder
    shutil.copytree(path, target_dir)
    print("Move complete.")
else:
    print(f"Target directory {target_dir} already exists.")
