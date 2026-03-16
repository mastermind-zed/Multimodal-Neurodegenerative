import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

def preprocess_mri_image(image_path, target_size=(224, 224)):
    """
    Load, resize, and normalize an MRI image slice.
    """
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    
    # Resize
    image_resized = cv2.resize(image, target_size)
    
    # Normalize pixel values to [0, 1]
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    return image_normalized

def batch_preprocess(source_dir, output_dir, target_size=(224, 224)):
    """
    Process all images in the source directory and save to output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    image_paths = glob(os.path.join(source_dir, "**", "*.jpg"), recursive=True) + \
                  glob(os.path.join(source_dir, "**", "*.png"), recursive=True)
    
    print(f"Found {len(image_paths)} images. Starting preprocessing...")
    
    for img_path in tqdm(image_paths):
        processed = preprocess_mri_image(img_path, target_size)
        if processed is not None:
            # Create subdirectories in output_dir to maintain structure
            rel_path = os.path.relpath(img_path, source_dir)
            target_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            # Save as numpy array or image
            cv2.imwrite(target_path, (processed * 255).astype(np.uint8))

if __name__ == "__main__":
    SOURCE_DIR = r"d:\Machine Learning\Multimodal Neurodegenerative Research\data\alzheimer"
    OUTPUT_DIR = r"d:\Machine Learning\Multimodal Neurodegenerative Research\data\processed_alzheimer"
    
    # This will be run after the download completes
    if os.path.exists(SOURCE_DIR):
        batch_preprocess(SOURCE_DIR, OUTPUT_DIR)
    else:
        print("Source directory not found. Please wait for the download to complete.")
