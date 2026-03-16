import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

def preprocess_mri_image(image_path, target_size=(224, 224)):
    """
    Load, resize, and normalize an MRI image slice.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    
    # Resize
    image_resized = cv2.resize(image, target_size)
    
    # Normalization (0-1)
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    return image_normalized

def batch_preprocess(source_dir, output_dir, target_size=(224, 224)):
    """
    Process all images in the source directory and save to output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    image_paths = glob(os.path.join(source_dir, "**", "*.jpg"), recursive=True) + \
                  glob(os.path.join(source_dir, "**", "*.png"), recursive=True) + \
                  glob(os.path.join(source_dir, "**", "*.jpeg"), recursive=True)
    
    print(f"Found {len(image_paths)} Parkinson's images. Starting preprocessing...")
    
    for img_path in tqdm(image_paths):
        processed = preprocess_mri_image(img_path, target_size)
        if processed is not None:
            rel_path = os.path.relpath(img_path, source_dir)
            target_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            cv2.imwrite(target_path, (processed * 255).astype(np.uint8))

if __name__ == "__main__":
    SOURCE_DIR = r"d:\Machine Learning\Multimodal Neurodegenerative Research\data\parkinsons\parkinsons_dataset"
    OUTPUT_DIR = r"d:\Machine Learning\Multimodal Neurodegenerative Research\data\processed_parkinsons"
    
    if os.path.exists(SOURCE_DIR):
        batch_preprocess(SOURCE_DIR, OUTPUT_DIR)
    else:
        print(f"Source directory {SOURCE_DIR} not found.")
