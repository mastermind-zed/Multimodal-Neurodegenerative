import pandas as pd
import numpy as np
import os
from glob import glob

def generate_alzheimer_metadata(processed_dir, output_path):
    """
    Generate simulated clinical data for the Alzheimer's cohort.
    OASIS classes: Mild Dementia, Moderate Dementia, Non Demented, Very mild Dementia
    """
    subdirs = ["Mild Dementia", "Moderate Dementia", "Non Demented", "Very mild Dementia"]
    data = []
    
    for label in subdirs:
        path = os.path.join(processed_dir, label)
        if not os.path.exists(path):
            continue
            
        images = glob(os.path.join(path, "*.jpg"))
        for img in images:
            img_id = os.path.basename(img)
            # Simulate Age (60-90)
            age = np.random.randint(60, 91)
            # Gender (0: Male, 1: Female)
            gender = np.random.randint(0, 2)
            
            # MMSE (Mini-Mental State Exam): 0-30 (Lower is worse)
            # CDR (Clinical Dementia Rating): 0, 0.5, 1, 2
            if label == "Non Demented":
                mmse = np.random.randint(27, 31)
                cdr = 0.0
            elif label == "Very mild Dementia":
                mmse = np.random.randint(20, 27)
                cdr = 0.5
            elif label == "Mild Dementia":
                mmse = np.random.randint(15, 21)
                cdr = 1.0
            elif label == "Moderate Dementia":
                mmse = np.random.randint(5, 16)
                cdr = 2.0
                
            data.append({
                "image_path": os.path.relpath(img, processed_dir),
                "label": label,
                "age": age,
                "gender": gender,
                "mmse": mmse,
                "cdr": cdr
            })
            
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Alzheimer metadata generated: {len(df)} records at {output_path}")

def generate_parkinsons_metadata(processed_dir, output_path):
    """
    Generate simulated clinical data for the Parkinson's cohort.
    Classes: normal, parkinson
    """
    subdirs = ["normal", "parkinson"]
    data = []
    
    for label in subdirs:
        path = os.path.join(processed_dir, label)
        if not os.path.exists(path):
            continue
            
        images = glob(os.path.join(path, "*.jpg")) + glob(os.path.join(path, "*.png"))
        for img in images:
            img_id = os.path.basename(img)
            age = np.random.randint(50, 85)
            gender = np.random.randint(0, 2)
            
            # UPDRS (Unified Parkinson's Disease Rating Scale) simulated
            if label == "normal":
                updrs = np.random.uniform(0, 5)
                state = 0
            else:
                updrs = np.random.uniform(20, 80)
                state = 1
                
            data.append({
                "image_path": os.path.relpath(img, processed_dir),
                "label": label,
                "age": age,
                "gender": gender,
                "updrs_score": updrs,
                "pd_state": state
            })
            
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Parkinson metadata generated: {len(df)} records at {output_path}")

if __name__ == "__main__":
    BASE_DIR = r"d:\Machine Learning\Multimodal Neurodegenerative Research"
    
    # Alzheimer's (Updated path to include 'Data' folder)
    generate_alzheimer_metadata(
        os.path.join(BASE_DIR, "data", "processed_alzheimer", "Data"),
        os.path.join(BASE_DIR, "metadata", "alzheimer_clinical.csv")
    )
    
    # Parkinson's (Updated path - removed extra subfolder)
    generate_parkinsons_metadata(
        os.path.join(BASE_DIR, "data", "processed_parkinsons"),
        os.path.join(BASE_DIR, "metadata", "parkinsons_clinical.csv")
    )
