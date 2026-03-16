import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fusion_model import MultimodalDataset, HybridFusionModel
from torch.utils.data import DataLoader
from torchvision import transforms

def perform_audit(model_path, disease_type="parkinsons"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Auditing on {device}...")

    base_dir = r"d:\Machine Learning\Multimodal Neurodegenerative Research"
    if disease_type == "alzheimer":
        csv_file = os.path.join(base_dir, "metadata", "alzheimer_clinical.csv")
        root_dir = os.path.join(base_dir, "data", "processed_alzheimer", "Data")
        num_classes = 4
        clinical_features = ["age", "gender", "mmse", "cdr"]
    else:
        csv_file = os.path.join(base_dir, "metadata", "parkinsons_clinical.csv")
        root_dir = os.path.join(base_dir, "data", "processed_parkinsons")
        num_classes = 2
        clinical_features = ["age", "gender", "updrs_score"]

    # Load data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = MultimodalDataset(csv_file, root_dir, transform=transform, clinical_features=clinical_features)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Load model
    # Note: Using HybridFusionModel as default moving forward
    model = HybridFusionModel(num_classes=num_classes, clinical_dim=len(clinical_features)).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        print(f"Warning: Model path {model_path} not found. Running with random weights for pipeline test.")

    all_labels = []
    all_preds = []
    all_genders = []
    all_ages = []

    with torch.no_grad():
        for imgs, clinical, labels in loader:
            imgs, clinical, labels = imgs.to(device), clinical.to(device), labels.to(device)
            outputs = model(imgs, clinical)
            preds = torch.argmax(outputs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            # For audit, we extract gender (idx 1) and age (idx 0) from clinical data
            all_ages.extend(clinical[:, 0].cpu().numpy())
            all_genders.extend(clinical[:, 1].cpu().numpy())

    df = pd.DataFrame({
        'label': all_labels,
        'pred': all_preds,
        'age': all_ages,
        'gender': all_genders
    })

    # Grouped Metrics
    print("\n--- Fairness Audit: Gender Parity ---")
    for g in df['gender'].unique():
        gender_str = "Male" if g == 1 else "Female"
        subset = df[df['gender'] == g]
        acc = (subset['label'] == subset['pred']).mean()
        print(f"Accuracy for {gender_str}: {acc:.4f}")

    print("\n--- Fairness Audit: Age Tiers ---")
    df['age_group'] = pd.cut(df['age'], bins=[0, 60, 75, 100], labels=['Younger', 'Middle', 'Elderly'])
    for age_bin in df['age_group'].unique():
        subset = df[df['age_group'] == age_bin]
        acc = (subset['label'] == subset['pred']).mean()
        print(f"Accuracy for {age_bin}: {acc:.4f}")

if __name__ == "__main__":
    # Path to the hybrid model checkpoint
    model_checkpoint = r"d:\Machine Learning\Multimodal Neurodegenerative Research\models\parkinsons_hybrid_model.pth"
    perform_audit(model_checkpoint, disease_type="parkinsons")
