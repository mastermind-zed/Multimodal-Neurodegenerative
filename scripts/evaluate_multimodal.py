import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import torchvision.transforms as transforms
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fusion_model import MultimodalDataset, FusionModel

def evaluate_model(disease_type="parkinsons"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # Model
    model = FusionModel(num_classes=num_classes, clinical_dim=len(clinical_features)).to(device)
    model_path = os.path.join(base_dir, "models", f"{disease_type}_fusion_model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print("Model file not found!")
        return

    # Transforms (Must match training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset
    dataset = MultimodalDataset(csv_file, root_dir, transform=transform, clinical_features=clinical_features)
    # Note: Use full dataset for quick test or separate test split
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, clinical, labels in loader:
            imgs, clinical = imgs.to(device), clinical.to(device)
            outputs = model(imgs, clinical)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Metrics
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {disease_type.capitalize()}")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    save_fig = os.path.join(base_dir, "results", f"{disease_type}_confusion_matrix.png")
    plt.savefig(save_fig)
    print(f"Confusion matrix saved to {save_fig}")

if __name__ == "__main__":
    evaluate_model(disease_type="parkinsons")
