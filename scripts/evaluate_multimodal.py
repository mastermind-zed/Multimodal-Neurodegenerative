import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import torchvision.transforms as transforms
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fusion_model import MultimodalDataset, FusionModel, HybridFusionModel

def evaluate_model(disease_type="parkinson", model_type="hybrid", batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}...")
    
    base_dir = r"d:\Machine Learning\Multimodal Neurodegenerative Research"
    results_dir = os.path.join(base_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    if disease_type == "alzheimer":
        csv_file = os.path.join(base_dir, "metadata", "alzheimer_clinical.csv")
        root_dir = os.path.join(base_dir, "data", "processed_alzheimer", "Data")
        num_classes = 4
        clinical_features = ["age", "gender", "mmse", "cdr"]
        label_map = {"Non Demented": 0, "Very mild Dementia": 1, "Mild Dementia": 2, "Moderate Dementia": 3}
    else:
        csv_file = os.path.join(base_dir, "metadata", "parkinsons_clinical.csv")
        root_dir = os.path.join(base_dir, "data", "processed_parkinsons")
        num_classes = 2
        clinical_features = ["age", "gender", "updrs_score"]
        label_map = {"normal": 0, "parkinson": 1}

    # Model
    if model_type == "hybrid":
        model = HybridFusionModel(num_classes=num_classes, clinical_dim=len(clinical_features)).to(device)
    else:
        model = FusionModel(num_classes=num_classes, clinical_dim=len(clinical_features)).to(device)
        
    model_path = os.path.join(base_dir, "models", f"{disease_type}_{model_type}_model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Loaded model from {model_path}")
    else:
        print(f"❌ ERROR: Model file not found at {model_path}")
        return

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset
    dataset = MultimodalDataset(csv_file, root_dir, transform=transform, clinical_features=clinical_features)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_preds = []
    all_labels = []
    
    print(f"Running evaluation on {len(dataset)} samples...")
    with torch.no_grad():
        for imgs, clinical, labels in loader:
            imgs, clinical = imgs.to(device), clinical.to(device)
            outputs = model(imgs, clinical)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Generate Report
    target_names = list(label_map.keys())
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)
    print("\nClassification Report:")
    print(report)
    
    # Save Report
    report_path = os.path.join(results_dir, f"{disease_type}_eval_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Confusion Matrix - {disease_type.capitalize()} ({model_type})")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    save_fig = os.path.join(results_dir, f"{disease_type}_eval_cm.png")
    plt.savefig(save_fig)
    plt.close()
    
    print(f"✅ Results saved to {results_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Trained Multimodal Models")
    parser.add_argument("--disease", type=str, default="parkinson", choices=["alzheimer", "parkinson"], help="Disease type")
    parser.add_argument("--model", type=str, default="hybrid", choices=["fusion", "hybrid"], help="Model type")
    
    args = parser.parse_args()
    evaluate_model(disease_type=args.disease, model_type=args.model)
