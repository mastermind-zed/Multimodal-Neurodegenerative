import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import classification_report
import numpy as np
import sys
import os

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fusion_model import MultimodalDataset, FusionModel, HybridFusionModel

def train_model(disease_type="alzheimer", model_type="hybrid", epochs=10, batch_size=64, lr=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")
    
    # Paths
    base_dir = r"d:\Machine Learning\Multimodal Neurodegenerative Research"
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

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset
    full_dataset = MultimodalDataset(csv_file, root_dir, transform=transform, clinical_features=clinical_features)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # --- HANDLING CLASS IMBALANCE ---
    # Get labels for the training subset
    train_labels = []
    print("Calculating class weights...")
    for i in train_dataset.indices:
        label_str = full_dataset.metadata.iloc[i]["label"]
        train_labels.append(label_map[label_str])
    
    class_counts = np.bincount(train_labels)
    class_weights = 1. / class_counts
    sample_weights = np.array([class_weights[t] for t in train_labels])
    
    # Sampler for training
    sampler = WeightedRandomSampler(torch.from_numpy(sample_weights).type(torch.DoubleTensor), len(sample_weights))
    
    # Loss weights (normalized)
    loss_weights = torch.FloatTensor(class_weights / class_weights.sum() * num_classes).to(device)
    # --------------------------------

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model selection
    if model_type == "hybrid":
        model = HybridFusionModel(num_classes=num_classes, clinical_dim=len(clinical_features)).to(device)
    else:
        model = FusionModel(num_classes=num_classes, clinical_dim=len(clinical_features)).to(device)
        
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs, clinical, labels in progress_bar:
            imgs, clinical, labels = imgs.to(device), clinical.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs, clinical)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': running_loss/len(train_loader)})
            
        # Validation Phase
        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for imgs, clinical, labels in val_loader:
                imgs, clinical, labels = imgs.to(device), clinical.to(device), labels.to(device)
                outputs = model(imgs, clinical)
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        
        print(f"\nValidation Report Epoch {epoch+1}:")
        print(classification_report(all_labels, all_preds, target_names=list(label_map.keys()), digits=4))
        
    print("Training complete.")
    
    # Save model
    save_path = os.path.join(base_dir, "models", f"{disease_type}_{model_type}_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Multimodal Neurodegenerative Models")
    parser.add_argument("--disease", type=str, default="alzheimer", choices=["alzheimer", "parkinson"], help="Disease type to train")
    parser.add_argument("--model", type=str, default="hybrid", choices=["fusion", "hybrid"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    
    args = parser.parse_args()
    
    # Run training
    train_model(
        disease_type=args.disease, 
        model_type=args.model, 
        epochs=args.epochs, 
        batch_size=64
    )
