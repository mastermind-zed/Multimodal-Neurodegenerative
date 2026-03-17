import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import sys
import os

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fusion_model import MultimodalDataset, FusionModel, HybridFusionModel

def train_model(disease_type="alzheimer", model_type="hybrid", epochs=5, batch_size=64, lr=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")
    
    # Paths
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model selection
    if model_type == "hybrid":
        model = HybridFusionModel(num_classes=num_classes, clinical_dim=len(clinical_features)).to(device)
    else:
        model = FusionModel(num_classes=num_classes, clinical_dim=len(clinical_features)).to(device)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Simple training loop
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
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
        
    print("Training complete.")
    
    # Save model
    save_path = os.path.join(base_dir, "models", f"{disease_type}_{model_type}_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    # Full Alzheimer's Research Run
    train_model(disease_type="alzheimer", model_type="hybrid", epochs=10, batch_size=64)
