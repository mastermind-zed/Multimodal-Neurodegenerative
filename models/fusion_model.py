import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
import torchvision.transforms as transforms

class MultimodalDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, clinical_features=None):
        self.metadata = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.clinical_cols = clinical_features or ["age", "gender", "mmse", "cdr"]
        
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx):
        # Image
        img_name = os.path.join(self.root_dir, self.metadata.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Clinical features
        clinical_data = self.metadata.iloc[idx][self.clinical_cols].values.astype('float32')
        clinical_data = torch.tensor(clinical_data)
        
        # Label
        # For AD, labels are categories; for simplicity, we map them
        label_map = {
            "Non Demented": 0,
            "Very mild Dementia": 1,
            "Mild Dementia": 2,
            "Moderate Dementia": 3,
            "normal": 0,
            "parkinson": 1
        }
        label_str = self.metadata.iloc[idx]["label"]
        label = torch.tensor(label_map[label_str], dtype=torch.long)
        
        return image, clinical_data, label

class FusionModel(nn.Module):
    def __init__(self, num_classes, clinical_dim, backbone="resnet18"):
        super(FusionModel, self).__init__()
        
        # Image Branch
        if backbone == "resnet18":
            self.image_branch = models.resnet18(pretrained=True)
            self.img_feature_dim = self.image_branch.fc.in_features
            self.image_branch.fc = nn.Identity() # Remove top layer
        else:
            raise ValueError("Backbone not supported")
            
        # Clinical Branch
        self.clinical_branch = nn.Sequential(
            nn.Linear(clinical_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Fusion Layer
        self.fusion_head = nn.Sequential(
            nn.Linear(self.img_feature_dim + 16, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, img, clinical):
        img_features = self.image_branch(img)
        clinical_features = self.clinical_branch(clinical)
        
        # Concatenate
        combined = torch.cat((img_features, clinical_features), dim=1)
        
        # Classification
        output = self.fusion_head(combined)
        return output

if __name__ == "__main__":
    # Test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModel(num_classes=4, clinical_dim=4).to(device)
    dummy_img = torch.randn(2, 3, 224, 224).to(device)
    dummy_clinical = torch.randn(2, 4).to(device)
    
    output = model(dummy_img, dummy_clinical)
    print(f"Fusion Model output shape: {output.shape}")
