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

class CrossAttention(nn.Module):
    def __init__(self, img_dim, clinical_dim, hidden_dim=256):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(clinical_dim, hidden_dim)
        self.key_proj = nn.Linear(img_dim, hidden_dim)
        self.value_proj = nn.Linear(img_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.out_proj = nn.Linear(hidden_dim, img_dim)

    def forward(self, img_features, clinical_features):
        """
        img_features: [batch, 512, 49] (spatial features)
        clinical_features: [batch, clinical_dim]
        """
        # img_features is [B, C, H*W], transpose to [B, H*W, C]
        img_features = img_features.transpose(1, 2)
        
        q = self.query_proj(clinical_features).unsqueeze(1) # [B, 1, H]
        k = self.key_proj(img_features) # [B, N, H]
        v = self.value_proj(img_features) # [B, N, H]

        # Attention scores
        attn_weights = self.softmax(torch.bmm(q, k.transpose(1, 2)) / (k.size(-1)**0.5)) # [B, 1, N]
        
        # Weighted sum of values
        attended = torch.bmm(attn_weights, v) # [B, 1, H]
        
        # Project back to original image dim
        out = self.out_proj(attended.squeeze(1)) # [B, img_dim]
        return out

class HybridFusionModel(nn.Module):
    def __init__(self, num_classes, clinical_dim, backbone="resnet18"):
        super(HybridFusionModel, self).__init__()
        
        # Image Branch (Spatial features)
        model = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*(list(model.children())[:-2])) # Remove avgpool and fc
        self.img_feature_dim = 512
        
        # Clinical Branch
        self.clinical_branch = nn.Sequential(
            nn.Linear(clinical_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Cross Attention
        self.cross_attn = CrossAttention(self.img_feature_dim, 32)
        
        # Fusion Layer
        self.fusion_head = nn.Sequential(
            nn.Linear(self.img_feature_dim + 32, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, img, clinical):
        # Extract spatial features [B, 512, 7, 7]
        spatial_features = self.backbone(img)
        B, C, H, W = spatial_features.shape
        spatial_features = spatial_features.view(B, C, -1) # [B, 512, 49]
        
        # Extract clinical features
        clin_features = self.clinical_branch(clinical)
        
        # Cross Attention: Clinical context attends to image patches
        attended_img = self.cross_attn(spatial_features, clin_features)
        
        # Concatenate and classify
        combined = torch.cat((attended_img, clin_features), dim=1)
        output = self.fusion_head(combined)
        return output

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
