import torch
import torch.nn as nn
import torchvision.models as models

class AlzheimerClassifier(nn.Module):
    def __init__(self, num_classes=4, model_name="resnet18", pretrained=True):
        super(AlzheimerClassifier, self).__init__()
        
        if model_name == "resnet18":
            self.base_model = models.resnet18(pretrained=pretrained)
            # Modify first layer if we use grayscale MRI slices (OASIS is often grayscale)
            # But the dataset might provide them as 3-channel JPGs
            num_ftrs = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        elif model_name == "vgg16":
            self.base_model = models.vgg16(pretrained=pretrained).features
            # Add custom classifier
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, 512),
                nn.ReLU(),
                nn.Linear(512, num_classes)
            )
        else:
            raise ValueError(f"Model {model_name} not supported")

    def forward(self, x):
        if hasattr(self, 'classifier'):
            x = self.base_model(x)
            x = self.classifier(x)
        else:
            x = self.base_model(x)
        return x

if __name__ == "__main__":
    # Test architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlzheimerClassifier(num_classes=4).to(device)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")
