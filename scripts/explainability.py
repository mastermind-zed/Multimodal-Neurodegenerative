import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate_heatmap(self, input_image, clinical_data, class_idx=None):
        self.model.eval()
        output = self.model(input_image, clinical_data)
        
        if class_idx is None:
            class_idx = torch.argmax(output)
            
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        gradients = self.gradients
        activations = self.activations
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activations
        cam = torch.sum(weights * activations, dim=1).squeeze()
        cam = nn.ReLU()(cam)
        cam = cam.detach().cpu().numpy()
        
        # Normalization
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

def overlay_heatmap(img_path, heatmap, save_path):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(save_path, superimposed_img)
    return save_path

def find_resnet_layer(model):
    """Find the last convolutional layer in a ResNet-based Hybrid Model."""
    if hasattr(model, 'backbone'):
        # For our HybridFusionModel, it's in the backbone
        return list(model.backbone.children())[-1]
    return None

def run_explainability(model_path, img_path, clinical_data, disease_type="alzheimer"):
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.fusion_model import HybridFusionModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 4 if disease_type == "alzheimer" else 2
    clinical_dim = len(clinical_data)

    model = HybridFusionModel(num_classes=num_classes, clinical_dim=clinical_dim).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print("Checkpoint not found. Running with random weights.")

    target_layer = find_resnet_layer(model)
    cam = GradCAM(model, target_layer)

    # Process Image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(img_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    clinical_tensor = torch.tensor([clinical_data], dtype=torch.float32).to(device)

    heatmap = cam.generate_heatmap(input_tensor, clinical_tensor)
    
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "roi_visualization.png")
    overlay_heatmap(img_path, heatmap, save_path)
    print(f"ROI Visualization saved to {save_path}")

if __name__ == "__main__":
    from torchvision import transforms
    print("Explainability module ready for Hybrid Fusion ROI checks.")
    # Example usage (uncomment after training ends):
    # model_chk = r"d:\Machine Learning\Multimodal Neurodegenerative Research\models\alzheimer_hybrid_model.pth"
    # sample_img = r"d:\Machine Learning\Multimodal Neurodegenerative Research\data\sample.jpg"
    # sample_clin = [75.0, 1.0, 24.0, 0.5] # Age, Gender, MMSE, CDR
    # run_explainability(model_chk, sample_img, sample_clin)
