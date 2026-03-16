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

if __name__ == "__main__":
    # This will be integrated into the evaluation script
    print("Grad-CAM module implemented.")
