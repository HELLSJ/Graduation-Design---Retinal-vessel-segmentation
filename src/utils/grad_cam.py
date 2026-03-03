import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()
        
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        self.model.zero_grad()
        
        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, target_class.unsqueeze(1), 1.0)
        
        output.backward(gradient=one_hot)
        
        gradients = self.gradients
        activations = self.activations
        
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        cam = F.relu(cam)
        
        cam = cam.squeeze().cpu().numpy()
        
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def generate_cam_batch(self, input_tensor):
        batch_size = input_tensor.shape[0]
        cams = []
        
        for i in range(batch_size):
            single_input = input_tensor[i:i+1]
            cam = self.generate_cam(single_input)
            cams.append(cam)
        
        return np.array(cams)


class SegmentationGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor):
        self.model.eval()
        
        output = self.model(input_tensor)
        
        self.model.zero_grad()
        
        output.backward(gradient=torch.ones_like(output))
        
        gradients = self.gradients
        activations = self.activations
        
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        cam = F.relu(cam)
        
        cam = cam.squeeze().cpu().numpy()
        
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def overlay_cam(self, image, cam, alpha=0.4, colormap=cv2.COLORMAP_JET):
        image_np = image.cpu().numpy().transpose(1, 2, 0)
        image_np = (image_np * 255).astype(np.uint8)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        return overlay
    
    def save_cam_visualization(self, image, mask, pred, cam, save_path):
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        image_np = image.cpu().numpy().transpose(1, 2, 0)
        image_np = np.clip(image_np, 0, 1)
        
        mask_np = mask.cpu().numpy().squeeze()
        pred_np = torch.sigmoid(pred).cpu().numpy().squeeze()
        
        axes[0, 0].imshow(image_np)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(mask_np, cmap='gray')
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(pred_np, cmap='gray')
        axes[1, 0].set_title('Prediction')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(cam, cmap='jet')
        axes[1, 1].set_title('Grad-CAM')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def get_target_layer(model, layer_name='decoder4'):
    if hasattr(model, layer_name):
        return getattr(model, layer_name)
    else:
        for name, module in model.named_modules():
            if layer_name in name:
                return module
    raise ValueError(f'Layer {layer_name} not found in model')


def analyze_model_attention(model, dataloader, device, num_samples=5, save_dir='results/grad_cam'):
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    target_layer = get_target_layer(model, 'decoder4')
    grad_cam = SegmentationGradCAM(model, target_layer)
    
    model.eval()
    
    sample_count = 0
    with torch.no_grad():
        for images, masks, datasets in dataloader:
            if sample_count >= num_samples:
                break
            
            images = images.to(device)
            masks = masks.to(device)
            
            for i in range(images.shape[0]):
                if sample_count >= num_samples:
                    break
                
                image = images[i:i+1]
                mask = masks[i:i+1]
                dataset = datasets[i]
                
                with torch.enable_grad():
                    output = model(image)
                    cam = grad_cam.generate_cam(image)
                
                save_path = os.path.join(save_dir, f'grad_cam_{sample_count}_{dataset}.png')
                grad_cam.save_cam_visualization(image[0], mask[0], output[0], cam, save_path)
                
                sample_count += 1
    
    print(f'Grad-CAM visualizations saved to {save_dir}')