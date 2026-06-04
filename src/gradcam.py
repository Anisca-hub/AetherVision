import torch
import numpy as np
import cv2

def get_gradcam(model, input_tensor, target_class):
    model.eval()
    features = []
    def hook(module, input, output): features.append(output)
    model.backbone.blocks[-1].register_forward_hook(hook)
    
    output = model(input_tensor)
    model.zero_grad()
    output[0, target_class].backward()
    
    grads = features[0].detach().cpu().numpy()
    heatmap = np.mean(grads, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    return cv2.resize(heatmap[0], (224, 224))