import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

class WeatherDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.weather_to_idx = {"Cloudy": 0, "Sunny": 1, "Rainy": 2, "Foggy": 3}
        
        # 1. Setup paths
        json_path = os.path.join(root_dir, f"{split}_dataset", f"{split}.json")
        self.image_dir = os.path.join(root_dir, f"{split}_dataset", f"{split}_images")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        annotations = data.get('annotations', [])
        
        # 2. Map filenames to labels
        self.labels = {}
        for entry in annotations:
            filename = entry.get('filename')
            weather = entry.get('weather')
            if filename and weather in self.weather_to_idx:
                # Clean filename (removes folder prefixes like 'train_images/')
                clean_name = os.path.basename(filename)
                self.labels[clean_name] = self.weather_to_idx[weather]
        
        # 3. Store valid file list
        self.image_files = list(self.labels.keys())
        print(f"DEBUG: Successfully loaded {len(self.image_files)} images.")
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        label = self.labels[filename]
        
        # 1. Clean the filename: replace backslashes with forward slashes
        # and extract only the actual file name
        clean_filename = os.path.basename(filename.replace('\\', '/'))
        
        # 2. Construct path: ensure we only join the root directory and the filename
        img_path = os.path.join(self.image_dir, clean_filename)
        
        # 3. Load
        try:
            img = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"DEBUG: Tried to find image at: {img_path}")
            raise
            
        if self.transform:
            img = self.transform(img)
            
        return img, label