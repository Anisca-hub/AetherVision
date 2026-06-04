import torch
from lightning import Trainer
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from src.model import AetherModel
from src.data_loader import WeatherDataset, setup_data

def run_pipeline():
    setup_data()
    
    # Standard normalization for image models
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    DATA_ROOT = "./data/raw"
    
    # 1. Load full training set
    full_train_ds = WeatherDataset(root_dir=DATA_ROOT, split='train', transform=transform)
    
    # 2. 80/20 Train-Validation Split
    train_size = int(0.8 * len(full_train_ds))
    val_size = len(full_train_ds) - train_size
    train_ds, val_ds = random_split(full_train_ds, [train_size, val_size])
    
    # 3. Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    # 4. Model & Training
    model = AetherModel(num_classes=4)
    trainer = Trainer(max_epochs=10, accelerator="auto")
    
    print("Training started...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # 5. Save Artifact
    torch.save(model.state_dict(), "model.pth")
    print("Model saved as 'model.pth'.")

    # --- INFERENCE STEP ---
    print("Running inference on a validation batch...")
    model.eval()  # Set model to evaluation mode
    
    # Grab one batch from the validation loader
    data_iter = iter(val_loader)
    images, labels = next(data_iter)
    
    # Move to the same device as the model (GPU/CPU)
    images = images.to(model.device)
    
    # Run inference
    with torch.no_grad():
        result = model(images)
    
    print("Model Output (logits for the first image in batch):", result[0])

if __name__ == "__main__":
    run_pipeline()