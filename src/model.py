import lightning as L
import torch
import timm
from .loss import FocalLoss
from torchmetrics.classification import MulticlassF1Score

class AetherModel(L.LightningModule):
    def __init__(self, num_classes=4, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        self.criterion = FocalLoss()
        
        # Initialize Metrics
        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')

    def forward(self, x): 
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        # FIX: Convert indices [32] to one-hot [32, 4]
        y_one_hot = torch.nn.functional.one_hot(y, num_classes=4).float()
        
        loss = self.criterion(logits, y_one_hot)
        
        # Log F1 (use original y here for MulticlassF1Score)
        self.train_f1(logits, y) 
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)