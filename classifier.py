import pytorch_lightning as pl
import torchvision.models as models
import torch


class FathomNetClassifier(pl.LightningModule):
    """
    A PyTorch Lightning module for classifying marine species using a pretrained EfficientNetV2-M model.
    This module adapts the EfficientNetV2-M architecture for custom classification tasks with a specified 
    number of output classes. It includes training and validation steps, uses cross-entropy loss, 
    and sets up an AdamW optimizer with learning rate scheduling.
    """
    def __init__(self, num_classes):
        """
        Args:
            num_classes (int): Number of output classes.
        """
        super().__init__()
        # Load a pretrained EfficientNetV2-L model
        self.model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)

        # Replace the classifier head with a more advanced version
        in_features = self.model.classifier[1].in_features
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features, 100),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(100, num_classes)
        )

        # Define loss function (Cross-Entropy Loss with label smoothing)
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, x):
        """Forward pass of the model."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Defines the training step."""
        x, y = batch  # Extract images and labels from batch
        y_hat = self(x)  # Forward pass
        loss = self.criterion(y_hat, y)  # Compute loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Defines the validation step."""
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Sets up the optimizer and learning rate scheduler for training."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=3,
                verbose=True,
                min_lr=1e-6
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
