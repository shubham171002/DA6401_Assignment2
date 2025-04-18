import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics

class LitModel(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3, optimizer_name='adam'):
        super().__init__()
        self.model = model
        self.lr = learning_rate
        self.optimizer_name = optimizer_name.lower()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        acc = self.train_acc(preds, y)

        self.log("loss", loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log("accuracy", acc, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        val_loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        val_acc = self.val_acc(preds, y)

        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log("val_accuracy", val_acc, prog_bar=True, on_epoch=True, on_step=False,sync_dist=True)

        return val_loss

    def configure_optimizers(self):
        if self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        elif self.optimizer_name == 'nadam':
            optimizer = torch.optim.NAdam(self.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }