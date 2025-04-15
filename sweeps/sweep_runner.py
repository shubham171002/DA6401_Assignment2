import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

root_dir = Path(__file__).resolve().parent.parent
data_dir = root_dir / "data" / "train"
if not data_dir.exists():
    raise FileNotFoundError(f"Data directory {data_dir} does not exist. Please check the path.")

import os
from models.cnn import CNN
from dataloaders.dataloader import get_dataloaders
from trainer.lightning_wrapper import LitModel
from sweeps.sweep_config import sweep_config
import wandb
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger



activation_map = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "mish": nn.Mish
}

def sweep_train():
    wandb.init()
    wandb_logger = WandbLogger(project="DA6401_A2", log_model=False, experiment=wandb.run)
    config = wandb.config

    wandb.run.name = (
        f"filt_{config.num_filters}_act_{config.activation_fn}_"
        f"org_{config.filter_organization}_opt_{config.optimizer}_"
        f"bn_{config.batchnorm}_aug_{config.data_aug}_"
        f"do_{config.dropout}_dense_{config.dense_neurons}_"
        f"lr_{config.learning_rate}_bs_{config.batch_size}"
    )

    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=config.batch_size,
        augment=config.data_aug,
        val_split=0.2,
        num_workers=4
    )

    model = CNN(
        num_classes=10,
        num_filters=config.num_filters,
        filter_size=3,
        activation_fn=activation_map[config.activation_fn],
        dense_neurons=config.dense_neurons,
        dropout=config.dropout,
        batchnorm=config.batchnorm,
        filter_organization=config.filter_organization
    )

    lit_model = LitModel(
        model=model,
        learning_rate=config.learning_rate,
        optimizer_name=config.optimizer
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="gpu",
        devices=2,
        precision="16-mixed",
        logger=wandb_logger,
        callbacks=[early_stopping, lr_monitor],
        log_every_n_steps=10,
    )

    trainer.fit(lit_model, train_loader, val_loader)

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="DA6401_A2")
    wandb.agent(sweep_id, function=sweep_train)