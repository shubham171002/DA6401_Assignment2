import gc
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import wandb
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10
data_dir = "data/train"  # Adjust if needed

def freeze_layers(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
        if "layer4" in name or "fc" in name:
            param.requires_grad = True

def create_resnet50():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    freeze_layers(model)
    return model.to(device)

def get_dataloaders(batch_size=32, val_split=0.2):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageFolder(data_dir, transform=transform)
    targets = [s[1] for s in dataset.samples]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    train_idx, val_idx = next(splitter.split(np.zeros(len(targets)), targets))
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader

def evaluate(model, loader, criterion):
    model.eval()
    correct, total, total_loss = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total_loss += loss.item()
            total += labels.size(0)
    return correct / total, total_loss / len(loader)

def train_model():
    config = wandb.config
    model = create_resnet50()
    train_loader, val_loader = get_dataloaders(batch_size=config.batch_size)
    params = [p for p in model.parameters() if p.requires_grad]

    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == "nadam":
        optimizer = torch.optim.NAdam(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
    else:
        raise ValueError("Invalid optimizer")

    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0
        correct, total = 0, 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")

        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress.set_postfix({
                "loss": running_loss / (progress.n + 1),
                "acc": 100 * correct / total
            })

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        val_acc, val_loss = evaluate(model, val_loader, criterion)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "learning_rate": optimizer.param_groups[0]["lr"]
        })

        torch.cuda.empty_cache()
        gc.collect()

    wandb.run.summary["best_val_accuracy"] = best_val_acc