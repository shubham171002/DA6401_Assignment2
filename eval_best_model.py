import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import accuracy_score

from models.cnn import CNN
from dataloaders.dataloader import get_dataloaders
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Best hyperparameters
config = {
    "activation_fn": "silu",
    "batch_size": 64,
    "batchnorm": True,
    "data_aug": True,
    "dense_neurons": 128,
    "dropout": 0.3,
    "filter_organization": "double",
    "learning_rate": 0.00021049670003157813,
    "num_filters": 64,
    "optimizer": "nadam",
}

# Map activation function
activation_map = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "mish": nn.Mish
}

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load data — full train set for training, val folder for test set
train_loader, _ = get_dataloaders(
    data_dir="data/train",
    batch_size=config["batch_size"],
    augment=config["data_aug"],
    val_split=0.0,  # full train data
    num_workers=4
)


test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
test_dataset = datasets.ImageFolder("data/val", transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)

# Build model
model = CNN(
    num_classes=10,
    num_filters=config["num_filters"],
    activation_fn=activation_map[config["activation_fn"]],
    dense_neurons=config["dense_neurons"],
    dropout=config["dropout"],
    batchnorm=config["batchnorm"],
    filter_organization=config["filter_organization"]
).to(device)

# Optimizer
if config["optimizer"] == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
elif config["optimizer"] == "rmsprop":
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config["learning_rate"])
elif config["optimizer"] == "nadam":
    optimizer = torch.optim.NAdam(model.parameters(), lr=config["learning_rate"])

# Loss function
criterion = nn.CrossEntropyLoss()

# Train on full train set
def train_model(model, loader, optimizer, criterion, device, epochs=30):
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/30] - Loss: {running_loss / len(loader):.4f}")


# Evaluate on test set
def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")
    return acc


# Visualize 10x3 Prediction Grid
def visualize_predictions(model, dataset, class_names, device):
    model.eval()

    # Collect all predictions with their true labels
    all_preds = []
    for idx in range(len(dataset)):
        image, label = dataset[idx]
        input_img = image.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_img)
            pred = output.argmax(dim=1).item()

        all_preds.append((image, label, pred))

    # Shuffle the data
    random.shuffle(all_preds)

    # Collect 3 random samples per class
    class_to_images = {i: [] for i in range(len(class_names))}
    for image, label, pred in all_preds:
        if len(class_to_images[label]) < 3:
            class_to_images[label].append((image, label, pred))
        if all(len(v) == 3 for v in class_to_images.values()):
            break

    # Plot the grid
    fig, axes = plt.subplots(10, 3, figsize=(8, 24))

    for row in range(10):
        for col in range(3):
            ax = axes[row, col]
            image, true_label, pred_label = class_to_images[row][col]

            ax.imshow(np.transpose(image.numpy(), (1, 2, 0)))
            color = "green" if pred_label == true_label else "red"
            ax.set_title(
                f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}",
                fontsize=9,
                color=color
            )
            ax.axis("off")

    plt.tight_layout()
    plt.show()

    
# Execution  of all steps 
print("Training model on full train set")
train_model(model, train_loader, optimizer, criterion, device)

print("\nEvaluating on test set")
evaluate_model(model, test_loader, device)

print("\nGenerating 10x3 grid of predictions")
visualize_predictions(model, test_dataset, test_dataset.classes, device)
