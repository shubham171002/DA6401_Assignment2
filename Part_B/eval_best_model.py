import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import gc

# Hyperparameters (from best sweep)
batch_size = 32
lr = 1.6848e-5
optimizer_name = 'nadam'
weight_decay = 1e-4
num_classes = 10
epochs = 10

# Directories
train_dir = "data/train"
test_dir = "data/test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = ImageFolder(train_dir, transform=train_transform)
test_dataset = ImageFolder(test_dir, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Create and modify model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
model = model.to(device)

# Freeze all except layer4 and fc (partial fine-tuning strategy)
def freeze_layers(model):
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

freeze_layers(model)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
if optimizer_name == 'nadam':
    optimizer = torch.optim.NAdam(params, lr=lr, weight_decay=weight_decay)
elif optimizer_name == 'adam':
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
elif optimizer_name == 'rmsprop':
    optimizer = torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
else:
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)

# Loss function
criterion = nn.CrossEntropyLoss()

# Training
print("Model Training")
model.train()
for epoch in range(epochs):
    running_loss, correct, total = 0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    print(f"Epoch {epoch+1}: Train Loss={running_loss/len(train_loader):.4f}, Train Acc={train_acc:.4f}")

    torch.cuda.empty_cache()
    gc.collect()

# Evaluation
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_acc = correct / total
print(f"Final Test Accuracy: {test_acc:.4f}")
