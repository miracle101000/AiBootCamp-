import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import models

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained MobileNetV2
base_model = models.mobilenet_v2(pretrained=True)

# Freeze all layers
for param in base_model.features.parameters():
    param.requires_grad = False

# Modify classification head
num_classes = 5
base_model.classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(1280, num_classes),
    nn.Softmax(dim=1)
)

base_model = base_model.to(device)

# Define data transformations
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}

# Load datasets
data_dir = "PATH_TO_DATASET"
datasets_dict = {x: datasets.ImageFolder(root=f"{data_dir}/{x}", transform=data_transforms[x]) for x in ["train", "val"]}
dataloaders = {x: DataLoader(datasets_dict[x], batch_size=32, shuffle=True, num_workers=4) for x in ["train", "val"]}

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(base_model.classifier.parameters(), lr=1e-5)

# Training loop
epochs = 10
for epoch in range(epochs):
    base_model.train()
    running_loss, correct = 0.0, 0
    for images, labels in dataloaders["train"]:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = base_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
    
    # Validation
    base_model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for images, labels in dataloaders["val"]:
            images, labels = images.to(device), labels.to(device)
            outputs = base_model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_correct += (outputs.argmax(dim=1) == labels).sum().item()
    
    # Print epoch results
    print(f"Epoch {epoch+1}/{epochs}: ")
    print(f"Train Loss: {running_loss/len(dataloaders['train'])}, Accuracy: {correct/len(datasets_dict['train'])}")
    print(f"Val Loss: {val_loss/len(dataloaders['val'])}, Accuracy: {val_correct/len(datasets_dict['val'])}")
