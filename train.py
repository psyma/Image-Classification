import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ----------------------------
# 1. Hyperparameters
# ----------------------------
IMG_SIZE = 420
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 10  # <-- change this to match your dataset

# ----------------------------
# 2. Data Preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225])
])

# Assuming dataset is structured like:
# root/train/class_x/xxx.png
# root/train/class_y/yyy.png
train_dataset = datasets.ImageFolder(root="./datasets/dataset/train", transform=transform)
val_dataset   = datasets.ImageFolder(root="./datasets/dataset/val", transform=transform)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Classes:", train_dataset.classes)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------
# 3. Model (transfer learning with ResNet18)
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = None #models.ResNet18_Weights.IMAGENET1K_V1  # or ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)  # adjust classifier
model = model.to(device)

# ----------------------------
# 4. Loss and Optimizer
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ----------------------------
# 5. Training Loop
# ----------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # tqdm wrapper
    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # update tqdm bar
        progress_bar.set_postfix(loss=loss.item())

    train_acc = correct / total
    train_loss = running_loss / total

    # Validation
    model.eval()
    val_correct, val_total, val_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    val_loss /= val_total

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# ----------------------------
# 6. Save Model
# ----------------------------
torch.save(model.state_dict(), "image_classifier.pth")
print("✅ Training complete, model saved as image_classifier.pth")
