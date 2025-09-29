import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ----------------------------
# 1. Settings
# ----------------------------
IMG_SIZE = 420
BATCH_SIZE = 24
NUM_CLASSES = 10  # update if you change dataset
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 2. Data Preprocessing (must match training!)
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(root="./datasets/dataset/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Test samples:", len(test_dataset))
print("Classes:", test_dataset.classes)

# ----------------------------
# 3. Model Definition (same as training)
# ----------------------------
model = models.resnet18(weights=None)  # must match training
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load("image_classifier.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ----------------------------
# 4. Evaluation
# ----------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    correct, total = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_acc = correct / total
print(f"\n✅ Test Accuracy: {test_acc:.4f}")

# ----------------------------
# 5. Detailed Report
# ----------------------------
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
