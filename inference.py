import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ----------------------------
# 1. Settings
# ----------------------------
IMG_SIZE = 420
NUM_CLASSES = 10  # update if needed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names (must match dataset order!)
CLASS_NAMES = ['butterfly', 'cat', 'chicken', 'cow', 'dog',
               'elephant', 'horse', 'sheep', 'spider', 'squirrel']

# ----------------------------
# 2. Preprocessing (same as training)
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------------------
# 3. Load Model
# ----------------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load("image_classifier.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ----------------------------
# 4. Inference Function
# ----------------------------
def predict(image_path, topk=3):
    # Load image
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)  # add batch dimension

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)  # convert logits -> probabilities
        top_probs, top_idxs = probs.topk(topk, dim=1)

    results = []
    for i in range(topk):
        class_name = CLASS_NAMES[top_idxs[0][i].item()]
        confidence = top_probs[0][i].item() * 100
        results.append((class_name, confidence))

    return results

# ----------------------------
# 5. Example Usage
# ----------------------------
if __name__ == "__main__":
    test_image = "./cat-test.webp"  # change to your image path
    predictions = predict(test_image, topk=3)

    print("✅ Predictions:")
    for cls, conf in predictions:
        print(f" - {cls}: {conf:.2f}%")

    print("\n")
    
    test_image = "./horse-test.jpg"  # change to your image path
    predictions = predict(test_image, topk=3)

    print("✅ Predictions:")
    for cls, conf in predictions:
        print(f" - {cls}: {conf:.2f}%")
