import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Image preprocessing used for ResNet50
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Load ResNet50 architecture
model = models.resnet50(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    1
)


import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(
    BASE_DIR,
    "models",
    "resnet50_pneumonia.pth"
)

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model = model.to(device)
model.eval()


def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        output = model(image_tensor)

        probability = torch.sigmoid(output).item()

    if probability >= 0.5:
        prediction = "Pneumonia"
    else:
        prediction = "Normal"

    return prediction, probability

if __name__ == "__main__":

    test_image = "../test_image.jpeg"

    prediction, probability = predict_image(test_image)

    print("Prediction:", prediction)
    print("Pneumonia Probability:", f"{probability:.4f}")