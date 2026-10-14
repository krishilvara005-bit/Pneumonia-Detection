
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


baseline_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

rgb_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_checkpoint(model, filename):
    path = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


def load_all_models():
    baseline = BaselineCNN()
    baseline = load_checkpoint(baseline, "baseline_cnn.pth")

    resnet = models.resnet50(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, 1)
    resnet = load_checkpoint(resnet, "resnet50_pneumonia.pth")

    mobilenet = models.mobilenet_v2(weights=None)
    mobilenet.classifier[1] = nn.Linear(
        mobilenet.classifier[1].in_features, 1
    )
    mobilenet = load_checkpoint(
        mobilenet, "mobilenetv2_pneumonia.pth"
    )

    return baseline, resnet, mobilenet


def predict_all(image, loaded_models):
    baseline, resnet, mobilenet = loaded_models

    gray_tensor = baseline_transform(
        image.convert("L")
    ).unsqueeze(0).to(DEVICE)

    rgb_tensor = rgb_transform(
        image.convert("RGB")
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        baseline_prob = torch.sigmoid(baseline(gray_tensor)).item()
        resnet_prob = torch.sigmoid(resnet(rgb_tensor)).item()
        mobilenet_prob = torch.sigmoid(mobilenet(rgb_tensor)).item()

    probabilities = {
        "Baseline CNN": baseline_prob,
        "ResNet50": resnet_prob,
        "MobileNetV2": mobilenet_prob
    }

    return {
        name: {
            "prediction": "Pneumonia" if prob >= 0.5 else "Normal",
            "probability": prob
        }
        for name, prob in probabilities.items()
    }

def generate_gradcam_for_model(image, model, target_layer, grayscale=False):
    """
    Generate a Grad-CAM overlay for one model.
    Returns a PIL RGB image.
    """
    activations = {}
    gradients = {}

    def forward_hook(module, inputs, output):
        activations["value"] = output

        def save_gradient(gradient):
            gradients["value"] = gradient

        output.register_hook(save_gradient)

    hook = target_layer.register_forward_hook(forward_hook)

    try:
        if grayscale:
            processed_image = image.convert("L")
            input_tensor = baseline_transform(processed_image)
        else:
            processed_image = image.convert("RGB")
            input_tensor = rgb_transform(processed_image)

        input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

        model.eval()
        model.zero_grad(set_to_none=True)

        output = model(input_tensor)
        score = output.reshape(-1)[0]
        score.backward()

        feature_maps = activations["value"].detach()
        grads = gradients["value"].detach()

        # Average gradients across the spatial dimensions
        weights = grads.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of feature maps
        cam = (weights * feature_maps).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        # Normalize the heatmap
        cam = torch.nn.functional.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0].cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()

        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Resize heatmap to the original image dimensions
        original = np.array(image.convert("RGB"))
        height, width = original.shape[:2]

        cam = cv2.resize(cam, (width, height))
        heatmap = np.uint8(255 * cam)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Blend original X-ray and heatmap
        overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

        return Image.fromarray(overlay)

    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)


def generate_gradcam_all(image, loaded_models):
    """
    Generate Grad-CAM overlays for all three models.
    """
    baseline, resnet, mobilenet = loaded_models

    targets = {
        "Baseline CNN": (baseline, baseline.features[6], True),
        "ResNet50": (resnet, resnet.layer4[-1].conv3, False),
        "MobileNetV2": (mobilenet, mobilenet.features[-1], False),
    }

    results = {}

    for name, (model, target_layer, grayscale) in targets.items():
        results[name] = generate_gradcam_for_model(
            image=image,
            model=model,
            target_layer=target_layer,
            grayscale=grayscale
        )

    return results