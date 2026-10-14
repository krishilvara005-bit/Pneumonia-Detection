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


def predict_image(image_input):
    # Accept either an uploaded image or a file path
    if isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        image = Image.open(image_input).convert("RGB")

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probability = torch.sigmoid(output).item()

    prediction = "Pneumonia" if probability >= 0.5 else "Normal"

    return prediction, probability

if __name__ == "__main__":

    test_image = "../test_image.jpeg"

    prediction, probability = predict_image(test_image)

    print("Prediction:", prediction)
    print("Pneumonia Probability:", f"{probability:.4f}")


import numpy as np
import cv2
def generate_gradcam(image_input):
    """
    Generate a Grad-CAM heatmap for the uploaded X-ray.
    Returns:
        original_image: RGB PIL image
        overlay_image: RGB numpy array with heatmap overlay
    """

    # 1. Load the image
    if isinstance(image_input, Image.Image):
        original_image = image_input.convert("RGB")
    else:
        original_image = Image.open(image_input).convert("RGB")

    # 2. Apply the same preprocessing used by ResNet50
    input_tensor = transform(original_image)
    input_tensor = input_tensor.unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    # 3. Store activations and gradients from ResNet50's last convolutional layer
    activations = []
    gradients = []

    target_layer = model.layer4[-1].conv3

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        # 4. Forward pass
        model.zero_grad(set_to_none=True)
        output = model(input_tensor)

        # Explain the Pneumonia score
        score = output.squeeze()
        score.backward()

        # 5. Calculate Grad-CAM weights
        feature_maps = activations[0]
        gradient_maps = gradients[0]

        weights = gradient_maps.mean(dim=(2, 3), keepdim=True)

        cam = (weights * feature_maps).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        # Normalize the heatmap
        cam = cam.squeeze().detach().cpu().numpy()

        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # 6. Resize heatmap to original image dimensions
        width, height = original_image.size

        heatmap = cv2.resize(
            cam,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

        heatmap_uint8 = np.uint8(255 * heatmap)
        colored_heatmap = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )
        colored_heatmap = cv2.cvtColor(
            colored_heatmap,
            cv2.COLOR_BGR2RGB
        )

        # 7. Overlay heatmap on the original image
        original_array = np.array(original_image)

        overlay_image = cv2.addWeighted(
            original_array,
            0.6,
            colored_heatmap,
            0.4,
            0
        )

        return original_image, overlay_image

    finally:
        forward_handle.remove()
        backward_handle.remove()