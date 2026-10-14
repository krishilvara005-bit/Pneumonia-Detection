
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.inference import generate_gradcam

# Original image location
image_path = (
    Path.home()
    / "Downloads"
    / "PneumoVision Dataset"
    / "chest_xray"
    / "chest_xray"
    / "test"
    / "NORMAL"
    / "IM-0022-0001.jpeg"
)

# Output location
output_dir = (
    Path(__file__).resolve().parent
    / "results"
    / "dataset_audit"
    / "gradcam"
)
output_dir.mkdir(parents=True, exist_ok=True)

if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

# Generate Grad-CAM
original_image, overlay_array = generate_gradcam(str(image_path))

# Convert overlay to a PIL image
overlay_image = Image.fromarray(
    np.asarray(overlay_array).astype(np.uint8)
).convert("RGB")

original_image = original_image.convert("RGB")

# Resize both images to the same dimensions
original_image = original_image.resize((400, 400))
overlay_image = overlay_image.resize((400, 400))

# Create a side-by-side comparison
canvas = Image.new("RGB", (820, 455), "white")
canvas.paste(original_image, (5, 40))
canvas.paste(overlay_image, (415, 40))

draw = ImageDraw.Draw(canvas)
draw.text((5, 12), "Original X-ray", fill="black")
draw.text((415, 12), "ResNet50 Grad-CAM Overlay", fill="black")

output_path = output_dir / "IM-0022-0001_gradcam.png"
canvas.save(output_path)

print("Grad-CAM generated successfully!")
print(f"Saved to: {output_path}")