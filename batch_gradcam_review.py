
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.inference import generate_gradcam

PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = (
    PROJECT_DIR / "results" / "dataset_audit"
    / "resnet50_mismatches.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR / "results" / "dataset_audit"
    / "batch_gradcam"
)

# Contact sheet settings
COLUMNS = 4
ROWS_PER_SHEET = 5
CASES_PER_SHEET = COLUMNS * ROWS_PER_SHEET

CARD_W = 340
CARD_H = 175
IMAGE_W = 160
IMAGE_H = 125

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_font(size=12):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def create_contact_sheets(records, category):
    if not records:
        print(f"No images for category: {category}")
        return

    category_dir = OUTPUT_DIR / category
    category_dir.mkdir(parents=True, exist_ok=True)

    font = get_font()

    for start in range(0, len(records), CASES_PER_SHEET):
        batch = records[start:start + CASES_PER_SHEET]
        sheet_number = start // CASES_PER_SHEET + 1

        sheet_rows = math.ceil(len(batch) / COLUMNS)

        sheet = Image.new(
            "RGB",
            (COLUMNS * CARD_W, sheet_rows * CARD_H),
            "white",
        )
        draw = ImageDraw.Draw(sheet)

        for index, record in enumerate(batch):
            x = (index % COLUMNS) * CARD_W
            y = (index // COLUMNS) * CARD_H

            original_path = Path(record["image_path"])

            try:
                # Generate Grad-CAM using the existing trained model
                original, overlay_array = generate_gradcam(
                    str(original_path)
                )

                original = original.convert("RGB")
                overlay = Image.fromarray(
                    np.asarray(overlay_array).astype(np.uint8)
                ).convert("RGB")

                original.thumbnail((IMAGE_W, IMAGE_H))
                overlay.thumbnail((IMAGE_W, IMAGE_H))

                original_x = x + 5 + (IMAGE_W - original.width) // 2
                overlay_x = x + 170 + (
                    IMAGE_W - overlay.width
                ) // 2

                image_y = y + 5

                sheet.paste(original, (original_x, image_y))
                sheet.paste(overlay, (overlay_x, image_y))

                draw.text(
                    (x + 5, y + 133),
                    "Original",
                    fill="black",
                    font=font,
                )
                draw.text(
                    (x + 170, y + 133),
                    "Grad-CAM",
                    fill="black",
                    font=font,
                )

                details = (
                    f"{record['actual_folder_label']} -> "
                    f"{record['predicted_label']} | "
                    f"Score: {float(record['pneumonia_score']):.3f}"
                )

                draw.text(
                    (x + 5, y + 151),
                    details,
                    fill="black",
                    font=font,
                )

            except Exception as exc:
                draw.text(
                    (x + 5, y + 10),
                    f"Failed: {original_path.name}",
                    fill="red",
                    font=font,
                )
                print(f"Error processing {original_path}: {exc}")

        output_path = category_dir / f"sheet_{sheet_number:02d}.jpg"
        sheet.save(output_path, quality=90)
        print(f"Saved: {output_path}")


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Audit CSV not found: {CSV_PATH}")

    with open(CSV_PATH, newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    test_rows = [
        row for row in rows
        if row["split"].lower() == "test"
    ]

    false_positives = [
        row for row in test_rows
        if row["actual_folder_label"].upper() == "NORMAL"
        and row["predicted_label"].upper() == "PNEUMONIA"
    ]

    false_negatives = [
        row for row in test_rows
        if row["actual_folder_label"].upper() == "PNEUMONIA"
        and row["predicted_label"].upper() == "NORMAL"
    ]

    print(f"Test false positives: {len(false_positives)}")
    print(f"Test false negatives: {len(false_negatives)}")
    print("\nGenerating Grad-CAM contact sheets...")
    print("This may take some time.")

    create_contact_sheets(false_positives, "false_positives")
    create_contact_sheets(false_negatives, "false_negatives")

    print("\n========== COMPLETE ==========")
    print(f"Output folder: {OUTPUT_DIR}")
    print("Original dataset and labels were not modified.")


if __name__ == "__main__":
    main()