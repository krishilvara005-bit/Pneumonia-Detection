
import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "results" / "dataset_audit" / "resnet50_mismatches.csv"
OUTPUT_DIR = PROJECT_DIR / "results" / "dataset_audit" / "review_sheets"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THUMB_W = 220
THUMB_H = 190
TEXT_H = 65
COLUMNS = 4
IMAGES_PER_SHEET = 20


def load_font(size=13):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def main():
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    rows = [
        row for row in rows
        if row["split"].lower() == "test"
    ]

    rows.sort(
        key=lambda row: (
            row["actual_folder_label"],
            row["predicted_label"],
            row["image_path"],
        )
    )

    if not rows:
        print("No test mismatches found.")
        return

    font = load_font()
    sheet_height = math.ceil(IMAGES_PER_SHEET / COLUMNS) * (
        THUMB_H + TEXT_H
    )

    for start in range(0, len(rows), IMAGES_PER_SHEET):
        batch = rows[start:start + IMAGES_PER_SHEET]
        sheet_number = start // IMAGES_PER_SHEET + 1

        sheet = Image.new(
            "RGB",
            (COLUMNS * THUMB_W, sheet_height),
            "white",
        )
        draw = ImageDraw.Draw(sheet)

        for index, row in enumerate(batch):
            image_path = Path(row["image_path"])
            x = (index % COLUMNS) * THUMB_W
            y = (index // COLUMNS) * (THUMB_H + TEXT_H)

            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    image.thumbnail((THUMB_W - 10, THUMB_H - 10))
                    image_x = x + (THUMB_W - image.width) // 2
                    image_y = y + (THUMB_H - image.height) // 2
                    sheet.paste(image, (image_x, image_y))
            except Exception as exc:
                draw.text(
                    (x + 5, y + 5),
                    f"Could not open image: {exc}",
                    fill="red",
                    font=font,
                )

            score = float(row["pneumonia_score"])
            text_lines = [
                f"Actual: {row['actual_folder_label']}",
                f"Predicted: {row['predicted_label']}",
                f"Pneumonia score: {score:.3f}",
                f"File: {image_path.name}",
            ]

            for line_number, line in enumerate(text_lines):
                draw.text(
                    (x + 5, y + THUMB_H + line_number * 15),
                    line,
                    fill="black",
                    font=font,
                )

        output_path = OUTPUT_DIR / f"test_mismatches_{sheet_number:02d}.jpg"
        sheet.save(output_path, quality=90)
        print(f"Saved: {output_path}")

    print(f"\nTotal images: {len(rows)}")
    print(f"Contact sheets saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()