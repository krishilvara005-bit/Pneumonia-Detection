
import csv
from pathlib import Path
from collections import Counter

from src.inference import predict_image

# Actual dataset location
DATASET_DIR = (
    Path.home()
    / "Downloads"
    / "PneumoVision Dataset"
    / "chest_xray"
    / "chest_xray"
)

# Save reports inside the project
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "dataset_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ["train", "val", "test"]
CLASSES = ["NORMAL", "PNEUMONIA"]


def main():
    if not DATASET_DIR.exists():
        print(f"Dataset folder not found: {DATASET_DIR}")
        return

    csv_path = OUTPUT_DIR / "resnet50_dataset_audit.csv"
    mismatches = []
    counts = Counter()
    errors = []

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split",
                "actual_folder_label",
                "predicted_label",
                "pneumonia_score",
                "prediction_matches_folder",
                "image_path",
            ],
        )
        writer.writeheader()

        for split in SPLITS:
            split_dir = DATASET_DIR / split

            if not split_dir.exists():
                print(f"Skipping missing split: {split_dir}")
                continue

            for actual_label in CLASSES:
                class_dir = split_dir / actual_label

                if not class_dir.exists():
                    print(f"Skipping missing folder: {class_dir}")
                    continue

                images = sorted(
                    p for p in class_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                )

                print(f"\n{split}/{actual_label}: {len(images)} images")

                for index, image_path in enumerate(images, start=1):
                    try:
                        predicted_label, score = predict_image(str(image_path))

                        matches = predicted_label.upper() == actual_label

                        row = {
                            "split": split,
                            "actual_folder_label": actual_label,
                            "predicted_label": predicted_label,
                            "pneumonia_score": round(float(score), 6),
                            "prediction_matches_folder": matches,
                            "image_path": str(image_path),
                        }

                        writer.writerow(row)
                        counts[f"{split}_total"] += 1

                        if matches:
                            counts[f"{split}_matching"] += 1
                        else:
                            counts[f"{split}_mismatch"] += 1
                            mismatches.append(row)

                        if index % 100 == 0:
                            print(f"  Processed {index}/{len(images)}")

                    except Exception as exc:
                        errors.append({
                            "split": split,
                            "image_path": str(image_path),
                            "error": str(exc),
                        })
                        print(f"  Error processing: {image_path.name}")

    mismatch_path = OUTPUT_DIR / "resnet50_mismatches.csv"
    with open(mismatch_path, "w", newline="", encoding="utf-8-sig") as file:
        fields = [
            "split",
            "actual_folder_label",
            "predicted_label",
            "pneumonia_score",
            "prediction_matches_folder",
            "image_path",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(mismatches)

    error_path = OUTPUT_DIR / "processing_errors.csv"
    with open(error_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file, fieldnames=["split", "image_path", "error"]
        )
        writer.writeheader()
        writer.writerows(errors)

    print("\n========== AUDIT SUMMARY ==========")

    for split in SPLITS:
        total = counts[f"{split}_total"]
        matching = counts[f"{split}_matching"]
        mismatch = counts[f"{split}_mismatch"]

        print(f"{split.upper()}:")
        print(f"  Processed: {total}")
        print(f"  Predictions matching folder: {matching}")
        print(f"  Predictions differing from folder: {mismatch}")

    print(f"\nTotal mismatches: {len(mismatches)}")
    print(f"Processing errors: {len(errors)}")
    print(f"\nFull report: {csv_path}")
    print(f"Mismatch report: {mismatch_path}")
    print(f"Error report: {error_path}")
    print("\nOriginal dataset files were not modified.")


if __name__ == "__main__":
    main()