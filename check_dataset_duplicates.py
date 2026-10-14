
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = (
    Path.home()
    / "Downloads"
    / "PneumoVision Dataset"
    / "chest_xray"
    / "chest_xray"
)

OUTPUT_DIR = PROJECT_DIR / "results" / "dataset_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ["train", "val", "test"]


def calculate_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_DIR}")

    hashes = defaultdict(list)
    total_images = 0

    for split in SPLITS:
        split_dir = DATASET_DIR / split

        if not split_dir.exists():
            print(f"Skipping missing folder: {split_dir}")
            continue

        for image_path in split_dir.rglob("*"):
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            try:
                digest = calculate_hash(image_path)
                relative_path = image_path.relative_to(DATASET_DIR)

                hashes[digest].append({
                    "split": split,
                    "label": image_path.parent.name,
                    "path": str(relative_path),
                })

                total_images += 1

            except Exception as exc:
                print(f"Could not read {image_path}: {exc}")

        print(f"Finished scanning {split}")

    duplicate_groups = [
        entries for entries in hashes.values()
        if len(entries) > 1
    ]

    report_path = OUTPUT_DIR / "duplicate_images_report.csv"

    with open(report_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "duplicate_group",
                "split",
                "label",
                "relative_path",
            ],
        )
        writer.writeheader()

        for group_number, entries in enumerate(duplicate_groups, start=1):
            for entry in entries:
                writer.writerow({
                    "duplicate_group": group_number,
                    "split": entry["split"],
                    "label": entry["label"],
                    "relative_path": entry["path"],
                })

    cross_split_groups = 0
    conflicting_label_groups = 0

    for entries in duplicate_groups:
        splits = {entry["split"] for entry in entries}
        labels = {entry["label"].upper() for entry in entries}

        if len(splits) > 1:
            cross_split_groups += 1

        if len(labels) > 1:
            conflicting_label_groups += 1

    print("\n========== DUPLICATE CHECK SUMMARY ==========")
    print(f"Images scanned: {total_images}")
    print(f"Unique file hashes: {len(hashes)}")
    print(f"Duplicate groups: {len(duplicate_groups)}")
    print(f"Groups repeated across splits: {cross_split_groups}")
    print(f"Groups with conflicting folder labels: {conflicting_label_groups}")
    print(f"\nReport saved to: {report_path}")
    print("\nNo images or labels were modified.")


if __name__ == "__main__":
    main()