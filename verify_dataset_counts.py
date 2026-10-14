
from pathlib import Path

DATASET_DIR = (
    Path.home()
    / "Downloads"
    / "PneumoVision Dataset"
    / "chest_xray"
    / "chest_xray"
)

SPLITS = ["train", "val", "test"]
CLASSES = ["NORMAL", "PNEUMONIA"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    print("========== DATASET SPLIT VERIFICATION ==========")

    if not DATASET_DIR.exists():
        print(f"Dataset not found: {DATASET_DIR}")
        return

    grand_total = 0

    for split in SPLITS:
        split_dir = DATASET_DIR / split
        split_total = 0

        print(f"\n{split.upper()}")

        for class_name in CLASSES:
            class_dir = split_dir / class_name

            if not class_dir.exists():
                print(f"  {class_name}: MISSING FOLDER")
                continue

            count = sum(
                1
                for path in class_dir.rglob("*")
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )

            print(f"  {class_name}: {count}")
            split_total += count

        print(f"  Split total: {split_total}")
        grand_total += split_total

    print(f"\nGRAND TOTAL: {grand_total}")
    print("No files or labels were modified.")


if __name__ == "__main__":
    main()