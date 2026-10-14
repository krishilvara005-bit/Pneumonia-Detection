
import csv
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = (
    PROJECT_DIR / "results" / "dataset_audit"
    / "resnet50_dataset_audit.csv"
)

THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]


def main():
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as file:
        rows = [
            row for row in csv.DictReader(file)
            if row["split"].lower() == "test"
        ]

    if not rows:
        print("No test data found.")
        return

    actual_labels = [
        1 if r["actual_folder_label"].upper() == "PNEUMONIA" else 0
        for r in rows
    ]
    scores = [float(r["pneumonia_score"]) for r in rows]

    print("\n========== THRESHOLD COMPARISON ==========")
    print("Threshold | Accuracy | Precision | Recall | Specificity | FP | FN")

    for threshold in THRESHOLDS:
        tn = fp = fn = tp = 0

        for actual, score in zip(actual_labels, scores):
            predicted = 1 if score >= threshold else 0

            if actual == 0 and predicted == 0:
                tn += 1
            elif actual == 0 and predicted == 1:
                fp += 1
            elif actual == 1 and predicted == 0:
                fn += 1
            else:
                tp += 1

        accuracy = (tn + tp) / len(rows)
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        specificity = tn / (tn + fp) if tn + fp else 0

        print(
            f"{threshold:9.2f} | "
            f"{accuracy:7.2%} | "
            f"{precision:9.2%} | "
            f"{recall:6.2%} | "
            f"{specificity:11.2%} | "
            f"{fp:2d} | {fn:2d}"
        )

    print("\nAnalysis only: model weights and dataset labels are unchanged.")
    print("Thresholds are being compared on the existing test set.")
    print("Do not select a final threshold based on test results alone.")


if __name__ == "__main__":
    main()