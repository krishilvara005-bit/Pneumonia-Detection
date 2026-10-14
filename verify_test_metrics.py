
import csv
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = (
    PROJECT_DIR / "results" / "dataset_audit"
    / "resnet50_dataset_audit.csv"
)


def calculate_auc(labels, scores):
    positives = [
        score for label, score in zip(labels, scores) if label == 1
    ]
    negatives = [
        score for label, score in zip(labels, scores) if label == 0
    ]

    if not positives or not negatives:
        return None

    wins = sum(
        1 if p > n else 0.5 if p == n else 0
        for p in positives
        for n in negatives
    )

    return wins / (len(positives) * len(negatives))


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Audit CSV not found: {CSV_PATH}")

    with open(CSV_PATH, newline="", encoding="utf-8-sig") as file:
        rows = [
            row for row in csv.DictReader(file)
            if row["split"].lower() == "test"
        ]

    if not rows:
        print("No test records found.")
        return

    tn = fp = fn = tp = 0
    labels = []
    scores = []

    for row in rows:
        actual = 1 if row["actual_folder_label"].upper() == "PNEUMONIA" else 0
        predicted = 1 if row["predicted_label"].upper() == "PNEUMONIA" else 0
        score = float(row["pneumonia_score"])

        labels.append(actual)
        scores.append(score)

        if actual == 0 and predicted == 0:
            tn += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        elif actual == 1 and predicted == 0:
            fn += 1
        else:
            tp += 1

    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    specificity = tn / (tn + fp) if tn + fp else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )
    auc = calculate_auc(labels, scores)

    print("\n========== RESNET50 TEST METRICS ==========")
    print(f"Test images: {total}")
    print("\nConfusion matrix (actual rows, predicted columns):")
    print("                 Pred Normal  Pred Pneumonia")
    print(f"Actual Normal       {tn:4d}          {fp:4d}")
    print(f"Actual Pneumonia    {fn:4d}          {tp:4d}")

    print("\nMetrics:")
    print(f"Accuracy:    {accuracy:.4%}")
    print(f"Precision:   {precision:.4%}")
    print(f"Recall:      {recall:.4%}")
    print(f"Specificity: {specificity:.4%}")
    print(f"F1-score:    {f1:.4%}")
    print(f"ROC-AUC:     {auc:.4%}" if auc is not None else "ROC-AUC: N/A")

    print("\nThese metrics describe predictions in the audit CSV.")
    print("They do not independently validate medical labels.")


if __name__ == "__main__":
    main()