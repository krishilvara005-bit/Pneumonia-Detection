
import pandas as pd
from pathlib import Path
from src.inference import predict_image

CSV_PATH = Path("results/dataset_audit/resnet50_dataset_audit.csv")

df = pd.read_csv(CSV_PATH)

# Select 10 existing test images; do not modify the dataset.
test_df = df[df["split"].str.lower() == "test"].head(10).copy()

if test_df.empty:
    raise ValueError("No test records found in the audit CSV.")

print("\nSAVED MODEL vs AUDIT CSV")
print("=" * 85)

results = []

for _, row in test_df.iterrows():
    image_path = Path(row["image_path"])

    if not image_path.exists():
        print(f"\nMISSING IMAGE: {image_path}")
        continue

    predicted_label, current_score = predict_image(str(image_path))

    saved_score = float(row["pneumonia_score"])
    difference = abs(float(current_score) - saved_score)

    matches = difference < 0.0001

    results.append(matches)

    print(f"\nImage: {image_path.name}")
    print(f"CSV label:       {row['predicted_label']}")
    print(f"Current label:   {predicted_label}")
    print(f"CSV score:       {saved_score:.8f}")
    print(f"Current score:   {float(current_score):.8f}")
    print(f"Absolute diff:   {difference:.8f}")
    print(f"Score match:     {'YES' if matches else 'NO'}")

print("\n" + "=" * 85)
print(f"Images compared: {len(results)}")
print(f"Scores matching within tolerance: {sum(results)}")

if results and all(results):
    print("RESULT: Strong evidence that the saved model reproduces these CSV predictions.")
elif results:
    print("RESULT: Differences found. Further investigation is needed.")
else:
    print("RESULT: No images could be compared.")