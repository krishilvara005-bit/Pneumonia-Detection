# Final Dataset and Model Audit Report

## 1. Project Information

**Project Title:** Deep Learning-Based Pneumonia Detection from Chest X-ray Images: A Comparative Evaluation of CNN and Transfer Learning Models

**Primary Model Audited:** ResNet50
**Classification Task:** Binary classification — Normal vs Pneumonia
**Input Image Size:** 224 × 224 pixels
**Decision Threshold:** 0.50

## 2. Dataset Verification

The dataset was checked for image counts, duplicate files, processing errors, and model prediction disagreements.

| Dataset Split |    Normal | Pneumonia |     Total |
| ------------- | --------: | --------: | --------: |
| Train         |     1,341 |     3,875 |     5,216 |
| Validation    |         8 |         8 |        16 |
| Test          |       234 |       390 |       624 |
| **Total**     | **1,583** | **4,273** | **5,856** |

The training workflow used a stratified 80:20 split of the training folder, producing 4,172 training images and 1,044 validation images. The original test set of 624 images was kept separate.

## 3. Duplicate Image Audit

* Images scanned: 5,856
* Unique file hashes: 5,824
* Duplicate groups: 30
* Duplicate groups found across dataset splits: 0
* Duplicate groups with conflicting folder labels: 0

**Interpretation:** No byte-identical duplicate groups were detected across the dataset splits. This check does not identify visually similar images stored with different file encodings.

## 4. ResNet50 Test Performance

The following metrics were calculated from the existing test prediction audit CSV.

| Metric               | Result |
| -------------------- | -----: |
| Accuracy             | 86.54% |
| Precision            | 82.97% |
| Recall (Sensitivity) | 98.72% |
| Specificity          | 66.24% |
| F1-score             | 90.16% |
| ROC-AUC              | 97.07% |

### Confusion Matrix

Rows represent actual folder labels; columns represent model predictions.

|                  | Predicted Normal | Predicted Pneumonia |
| ---------------- | ---------------: | ------------------: |
| Actual Normal    |              155 |                  79 |
| Actual Pneumonia |                5 |                 385 |

The model correctly classified 155 Normal images and 385 Pneumonia images. It predicted Pneumonia for 79 images labelled Normal and predicted Normal for 5 images labelled Pneumonia.

## 5. Prediction Disagreement Audit

Across the dataset audit:

| Split      | Images Processed | Matching Predictions | Disagreements |
| ---------- | ---------------: | -------------------: | ------------: |
| Train      |            5,216 |                5,182 |            34 |
| Validation |               16 |                   16 |             0 |
| Test       |              624 |                  540 |            84 |
| **Total**  |        **5,856** |            **5,738** |       **118** |

No image processing errors were reported.

**Important:** A disagreement between the model prediction and a folder label does not establish that the folder label is incorrect. It may represent a model error, an ambiguous image, a label issue, or other factors requiring independent review.

Training-split agreement is not an independent measure of generalization because the model may have seen these images during training. The 16-image original validation folder is also too small to provide a reliable performance estimate.

## 6. Threshold Comparison

The existing test predictions were examined at several decision thresholds.

| Threshold | Accuracy | Precision | Recall | Specificity | False Positives | False Negatives |
| --------- | -------: | --------: | -----: | ----------: | --------------: | --------------: |
| 0.30      |   84.13% |    80.00% | 99.49% |      58.55% |              97 |               2 |
| 0.40      |   85.42% |    81.61% | 98.97% |      62.82% |              87 |               4 |
| 0.50      |   86.54% |    82.97% | 98.72% |      66.24% |              79 |               5 |
| 0.60      |   87.34% |    83.88% | 98.72% |      68.38% |              74 |               5 |
| 0.70      |   87.98% |    84.77% | 98.46% |      70.51% |              69 |               6 |

Increasing the threshold reduced false positives in this comparison but also slightly increased false negatives. These results are exploratory; the final threshold should not be selected solely using the test set.

## 7. Limitations

1. Folder labels were treated as the reference labels; they were not independently confirmed by a radiologist.
2. Prediction disagreements do not prove that dataset labels are incorrect.
3. The original validation folder contains only 16 images.
4. The audit did not detect byte-identical duplicates across splits, but near-duplicate images may remain.
5. Performance on this dataset does not establish clinical safety or suitability for medical diagnosis.
6. Threshold comparisons on the test set should be considered exploratory.
7. Model weights and dataset labels were not modified during the audit.

## 8. Conclusion

The dataset counts, duplicate-image checks, processing status, test metrics, and prediction disagreements were reviewed. The ResNet50 model achieved 86.54% test accuracy, 98.72% recall, and a ROC-AUC of approximately 97.07% on the audited test predictions.

The audit identified 84 test-set prediction disagreements. These should be described as model-label disagreements unless independent evidence establishes a labeling problem.

The results support a comparative academic evaluation of the tested models, but they do not establish clinical readiness.

**Audit status:** Dataset and prediction audit completed. Independent verification of disputed medical labels remains unresolved.
