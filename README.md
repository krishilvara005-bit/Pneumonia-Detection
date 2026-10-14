# PneumoAI --- Pneumonia Detection from Chest X-rays

PneumoAI is an academic deep-learning project that compares three
image-classification models for identifying patterns associated with
pneumonia in chest X-ray images:

-   **Baseline CNN**
-   **ResNet50**
-   **MobileNetV2**

The Streamlit application lets a user upload one chest X-ray, view each
model's prediction and pneumonia score, and inspect a Grad-CAM
visualization for each model.

> **Important:** PneumoAI is an educational research prototype, not a
> clinically validated medical device. Its predictions and
> visualizations must not be used to diagnose, rule out, or treat
> pneumonia.

## Features

-   Upload a chest X-ray image (`.png`, `.jpg`, or `.jpeg`).
-   View predictions from Baseline CNN, ResNet50, and MobileNetV2 for
    the same image.
-   Compare each model's sigmoid output for the Pneumonia class.
-   View a Grad-CAM overlay generated for each model.
-   Explore project information, methodology, model comparison, and
    explainability through the Streamlit navigation pages.

## Models

  -----------------------------------------------------------------------
  Model                   Description             Input preprocessing
  ----------------------- ----------------------- -----------------------
  Baseline CNN            Custom convolutional    Grayscale, resized to
                          neural network trained  224 × 224, converted to
                          for binary              a tensor
                          classification          

  ResNet50                ResNet50 with a         RGB, resized to 224 ×
                          one-output              224, ImageNet
                          classification layer,   normalization
                          fine-tuned for the task 

  MobileNetV2             MobileNetV2 with a      RGB, resized to 224 ×
                          one-output              224, ImageNet
                          classification layer,   normalization
                          fine-tuned for the task 
  -----------------------------------------------------------------------

The app uses a threshold of **0.5** for its displayed class prediction:
scores at or above 0.5 are labelled `Pneumonia`; scores below 0.5 are
labelled `Normal`.

The displayed score is the model's sigmoid output for the Pneumonia
class. It is **not necessarily a calibrated probability of disease**.

## Grad-CAM Explainability

Grad-CAM creates a heatmap showing image regions that influenced a
model's output. PneumoAI displays an overlay for each of the three
models.

Grad-CAM is not an infection-segmentation mask. A highlighted region
does not confirm that pneumonia is present there, and the heatmap does
not establish the exact location or extent of an infection. Different
models may highlight different regions and may disagree on the same
image.

## Dataset

The project uses the **Chest X-Ray Images (Pneumonia)** dataset from
Kaggle:

-   Dataset: [Chest X-Ray Images
    (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
-   Classes: `NORMAL` and `PNEUMONIA`
-   The project uses training, validation, and test image folders.

The recorded final split used in the project was:

  Split          Number of images
  ------------ ------------------
  Training                  4,172
  Validation                1,044
  Test                        624
  **Total**             **5,840**

Dataset copies and notebook runs can differ. Use the split and metrics
produced by the specific experiment you are reporting.

## Project Structure

``` text
Pneumonia-Detection/
├── models/
│   ├── baseline_cnn.pth
│   ├── resnet50_pneumonia.pth
│   └── mobilenetv2_pneumonia.pth
├── src/
│   ├── inference.py
│   └── multi_model_inference.py
├── results/
│   ├── roc_curve_comparison.png
│   ├── dataset_audit/
│   └── ...
├── app.py
├── requirements.txt
└── README.md
```

Your local folder may contain additional notebooks, audit files, images,
or result folders.

## Requirements

-   Python 3.10 or a compatible Python version supported by your
    installed PyTorch and Streamlit packages
-   PyTorch
-   torchvision
-   Streamlit
-   Pillow
-   NumPy
-   OpenCV
-   pandas and scikit-learn if running the evaluation or audit
    notebooks/scripts

Install the packages listed in `requirements.txt` from the project root.

``` bash
pip install -r requirements.txt
```

If installation fails, check that the PyTorch and torchvision versions
are compatible with each other and with your Python version.

## Run the Application

1.  Clone or download this repository.
2.  Confirm that all three trained model checkpoints are present in the
    `models/` directory.
3.  Create and activate a virtual environment if desired.
4.  Install dependencies.
5.  From the project root, run:

``` bash
streamlit run app.py
```

6.  Open the local URL printed by Streamlit.
7.  Navigate to **Try Detection**, upload a chest X-ray, and click
    **Analyze X-ray**.

The app loads the saved checkpoints; it does not retrain the models
during prediction. The multi-model inference module must be present at
`src/multi_model_inference.py` and should match the model architectures
and preprocessing used when the checkpoints were trained.

## Model Comparison

The project includes a Model Comparison page for presenting recorded
evaluation metrics for the three models. Report metrics from the same
test split and experiment when comparing models.

Common evaluation metrics include:

-   **Accuracy:** fraction of all predictions that are correct.
-   **Precision:** fraction of predicted pneumonia cases that are
    actually positive according to the dataset labels.
-   **Recall (Sensitivity):** fraction of pneumonia-labelled images
    correctly identified.
-   **F1-score:** harmonic mean of precision and recall.
-   **Specificity:** fraction of normal-labelled images correctly
    identified.
-   **ROC-AUC:** measures how well the model scores separate the two
    labelled classes across thresholds.

Results on this dataset do not establish real-world clinical
performance. Performance can change with data sources, image quality,
patient populations, and dataset bias.

## Limitations

-   This is an academic prototype, not a diagnostic system.
-   Model outputs can disagree for the same image.
-   Sigmoid outputs are not necessarily calibrated disease
    probabilities.
-   Grad-CAM highlights influential image regions but does not prove
    disease location or explain the model completely.
-   Dataset labels and distribution may contain limitations or bias.
-   Performance on a held-out dataset does not guarantee performance in
    hospitals or on images from other sources.

## Responsible Use

Use this project for educational demonstrations and research only. Do
not use its output to make medical decisions, delay care, or replace
professional assessment. Consult a qualified healthcare professional for
medical concerns.

## Acknowledgements

-   Dataset: [Chest X-Ray Images (Pneumonia),
    Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
-   PyTorch and torchvision for model implementation.
-   Streamlit for the interactive web application.
