import streamlit as st
from PIL import Image
from src.inference import predict_image, generate_gradcam

from src.multi_model_inference import (
    load_all_models,
    predict_all,
    generate_gradcam_all,
)

@st.cache_resource
def get_all_models():
    return load_all_models()
# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="PneumoAI | Pneumonia Research",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# Global styling
# --------------------------------------------------

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .hero {
        padding: 2.5rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #102a43, #176b87);
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: white;
        font-size: 2.5rem;
    }

    .hero p {
        color: #e0f2fe;
        font-size: 1.1rem;
    }

    .info-card {
        padding: 1.2rem;
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 12px;
        min-height: 140px;
    }

    .section-label {
        color: #168aad;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.8rem;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Sidebar navigation
# --------------------------------------------------

st.sidebar.title("🫁 PneumoAI")
st.sidebar.caption("Pneumonia Detection Research Project")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Understanding Pneumonia",
        "Chest X-rays",
        "Project Methodology",
        "Model Comparison",
        "Model Explainability (Grad-CAM)",
        "Try Detection",
        "Limitations"
    ]
)

st.sidebar.divider()
st.sidebar.caption(
    "Educational and research prototype. "
    "Not intended for medical diagnosis."
)


# --------------------------------------------------
# HOME
# --------------------------------------------------

if page == "Home":

    st.markdown("""
    <div class="hero">
        <div class="section-label">Deep Learning Research Project</div>
        <h1>Understanding Pneumonia Through AI</h1>
        <p>
            Exploring chest X-ray classification using convolutional
            neural networks and transfer learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Welcome to PneumoAI")

    st.write(
        "This project investigates how deep learning models can "
        "classify chest X-ray images into Normal and Pneumonia "
        "categories. It also compares different model architectures "
        "and examines model predictions using explainability tools."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🫁 Learn</h3>
            <p>Understand pneumonia, lung anatomy, and chest X-rays.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🧠 Explore AI</h3>
            <p>Learn about preprocessing, Baseline CNN, ResNet50, and MobileNetV2.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-card">
            <h3>📊 Compare</h3>
            <p>Explore evaluation metrics and model comparison results.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.subheader("Our Research Workflow")

    steps = [
        ("01", "Dataset", "Chest X-ray image collection"),
        ("02", "Preprocessing", "Resize and normalize images"),
        ("03", "Model Training", "CNN, ResNet50, and MobileNetV2"),
        ("04", "Evaluation", "Compare classification metrics"),
        ("05", "Explainability", "Visualize influential image regions"),
        ("06", "Prediction", "Compare predictions from all three models")
    ]

    cols = st.columns(3)

    for index, (number, title, description) in enumerate(steps):
        with cols[index % 3]:
            st.markdown(f"**{number} · {title}**")
            st.caption(description)

    st.info(
        "Use the sidebar to explore the educational sections, "
        "research methodology, model comparison, and detection demo."
    )


# --------------------------------------------------
# UNDERSTANDING PNEUMONIA
# --------------------------------------------------


elif page == "Understanding Pneumonia":
    st.title("Understanding Pneumonia")
    st.write(
        "Pneumonia is a lung infection that causes inflammation "
        "in the air sacs of one or both lungs. These air sacs may "
        "fill with fluid or pus, which can make breathing difficult."
    )

    st.divider()
    st.subheader("How Does Pneumonia Affect the Lungs?")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
        """
        <div style="text-align:center; background:#f0f7fa;
                    padding:20px; border-radius:12px;">
          <svg viewBox="0 0 320 240" width="100%"
                     style="max-width:320px;" xmlns="http://www.w3.org/2000/svg">
                  <path d="M155 25 L155 90 M165 25 L165 90"
                        stroke="#64748b" stroke-width="8" fill="none"
                        stroke-linecap="round"/>
                  <path d="M160 85 L125 115 M160 85 L195 115"
                        stroke="#64748b" stroke-width="6" fill="none"
                        stroke-linecap="round"/>
                  <g fill="#8ecae6" stroke="#277da1" stroke-width="2">
                    <circle cx="115" cy="125" r="24"/>
                    <circle cx="145" cy="125" r="24"/>
                    <circle cx="130" cy="153" r="24"/>
                    <circle cx="190" cy="125" r="24"/>
                    <circle cx="220" cy="125" r="24"/>
                    <circle cx="205" cy="153" r="24"/>
                    <circle cx="160" cy="177" r="22"/>
                  </g>
                  <path d="M105 130 Q130 110 150 135 T200 135 T225 130"
                        stroke="#d1495b" stroke-width="3" fill="none"/>
                  <text x="180" y="218" text-anchor="middle"
                        font-size="13" fill="#334155">
                    Lung air sacs (alveoli)
                  </text>
                </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    
    st.caption("Simplified illustration of lung air sacs (alveoli).")

    with col2:
        st.markdown("### What happens?")
        st.write(
            "The infection can cause inflammation and fluid buildup "
            "in the air sacs, reducing the efficiency of oxygen exchange."
        )
        st.markdown("### Common symptoms")
        st.markdown(
            "- Cough, sometimes with mucus\n"
            "- Fever and chills\n"
            "- Shortness of breath\n"
            "- Chest pain while breathing or coughing\n"
            "- Fatigue"
        )

    st.divider()
    st.subheader("Causes and Risk Factors")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Possible causes")
        st.markdown(
            "- Bacterial infections\n"
            "- Viral infections\n"
            "- Certain fungal infections"
        )

    with col2:
        st.markdown("### Higher-risk groups")
        st.markdown(
            "- Young children\n"
            "- Older adults\n"
            "- People with weakened immune systems\n"
            "- People with certain chronic conditions"
        )

    st.info(
        "Important: This website is an educational research project. "
        "Its AI predictions cannot confirm or rule out pneumonia. "
        "Consult a qualified healthcare professional for medical concerns."
    )


# --------------------------------------------------
# CHEST X-RAYS
# --------------------------------------------------

elif page == "Chest X-rays":
    from pathlib import Path

    st.title("Chest X-rays")

    st.write(
        "Chest X-rays are medical images that help healthcare "
        "professionals examine the lungs, heart, and chest. "
        "In this project, X-ray images are analyzed by three "
        "deep learning classification models."
    )

    # --------------------------------------------------
    # Normal vs Pneumonia X-ray Images
    # --------------------------------------------------
    st.divider()
    st.subheader("Normal vs Pneumonia-Affected X-rays")

    from pathlib import Path
    from PIL import Image, ImageOps

    base_dir = Path(__file__).resolve().parent
    xray_dir = base_dir / "results" / "xray_examples"

    normal_path = xray_dir / "normal_example.jpeg"
    pneumonia_path = xray_dir / "pneumonia_example.jpeg"

    def prepare_xray(image_path):
        image = Image.open(image_path).convert("RGB")

        # Fit the image inside a fixed-size canvas without distortion
        image.thumbnail((600, 450), Image.Resampling.LANCZOS)

        # Add black padding so both images have identical dimensions
        canvas = Image.new("RGB", (600, 450), color="black")

        x = (600 - image.width) // 2
        y = (450 - image.height) // 2

        canvas.paste(image, (x, y))
        return canvas

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Normal Chest X-ray")

        if normal_path.exists():
            normal_image = prepare_xray(normal_path)

            st.image(
                normal_image,
                caption="Example of a Normal Chest X-ray",
                use_container_width=True,
            )
        else:
            st.warning("Normal X-ray image not found.")

    with col2:
        st.markdown("### Pneumonia Chest X-ray")

        if pneumonia_path.exists():
            pneumonia_image = prepare_xray(pneumonia_path)

            st.image(
                pneumonia_image,
                caption="Example of a Pneumonia-affected Chest X-ray",
                use_container_width=True,
            )
        else:
            st.warning("Pneumonia X-ray image not found.")

    st.caption(
        "These images are educational examples. "
        "Visual appearance alone cannot confirm a diagnosis."
    )

    # --------------------------------------------------
    # Explanatory Cards
    # --------------------------------------------------
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div style="background:#eaf7ef; padding:20px;
                        border-radius:12px; min-height:200px;">
                <h3 style="color:#176b3a;">Normal X-ray</h3>
                <p style="color:#244b35;">
                    • Lungs generally appear darker due to air.<br><br>
                    • No obvious large areas of abnormal opacity.<br><br>
                    • Overall lung appearance may be relatively clear.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div style="background:#fff0e9; padding:20px;
                        border-radius:12px; min-height:200px;">
                <h3 style="color:#a23b22;">
                    Pneumonia-Affected X-ray
                </h3>
                <p style="color:#693b30;">
                    • Some areas may appear whiter or more opaque.<br><br>
                    • Inflammation or fluid may affect lung appearance.<br><br>
                    • Patterns can vary depending on the cause and severity.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------
    # What Does the Model Learn?
    # --------------------------------------------------
    st.divider()
    st.subheader("What Does the Model Learn?")

    st.write(
        "Each model learns visual patterns from labelled training "
        "images and produces a score for the Pneumonia class. "
        "The models can focus on different patterns and may disagree. "
        "They do not interpret an X-ray in the same way a "
        "radiologist does."
    )

    # --------------------------------------------------
    # Medical Disclaimer
    # --------------------------------------------------
    st.info(
        "Important: These descriptions are general educational "
        "observations. A chest X-ray must be interpreted in its "
        "clinical context by a qualified healthcare professional. "
        "The model's prediction is not a medical diagnosis."
    )

# --------------------------------------------------
# PROJECT METHODOLOGY
# --------------------------------------------------


elif page == "Project Methodology":
    st.title("Project Methodology")

    st.write(
        "This project compares three deep learning models for "
        "classifying chest X-ray images into Normal and Pneumonia "
        "categories."
    )

    st.divider()
    st.subheader("Project Objectives")

    st.markdown(
        """
        - Develop a baseline Convolutional Neural Network (CNN).
        - Apply transfer learning using ResNet50 and MobileNetV2.
        - Compare model performance using multiple evaluation metrics.
        - Build an interactive web application for demonstration.
        - Examine model errors and visualize selected predictions.
        """
    )

    st.divider()
    st.subheader("Research Workflow")

    steps = [
        (
            "01",
            "Dataset Collection",
            "Use the Chest X-Ray Images (Pneumonia) dataset "
            "containing Normal and Pneumonia images."
        ),
        (
            "02",
            "Data Preparation",
            "Separate training, validation, and test data. "
            "Resize images to 224 × 224 pixels."
        ),
        (
            "03",
            "Image Preprocessing",
            "Convert images to tensors. Normalize RGB images "
            "using ImageNet mean and standard deviation for "
            "the transfer-learning models."
        ),
        (
            "04",
            "Model Training",
            "Train a baseline CNN and fine-tune pretrained ResNet50 "
            "and MobileNetV2 models for binary classification."
        ),
        (
            "05",
            "Model Evaluation",
            "Compare accuracy, precision, recall, F1-score, "
            "specificity, ROC-AUC, and confusion matrices."
        ),
        (
            "06",
            "Web Application",
            "Load the three saved checkpoints and allow users to "
            "compare predictions and Grad-CAM overlays for an X-ray."
        ),
    ]

    for number, heading, description in steps:
        with st.container(border=True):
            st.markdown(f"### {number}. {heading}")
            st.write(description)

    st.divider()
    st.subheader("Dataset Split")

    col1, col2, col3 = st.columns(3)

    col1.metric("Training Images", "4,172")
    col2.metric("Validation Images", "1,044")
    col3.metric("Test Images", "624")

    st.caption(
        "The training data was split into training and validation "
        "subsets using a stratified 80/20 split with random_state=42. "
        "The original test set was kept separate for final evaluation."
    )

    st.divider()
    st.subheader("Image Preprocessing")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Baseline CNN")
        st.markdown(
            """
            - Grayscale input images
            - Resize to 224 × 224
            - Convert to tensors
            """
        )

    with col2:
        st.markdown("### ResNet50 and MobileNetV2")
        st.markdown(
            """
            - Convert images to RGB
            - Resize to 224 × 224
            - Convert to tensors
            - Normalize using ImageNet statistics
            """
        )

    st.info(
        "Research note: Model performance depends on the dataset, "
        "preprocessing, training setup, and evaluation procedure. "
        "Results from this dataset alone do not establish clinical "
        "reliability."
    )


# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------


elif page == "Model Comparison":
    import pandas as pd

    st.title("Model Comparison")
    st.write(
        "This section compares the three trained models using "
        "their performance on the held-out test dataset."
    )

    # Test-set results
    metrics = pd.DataFrame(
        {
            "Baseline CNN": {
                "Accuracy": 78.04,
                "Precision": 74.56,
                "Recall": 98.46,
                "F1-Score": 84.86,
                "Specificity": 44.02,
                "ROC-AUC": 92.98,
            },
            "ResNet50": {
                "Accuracy": 86.54,
                "Precision": 82.97,
                "Recall": 98.72,
                "F1-Score": 90.16,
                "Specificity": 66.24,
                "ROC-AUC": 97.03,
            },
            "MobileNetV2": {
                "Accuracy": 83.65,
                "Precision": 79.51,
                "Recall": 99.49,
                "F1-Score": 88.38,
                "Specificity": 57.26,
                "ROC-AUC": 96.69,
            },
        }
    )

    st.divider()
    st.subheader("Test-Set Performance Metrics")

    st.dataframe(
        metrics.style.format("{:.2f}%"),
        use_container_width=True,
    )

    st.caption(
        "Results are from the 624-image test set. "
        "Higher values indicate better performance for each metric."
    )

    st.divider()
    st.subheader("Compare Model Performance")

    selected_metrics = st.multiselect(
        "Select metrics to display",
        options=[
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "Specificity",
            "ROC-AUC",

        ],
        default=["Accuracy", "Recall", "F1-Score", "ROC-AUC"],
    )


    if selected_metrics:
        import altair as alt

        chart_data = (
        metrics.loc[selected_metrics]
        .T
        .rename_axis("Model")
        .reset_index()
        .melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Score",
        )
        )

        chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
        x=alt.X(
            "Model:N",
            title="Model",
            axis=alt.Axis(labelAngle=0),
        ),
        y=alt.Y(
            "Score:Q",
            title="Score (%)",
            scale=alt.Scale(domain=[40, 100]),
        ),
        xOffset=alt.XOffset("Metric:N"),
        color=alt.Color("Metric:N", title="Metric"),
        tooltip=[
            alt.Tooltip("Model:N"),
            alt.Tooltip("Metric:N"),
            alt.Tooltip("Score:Q", format=".2f"),
        ],
        )
        .properties(height=420)
        )

        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Select at least one metric to display the chart.")

    st.divider()
    st.subheader("ROC Curve Comparison")

    from pathlib import Path

    roc_curve_path = (
        Path(__file__).resolve().parent
        / "results"
        / "roc_curve_comparison.png"
    )

    if roc_curve_path.exists():
        st.image(
            str(roc_curve_path),
            caption="ROC curves for Baseline CNN, ResNet50, and MobileNetV2",
            use_container_width=True,
        )
    else:
        st.warning(
            "ROC curve image not found. Check that "
            "results/roc_curve_comparison.png exists."
        )

    st.write(
        "The ROC curve illustrates the trade-off between the "
        "true positive rate and false positive rate at different "
        "classification thresholds. A higher ROC-AUC indicates "
        "better discrimination between the two classes on the "
        "evaluated dataset."
    )
    st.subheader("Confusion Matrices")

    st.write(
        "Rows represent the actual class, and columns represent "
        "the predicted class."
    )

    matrices = {
        "Baseline CNN": [[103, 131], [6, 384]],
        "ResNet50": [[155, 79], [5, 385]],
        "MobileNetV2": [[134, 100], [2, 388]],
    }

    for model_name, matrix in matrices.items():
        st.markdown(f"### {model_name}")

        confusion_df = pd.DataFrame(
            matrix,
            index=["Actual Normal", "Actual Pneumonia"],
            columns=["Predicted Normal", "Predicted Pneumonia"],
        )

        st.dataframe(
            confusion_df,
            use_container_width=True,
        )

    st.caption(
        "False positives: Normal images classified as Pneumonia. "
        "False negatives: Pneumonia images classified as Normal."
    )

    st.divider()
    st.subheader("Interpreting the Comparison")

    st.write(
        "In the recorded test-set results, ResNet50 has the highest "
        "accuracy, precision, F1-score, specificity, and ROC-AUC. "
        "MobileNetV2 has the highest recall, while the Baseline CNN "
        "provides a reference point for comparison with transfer learning."
    )

    st.write(
        "No single metric describes every aspect of model performance. "
        "For example, recall measures how many Pneumonia cases are "
        "detected, while specificity measures how many Normal images "
        "are correctly identified. The Try Detection page now lets you "
        "compare predictions and Grad-CAM overlays from all three models."
    )

    st.info(
        "These results describe performance on this particular test set. "
        "They do not establish clinical reliability; independent external "
        "validation would be required before any clinical use."
    )


# --------------------------------------------------
# TRY DETECTION
# --------------------------------------------------

elif page == "Try Detection":
    from PIL import Image
    from io import BytesIO

    st.title("Try Pneumonia Detection")
    st.write(
        "Upload a chest X-ray image to compare predictions "
        "from Baseline CNN, ResNet50, and MobileNetV2."
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image",
        type=["png", "jpg", "jpeg"],
        help="Supported formats: PNG, JPG, and JPEG.",
        key="xray_uploader",
    )

    # Save a newly uploaded image in session state
    if uploaded_file is not None:
        new_image_bytes = uploaded_file.getvalue()

        if (
            st.session_state.get("detection_image_bytes")
            != new_image_bytes
        ):
            st.session_state["detection_image_bytes"] = new_image_bytes
            st.session_state["detection_filename"] = uploaded_file.name

            # Clear results belonging to the previous image
            st.session_state.pop("detection_result", None)
            st.session_state.pop("multi_model_results", None)
            st.session_state.pop("gradcam_images", None)
            st.session_state.pop("gradcam_image", None)

    # Restore the image after navigating between pages
    saved_image_bytes = st.session_state.get("detection_image_bytes")

    if saved_image_bytes is not None:
        image = Image.open(BytesIO(saved_image_bytes)).convert("RGB")
        filename = st.session_state.get(
            "detection_filename", "Uploaded X-ray"
        )

        col1, col2 = st.columns([1.1, 1])

        with col1:
            st.subheader("Uploaded X-ray")
            st.image(
                image,
                caption=filename,
                use_container_width=True,
            )

        with col2:
            st.subheader("Analysis")
            st.write(
                "The same X-ray will be analyzed by all three "
                "saved models."
            )

            if st.button(
                "Analyze X-ray",
                type="primary",
                use_container_width=True,
            ):
                try:
                    with st.spinner(
                        "Loading models and analyzing X-ray..."
                    ):
                        loaded_models = get_all_models()

                        # Predictions from all three models
                        results = predict_all(image, loaded_models)

                        # Preserve ResNet50's existing Grad-CAM
                        prediction, probability = predict_image(image)

                        # Generate Grad-CAM overlays for all three models
                        gradcam_images = generate_gradcam_all(image, loaded_models)

                        st.session_state["multi_model_results"] = {
                            "filename": filename,
                            "results": results,
                        }

                        st.session_state["detection_result"] = {
                            "filename": filename,
                            "prediction": prediction,
                            "probability": probability,
                        }

                        st.session_state["gradcam_images"] = gradcam_images

                except Exception as e:
                    st.error(f"Could not analyze the X-ray: {e}")

        # --------------------------------------------------
        # DISPLAY ALL THREE MODEL PREDICTIONS
        # --------------------------------------------------

        saved_results = st.session_state.get("multi_model_results")

        if saved_results and saved_results["filename"] == filename:
            st.divider()
            st.subheader("Model Comparison")

            st.caption(
                "All three models analyzed the same uploaded X-ray."
            )

            results = saved_results["results"]
            model_names = ["Baseline CNN", "ResNet50", "MobileNetV2"]

            result_columns = st.columns(3)

            for column, model_name in zip(result_columns, model_names):
                model_result = results[model_name]
                prediction = model_result["prediction"]
                probability = float(model_result["probability"])

                with column:
                    st.markdown(f"### {model_name}")

                    if prediction == "Pneumonia":
                        st.warning(f"Prediction: {prediction}")
                    else:
                        st.success(f"Prediction: {prediction}")

                    st.metric(
                        "Pneumonia score",
                        f"{probability * 100:.2f}%",
                    )

                    st.progress(
                        min(max(probability, 0.0), 1.0)
                    )

            st.caption(
                "Scores represent each model's sigmoid output for "
                "the Pneumonia class. They are not necessarily "
                "calibrated probabilities of disease."
            )

            # --------------------------------------------------
            # GRAD-CAM FOR ALL THREE MODELS
            # --------------------------------------------------

            gradcam_images = st.session_state.get("gradcam_images")

            if gradcam_images:
                st.divider()
                st.subheader("Model Explainability — Grad-CAM")

                st.caption(
                    "Each heatmap highlights image regions that influenced "
                    "the respective model output. These are not infection "
                    "segmentation masks or proof of disease location."
                )

                gradcam_columns = st.columns(3)
                model_names = ["Baseline CNN", "ResNet50", "MobileNetV2"]

                for column, model_name in zip(gradcam_columns, model_names):
                    with column:
                        st.markdown(f"**{model_name}**")
                        st.image(
                            gradcam_images[model_name],
                            caption=f"{model_name} Grad-CAM Overlay",
                            use_container_width=True,
                        )

            st.info(
                "Research demonstration only. These predictions "
                "cannot confirm or rule out pneumonia. Do not use "
                "them to make treatment decisions. Consult a "
                "qualified healthcare professional for medical concerns."
            )

    else:
        st.info(
            "Upload a chest X-ray to begin. The analysis button "
            "will appear after you select an image."
        )
# --------------------------------------------------
# LIMITATIONS
# --------------------------------------------------

elif page == "Limitations":

    st.title("Limitations and Responsible Use")

    st.write(
        "This project is an educational deep learning experiment. "
        "All three models were evaluated on a specific dataset, and "
        "their results may not generalize to all hospitals, equipment, "
        "patient populations, or imaging conditions."
    )

    st.write(
        "Any model prediction can be incorrect, and the three models "
        "may disagree on the same X-ray. A Normal prediction does not "
        "rule out pneumonia, and a Pneumonia prediction does not "
        "confirm the disease. Do not treat majority agreement as a diagnosis."
    )

    st.write(
        "Grad-CAM visualizations can help explore which image regions "
        "influenced each model prediction, but they do not establish "
        "that a model identified a medically meaningful feature."
    )

    st.warning(
        "Do not use this application for diagnosis, treatment, "
        "or emergency decisions. Seek professional medical advice."
    )


elif page == "Model Explainability (Grad-CAM)":
    from pathlib import Path

    st.title("Model Explainability: Grad-CAM")

    st.write(
        "Grad-CAM (Gradient-weighted Class Activation Mapping) "
        "is a visualization technique that highlights image regions "
        "that influenced a model's output for a selected class."
    )

    st.info(
        "For a newly uploaded X-ray, open Try Detection to view "
        "predictions and Grad-CAM overlays from Baseline CNN, ResNet50, "
        "and MobileNetV2. The saved error examples below are ResNet50 "
        "case studies."
    )

    st.divider()
    st.subheader("How to Read a Grad-CAM Visualization")

    st.markdown(
        """
        - **Original image:** The input chest X-ray.
        - **Heatmap overlay:** Highlights regions that contributed
          to the model's output for the selected class.
        - **Interpretation:** These highlighted regions can help
          investigate model behavior and classification errors.
        """
    )

    # Build paths relative to the project folder
    base_dir = Path(__file__).resolve().parent
    gradcam_dir = base_dir / "results" / "gradcam"

    false_positive_path = (
        gradcam_dir / "resnet50_false_positive_gradcam.png"
    )
    false_negative_path = (
        gradcam_dir / "resnet50_false_negative_gradcam.png"
    )

    st.divider()
    st.subheader("Case 1: False Positive")

    st.write(
        "The actual image belongs to the Normal class, but "
        "ResNet50 classified it as Pneumonia. The model's "
        "Pneumonia score for this example was approximately 74.52%."
    )

    if false_positive_path.exists():
        st.image(
            str(false_positive_path),
            caption="ResNet50 false-positive Grad-CAM visualization",
            use_container_width=True,
        )
    else:
        st.error(
            "False-positive image not found. Check the file in "
            "results/gradcam/resnet50_false_positive_gradcam.png."
        )

    st.divider()
    st.subheader("Case 2: False Negative")

    st.write(
        "The actual image belongs to the Pneumonia class, but "
        "ResNet50 classified it as Normal. The model's "
        "Pneumonia score for this example was approximately 38.02%."
    )

    if false_negative_path.exists():
        st.image(
            str(false_negative_path),
            caption="ResNet50 false-negative Grad-CAM visualization",
            use_container_width=True,
        )
    else:
        st.error(
            "False-negative image not found. Check the file in "
            "results/gradcam/resnet50_false_negative_gradcam.png."
        )

    st.divider()
    st.subheader("What Can We Learn From These Examples?")

    st.markdown(
        """
        - A false positive occurs when a Normal image is classified
          as Pneumonia.
        - A false negative occurs when a Pneumonia image is
          classified as Normal.
        - These examples help investigate mistakes made by the model.
        - Grad-CAM highlights influential image regions, but does
          not prove that the model is focusing on medically correct
          features or explain the model's decision completely.
        """
    )

    st.info(
        "These visualizations are research aids, not medical evidence. "
        "The model and Grad-CAM outputs must not be used to diagnose "
        "or rule out pneumonia."
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "PneumoAI · Deep Learning-Based Pneumonia Detection Research "
    "· Educational prototype, not a clinical diagnostic tool"
)

    