import streamlit as st
from PIL import Image
from src.inference import predict_image


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🫁 Pneumonia Detection from Chest X-rays")

st.write(
    "Upload a chest X-ray image to obtain a prediction "
    "using the trained ResNet50 deep learning model."
)


# --------------------------------------------------
# Disclaimer
# --------------------------------------------------

st.warning(
    "⚠️ Educational/Research Prototype: "
    "This application is not a medical diagnostic tool "
    "and should not be used for clinical decisions."
)


# --------------------------------------------------
# Upload image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded X-ray")

    st.image(
        image,
        caption="Input Chest X-ray",
        width="stretch"
    )

    # Save temporary uploaded image
    temp_path = "temp_xray.png"

    image.save(temp_path)

    # Predict
    prediction, probability = predict_image(temp_path)

    st.subheader("Prediction Result")

    if prediction == "Pneumonia":

        st.error("Prediction: Pneumonia")

    else:

        st.success("Prediction: Normal")

    st.write(
        f"Pneumonia Probability: **{probability:.2%}**"
    )