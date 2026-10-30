import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.set_page_config(
    page_title="Fruit Detection System",
    page_icon="🍎",
    layout="wide"
)

st.title("🍎 Fruit Detection System")
st.write("Detect apples, bananas, and oranges using YOLOv8n.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload a fruit image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    if st.button("Detect Fruits"):
        with st.spinner("Detecting fruits..."):
            results = model.predict(image, conf=0.25)

            result_image = results[0].plot()

            st.subheader("Detection Results")
            st.image(result_image, channels="BGR", use_container_width=True)

            st.subheader("Detected Fruits")

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    fruit = model.names[class_id]

                    st.write(
                        f"**{fruit.capitalize()}** — "
                        f"Confidence: {confidence:.2%}"
                    )
            else:
                st.warning("No fruits detected.")
