import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd

st.set_page_config(
    page_title="GROUP MEMBERS NAME AND REG NO",
    page_title="MUGISHA EMMANUEL 20RP00370",
    page_title="NDAYIMENYE SAMUEL 26RP02292",
    page_title="Fruit Detection System",
    page_icon="🍎",
    layout="wide"
)

st.title("🍎 Fruit Detection System")
st.write("AI-powered fruit detection using YOLOv8n.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.20,
    step=0.05
)

iou_threshold = st.sidebar.slider(
    "Duplicate Box Reduction (IoU)",
    min_value=0.10,
    max_value=0.90,
    value=0.45,
    step=0.05
)

uploaded_file = st.file_uploader(
    "Upload a fruit image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    if st.button("🔍 Detect Fruits", type="primary"):

        with st.spinner("Analyzing image..."):

            results = model.predict(
                image,
                conf=confidence,
                iou=iou_threshold,
                verbose=False
            )

            result = results[0]
            boxes = result.boxes

            st.subheader("Detection Results")

            annotated_image = result.plot()
            st.image(
                annotated_image,
                channels="BGR",
                use_container_width=True
            )

            fruit_counts = {
                "apple": 0,
                "banana": 0,
                "orange": 0
            }

            detection_data = []

            if boxes is not None and len(boxes) > 0:

                for box in boxes:
                    class_id = int(box.cls[0])
                    fruit = model.names[class_id].lower()
                    score = float(box.conf[0])

                    fruit_counts[fruit] = (
                        fruit_counts.get(fruit, 0) + 1
                    )

                    detection_data.append({
                        "Fruit": fruit.capitalize(),
                        "Confidence": round(score * 100, 2)
                    })

            total = sum(fruit_counts.values())

            st.subheader("📊 Fruit Detection Summary")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Total Fruits", total)
            col2.metric("🍎 Apples", fruit_counts["apple"])
            col3.metric("🍌 Bananas", fruit_counts["banana"])
            col4.metric("🍊 Oranges", fruit_counts["orange"])

            st.subheader("Detection Details")

            if detection_data:

                df = pd.DataFrame(detection_data)

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )

                csv = df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Detection Results (CSV)",
                    data=csv,
                    file_name="fruit_detection_results.csv",
                    mime="text/csv"
                )

                image_to_download = Image.fromarray(
                    annotated_image[:, :, ::-1]
                )

                import io
                image_buffer = io.BytesIO()
                image_to_download.save(image_buffer, format="PNG")

                st.download_button(
                    "Download Annotated Image",
                    data=image_buffer.getvalue(),
                    file_name="fruit_detection_result.png",
                    mime="image/png"
                )

            else:
                st.warning("No fruits detected. Try lowering the confidence threshold.")
