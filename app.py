import streamlit as st
from PIL import Image
import numpy as np
import torch
import cv2

from utils.report_generator import ReportGenerator
from utils.object_detector import TBObjectDetector
from utils.classifier_model import TBClassifier  

# ===========================
# Load CSS
# ===========================
def load_css():
    try:
        with open("styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        pass

load_css()

# ===========================
# SIDEBAR MENU
# ===========================
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Choose a task",
    ["TB Classification & Detection", "Report Generation"]
)



# ===========================
# MODEL LOADERS
# ===========================
@st.cache_resource
def load_classifier():
    return TBClassifier("models/xray_classifier_pytorch.pth")

@st.cache_resource
def load_detector():
    return TBObjectDetector("models/detection_model.pth", device="cpu")

@st.cache_resource
def load_report_model():
    return ReportGenerator("models/report_generator.pt", device="cpu")

classifier = load_classifier()
detector = load_detector()
report_model = load_report_model()

# ======================================================
# PAGE 1: CLASSIFICATION + DETECTION (ONE IMAGE)
# ======================================================
if app_mode == "TB Classification & Detection":
    st.title("🩺 TB Classification & Lesion Detection")
    st.write("Upload *one* chest X-ray to run our AI classifier and object detector.")

    # Image upload
    uploaded_img = st.file_uploader("Upload Chest X-ray", type=["jpg", "jpeg", "png"])

    if uploaded_img:
        img_display = Image.open(uploaded_img).convert("RGB")
        st.image(img_display, width=350)

    analyze = st.button("🔬 Analyze X-ray", use_container_width=True)

    if analyze:
        if not uploaded_img:
            st.error("❌ Please upload an X-ray image.")
        else:
            st.success("Processing X-ray...")

            # -----------------------------
            # 1️⃣ TB CLASSIFICATION
            # -----------------------------
            with st.spinner("Running TB classification..."):
                tb_prob, cls = classifier.predict(img_display)

            # -----------------------------
            # 2️⃣ TB OBJECT DETECTION
            # -----------------------------
            with st.spinner("Detecting possible TB lesions..."):
                boxes = detector.predict(img_display)

            img_cv = np.array(img_display)

            for (x1, y1, x2, y2, score) in boxes:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img_cv, f"TB {score:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 0, 0),
                            2)

            # -----------------------------
            # DISPLAY RESULTS
            # -----------------------------
            st.markdown("---")
            st.header("📊 Results")

            st.subheader("🩻 TB Classification")
            st.metric("Probability of TB", f"{tb_prob:.2%}")

            st.subheader("🎯 Detected TB Lesions")
            st.image(img_cv, width=500)

# ======================================================
# PAGE 2: REPORT GENERATION (TWO IMAGES)
# ======================================================
elif app_mode == "Report Generation":
    st.title("📝 AI Radiology Report")
    st.write("Upload *both* frontal and lateral X-rays to generate a full AI-assisted report.")

    col1, col2 = st.columns(2)

    with col1:
        front_img = st.file_uploader("Upload *Frontal* View", type=["jpg", "jpeg", "png"])

        if front_img:
            front_display = Image.open(front_img).convert("RGB")
            st.image(front_display, width=300)

    with col2:
        lateral_img = st.file_uploader("Upload *Lateral* View", type=["jpg", "jpeg", "png"])

        if lateral_img:
            lateral_display = Image.open(lateral_img).convert("RGB")
            st.image(lateral_display, width=300)

    generate_report = st.button("📝 Generate Report", use_container_width=True)

    if generate_report:
        if not front_img or not lateral_img:
            st.error("❌ Please upload BOTH frontal and lateral X-rays.")
        else:
            st.success("Generating report...")

            with st.spinner("AI is analyzing both images..."):
                report_text = report_model.generate(front_display, lateral_display)

            st.markdown("---")
            st.header("📄 Generated Radiology Report")
            st.write(report_text)

