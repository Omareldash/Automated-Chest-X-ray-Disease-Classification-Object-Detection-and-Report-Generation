import streamlit as st
from PIL import Image
import numpy as np
import torch

from utils.report_generator import ReportGenerator
from utils.object_detector import TBObjectDetector
from utils.classifier_model import TBClassifier  
import cv2

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
# Page Title
# ===========================
st.title("🩺 Tuberculosis AI Screening System")
st.write("Upload two chest X-ray views and get predictions from our 3 AI models.")

st.markdown("---")

# ---------------------------------------------------------
# PATIENT INFORMATION
# ---------------------------------------------------------
with st.container():
    st.header("👤 Patient Information")
    col1, col2 = st.columns(2)

    with col1:
        patient_name = st.text_input("Full Name")
        age = st.number_input("Age", 1, 120)
        gender = st.selectbox("Gender", ["Male", "Female"])

    with col2:
        patient_id = st.text_input("Patient ID")
        symptoms = st.text_area("Symptoms (optional)")

st.markdown("---")

# ---------------------------------------------------------
# X-RAY UPLOAD
# ---------------------------------------------------------
st.header("🩻 Upload X-ray Images")

colX1, colX2 = st.columns(2)

with colX1:
    front_img = st.file_uploader("Upload *FRONTAL* X-ray", type=["jpg", "jpeg", "png"])
    if front_img:
        front_img_display = Image.open(front_img).convert("RGB")
        st.image(front_img_display, width=300)

with colX2:
    side_img = st.file_uploader("Upload *LATERAL* X-ray", type=["jpg", "jpeg", "png"])
    if side_img:
        side_img_display = Image.open(side_img).convert("RGB")
        st.image(side_img_display, width=300)

st.markdown("---")

# ---------------------------------------------------------
# ANALYZE BUTTON
# ---------------------------------------------------------
analyze = st.button("🔬 Analyze X-rays", use_container_width=True)

# ---------------------------------------------------------
# MODEL LOADERS
# ---------------------------------------------------------

@st.cache_resource
def load_classifier():
    return TBClassifier(r"models\xray_classifier_pytorch.pth")

@st.cache_resource
def load_detector():
    return TBObjectDetector(r"models/detection_model.pth", device="cpu")

@st.cache_resource
def load_report_model():
    return ReportGenerator(r"models/report_generator.pth", device="cpu")

classifier = load_classifier()
detector = load_detector()
report_model = load_report_model()

# ---------------------------------------------------------
# RUN MODELS
# ---------------------------------------------------------

if analyze:
    if not front_img or not side_img:
        st.error("❌ Please upload BOTH X-rays.")
    else:
        st.success("Processing X-rays...")

        # ==============================
        # 1️⃣ TB CLASSIFICATION
        # ==============================
        with st.spinner("Running TB classification..."):
            tb_prob, cls = classifier.predict(front_img_display)

        # ==============================
        # 2️⃣ OBJECT DETECTION
        # ==============================
        with st.spinner("Detecting TB lesions..."):
            boxes = detector.predict(front_img_display)

        # Draw bounding boxes
        img_cv = np.array(front_img_display)
        for (x1, y1, x2, y2, score) in boxes:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_cv, f"TB {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


        # ==============================
        # 3️⃣ REPORT GENERATION
        # ==============================
        with st.spinner("Generating radiology report..."):
            report_text = report_model.generate(front_img_display, side_img_display)

        # ---------------------------------------------------------
        # DISPLAY RESULTS
        # ---------------------------------------------------------

        st.markdown("---")
        st.header("📊 Results")

        # Classification
        st.subheader("🩻 TB Classification")
        st.metric("Probability of TB", f"{tb_prob:.2%}")

        st.markdown("---")

        # Detection
        st.subheader("🎯 Detected TB Lesions")
        st.image(img_cv, width=500)

        st.markdown("---")

        # Report
        st.subheader("📝 AI-Generated Radiology Report")
        st.write(report_text)

        st.info(f"Analysis complete for **{patient_name}** (ID: {patient_id}).")
