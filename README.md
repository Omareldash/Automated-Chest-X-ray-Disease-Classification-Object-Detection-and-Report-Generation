# Automated Chest X-ray Disease Classification, Object Detection, and Report Generation

## Executive Summary
This project introduces an integrated AI assistant designed to support, rather than replace, radiologists. The system automatically classifies chest diseases, detects focal abnormalities using computer vision, and generates preliminary radiology-style reports using Natural Language Processing (NLP). The primary goal is to speed up triage, reduce turnaround time, and provide explainable diagnostic support.

## Problem Statement & Motivation
* Radiologists currently face significant challenges due to growing workloads and the high volume of chest X-rays. 
* These pressures can lead to delayed diagnoses and an increased risk of interpretation errors.
* Tuberculosis (TB) remains a critical global disease, and fast, automated radiology assistance helps in patient triage.
* Many clinics lack radiologists; having automatic localization and report writing can speed up the diagnosis process.

## System Architecture 
The system follows a multi-stage pipeline methodology that integrates dataset preprocessing, model training, and web-based integration. The pipeline takes raw Chest X-ray images (Frontal and Lateral views) as input and processes them through three parallel model branches.

### 1. Disease Classification
* Determines the presence of disease (Tuberculosis vs. Normal).
* **Architecture**: Uses a DenseNet121 backbone fine-tuned at 300 layers.
* **Performance**: Achieved a validation accuracy of 94.89%.

### 2. Object Detection
* Identifies the coordinates of abnormalities and generates bounding boxes.
* **Architecture**: The winning strategy utilizes a MobileNetV3-Large backbone, which acts as a lightweight and fast baseline model. 
* **Other Tested Models**: ResNet-50 with tuned anchors, DenseNet121 pretrained on medical X-rays, and EfficientNet-B3.
* **Performance**: The MobileNetV3 baseline achieved an F1 Score of 0.7310.

### 3. Medical Report Generation
* Generates textual descriptions based on image features.
* **Architecture**: Employs a hybrid `CNN_GPT2` model.
* **Workflow**: A ResNet-18 CNN encoder extracts visual features from frontal and lateral images, which are then projected into the embedding space of a GPT-2 language decoder.
* **Performance Metrics**: Achieved a BLEU-1 score of 0.4735, a BLEU-2 score of 0.2463, and a ROUGE-1 F-score of 0.5714.

## Datasets
* **TBX11K**: A large-scale dataset containing 11,200 chest X-ray images with bounding boxes marking TB-affected regions. This was utilized to train the object detection models for reliable TB identification.
* **Open-i (Indiana University)**: An open-access chest X-ray collection. This dataset supports medical report generation by pairing the images with structured text reports.

## Installation & Usage
The user interface is built using Streamlit and designed for simplicity and speed. Follow these steps to run the application locally:

1.  **Clone the repository and install dependencies**:
    ```bash
    git clone [https://github.com/Omareldash/Automated-Chest-X-ray-Disease-Classification-Obje](https://github.com/Omareldash/Automated-Chest-X-ray-Disease-Classification-Obje)
    cd Automated-Chest-X-ray...
    pip install -r requirements.txt
    ```
    *(Note: Ensure you have Python installed, and Docker Desktop is optional but recommended)*.

2.  **Launch the Backend (if decoupled)**:
    ```bash
    uvicorn main:app --reload
    ```

3.  **Launch the Streamlit Interface**:
    ```bash
    streamlit run app.py
    ```

4.  **Using the App**:
    * Open your browser to the provided localhost URL (usually `http://localhost:8501`).
    * Upload a chest X-ray image (supported formats: `.jpg`, `.png`).
    * Wait for the processing indicator to finish, then view the classification, bounding boxes, and generated report.

## Team X-perts
This project was developed by:
* Omar Eldash
* Mennat Allah Khalifa
* Mostafa Asharaf
* Abdullah Elsayed
* Seif Allah Mohamed
* Marco Ayman
