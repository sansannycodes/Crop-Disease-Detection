# 🌿 Crop-Disease-Detection

A deep learning project for identifying plant diseases from leaf images using Convolutional Neural Networks (CNN), OpenCV, and TensorFlow/Keras. Structured with clean `backend/` and `frontend/` components, interactive Streamlit web interface, CLI inference capabilities, and automated unit testing.

[![GitHub Repository](https://img.shields.io/badge/GitHub-Crop--Disease--Detection-blue?logo=github)](https://github.com/sansannycodes/Crop-Disease-Detection)

---

## 📌 Problem Statement

Plant diseases pose a severe threat to global food security and cause substantial yield loss in agriculture. Traditional methods of plant disease diagnosis rely on manual inspection by agricultural experts, which is labor-intensive, time-consuming, expensive, and susceptible to human error.

Automated image-based plant disease detection using deep learning enables early diagnosis, helping farmers and agronomists take timely, targeted corrective measures to prevent crop losses.

---

## 🎯 Project Objective

To build an accurate, end-to-end Computer Vision system that processes leaf images, extracts visual features using a multi-layer Convolutional Neural Network (CNN), and classifies the plant health status across **38 distinct crop disease categories**.

---

## ✨ Features

- **Modular Backend Architecture**: Decoupled backend modules (`backend/config.py`, `backend/dataset.py`, `backend/model.py`, `backend/train.py`, `backend/predict.py`).
- **Interactive Frontend UI**: Streamlit web application (`frontend/app.py`) for drag-and-drop leaf image upload, prediction breakdown, and top-3 probability visualization.
- **Automated Preprocessing**: OpenCV/PIL image reading, color space conversion (BGR to RGB), spatial resizing ($224 \times 224$), and pixel normalization $[0.0, 1.0]$.
- **Data Augmentation**: Real-time augmentation (random rotations, width/height shifts, shear, zoom, and horizontal flips) using Keras `ImageDataGenerator`.
- **Command Line Inference**: CLI tool (`backend/predict.py`) for quick single-image diagnosis.
- **Automated Testing**: Unit test suite (`tests/test_pipeline.py`) verifying model compilation, input tensor shapes, and preprocessing routines via `pytest` or `unittest`.
- **Portable & Reproducible**: Fully relative paths without system-specific hardcoded dependencies.

---

## 📊 Dataset Description

The system is trained and evaluated on the **PlantVillage / New Plant Diseases Dataset**, containing high-resolution leaf photographs across 14 crop species.

- **Total Classes**: 38 plant health and disease categories
- **Crops Included**: Apple, Blueberry, Cherry, Corn (Maize), Grape, Orange (Citrus greening), Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.
- **Image Specifications**: RGB format, preprocessed to $224 \times 224 \times 3$ resolution.

### 🏷️ 38 Confirmed Crop Disease Classes

1. `Apple___Apple_scab`
2. `Apple___Black_rot`
3. `Apple___Cedar_apple_rust`
4. `Apple___healthy`
5. `Blueberry___healthy`
6. `Cherry_(including_sour)___Powdery_mildew`
7. `Cherry_(including_sour)___healthy`
8. `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot`
9. `Corn_(maize)___Common_rust_`
10. `Corn_(maize)___Northern_Leaf_Blight`
11. `Corn_(maize)___healthy`
12. `Grape___Black_rot`
13. `Grape___Esca_(Black_Measles)`
14. `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)`
15. `Grape___healthy`
16. `Orange___Haunglongbing_(Citrus_greening)`
17. `Peach___Bacterial_spot`
18. `Peach___healthy`
19. `Pepper,_bell___Bacterial_spot`
20. `Pepper,_bell___healthy`
21. `Potato___Early_blight`
22. `Potato___Late_blight`
23. `Potato___healthy`
24. `Raspberry___healthy`
25. `Soybean___healthy`
26. `Squash___Powdery_mildew`
27. `Strawberry___Leaf_scorch`
28. `Strawberry___healthy`
29. `Tomato___Bacterial_spot`
30. `Tomato___Early_blight`
31. `Tomato___Late_blight`
32. `Tomato___Leaf_Mold`
33. `Tomato___Septoria_leaf_spot`
34. `Tomato___Spider_mites Two-spotted_spider_mite`
35. `Tomato___Target_Spot`
36. `Tomato___Tomato_Yellow_Leaf_Curl_Virus`
37. `Tomato___Tomato_mosaic_virus`
38. `Tomato___healthy`

---

## 📁 Project Structure

```text
Crop-Disease-Detection/
├── .gitignore               # Git exclusion rules (venv, cache, models, datasets)
├── README.md                # Comprehensive project documentation
├── requirements.txt         # Project dependencies manifest
├── backend/                 # Backend Processing & Model Pipeline
│   ├── __init__.py          # Package initializer
│   ├── config.py            # Hyperparameters, class labels, and relative paths
│   ├── dataset.py           # Preprocessing & data generator functions
│   ├── model.py             # Keras CNN architecture definition
│   ├── train.py             # Model training execution script
│   └── predict.py           # CLI inference script
├── frontend/                # User Interface Components
│   └── app.py               # Streamlit web application
├── data/
│   └── README.md            # Dataset placement & download instructions
├── models/
│   └── README.md            # Model file storage & weights instructions
├── notebooks/
│   └── leaf_preprocessing_data.ipynb  # Cleaned Jupyter notebook for EDA
└── tests/
    ├── __init__.py          # Test module initializer
    └── test_pipeline.py     # Automated Pytest / Unittest suite
```

---

## 🚀 How to Run Everything (Step-by-Step)

### Step 1: Clone Repository & Install Dependencies

```bash
git clone https://github.com/sansannycodes/Crop-Disease-Detection.git
cd Crop-Disease-Detection
pip install -r requirements.txt
```

### Step 2: Download & Prepare the Dataset

Raw dataset images are excluded from Git to keep the repository lightweight. Download the dataset into `data/dataset/`:

```bash
# Download dataset using KaggleHub snippet:
python -c "import kagglehub; path = kagglehub.dataset_download('vipoooool/new-plant-diseases-dataset'); print('Downloaded to:', path)"
```

Ensure the folders are organized under `data/dataset/`:
- `data/dataset/train/<class_folders>`
- `data/dataset/valid/<class_folders>`

### Step 3: Run Model Training

Train the CNN model from scratch on the dataset:

```bash
python backend/train.py
```

- Trained model output: `models/crop_disease_cnn.keras`
- Loss & Accuracy training plots: `output/training_history.png`

### Step 4: Run Automated Tests

Run unit tests to verify backend module configurations, model architecture compilation, and preprocessing logic:

```bash
python -m unittest tests/test_pipeline.py -v
```

### Step 5: Run CLI Single-Image Inference

Predict crop disease on an input leaf image via command line:

```bash
python backend/predict.py --image path/to/leaf_image.jpg
```

### Step 6: Launch Frontend Streamlit Web Application

Launch the interactive web user interface:

```bash
streamlit run frontend/app.py
```

Open `http://localhost:8501` in your web browser to upload leaf photographs and get AI plant disease diagnosis.

---

## 💻 Example CLI Prediction Output

```text
============================================================
 CROP DISEASE DETECTION INFERENCE
============================================================
Input Image : sample_leaf.jpg
Model Path  : models/crop_disease_cnn.keras

Top Predictions:
------------------------------------------------------------
  1. Tomato___Early_blight                         [94.85%]
  2. Tomato___Late_blight                          [ 3.12%]
  3. Tomato___Bacterial_spot                       [ 1.05%]
============================================================
```

---

## 📜 License & Acknowledgments

- GitHub Repository: [sansannycodes/Crop-Disease-Detection](https://github.com/sansannycodes/Crop-Disease-Detection)
- Dataset Source: **PlantVillage / New Plant Diseases Dataset** on Kaggle.
- Technology Stack: Python, TensorFlow, Keras, OpenCV, Streamlit, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn.
