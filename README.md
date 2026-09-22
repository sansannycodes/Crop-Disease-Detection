# 🌿 Crop Disease Detection System using Deep Learning

A deep learning project for identifying plant diseases from leaf images using Convolutional Neural Networks (CNN), OpenCV, and TensorFlow/Keras. Structured with clean `backend/` and `frontend/` components, interactive Streamlit web interface, CLI inference capabilities, and automated unit testing.

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

## 🛠️ Data Preprocessing & Augmentation

1. **Image Loading & Resizing**: Input images are loaded via OpenCV/PIL, converted to RGB space, and resized to $224 \times 224$ pixels.
2. **Normalization**: Pixel intensities are rescaled from $[0, 255]$ to floating-point values in $[0.0, 1.0]$.
3. **Training Data Augmentation**: Applied during training to prevent overfitting:
   - Rotation Range: $\pm 20^\circ$
   - Width & Height Shifts: $\pm 10\%$
   - Shear Range: $10\%$
   - Zoom Range: $10\%$
   - Horizontal Flip: Enabled

---

## 🧠 CNN Model Architecture

The custom CNN architecture is designed for multi-class image classification using Keras Sequential API:

```text
Input Layer: (224, 224, 3)
│
├── Conv2D (32 filters, 3x3) + BatchNorm + ReLU + MaxPooling (2x2) + Dropout (0.2)
├── Conv2D (64 filters, 3x3) + BatchNorm + ReLU + MaxPooling (2x2) + Dropout (0.2)
├── Conv2D (128 filters, 3x3) + BatchNorm + ReLU + MaxPooling (2x2) + Dropout (0.3)
├── Conv2D (256 filters, 3x3) + BatchNorm + ReLU + MaxPooling (2x2) + Dropout (0.3)
│
├── Flatten Layer
├── Dense (256 units) + BatchNorm + ReLU + Dropout (0.5)
└── Output Layer: Dense (38 units, Softmax)
```

- **Loss Function**: Categorical Crossentropy (`categorical_crossentropy`)
- **Optimizer**: Adam ($\text{learning rate} = 0.001$)
- **Evaluation Metrics**: Categorical Accuracy, Top-3 Categorical Accuracy

---

## 🏋️ Training & Evaluation Procedure

- **Callbacks**:
  - `ModelCheckpoint`: Saves the best model weights to `models/crop_disease_cnn.keras` based on validation accuracy.
  - `EarlyStopping`: Stops training if validation loss does not improve for 5 consecutive epochs.
  - `ReduceLROnPlateau`: Reduces learning rate by a factor of $0.2$ when validation loss plateaus.
- **Evaluation Metrics Generated**: Loss curves, Accuracy progression plots (`output/training_history.png`), and Top-3 accuracy metrics upon validation completion.

---

## 📁 Project Structure

```text
cropdiseaseprediction/
├── .gitignore               # Git exclusion rules (venv, cache, models, datasets)
├── README.md                # Comprehensive project documentation
├── requirements.txt         # Project dependencies list
├── backend/                 # Core Backend Modules (CNN Model, Data, Pipeline)
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

## 🚀 How to Run the Project

### 1. Prerequisites & Installation

Clone the repository and install required packages:

```bash
git clone https://github.com/your-username/cropdiseaseprediction.git
cd cropdiseaseprediction
pip install -r requirements.txt
```

### 2. Dataset Setup

Large raw datasets are excluded from Git. Download the dataset and place it in `data/dataset/`:

```bash
# Follow instructions in data/README.md or use kagglehub:
python -c "import kagglehub; path = kagglehub.dataset_download('vipoooool/new-plant-diseases-dataset'); print('Downloaded to:', path)"
```

Ensure the directory structure matches:
- `data/dataset/train/<class_folders>`
- `data/dataset/valid/<class_folders>`

### 3. Model Training

To train the CNN model from scratch:

```bash
python backend/train.py
```

The best trained model will be saved to `models/crop_disease_cnn.keras` and training history plots to `output/training_history.png`.

### 4. Running Unit Tests

Verify the project setup and model pipeline:

```bash
python -m unittest tests/test_pipeline.py -v
```

### 5. Running CLI Inference

Predict disease class for a single leaf image:

```bash
python backend/predict.py --image path/to/leaf_image.jpg
```

### 6. Launching the Frontend Web App

Start the interactive Streamlit web interface:

```bash
streamlit run frontend/app.py
```

Open `http://localhost:8501` in your browser to test leaf image diagnosis.

---

## 💻 Example CLI Output

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

## 🔮 Future Enhancements

- **Transfer Learning**: Integrate MobileNetV3 / ResNet50 backbones for improved feature extraction and lightweight deployment on mobile devices.
- **Severity Assessment**: Quantify leaf infection coverage percentages using color segmentation.
- **Treatment Suggestions**: Integrate automated agronomic recommendation guidelines based on diagnosed diseases.
- **REST API Endpoint**: Package the inference pipeline as a FastAPI service for mobile application integration.

---

## 📜 License & Acknowledgments

- Dataset source: **PlantVillage / New Plant Diseases Dataset** on Kaggle.
- Built with TensorFlow, OpenCV, and Streamlit.
