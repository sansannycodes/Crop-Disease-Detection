# Crop Disease Detection

A machine learning project for classifying plant leaf diseases using Convolutional Neural Networks (CNN), OpenCV, and TensorFlow/Keras. Includes a modular backend pipeline, a Streamlit web application, CLI inference tools, and unit tests.

---

## Overview

Plant diseases cause significant yield loss in agriculture. Traditional disease diagnosis relies on manual inspection, which can be slow and prone to errors. This project provides an automated image classification pipeline to help identify crop leaf diseases early.

---

## Objectives

- Build an end-to-end image classification pipeline for plant leaf health assessment.
- Classify leaf images across 38 distinct crop disease categories using a multi-layer CNN.
- Provide both a command-line interface (CLI) and a web app interface for running predictions.

---

## Features

- Modular Backend: Clean separation of dataset loading, model architecture, training routines, and inference scripts in `backend/`.
- Frontend Web App: Streamlit interface (`frontend/app.py`) for uploading leaf images and viewing classification results.
- Preprocessing Pipeline: Image reading and resizing via OpenCV/PIL, color space conversion, and pixel normalization.
- Data Augmentation: Real-time image augmentation using Keras `ImageDataGenerator`.
- Testing: Automated test suite in `tests/` verifying model setup and data preprocessing.

---

## Dataset Information

The project uses the PlantVillage / New Plant Diseases Dataset, which contains leaf images across 14 crop species.

- Total Classes: 38 plant health and disease categories
- Included Crops: Apple, Blueberry, Cherry, Corn (Maize), Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.
- Image Dimensions: Resized to 224 x 224 pixels with 3 RGB color channels.

### Target Classes

1. Apple___Apple_scab
2. Apple___Black_rot
3. Apple___Cedar_apple_rust
4. Apple___healthy
5. Blueberry___healthy
6. Cherry_(including_sour)___Powdery_mildew
7. Cherry_(including_sour)___healthy
8. Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot
9. Corn_(maize)___Common_rust_
10. Corn_(maize)___Northern_Leaf_Blight
11. Corn_(maize)___healthy
12. Grape___Black_rot
13. Grape___Esca_(Black_Measles)
14. Grape___Leaf_blight_(Isariopsis_Leaf_Spot)
15. Grape___healthy
16. Orange___Haunglongbing_(Citrus_greening)
17. Peach___Bacterial_spot
18. Peach___healthy
19. Pepper,_bell___Bacterial_spot
20. Pepper,_bell___healthy
21. Potato___Early_blight
22. Potato___Late_blight
23. Potato___healthy
24. Raspberry___healthy
25. Soybean___healthy
26. Squash___Powdery_mildew
27. Strawberry___Leaf_scorch
28. Strawberry___healthy
29. Tomato___Bacterial_spot
30. Tomato___Early_blight
31. Tomato___Late_blight
32. Tomato___Leaf_Mold
33. Tomato___Septoria_leaf_spot
34. Tomato___Spider_mites Two-spotted_spider_mite
35. Tomato___Target_Spot
36. Tomato___Tomato_Yellow_Leaf_Curl_Virus
37. Tomato___Tomato_mosaic_virus
38. Tomato___healthy

---

## Project Structure

```text
Crop-Disease-Detection/
├── .gitignore
├── README.md
├── requirements.txt
├── backend/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
├── frontend/
│   └── app.py
├── data/
│   └── README.md
├── models/
│   └── README.md
├── notebooks/
│   └── leaf_preprocessing_data.ipynb
└── tests/
    ├── __init__.py
    └── test_pipeline.py
```

---

## How to Run the Project

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Dataset Setup

Download the dataset locally into `data/dataset/`:

```bash
python -c "import kagglehub; path = kagglehub.dataset_download('vipoooool/new-plant-diseases-dataset'); print('Downloaded to:', path)"
```

Organize the extracted data as follows:
- `data/dataset/train/<class_folders>`
- `data/dataset/valid/<class_folders>`

### 3. Train the Model

```bash
python backend/train.py
```

- Trained weights are saved to `models/crop_disease_cnn.keras`.
- Training history curves are saved to `output/training_history.png`.

### 4. Run Unit Tests

```bash
python -m unittest tests/test_pipeline.py -v
```

### 5. Run CLI Inference

```bash
python backend/predict.py --image path/to/sample_leaf.jpg
```

### 6. Run Web App

```bash
streamlit run frontend/app.py
```

Open `http://localhost:8501` in your browser.

---

## License & Acknowledgments

- GitHub Repository: [sansannycodes/Crop-Disease-Detection](https://github.com/sansannycodes/Crop-Disease-Detection)
- Dataset Source: PlantVillage / New Plant Diseases Dataset (Kaggle).
