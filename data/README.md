# Dataset Placement & Setup Guide

This directory holds the training, validation, and test datasets for the **Crop Disease Detection** project.
Because raw image datasets (~2 GB) are too large to commit directly to GitHub, follow the instructions below to download and set up the dataset locally.

---

## 1. Expected Dataset Directory Structure

Place the leaf images into `data/dataset/` with subdirectories matching the 38 class names:

```text
data/
└── dataset/
    ├── train/
    │   ├── Apple___Apple_scab/
    │   ├── Apple___Black_rot/
    │   ├── ...
    │   └── Tomato___healthy/
    └── valid/
        ├── Apple___Apple_scab/
        ├── Apple___Black_rot/
        ├── ...
        └── Tomato___healthy/
```

---

## 2. Downloading the Dataset

### Option A: Via Python (KaggleHub - Recommended)

Run the following snippet in Python to download the **New Plant Diseases Dataset (Augmented)** from Kaggle:

```python
import kagglehub
import shutil

# Download latest version
path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
print("Dataset downloaded to:", path)

# Target directory: cropdiseaseprediction/data/dataset
# Copy or link the downloaded train and valid folders to data/dataset/
```

### Option B: Manual Download from Kaggle

1. Visit the Kaggle dataset page: [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).
2. Click **Download** and extract the ZIP archive.
3. Move the `New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train` and `valid` folders into `cropdiseaseprediction/data/dataset/`.

---

## 3. Dataset Overview

- **Source**: PlantVillage Dataset (Augmented)
- **Total Classes**: 38 plant leaf disease classes
- **Crops Included**: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato.
- **Image Format**: RGB Leaf Images (Resized to 224x224 during preprocessing).
