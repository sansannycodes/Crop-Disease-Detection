# Dataset Setup Guide - Crop-Disease-Detection

This directory stores the dataset for the **Crop-Disease-Detection** project.
The dataset is excluded from version control due to file size limits. Follow the instructions below to download and configure the dataset locally.

---

## 1. Directory Layout

Extract leaf images into `data/dataset/` under `train` and `valid` subfolders matching the class names:

```text
data/
└── dataset/
    ├── train/
    │   ├── Apple___Apple_scab/
    │   ├── Apple___Black_rot/
    │   └── ...
    └── valid/
        ├── Apple___Apple_scab/
        ├── Apple___Black_rot/
        └── ...
```

---

## 2. Download Options

### Option A: Using KaggleHub

Run the following Python command to download the dataset from Kaggle:

```python
import kagglehub

path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
print("Downloaded to:", path)
```

Copy or move the `train` and `valid` subdirectories to `data/dataset/`.

### Option B: Manual Download

1. Download the New Plant Diseases Dataset from Kaggle.
2. Extract the archive.
3. Move the `train` and `valid` folders into `data/dataset/`.

---

## 3. Dataset Summary

- Dataset: PlantVillage (Augmented)
- Total Classes: 38 leaf disease categories
- Resolution: Resized to 224 x 224 RGB during model input processing.
