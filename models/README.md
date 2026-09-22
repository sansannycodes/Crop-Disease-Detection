# Trained Models

This directory stores trained model files (`.keras` / `.h5`).

Model weights are excluded from Git repository tracking via `.gitignore`.

---

## Model Recreation and Usage

### Option A: Train Locally
To train the CNN model locally, place the dataset in `data/dataset/` and run:

```bash
python backend/train.py
```

The output file will be saved as `models/crop_disease_cnn.keras`.

### Option B: Pre-trained Weights
Place any pre-trained model file named `crop_disease_cnn.keras` in this directory before running inference or launching the Streamlit app.
