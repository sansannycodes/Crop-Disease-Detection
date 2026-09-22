# Trained Models Directory

This directory stores trained model weights and serialized Keras model files (`.keras` / `.h5`).

> [!NOTE]
> Trained CNN model weights are excluded from Git version control via `.gitignore` because binary model files are too large for standard GitHub repositories.

---

## Obtaining or Recreating the Model

### Option A: Train a New Model locally
To train the CNN model from scratch on your local system, place the dataset in `data/dataset/` and execute:

```bash
python src/train.py
```

Upon completion, the trained model file will automatically be saved to:
`models/crop_disease_cnn.keras`

### Option B: Pre-trained Weights (Release Downloads)
If pre-trained weights are provided, download `crop_disease_cnn.keras` from the GitHub Repository Releases page and place it into this `models/` directory before running inference or launching the Streamlit app.

---

## File Layout

- `models/crop_disease_cnn.keras` - Saved Keras Sequential CNN model (HDF5 / Native Keras format)
