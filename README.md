   ---
   title: Reclaim Waste Classifier
   sdk: docker
   app_port: 7860
   ---
# Reclaim — Waste Classification for Smart Recycling

A small web app that classifies a photo of waste (cardboard, glass, metal, paper, plastic, trash) and shows a disposal tip.

**Stack:** Flask, TensorFlow/Keras (MobileNetV2, transfer learning), Pillow, scikit-learn, Matplotlib, plain HTML/CSS/JS.

> **Status:** the code is written, but the model has **not been trained yet**. No accuracy figures are reported until you run training and evaluation yourself.

## 1. Setup (Windows, Python 3.10–3.12)

```
cd waste-classification
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Dataset

Use **TrashNet** (about 2,500 images, 6 classes). A copy is on Kaggle (search "TrashNet" or "garbage classification"). Extract it so it looks like this:

```
dataset/
  cardboard/ glass/ metal/ paper/ plastic/ trash/
```

Class names are read from the folder names, so other folder-per-class datasets also work.

## 3. Train and evaluate

```
python train.py
python evaluate.py
```

- `train.py` makes a stratified 70/15/15 train/val/test split, trains the new head, then fine-tunes the last 30 MobileNetV2 layers. It saves `models/model.keras`, `models/class_names.json` and the test split. Needs internet once to download ImageNet weights.
- `evaluate.py` evaluates on the held-out test images only and writes `models/metrics.json` (accuracy, precision, recall, F1) and `models/confusion_matrix.png`.

Options: `python train.py --epochs 5 --fine-tune-epochs 3` (faster on CPU).

## 4. Run the app

```
python app.py
```

Open http://127.0.0.1:5000. `GET /health` shows whether the model loaded.

## 5. Tests

```
python -m unittest discover tests
```

## Evaluation results

_Not measured yet. Paste your values from `models/metrics.json` here after running `evaluate.py`._

## Limitations and troubleshooting

- TrashNet has plain backgrounds; real-world photos can give lower accuracy. Confidence is not a guarantee.
- Recycling rules vary by location; advice is general.
- "Model is not available" → run `python train.py` first, then restart Flask.
- Slow training → use fewer epochs; CPU training of TrashNet takes several minutes per epoch.
- TensorFlow install fails → check you are on Python 3.10–3.12 (64-bit).
