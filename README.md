# Gesture Canvas Project

## Overview

This project is a real-time webcam-based gesture drawing and shape recognition app.
It uses MediaPipe hand landmarks to detect gestures and control a drawing canvas.
The app supports:

- drawing with an index-finger gesture
- erasing with a fist gesture
- predicting the drawn shape with a two-finger gesture
- selecting colours by hovering over a palette bar
- dynamic brush thickness via pinch distance
- real-time shape classification using a trained CNN model or a heuristic fallback

## Key Files

- `main.py` — main application loop, webcam capture, UI, gesture state, and prediction logic
- `gesture_detector.py` — MediaPipe hand detection and gesture mode classification
- `canvas.py` — canvas drawing engine, palette, thumbnail, and blending
- `shape_classifier.py` — loads Keras model or uses contour heuristic for shape prediction
- `train_model.py` — defines and trains the CNN model on shape data
- `prepare_dataset.py` — downloads and prepares Quick Draw dataset splits
- `config.py` — shared configuration values and shape labels
- `requirements.txt` — Python dependencies for the project

## Project Structure

- `data/` — preprocessed dataset arrays used for training and evaluation
- `models/` — trained model files and saved artifacts

## Requirements

This project is designed to work on Windows with Python 3.10 / 3.11 / 3.12.

> Python 3.14 is not supported for the main app because the installed MediaPipe package may no longer expose the classic `mp.solutions` API.

Required packages:

- `opencv-python`
- `mediapipe`
- `numpy`
- `scikit-learn`
- `matplotlib`

Optional but recommended for model training and best prediction quality:

- `tensorflow==2.13.0`

> Note: TensorFlow is not included in `requirements.txt` because it may not support Python 3.14.

## Setup

1. Create and activate a virtual environment using a supported Python interpreter (3.10 / 3.11 / 3.12):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
pip install mediapipe==0.10.35
```

3. (Optional) Install TensorFlow if you want to train the model or use the trained CNN:

```powershell
pip install tensorflow==2.13.0
```

## Run the App

Start the real-time gesture drawing application:

```powershell
python main.py
```

Controls while the app is running:

- `q` or `Esc` — quit the app
- `c` — clear the canvas manually

Gesture controls:

- `draw` mode: index finger up only → draw on the canvas
- `erase` mode: all fingers down (fist) → stop drawing and erase gesture
- `predict` mode: index + middle fingers up → trigger shape prediction after holding for a few seconds
- `idle` mode: any other finger combination

## Training the Model

If you want to train the CNN model from scratch:

1. Prepare the dataset:

```powershell
python prepare_dataset.py
```

2. Train the model:

```powershell
python train_model.py
```

This will save the best model to `models/shape_classifier.h5` and also create training plots in `models/`.

## How Prediction Works

`shape_classifier.py` can use two modes:

- TensorFlow model prediction from `models/shape_classifier.h5`
- contour-based heuristic fallback when TensorFlow or the model file is unavailable

The shape classes are defined in `config.py`:

- `circle`
- `square`
- `triangle`
- `star`

## Notes

- If `models/shape_classifier.h5` is missing, the app still runs with heuristic shape detection.
- The webcam feed is flipped horizontally to behave like a mirror.
- Colour is selected by hovering the index finger over the top palette for 1 second.
- Brush thickness is adjusted by the distance between thumb and index finger.

## Troubleshooting

- If the camera does not open, check your webcam index in `config.py` under `CAMERA_INDEX`.
- If you see `AttributeError: module 'mediapipe' has no attribute 'solutions'`, you are likely using an unsupported Python/MediaPipe combination.
  Use Python 3.10 / 3.11 / 3.12 and install MediaPipe with:

  ```powershell
  python -m pip install mediapipe==0.10.35
  ```

- If the app cannot load the TensorFlow model, it will print a warning and use the fallback heuristic.
- Make sure `data/` and `models/` folders exist when running training or prediction.

## Useful Commands

```powershell
python main.py
python prepare_dataset.py
python train_model.py
```
