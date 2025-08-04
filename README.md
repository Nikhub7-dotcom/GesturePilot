# GesturePilot
This project enables real-time computer control using hand gestures recognized through a webcam. By combining computer vision and deep learning, it provides an intuitive and contactless way to interact with your system.

# Sign Language Controlled Computer Interface

This project allows users to control basic computer actions using sign language gestures captured through a webcam. Trained on custom hand gesture data using TensorFlow and OpenCV, it recognizes five gestures: `click`, `scroll`, `swipe`, `pause`, and `tabchange` — and maps them to corresponding system functions in real time.

## 🚀 Features

- 🎯 Real-time hand gesture recognition using MediaPipe and OpenCV.
- 🧠 Deep learning model trained on sequence data (90 frames × 30 keypoints × 1662 features).
- 🖱️ Gesture-to-command mapping (e.g., click to open video, swipe to move cursor).
- 🖥️ Integrates seamlessly with the operating system using `pyautogui`.
- 📷 Live camera preview and screen action recording capability.
- 👋 Ignores input if no hands are detected — avoids false triggers.

## 🧠 Model Information

- **Architecture**: LSTM-based sequence classification.
- **Input shape**: `(90, 1662)` per sequence.
- **Classes**: `['click', 'scroll', 'swipe', 'pause', 'tabchange']`
- **Frameworks**: TensorFlow, Keras, MediaPipe, OpenCV.

## 🛠️ Requirements

Install dependencies via:

pip install -r requirements.txt



🗂️ Project Structure
├── ComputerControl.py         # Main script to run the controller
├── action.h5 / CC_action.h5   # Trained model files
├── requirements.txt
├── .gitignore
├── CC_DATA/                   # Preprocessed keypoint sequences
├── CC_Logs/                   # Model training logs
├── Sign Language Detection.ipynb
└── ComputerControl.ipynb



🖥️ Running the System
🔹 From Terminal:

conda activate py311_env  # or your env name
python ComputerControl.py



## 🎥 Demo

[![Watch the demo](https://img.youtube.com/vi/AzIjbG0mR5c/maxresdefault.jpg)](https://youtu.be/AzIjbG0mR5c)

