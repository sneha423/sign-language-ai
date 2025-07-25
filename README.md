
# 🤟 Real-Time Sign Language Detection using AI

This project implements a real-time system for recognizing hand gestures corresponding to sign language characters. It uses computer vision techniques and a machine learning model trained on hand keypoint data to classify gestures captured via webcam. The project is built in Python using OpenCV, MediaPipe, and scikit-learn.

## 🚀 Features

- Live hand gesture recognition from webcam feed
- Trained K-Nearest Neighbors (KNN) classifier
- Real-time display of predicted sign on video overlay
- Hand landmark extraction using MediaPipe
- Organized and modular codebase

## 🧰 Tech Stack

- **Language:** Python
- **Libraries:** OpenCV, MediaPipe, NumPy, scikit-learn, Pandas

## 📁 Project Structure

```
sign-language-ai/
├── train_model.py         # Script to train the KNN model from dataset
├── predict_gesture.py     # Script to capture webcam input and predict gestures
├── all_sign_data.csv      # Hand keypoints dataset with gesture labels
├── AI_PROJECT_REPORT.docx # Project documentation/report
└── README.md              # Project overview and instructions
```

## 🛠 How to Run the Project

### 1. Install dependencies

Make sure you have Python 3.7+ installed, then run:

```bash
pip install opencv-python mediapipe scikit-learn pandas numpy
```

### 2. Train the model

This will read the gesture data from `all_sign_data.csv` and save the model.

```bash
python train_model.py
```

### 3. Run the prediction

This launches the webcam and starts recognizing gestures in real time:

```bash
python predict_gesture.py
```

## 🧾 Dataset

- `all_sign_data.csv` contains rows of hand keypoints (x, y coordinates) for each landmark detected.
- Each row is labeled with the corresponding gesture class (e.g., A, B, C, ...).


## 🏷 License

This project is licensed under the MIT License. Feel free to use and modify with attribution.
#   s i g n - l a n g u a g e - a i  
 