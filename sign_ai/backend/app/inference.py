import os
import cv2
import joblib
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "sign_model.pkl")
HAND_TASK_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")

classifier = joblib.load(MODEL_PATH)

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_TASK_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

def extract_hand_keypoints(image_bgr: np.ndarray):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    result = landmarker.detect(mp_image)

    if not result.hand_landmarks or len(result.hand_landmarks) == 0:
        return None

    hand_landmarks = result.hand_landmarks[0]
    keypoints = []

    for lm in hand_landmarks:
        keypoints.extend([lm.x, lm.y])

    return np.array(keypoints, dtype=np.float32)

def predict_from_image_bytes(image_bytes: bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image file")

    features = extract_hand_keypoints(image)

    if features is None:
        return {
            "prediction": "No hand detected",
            "confidence": 0.0
        }

    features = features.reshape(1, -1)

    pred = classifier.predict(features)[0]

    confidence = 1.0
    if hasattr(classifier, "predict_proba"):
        confidence = float(np.max(classifier.predict_proba(features)))

    return {
        "prediction": str(pred),
        "confidence": round(confidence, 4)
    }