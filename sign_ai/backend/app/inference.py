import cv2
import joblib
import mediapipe as mp
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sign_model.pkl")
model = joblib.load(MODEL_PATH)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def predict_from_image_bytes(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return {"success": False, "error": "Invalid image"}

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return {"success": False, "error": "No hand detected"}

    landmarks = []
    for lm in results.multi_hand_landmarks[0].landmark:
        landmarks.extend([lm.x, lm.y])

    if len(landmarks) != 42:
        return {"success": False, "error": "Expected 42 features"}

    prediction = model.predict([landmarks])[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([landmarks])[0]
        confidence = float(np.max(probs))

    return {
        "success": True,
        "prediction": str(prediction),
        "confidence": confidence
    }