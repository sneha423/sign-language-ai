import cv2 as cv
import mediapipe as mp
import joblib
import numpy as np
import sys

# Load the model
try:
    model = joblib.load("sign_model.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Initialize webcam and MediaPipe
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit(1)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

print("Start signing. Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])

            if len(landmarks) == 42:
                prediction = model.predict([landmarks])[0]

                # Show prediction on screen
                cv.putText(frame, f"Gesture: {prediction}", (10, 50),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv.imshow("Sign Language Recognition", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
