import cv2 as cv
import mediapipe as mp
import time
import csv
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import sys

# Create folder if not exists
if not os.path.exists("gesture_data"):
    try:
        os.makedirs("gesture_data")
        print("Created folder 'gesture_data'")
    except OSError as e:
        print(f"Error creating folder 'gesture_data': {e}")
        sys.exit(1)

# Global variables
csv_file_name = "all_sign_data.csv"
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam. Please check your camera connection.")
    sys.exit(1)

detector = mp.solutions.hands.Hands(static_image_mode=False,
                                    max_num_hands=1,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# --- Data Collection Function ---
def collect_data():
    """Collects hand landmark data for different gestures and saves it to a single CSV file."""
    global cap, detector, csv_file_name

    # Write header only if file doesn't exist
    if not os.path.exists(csv_file_name):
        try:
            with open(csv_file_name, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                header = ["Sign"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)]
                csv_writer.writerow(header)
        except Exception as e:
            print(f"Error writing CSV header: {e}")
            sys.exit(1)

    while True:
        print("Collecting started...")
        gesture_name = input("Enter the gesture name (or type 'exit' to stop): ").strip()
        if gesture_name.lower() == "exit":
            break

        data = []
        print(f"Start showing gesture: {gesture_name}")
        print("Press 'x' to stop recording this gesture.")
        ptime = 0

        while True:
            success, frame = cap.read()
            if not success:
                print("Error: Could not read frame from webcam.")
                break

            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            try:
                results = detector.process(frame_rgb)
            except Exception as e:
                print(f"Error processing frame with MediaPipe: {e}")
                results = None

            if results and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                    row = []
                    for landmark in hand_landmarks.landmark:
                        row.extend([landmark.x, landmark.y])
                    data.append([gesture_name] + row)

            # Display FPS
            ctime = time.time()
            fps = 1 / (ctime - ptime + 1e-6)
            ptime = ctime
            cv.putText(frame, str(int(fps)), (10, 70), cv.FONT_ITALIC, 2, (0, 255, 0), 2)
            cv.imshow("Webcam", frame)

            if cv.waitKey(1) & 0xFF == ord("x"):
                break

        # Save to CSV file
        if data:
            try:
                with open(csv_file_name, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(data)
                print(f"Saved {len(data)} frames to {csv_file_name} ✅\n")
            except Exception as e:
                print(f"Error writing to CSV file: {e}")
                print("Data for this gesture was not saved.")
        else:
            print(f"No data was collected for gesture: {gesture_name} \n")

    cap.release()
    cv.destroyAllWindows()

# --- Model Training Function ---
def train_model():
    """Trains a Random Forest Classifier on the collected data and saves the model."""
    global csv_file_name

    # Load the dataset
    try:
        data = pd.read_csv(csv_file_name)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_name}' was not found. Please run the data collection step first.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

    if data.empty:
        print(f"Error: The CSV file '{csv_file_name}' is empty. Please collect data before training the model.")
        sys.exit(1)

    # Features and labels
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values

    if len(y) < 2:
        print("Error: Not enough data to train a model. Please collect more samples.")
        sys.exit(1)

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        print(f"Error splitting data: {e}")
        sys.exit(1)

    try:
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error training model: {e}")
        sys.exit(1)

    try:
        accuracy = model.score(X_test, y_test)
        print("Model Accuracy:", accuracy)
    except Exception as e:
        print(f"Error evaluating model: {e}")
        accuracy = None

    try:
        if accuracy is not None:
            joblib.dump(model, "sign_model.pkl")
            print("Trained model saved to sign_model.pkl")
        else:
            print("Model was not saved due to errors during evaluation.")
    except Exception as e:
        print(f"Error saving model: {e}")
        sys.exit(1)

# --- Main Execution ---
if __name__ == "__main__":
    collect_data()
    train_model()
