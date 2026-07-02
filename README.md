# Sign Language Recognition System

A full-stack AI web application that recognizes sign language gestures from an uploaded image or a live webcam capture.

## Overview

This project combines a React + Vite frontend with a FastAPI backend and an image-based inference pipeline. Users can either upload an image or capture a frame from the webcam, send it to the backend, and receive a predicted sign with confidence.

## Features

- Upload an image and run sign prediction.
- Capture a live webcam frame and predict instantly.
- Backend health check and API status indicator.
- Prediction history for recent results.
- FastAPI inference endpoint for image-based classification.
- Frontend and backend deployed separately.

## Tech Stack

### Frontend

- React
- Vite
- JavaScript
- Fetch API

### Backend

- FastAPI
- Python
- CORSMiddleware

### AI / Inference

- Custom image inference pipeline
- Model prediction from image bytes

## Project Structure

```text
sign_ai/
├── frontend/        # React + Vite client
├── backend/         # FastAPI API and inference logic
└── README.md
```

## How It Works

1. The user uploads an image or captures a frame from the webcam.
2. The frontend sends the image as `multipart/form-data` to the backend.
3. The backend processes the image and runs inference.
4. The API returns the predicted sign and confidence score.
5. The frontend displays the result and stores recent predictions in history.

## API Endpoints

### `GET /`
Returns a basic API message.

### `GET /health`
Returns backend health status.

### `POST /predict`
Accepts an image file and returns the prediction output.

## Local Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd sign_ai
```

### 2. Run the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Backend runs on `http://127.0.0.1:8000`.

### 3. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:5173`.

## Environment Variables

Create a `.env` file inside `frontend/`:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

For production, set `VITE_API_BASE_URL` to the deployed backend URL.

## Deployment Notes

- Frontend can be deployed on Vercel.
- Backend can be deployed on Render.
- CORS must allow the frontend production domain and local development origins.
- Preview deployment URLs can change, so a stable production domain is recommended.

## Challenges Solved

- Handled webcam startup issues with a safer camera flow.
- Fixed frontend-backend integration for image uploads.
- Resolved CORS issues between Vercel frontend and Render backend.
- Improved resilience for API status checks and prediction flow.

## Future Improvements

- Add support for more signs and model classes.
- Show uploaded/captured image preview with prediction.
- Save prediction history persistently.
- Add loading states, retry states, and better error messaging.
- Improve deployment with a stable production domain and automated CI/CD.

## License

This project is intended for learning, experimentation, and academic use.
