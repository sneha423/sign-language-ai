from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.inference import predict_from_image_bytes

app = FastAPI(title="Sign Language Recognition System API")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://your-frontend-name.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    return predict_from_image_bytes(image_bytes)