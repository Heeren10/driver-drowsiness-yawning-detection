"""Driver Drowsiness Detection API.

Run (from the backend/ folder):   ..\\.venv\\Scripts\\uvicorn main:app --port 8000
Interactive docs:                 http://127.0.0.1:8000/docs
Call the API via 127.0.0.1, not "localhost": on Windows "localhost" tries IPv6 first and adds ~2 s per request.
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from inference import MAX_IMAGE_BYTES, ImageError, analyze, decode_base64, decode_image
from model_loader import load_bundle
from schemas import HealthResponse, Prediction, StreamPrediction, StreamRequest
from tracker import DrowsinessTracker


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = load_bundle()  # model is loaded once, not per request
    app.state.tracker = DrowsinessTracker()
    yield


app = FastAPI(title="Driver Drowsiness Detection API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),  # Vite dev server
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(ImageError)
def image_error(_: Request, exc: ImageError):
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


@app.get("/health", response_model=HealthResponse)
def health():
    config = app.state.bundle.config
    return {"status": "ok", "model": config["model_file"], "classes": config["classes"],
            "test_accuracy": config["test_metrics"]["accuracy"]}


@app.post("/predict", response_model=Prediction)
def predict(file: UploadFile = File(...), detect_face: bool = True):
    """One image (multipart): eye state (CNN) + mouth opening. detect_face=false runs the CNN on the image as-is."""
    rgb = decode_image(file.file.read(MAX_IMAGE_BYTES + 1))
    return analyze(app.state.bundle, rgb, detect_face)


@app.post("/predict-stream", response_model=StreamPrediction)
def predict_stream(req: StreamRequest):
    """One webcam frame (base64) per call; the server tracks eye closure and yawns per session_id."""
    result = analyze(app.state.bundle, decode_image(decode_base64(req.image)))
    face = result["face_detected"]
    eyes_closed = result["eye_state"] == "Closed" if face else None
    mouth_open = result["yawn_state"] == "yawn" if face else None
    return {**result, **app.state.tracker.update(req.session_id, eyes_closed, mouth_open,
                                                 req.closed_threshold_sec, req.yawn_threshold_sec)}
