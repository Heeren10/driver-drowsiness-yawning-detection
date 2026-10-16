"""Request and response models (validated by FastAPI/Pydantic)."""
from typing import Literal

from pydantic import BaseModel, Field

from inference import MAX_IMAGE_BYTES


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model: str
    classes: list[str]
    test_accuracy: float


class Prediction(BaseModel):
    prediction: Literal["Closed", "Open", "no_yawn", "yawn"] | None = Field(
        description="CNN result: the eye state when a face is found; the raw 4-class result when detect_face=false")
    confidence: float | None
    face_detected: bool
    face_box: list[int] | None = Field(None, description="[x, y, width, height] of the face landmarks")
    eye_state: Literal["Closed", "Open"] | None
    eye_confidence: float | None
    yawn_state: Literal["no_yawn", "yawn"] | None = Field(description="Mouth wide open in THIS frame")
    mouth_open_ratio: float | None = Field(description="Mouth aspect ratio: ~0 closed, >= 0.30 wide open")
    latency_ms: float


class StreamRequest(BaseModel):
    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$", description="One id per browser tab / driver")
    image: str = Field(max_length=MAX_IMAGE_BYTES * 4 // 3 + 64, description="Base64 JPEG/PNG, data-URL prefix allowed")
    closed_threshold_sec: float = Field(2.0, ge=0.5, le=10.0, description="Eyes closed this long -> drowsy_alert")
    yawn_threshold_sec: float = Field(3.0, ge=1.0, le=8.0, description="Mouth wide open this long -> yawning")


class StreamPrediction(Prediction):
    closed_duration_sec: float
    perclos: float = Field(description="Share of closed-eye frames in the last 60 s")
    drowsy_alert: bool
    yawn_duration_sec: float
    yawning: bool = Field(description="A sustained yawn is in progress")
    yawn_count: int = Field(description="Yawns detected in this session")
