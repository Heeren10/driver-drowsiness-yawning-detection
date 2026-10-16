"""Image decoding, face landmarks, eye-state CNN and mouth-opening (yawn) measurement."""
import base64
import binascii
import time
from threading import Lock

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from model_loader import ModelBundle

MAX_IMAGE_BYTES = 5 * 1024 * 1024

# MediaPipe Face Mesh landmark indices
EYE_CORNERS = ((33, 133), (362, 263))           # (outer, inner) corner of each eye
LIP_PAIRS = ((13, 14), (82, 87), (312, 317))    # inner upper-lip / lower-lip points
MOUTH_CORNERS = (78, 308)                        # inner mouth corners

# Calibration knobs (set on the dataset test frames + the user's webcam photos; re-check on the live webcam):
EYE_CROP_SCALE = 1.8        # eye crop side = 1.8 x eye width, so it looks like the training eye crops (eye + brow)
EYE_CROP_LIFT = 0.10        # shift the crop up by 10 % of its side to include the eyebrow
EYE_CLOSED_MIN_PROB = 0.85  # call eyes Closed only when the CNN is sure: downcast/typing eyes score lower, shut eyes ~0.98
YAWN_MIN_MAR = 0.30         # mouth aspect ratio for "mouth wide open"; test frames: normal <= 0.24, yawns >= 0.25

# ponytail: one lock = one frame analysed at a time (~15 ms each, plenty for a webcam demo); use a worker pool to scale.
_lock = Lock()


class ImageError(Exception):
    """Bad image input; carries the HTTP status code to return."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def decode_base64(text: str) -> bytes:
    """Accept plain base64 or a data URL ("data:image/jpeg;base64,...")."""
    payload = text.split(",", 1)[1] if text.startswith("data:") else text
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise ImageError("Invalid base64 image") from None


def decode_image(data: bytes) -> np.ndarray:
    """Encoded image bytes -> RGB uint8 array."""
    if not data:
        raise ImageError("Empty image")  # cv2.imdecode crashes on an empty buffer
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageError("Image larger than 5 MB", status_code=413)
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ImageError("Could not decode image (send a JPEG or PNG)")
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))  # the models expect RGB


def mouth_aspect_ratio(pts: np.ndarray) -> float:
    """Average inner-lip gap divided by mouth width: ~0 when closed, ~0.7 in a typical yawn."""
    gap = np.mean([np.linalg.norm(pts[a] - pts[b]) for a, b in LIP_PAIRS])
    return float(gap / np.linalg.norm(pts[MOUTH_CORNERS[0]] - pts[MOUTH_CORNERS[1]]))


def _landmarks(bundle: ModelBundle, rgb: np.ndarray):
    """478 face landmarks in pixel coordinates, or None if no face is found."""
    result = bundle.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.face_landmarks:
        return None
    h, w = rgb.shape[:2]
    return np.array([(p.x * w, p.y * h) for p in result.face_landmarks[0]])


def _eye_crops(rgb: np.ndarray, pts: np.ndarray) -> list:
    """Square crop around each eye, sized from the eye's own corner landmarks."""
    crops = []
    for outer, inner in EYE_CORNERS:
        side = EYE_CROP_SCALE * np.linalg.norm(pts[outer] - pts[inner])
        cx, cy = (pts[outer] + pts[inner]) / 2
        cy -= EYE_CROP_LIFT * side
        x0, y0 = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
        crops.append(rgb[y0:y0 + int(side), x0:x0 + int(side)])
    return crops


def _classify(bundle: ModelBundle, images: list) -> np.ndarray:
    """CNN class probabilities (n, 4) — preprocessing exactly as in training: tf bilinear resize, raw 0-255 RGB."""
    size = bundle.config["img_size"]
    batch = tf.stack([tf.image.resize(img, (size, size), method="bilinear") for img in images])
    return bundle.model.predict_on_batch(batch)


def analyze(bundle: ModelBundle, rgb: np.ndarray, detect_face: bool = True) -> dict:
    """Face landmarks -> eye crops -> CNN eye state, and lip landmarks -> mouth opening (yawn).
    detect_face=False runs the CNN on the image as-is (e.g. a dataset eye crop)."""
    start = time.perf_counter()
    classes = bundle.config["classes"]
    result = {"prediction": None, "confidence": None, "face_detected": False, "face_box": None,
              "eye_state": None, "eye_confidence": None, "yawn_state": None, "mouth_open_ratio": None}

    with _lock:
        if not detect_face:
            probs = _classify(bundle, [rgb])[0]
            k = int(probs.argmax())
            result.update(prediction=classes[k], confidence=float(probs[k]))
        elif (pts := _landmarks(bundle, rgb)) is not None:
            probs = _classify(bundle, _eye_crops(rgb, pts)).mean(axis=0)  # average of both eyes
            closed, opened = probs[classes.index("Closed")], probs[classes.index("Open")]
            p_closed = float(closed / (closed + opened))
            eye_state, eye_conf = ("Closed", p_closed) if p_closed >= EYE_CLOSED_MIN_PROB else ("Open", 1 - p_closed)
            mar = mouth_aspect_ratio(pts)
            (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
            result.update(prediction=eye_state, confidence=eye_conf, face_detected=True,
                          face_box=[int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
                          eye_state=eye_state, eye_confidence=eye_conf,
                          yawn_state="yawn" if mar >= YAWN_MIN_MAR else "no_yawn", mouth_open_ratio=round(mar, 3))

    result["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
    return result
