"""Loads the exported CNN, its config and the MediaPipe face-landmark model once at startup."""
import json
import os
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

# The backend has no copy of any model: it reads the project's model/ folder (override with the MODEL_ROOT env var).
#   model/exported/                     <- our CNN (notebook 04)
#   model/pretrained/face_landmarker.task <- Google's pre-trained MediaPipe model, from
#   https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
MODEL_ROOT = Path(os.getenv("MODEL_ROOT", Path(__file__).resolve().parent.parent / "model"))


@dataclass(frozen=True)
class ModelBundle:
    model: keras.Model
    config: dict  # contents of model_config.json (classes, img_size, preprocessing)
    landmarker: vision.FaceLandmarker


def load_bundle(model_root: Path = MODEL_ROOT) -> ModelBundle:
    exported = model_root / "exported"
    config = json.loads((exported / "model_config.json").read_text())
    model = keras.models.load_model(exported / config["model_file"])

    size = config["img_size"]
    for batch in (1, 2):  # warm up the compiled predict path for both request shapes (one image / two eyes)
        model.predict_on_batch(np.zeros((batch, size, size, 3), dtype=np.float32))

    landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_root / "pretrained" / "face_landmarker.task")),
        num_faces=1))
    return ModelBundle(model, config, landmarker)
