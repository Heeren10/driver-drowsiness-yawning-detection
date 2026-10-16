"""Experiment (option B): can MediaPipe face landmarks detect yawns on new faces?

Isolated from the project: own venv (experiments/yawn_mediapipe/.venv), touches no project code.
Run:  .venv\\Scripts\\python test_yawn_mediapipe.py
"""
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEST_DIR = ROOT / "data" / "processed" / "test"

# Face Mesh landmark indices: inner-lip vertical pairs and the inner mouth corners
LIP_PAIRS = [(13, 14), (82, 87), (312, 317)]
MOUTH_CORNERS = (78, 308)

landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(ROOT / "model" / "pretrained" / "face_landmarker.task")),
    output_face_blendshapes=True, num_faces=1))


def measure(path: Path):
    """(MAR, jawOpen, landmarks in pixels) for one image, or None if no face is found."""
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.face_landmarks:
        return None
    h, w = rgb.shape[:2]
    pts = np.array([(p.x * w, p.y * h) for p in result.face_landmarks[0]])
    gap = np.mean([np.linalg.norm(pts[a] - pts[b]) for a, b in LIP_PAIRS])
    mar = gap / np.linalg.norm(pts[MOUTH_CORNERS[0]] - pts[MOUTH_CORNERS[1]])
    jaw_open = next(c.score for c in result.face_blendshapes[0] if c.category_name == "jawOpen")
    return float(mar), float(jaw_open), pts


def best_threshold(yawn: np.ndarray, normal: np.ndarray):
    """Threshold with the highest balanced accuracy for 'value >= t means yawn'."""
    candidates = np.unique(np.concatenate([yawn, normal]))
    scores = [((yawn >= t).mean() + (normal < t).mean()) / 2 for t in candidates]
    i = int(np.argmax(scores))
    return float(candidates[i]), float(scores[i])


# 1) Labelled dataset faces (many different drivers)
values = {}
for cls in ("yawn", "no_yawn"):
    files = sorted((TEST_DIR / cls).iterdir())
    found = [m[:2] for m in map(measure, files) if m]
    values[cls] = np.array(found)
    print(f"{cls:<8} faces found {len(found)}/{len(files)} | "
          f"MAR median {np.median(values[cls][:, 0]):.3f} | jawOpen median {np.median(values[cls][:, 1]):.3f}")

for k, name in enumerate(("MAR", "jawOpen")):
    t, acc = best_threshold(values["yawn"][:, k], values["no_yawn"][:, k])
    print(f"{name:<8} best threshold {t:.3f} -> balanced accuracy {acc:.1%} "
          f"(yawn caught {(values['yawn'][:, k] >= t).mean():.1%}, no_yawn kept {(values['no_yawn'][:, k] < t).mean():.1%})")

# 2) The user's own webcam photos (WIN_*.jpg in the project root), drawn with the lip landmarks
tiles = []
for path in sorted(ROOT.glob("WIN_*.jpg")):
    m = measure(path)
    if m is None:
        print(f"{path.name}: no face found")
        continue
    mar, jaw_open, pts = m
    print(f"{path.name}: MAR {mar:.3f} | jawOpen {jaw_open:.3f}")
    img = cv2.imread(str(path))
    for a, b in LIP_PAIRS + [MOUTH_CORNERS]:
        cv2.line(img, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), (0, 255, 0), 2)
    x, y = pts.min(axis=0).astype(int), pts.max(axis=0).astype(int)
    face = cv2.resize(img[max(y[1] - (y[1] - x[1]) - 40, 0):y[1] + 40, max(x[0] - 40, 0):y[0] + 40], (320, 320))
    cv2.putText(face, f"MAR {mar:.2f} jaw {jaw_open:.2f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    tiles.append(face)
if tiles:
    cv2.imwrite(str(HERE / "user_photos_landmarks.png"), np.hstack(tiles))
    print("saved user_photos_landmarks.png")
