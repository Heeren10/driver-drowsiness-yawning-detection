# Driver Drowsiness Detection System

Real-time driver drowsiness and yawn detection from a webcam: a custom-trained **CNN** classifies eye state, **MediaPipe** face landmarks measure mouth opening, and a **FastAPI** backend raises an alarm when the eyes stay closed for 2 seconds or more.

| Test accuracy | Macro F1 | CPU time per frame |
|---|---|---|
| **98.61 %** | **98.61 %** | **~18 ms** (full pipeline) |

📄 **Full documentation, architecture and all results:** [docs/README.md](docs/README.md)

## Quick start

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r model/requirements.txt -r backend/requirements.txt

cd backend
..\.venv\Scripts\uvicorn main:app --port 8000
```

Then, in a second terminal: `cd test_frontend` → `..\.venv\Scripts\python -m http.server 5173 --bind 127.0.0.1`, and open http://127.0.0.1:5173.

Dataset: [Yawn Eye Dataset (new)](https://www.kaggle.com/datasets/serenaraju/yawn-eye-dataset-new) on Kaggle (place the zip at `data/archive.zip` to retrain).
