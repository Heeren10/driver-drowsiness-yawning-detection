"""Self-check for the backend logic. Run from backend/:  ..\\.venv\\Scripts\\python test_backend.py"""
import base64

import numpy as np

from inference import LIP_PAIRS, MOUTH_CORNERS, ImageError, decode_base64, decode_image, mouth_aspect_ratio
from tracker import DrowsinessTracker


def test_eye_tracking():
    t = DrowsinessTracker(window_sec=60)
    up = lambda closed, now: t.update("s", closed, False, 2.0, 3.0, now=now)
    assert up(False, 0.0)["drowsy_alert"] is False
    up(True, 1.0)                                         # eyes close at t=1
    assert up(True, 2.5)["drowsy_alert"] is False         # closed 1.5 s < 2 s
    r = up(True, 3.0)                                     # closed 2.0 s -> alert
    assert r["drowsy_alert"] is True and r["closed_duration_sec"] == 2.0
    assert r["perclos"] == 0.75                           # 3 of 4 frames closed
    assert up(False, 3.5)["drowsy_alert"] is False        # eyes open -> streak resets
    up(True, 4.0)
    assert up(None, 9.0)["closed_duration_sec"] == 0.0    # face lost -> streak resets
    assert t.update("other", True, False, 2.0, 3.0, now=9.0)["closed_duration_sec"] == 0.0  # sessions independent
    assert up(False, 200.0)["perclos"] == 0.0             # old frames leave the 60 s window


def test_yawn_tracking():
    t = DrowsinessTracker()
    up = lambda mouth_open, now: t.update("y", False, mouth_open, 2.0, 3.0, now=now)
    up(True, 0.0)                                         # mouth opens wide at t=0
    assert up(True, 0.5)["yawning"] is False              # 0.5 s: could be talking
    assert up(False, 1.0)["yawn_count"] == 0              # closed again -> no yawn
    up(True, 2.0)
    r = up(True, 5.0)                                     # held 3 s -> yawn
    assert r["yawning"] is True and r["yawn_count"] == 1 and r["yawn_duration_sec"] == 3.0
    assert up(True, 6.0)["yawn_count"] == 1               # same yawn is counted once
    up(False, 7.0)
    up(True, 8.0)
    assert up(True, 11.5)["yawn_count"] == 2              # a second yawn
    assert up(None, 12.0)["yawning"] is False             # face lost -> yawn streak resets


def test_mouth_aspect_ratio():
    pts = np.zeros((478, 2))
    pts[MOUTH_CORNERS[0]], pts[MOUTH_CORNERS[1]] = (0, 0), (100, 0)  # mouth 100 px wide
    for upper, lower in LIP_PAIRS:
        pts[upper], pts[lower] = (50, -35), (50, 35)                  # lips 70 px apart
    assert abs(mouth_aspect_ratio(pts) - 0.7) < 1e-9


def test_decoding():
    for bad in (b"not an image", b""):
        try:
            decode_image(bad)
            raise AssertionError("bad bytes must raise ImageError")
        except ImageError as e:
            assert e.status_code == 400
    try:
        decode_base64("%%%not-base64%%%")
        raise AssertionError("bad base64 must raise ImageError")
    except ImageError:
        pass
    assert decode_base64("data:image/png;base64," + base64.b64encode(b"abc").decode()) == b"abc"


if __name__ == "__main__":
    test_eye_tracking()
    test_yawn_tracking()
    test_mouth_aspect_ratio()
    test_decoding()
    print("backend self-check: all assertions passed")
