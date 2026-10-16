"""Per-session drowsiness tracking: eye-closure time, PERCLOS and sustained yawns."""
import time
from collections import deque
from threading import Lock


class DrowsinessTracker:
    """Everything is measured in seconds, not frames, so it works at any frame rate.
    - drowsy_alert: eyes continuously closed for >= closed_threshold_sec
    - perclos: share of frames with closed eyes in the last `window_sec`
    - yawning: mouth continuously wide open for >= yawn_threshold_sec (a real yawn lasts 4+ s;
      talking or laughing opens the mouth for well under a second); yawn_count counts each one once."""

    def __init__(self, window_sec: float = 60.0, idle_sec: float = 600.0):
        self.window_sec, self.idle_sec = window_sec, idle_sec
        self._sessions: dict[str, dict] = {}
        # ponytail: in-memory, single process; move to Redis if the API ever runs with several workers.
        self._lock = Lock()

    def update(self, session_id: str, eyes_closed: bool | None, mouth_open: bool | None,
               closed_threshold_sec: float, yawn_threshold_sec: float, now: float | None = None) -> dict:
        now = time.monotonic() if now is None else now
        with self._lock:
            self._sessions = {k: s for k, s in self._sessions.items() if now - s["last_seen"] < self.idle_sec}
            s = self._sessions.setdefault(session_id, {"closed_since": None, "history": deque(), "last_seen": now,
                                                       "yawn_since": None, "yawn_counted": False, "yawn_count": 0})
            s["last_seen"] = now

            # Eyes. No face visible -> reset the streak rather than guess (avoids false alarms).
            if eyes_closed is None or not eyes_closed:
                s["closed_since"] = None
            elif s["closed_since"] is None:
                s["closed_since"] = now
            if eyes_closed is not None:
                s["history"].append((now, eyes_closed))
            while s["history"] and now - s["history"][0][0] > self.window_sec:
                s["history"].popleft()
            closed_for = 0.0 if s["closed_since"] is None else now - s["closed_since"]

            # Mouth. A yawn is counted once, when it first reaches the threshold.
            if not mouth_open:
                s["yawn_since"], s["yawn_counted"] = None, False
            elif s["yawn_since"] is None:
                s["yawn_since"] = now
            yawn_for = 0.0 if s["yawn_since"] is None else now - s["yawn_since"]
            yawning = yawn_for >= yawn_threshold_sec
            if yawning and not s["yawn_counted"]:
                s["yawn_count"] += 1
                s["yawn_counted"] = True

            closed_flags = [closed for _, closed in s["history"]]
            return {
                "closed_duration_sec": round(closed_for, 2),
                "perclos": round(sum(closed_flags) / len(closed_flags), 3) if closed_flags else 0.0,
                "drowsy_alert": closed_for >= closed_threshold_sec,
                "yawn_duration_sec": round(yawn_for, 2),
                "yawning": yawning,
                "yawn_count": s["yawn_count"],
            }
