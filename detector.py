from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_FINGERTIP_INDICES: tuple[int, ...] = (4, 8, 12, 16, 20)
_MAX_MISSED_FRAMES = 10
_DETECT_MAX_WIDTH = 400

type Point = tuple[int, int]
type Point3D = tuple[float, float, float]


def _ensure_model() -> str:
    cache_dir = Path.home() / ".fingermagic" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "hand_landmarker.task"
    if not model_path.is_file():
        print(f"Downloading hand landmarker model to {model_path} ...")
        urlretrieve(_MODEL_URL, str(model_path))
        print("Done.")
    return str(model_path)


@dataclass(slots=True)
class HandObservation:
    fingertips: list[Point]
    fingertips_3d: list[Point3D]
    spread_cm: float
    contour: np.ndarray | None

    @property
    def found(self) -> bool:
        return self.contour is not None


def _compute_spread(world_pts: list[Point3D]) -> float:
    """Return min 3D distance (cm) between adjacent x-sorted fingertips."""
    if len(world_pts) < 2:
        return 0.0
    ordered = sorted(world_pts, key=lambda p: p[0])
    dists: list[float] = []
    for a, b in pairwise(ordered):
        dists.append(float(np.linalg.norm(np.array(a) - np.array(b))))
    return min(dists)


class HandDetector:
    def __init__(self) -> None:
        model_path = _ensure_model()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._prev_fingertips: list[Point] = []
        self._prev_contour: np.ndarray | None = None
        self._missed_frames = 0
        self._smooth_alpha = 0.5
        self._frame_idx = 0

    def detect(self, frame: np.ndarray) -> HandObservation:
        self._frame_idx += 1
        h, w = frame.shape[:2]

        scale = 1.0
        if w > _DETECT_MAX_WIDTH:
            scale = _DETECT_MAX_WIDTH / w
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small = frame

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, self._frame_idx * 33)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            world_landmarks = (
                result.hand_world_landmarks[0] if result.hand_world_landmarks else []
            )

            raw_fingertips: list[Point] = []
            world_fingertips: list[Point3D] = []
            all_pts: list[Point] = []

            for i, lm in enumerate(landmarks):
                pt = (int(lm.x * w), int(lm.y * h))
                all_pts.append(pt)
                if i in _FINGERTIP_INDICES:
                    raw_fingertips.append(pt)
                    if world_landmarks and i < len(world_landmarks):
                        wlm = world_landmarks[i]
                        world_fingertips.append(
                            (float(wlm.x), float(wlm.y), float(wlm.z))
                        )

            spread = (
                _compute_spread(world_fingertips)
                if world_fingertips
                else float(np.std([p[0] for p in raw_fingertips]) * 0.05)
            )

            hull = cv2.convexHull(np.array(all_pts, dtype=np.int32))

            fingertips = self._smooth(raw_fingertips)
            self._prev_fingertips = fingertips
            self._prev_contour = hull
            self._missed_frames = 0
            return HandObservation(fingertips, world_fingertips, spread, hull)

        return self._recover_or_empty()

    def _recover_or_empty(self) -> HandObservation:
        self._missed_frames += 1
        if self._prev_contour is not None and self._missed_frames <= _MAX_MISSED_FRAMES:
            return HandObservation(self._prev_fingertips, [], 0.0, self._prev_contour)
        self._prev_contour = None
        self._prev_fingertips = []
        return HandObservation([], [], 0.0, None)

    def _smooth(self, current: list[Point]) -> list[Point]:
        if not current:
            return []
        if not self._prev_fingertips or len(self._prev_fingertips) != len(current):
            self._prev_fingertips = current
            return current

        alpha = self._smooth_alpha
        smoothed: list[Point] = []
        for (px, py), (cx, cy) in zip(self._prev_fingertips, current, strict=True):
            smoothed.append((int(px + alpha * (cx - px)), int(py + alpha * (cy - py))))
        self._prev_fingertips = smoothed
        return smoothed
