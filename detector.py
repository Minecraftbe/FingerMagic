from dataclasses import dataclass
from pathlib import Path
from typing import Any
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
_DETECT_MAX_WIDTH = 320
_DETECT_EVERY_N = 2

type Point = tuple[int, int]


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
    spread: float

    @property
    def found(self) -> bool:
        return len(self.fingertips) >= 2


class HandDetector:
    def __init__(self) -> None:
        model_path = _ensure_model()
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._frame_idx = 0
        self._cached: list[HandObservation] = []

    def detect(self, frame: np.ndarray) -> list[HandObservation]:
        self._frame_idx += 1
        if self._frame_idx % _DETECT_EVERY_N != 0 and self._cached:
            return self._cached

        h, w = frame.shape[:2]

        if w > _DETECT_MAX_WIDTH:
            scale = _DETECT_MAX_WIDTH / w
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small = frame

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        hands: list[HandObservation] = []
        if result.hand_landmarks:
            for landmarks in result.hand_landmarks:
                fingertips: list[Point] = []
                for i in _FINGERTIP_INDICES:
                    lm = landmarks[i]
                    fingertips.append((int(lm.x * w), int(lm.y * h)))
                spread = _compute_spread(landmarks, _FINGERTIP_INDICES)
                hands.append(HandObservation(fingertips, spread))

        if hands:
            self._cached = hands
            return hands
        return []


def _compute_spread(landmarks: list[Any], finger_indices: tuple[int, ...]) -> float:
    wrist = landmarks[0]
    middle_tip = landmarks[12]
    hand_size = float(np.hypot(wrist.x - middle_tip.x, wrist.y - middle_tip.y))
    if hand_size < 0.02:
        return 0.0
    xs = [landmarks[i].x for i in finger_indices]
    return (max(xs) - min(xs)) / hand_size
