from itertools import pairwise
from typing import ClassVar

import cv2
import numpy as np

type Point = tuple[int, int]

_SPREAD_THRESHOLD = 0.32


class CyberEffect:
    _PALETTES: ClassVar[
        dict[
            str,
            tuple[
                tuple[int, int, int],
                tuple[int, int, int],
                tuple[int, int, int],
            ],
        ]
    ] = {
        "arcane": ((255, 140, 30), (255, 60, 180), (255, 220, 40)),
        "plasma": ((255, 80, 200), (180, 40, 255), (255, 200, 60)),
        "matrix": ((50, 255, 100), (30, 200, 60), (180, 255, 120)),
        "cosmic": ((80, 180, 255), (140, 80, 255), (180, 255, 255)),
        "frost": ((100, 220, 255), (60, 180, 255), (200, 240, 255)),
        "neon": ((255, 30, 120), (255, 200, 0), (255, 60, 200)),
    }

    def __init__(self, style: str = "arcane") -> None:
        self._style = style
        self._phase = 0.0

    def apply(self, frame: np.ndarray, hands: list[list[Point]]) -> np.ndarray:
        self._phase += 0.1
        primary, secondary, accent = self._PALETTES[self._style]

        glow = np.zeros_like(frame)
        active_hands: list[list[Point]] = []

        for fingertips in hands:
            if len(fingertips) < 2:
                continue
            ordered = sorted(fingertips, key=lambda p: p[0])
            active_hands.append(ordered)

            pulse = 0.6 + 0.4 * np.sin(self._phase * 2.5)

            for a, b in pairwise(ordered):
                mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
                cv2.line(glow, a, b, primary, 3, cv2.LINE_AA)
                cv2.line(
                    glow,
                    a,
                    b,
                    (
                        int(primary[0] * pulse),
                        int(primary[1] * pulse),
                        int(primary[2] * pulse),
                    ),
                    1,
                    cv2.LINE_AA,
                )
                cv2.circle(glow, mid, 3, secondary, -1, cv2.LINE_AA)

            for i, (x, y) in enumerate(ordered):
                r = int(5 + 3 * np.sin(self._phase * 4.0 + i))
                cv2.circle(glow, (x, y), r, accent, -1, cv2.LINE_AA)
                cv2.circle(glow, (x, y), int(r * 2.4), secondary, 1, cv2.LINE_AA)

        if not active_hands:
            return frame.copy()

        blurred = cv2.GaussianBlur(glow, (0, 0), 6)
        result = cv2.addWeighted(frame, 1.0, blurred, 0.65, 0)
        result = cv2.addWeighted(result, 1.0, glow, 0.45, 0)

        h, w = frame.shape[:2]
        scan = np.zeros_like(result, dtype=np.uint8)
        offset = int(self._phase * 60 % 8)
        for y in range(offset, h, 8):
            cv2.line(scan, (0, y), (w, y), (40, 40, 60), 1)
        result = cv2.addWeighted(result, 1.0, scan, 0.08, 0)

        return result
