from dataclasses import dataclass
from itertools import pairwise
from typing import ClassVar

import cv2
import numpy as np

type Point = tuple[int, int]

_SPREAD_THRESHOLD = 0.35


@dataclass(slots=True)
class FingerWeb:
    panels: list[list[Point]]
    combined_mask: np.ndarray
    links: list[tuple[Point, Point]]
    fingertips: list[Point]

    @property
    def visible(self) -> bool:
        return len(self.panels) >= 1 and cv2.countNonZero(self.combined_mask) > 32


class FingerWebTracker:
    def __init__(self) -> None:
        self._prev_panels: list[list[Point]] | None = None

    def update(
        self, shape: tuple[int, int], fingertips: list[Point], spread: float
    ) -> FingerWeb:
        links = list(pairwise(sorted(fingertips, key=lambda p: p[0])))

        if len(fingertips) < 2 or spread < _SPREAD_THRESHOLD:
            return FingerWeb([], np.zeros(shape, dtype=np.uint8), links, fingertips)

        panels = [self._sub_panel(a, b) for a, b in links]

        if self._prev_panels and len(self._prev_panels) == len(panels):
            panels = [
                [
                    (
                        int(pc[0] * 0.45 + c[0] * 0.55),
                        int(pc[1] * 0.45 + c[1] * 0.55),
                    )
                    for pc, c in zip(pp, p, strict=True)
                ]
                for pp, p in zip(self._prev_panels, panels, strict=True)
            ]

        mask = np.zeros(shape, dtype=np.uint8)
        for corners in panels:
            cv2.fillConvexPoly(
                mask, np.array(corners, dtype=np.int32), 255, cv2.LINE_AA
            )

        self._prev_panels = panels
        return FingerWeb(panels, mask, links, fingertips)

    @staticmethod
    def _sub_panel(a: Point, b: Point) -> list[Point]:
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        length = max(float(np.hypot(dx, dy)), 1.0)
        height = max(40.0, length * 0.5)

        nx, ny = -dy / length, dx / length
        if ny < 0:
            nx, ny = -nx, -ny

        return [
            a,
            b,
            (int(bx + nx * height), int(by + ny * height)),
            (int(ax + nx * height), int(ay + ny * height)),
        ]


class FingerWebEffect:
    _PALETTES: ClassVar[
        dict[
            str,
            tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
        ]
    ] = {
        "arcane": ((60, 16, 8), (255, 220, 80), (255, 66, 210)),
        "plasma": ((50, 8, 40), (255, 120, 235), (255, 225, 120)),
        "matrix": ((16, 55, 4), (130, 255, 82), (225, 255, 185)),
        "cosmic": ((6, 10, 35), (80, 180, 255), (200, 128, 255)),
        "frost": ((10, 40, 70), (140, 230, 255), (200, 240, 255)),
    }

    def __init__(self, style: str = "arcane") -> None:
        self._style = style
        self._phase = 0.0

    def apply(self, frame: np.ndarray, web: FingerWeb) -> np.ndarray:
        if not web.visible:
            return frame.copy()

        self._phase += 0.09
        base, line, accent = self._PALETTES[self._style]

        soft = cv2.GaussianBlur(web.combined_mask.astype(np.float32), (0, 0), 8) / 255.0

        h, w = frame.shape[:2]
        fill = np.zeros_like(frame)
        fill[:] = base

        grid = np.zeros_like(frame)
        offset = int((self._phase * 42) % 28)
        for x in range(-h + offset, w + h, 28):
            cv2.line(grid, (x, 0), (x + h, h), line, 1, cv2.LINE_AA)

        blend = cv2.addWeighted(fill, 0.55, grid, 0.45, 0)
        result = _composite(frame, blend, soft * 0.75)

        for corners in web.panels:
            pts = np.array(corners, dtype=np.int32)
            cv2.polylines(result, [pts], True, accent, 2, cv2.LINE_AA)

        for start, end in web.links:
            cv2.line(result, start, end, line, 3, cv2.LINE_AA)

        glow = np.zeros_like(frame)
        for x, y in web.fingertips:
            cv2.circle(glow, (x, y), 14, accent, -1, cv2.LINE_AA)
            cv2.circle(glow, (x, y), 22, accent, -1, cv2.LINE_AA)
        glow_blur = cv2.GaussianBlur(glow, (0, 0), 12)
        result = _add_light(result, glow_blur, 0.45)
        result = _add_light(result, glow, 0.35)

        return result


def _composite(base: np.ndarray, layer: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    mixed = (
        base.astype(np.float32) * (1.0 - alpha[..., None])
        + layer.astype(np.float32) * alpha[..., None]
    )
    return np.clip(mixed, 0, 255).astype(np.uint8)


def _add_light(base: np.ndarray, light: np.ndarray, strength: float) -> np.ndarray:
    return np.clip(
        base.astype(np.float32) + light.astype(np.float32) * strength, 0, 255
    ).astype(np.uint8)
