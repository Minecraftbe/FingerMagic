from dataclasses import dataclass
from itertools import pairwise
from typing import ClassVar

import cv2
import numpy as np

type Point = tuple[int, int]

_SPREAD_THRESHOLD_CM = 2.8


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
        self._prev_smoothed: np.ndarray | None = None

    def update(
        self, shape: tuple[int, int], fingertips: list[Point], spread_cm: float
    ) -> FingerWeb:
        links = list(pairwise(sorted(fingertips, key=lambda p: p[0])))

        if len(fingertips) < 2 or spread_cm < _SPREAD_THRESHOLD_CM:
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
        "arcane": ((122, 32, 12), (255, 220, 80), (255, 66, 210)),
        "plasma": ((90, 12, 70), (255, 120, 235), (255, 225, 120)),
        "matrix": ((32, 88, 8), (130, 255, 82), (225, 255, 185)),
        "cosmic": ((8, 12, 42), (80, 180, 255), (200, 128, 255)),
        "frost": ((12, 48, 80), (140, 230, 255), (200, 240, 255)),
    }

    def __init__(self, style: str = "arcane") -> None:
        self._style = style
        self._phase = 0.0

    def apply(self, frame: np.ndarray, web: FingerWeb) -> np.ndarray:
        result = frame.copy()
        if not web.visible:
            return result

        self._phase += 0.09
        base, line, accent = self._PALETTES[self._style]
        result = self._draw_web_fill(result, web, base, line)

        for corners in web.panels:
            pts = np.array(corners, dtype=np.int32)
            cv2.polylines(result, [pts], True, accent, 2, cv2.LINE_AA)

        for start, end in web.links:
            cv2.line(result, start, end, line, 3, cv2.LINE_AA)

        for x, y in web.fingertips:
            cv2.circle(result, (x, y), 7, accent, -1, cv2.LINE_AA)

        if web.fingertips:
            cv2.line(
                result,
                web.fingertips[-1],
                web.links[-1][1] if web.links else web.fingertips[-1],
                line,
                3,
                cv2.LINE_AA,
            )

        return result

    def _draw_web_fill(
        self,
        frame: np.ndarray,
        web: FingerWeb,
        base: tuple[int, int, int],
        line: tuple[int, int, int],
    ) -> np.ndarray:
        soft = cv2.GaussianBlur(web.combined_mask.astype(np.float32), (0, 0), 6) / 255.0

        grid = frame.copy()
        offset = int((self._phase * 42) % 28)
        for x in range(-grid.shape[0] + offset, grid.shape[1] + grid.shape[0], 28):
            cv2.line(
                grid,
                (x, 0),
                (x + grid.shape[0], grid.shape[0]),
                line,
                1,
                cv2.LINE_AA,
            )

        fill = np.empty_like(frame)
        fill[:] = base
        blend = cv2.addWeighted(fill, 0.25, grid, 0.28, 0)
        return self._composite(frame, blend, soft * 0.78)

    @staticmethod
    def _composite(
        base: np.ndarray, layer: np.ndarray, alpha: np.ndarray
    ) -> np.ndarray:
        mixed = (
            base.astype(np.float32) * (1 - alpha[..., None])
            + layer.astype(np.float32) * alpha[..., None]
        )
        return np.clip(mixed, 0, 255).astype(np.uint8)
