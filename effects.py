from dataclasses import dataclass
from itertools import pairwise
from typing import ClassVar

import cv2
import numpy as np

type Point = tuple[int, int]

_SPREAD_THRESHOLD = 0.35
_BAND_COUNT = 5
_BAND_HEIGHT = 6


@dataclass(slots=True)
class FingerWeb:
    panels: list[tuple[int, int, int, int]]
    mask: np.ndarray
    links: list[tuple[Point, Point]]
    fingertips: list[Point]

    @property
    def visible(self) -> bool:
        return len(self.panels) >= 1 and cv2.countNonZero(self.mask) > 16


class FingerWebTracker:
    def __init__(self) -> None:
        self._prev: dict[int, list[list[Point]]] = {}

    def update_all(
        self,
        shape: tuple[int, int],
        hands: list[tuple[list[Point], float]],
    ) -> list[FingerWeb]:
        webs: list[FingerWeb] = []
        seen: set[int] = set()

        for idx, (fingertips, spread) in enumerate(hands):
            seen.add(idx)
            links = list(pairwise(sorted(fingertips, key=lambda p: p[0])))

            if len(fingertips) < 2 or spread < _SPREAD_THRESHOLD:
                webs.append(
                    FingerWeb([], np.zeros(shape, dtype=np.uint8), links, fingertips)
                )
                continue

            panels = self._build_panels(links)
            prev_panels = self._prev.get(idx)
            if prev_panels and len(prev_panels) == len(panels):
                panels = self._smooth_panels(prev_panels, panels)

            mask = np.zeros(shape, dtype=np.uint8)
            for x0, y0, x1, y1 in panels:
                cv2.rectangle(mask, (x0, y0), (x1, y1), 255, cv2.FILLED)

            self._prev[idx] = [
                [(x0, y0), (x1, y0), (x1, y1), (x0, y1)] for x0, y0, x1, y1 in panels
            ]
            webs.append(FingerWeb(panels, mask, links, fingertips))

        removed = [k for k in self._prev if k not in seen]
        for k in removed:
            del self._prev[k]

        return webs

    @staticmethod
    def _build_panels(
        links: list[tuple[Point, Point]],
    ) -> list[tuple[int, int, int, int]]:
        result: list[tuple[int, int, int, int]] = []
        for (ax, ay), (bx, by) in links:
            dx, dy = bx - ax, by - ay
            length = max(float(np.hypot(dx, dy)), 1.0)
            panel_h = max(55.0, length * 0.25)
            band_h = max(4, int(panel_h / _BAND_COUNT))
            gap = max(
                2, (int(panel_h) - band_h * _BAND_COUNT) // max(_BAND_COUNT - 1, 1)
            )

            nx, ny = -dy / length, dx / length
            if ny < 0:
                nx, ny = -nx, -ny

            for band_i in range(_BAND_COUNT):
                oy = int((band_h + gap) * band_i)
                b0 = (int(ax + nx * oy), int(ay + ny * oy))
                b1 = (int(bx + nx * oy), int(by + ny * oy))
                b3 = (int(ax + nx * (oy + band_h)), int(ay + ny * (oy + band_h)))
                b2 = (int(bx + nx * (oy + band_h)), int(by + ny * (oy + band_h)))
                x0 = min(b0[0], b1[0], b2[0], b3[0])
                y0 = min(b0[1], b1[1], b2[1], b3[1])
                x1 = max(b0[0], b1[0], b2[0], b3[0])
                y1 = max(b0[1], b1[1], b2[1], b3[1])
                if x1 > x0 and y1 > y0:
                    result.append((x0, y0, x1, y1))
        return result

    @staticmethod
    def _smooth_panels(
        prev: list[list[Point]],
        current: list[tuple[int, int, int, int]],
    ) -> list[tuple[int, int, int, int]]:
        smoothed: list[tuple[int, int, int, int]] = []
        for pp, (x0, y0, x1, y1) in zip(prev, current, strict=True):
            pc0, pc1, _, _ = pp
            sx0 = int(pc0[0] * 0.45 + x0 * 0.55)
            sy0 = int(pc0[1] * 0.45 + y0 * 0.55)
            sx1 = int(pc1[0] * 0.45 + x1 * 0.55)
            sy1 = int(pc1[1] * 0.45 + y1 * 0.55)
            smoothed.append((sx0, sy0, sx1, sy1))
        return smoothed


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

    def apply(self, frame: np.ndarray, webs: list[FingerWeb]) -> np.ndarray:
        self._phase += 0.09
        base, line, accent = self._PALETTES[self._style]
        result = frame

        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for web in webs:
            if web.visible:
                cv2.add(combined_mask, web.mask, dst=combined_mask)

        if cv2.countNonZero(combined_mask) == 0:
            return frame.copy()

        soft = cv2.GaussianBlur(combined_mask.astype(np.float32), (0, 0), 4) / 255.0

        fill = np.empty_like(frame)
        fill[:] = base

        blend = cv2.addWeighted(result, 0.78, fill, 0.22, 0)
        result = _composite(blend, fill, soft * 0.7)

        for web in webs:
            if not web.visible:
                continue

            for x0, y0, x1, y1 in web.panels:
                pulse = 0.5 + 0.5 * np.sin(self._phase * 3.0 + x0 * 0.01)
                c = (
                    int(line[0] * pulse + accent[0] * (1.0 - pulse)),
                    int(line[1] * pulse + accent[1] * (1.0 - pulse)),
                    int(line[2] * pulse + accent[2] * (1.0 - pulse)),
                )
                cv2.rectangle(result, (x0, y0), (x1, y1), c, cv2.FILLED)

                band_glow = np.zeros_like(result)
                cv2.rectangle(band_glow, (x0, y0), (x1, y1), c, cv2.FILLED)
                result = _add_light(
                    result, cv2.GaussianBlur(band_glow, (0, 0), 5), 0.25
                )

            for start, end in web.links:
                cv2.line(result, start, end, line, 3, cv2.LINE_AA)
                cv2.line(
                    result,
                    (start[0], start[1] - 1),
                    (end[0], end[1] - 1),
                    accent,
                    1,
                    cv2.LINE_AA,
                )

        for web in webs:
            for x, y in web.fingertips:
                cv2.circle(result, (x, y), 6, accent, -1, cv2.LINE_AA)
                cv2.circle(result, (x, y), 10, line, 1, cv2.LINE_AA)

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
