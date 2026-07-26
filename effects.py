from abc import ABC, abstractmethod
from typing import TypedDict, cast

import cv2
import numpy as np

# ============================================================
# Polygon helpers
# ============================================================


def polygon_mask(shape: tuple[int, int], points: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if len(points) >= 3:
        pts = np.array(points, dtype=np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillPoly(mask, [hull], 255)
    return mask


def convex_hull_of(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(points) < 3:
        return points
    pts = np.array(points, dtype=np.int32)
    hull = cv2.convexHull(pts)
    return [tuple(h[0]) for h in hull]


# ============================================================
# Visual Effects
# ============================================================


class BaseEffect(ABC):
    @abstractmethod
    def apply(
        self, frame: np.ndarray, arg: np.ndarray | list[tuple[int, int]]
    ) -> np.ndarray: ...


class StarfieldEffect(BaseEffect):
    """Drifting starfield inside the polygon."""

    def __init__(self, num_stars: int = 250) -> None:
        self.num_stars = num_stars
        self._stars: np.ndarray | None = None
        self._phases: np.ndarray | None = None
        self._last_shape: tuple[int, int] = (0, 0)

    def apply(
        self, frame: np.ndarray, arg: np.ndarray | list[tuple[int, int]]
    ) -> np.ndarray:
        mask = cast(np.ndarray, arg)
        h, w = frame.shape[:2]
        result = frame.copy()
        ys, _xs = np.where(mask > 0)
        if len(ys) < 10:
            return result

        if self._stars is None or (w, h) != self._last_shape:
            self._init(w, h)

        stars = self._stars
        phases = self._phases
        assert stars is not None
        assert phases is not None

        phases += np.random.uniform(0.04, 0.18, self.num_stars)
        stars[:, 0] += np.random.uniform(-0.6, 0.6, self.num_stars)
        stars[:, 1] += np.random.uniform(-0.6, 0.4, self.num_stars)
        self._phases = phases
        self._stars = stars

        brightness = (np.sin(phases) + 1.0) / 2.0

        for i in range(self.num_stars):
            x = int(stars[i, 0])
            y = int(stars[i, 1])
            if 0 <= x < w and 0 <= y < h and mask[y, x]:
                b = float(brightness[i])
                color = (int(180 * b + 75), int(200 * b + 55), int(255 * b))
                cv2.circle(result, (x, y), max(1, int(2.5 * b)), color, -1)
                if b > 0.75:
                    cv2.circle(result, (x, y), 5, (255, 255, 220), 1)

        return result

    def _init(self, w: int, h: int) -> None:
        self._stars = np.column_stack(
            [
                np.random.randint(0, w, self.num_stars),
                np.random.randint(0, h, self.num_stars),
            ]
        ).astype(np.float64)
        self._phases = np.random.uniform(0, 2 * np.pi, self.num_stars)
        self._last_shape = (w, h)


class RainbowEffect(BaseEffect):
    """HSV hue-cycling inside the polygon."""

    def __init__(self) -> None:
        self._offset = 0

    def apply(
        self, frame: np.ndarray, arg: np.ndarray | list[tuple[int, int]]
    ) -> np.ndarray:
        mask = cast(np.ndarray, arg)
        self._offset = (self._offset + 3) % 180
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int16)
        roi = mask > 0
        hsv[roi, 0] = (hsv[roi, 0] + self._offset) % 180
        hsv[roi, 1] = np.clip(hsv[roi, 1] + 40, 0, 255)
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class _Particle(TypedDict):
    x: float
    y: float
    vx: float
    vy: float
    life: float
    hue: int


class ParticleEffect(BaseEffect):
    """Sparkles emanating from fingertips."""

    def __init__(self) -> None:
        self._particles: list[_Particle] = []

    def apply(
        self, frame: np.ndarray, arg: np.ndarray | list[tuple[int, int]]
    ) -> np.ndarray:
        return self._do_apply(frame, cast(list[tuple[int, int]], arg))

    def _do_apply(
        self, frame: np.ndarray, fingertips: list[tuple[int, int]]
    ) -> np.ndarray:
        result = frame.copy()
        for fx, fy in fingertips:
            for _ in range(2):
                angle = np.random.uniform(0, 2 * np.pi)
                speed = np.random.uniform(1.5, 5)
                self._particles.append(
                    _Particle(
                        x=float(fx),
                        y=float(fy),
                        vx=np.cos(angle) * speed,
                        vy=np.sin(angle) * speed,
                        life=np.random.uniform(0.5, 1.2),
                        hue=np.random.randint(0, 180),
                    )
                )

        alive: list[_Particle] = []
        h, w = frame.shape[:2]
        for p in self._particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vy"] += 0.1
            p["life"] -= 0.018
            if p["life"] <= 0:
                continue
            x, y = int(p["x"]), int(p["y"])
            if 0 <= x < w and 0 <= y < h:
                alpha = min(float(p["life"]), 1.0)
                color_hsv = np.array(
                    [[[p["hue"], 200, int(255 * alpha)]]], dtype=np.uint8
                )
                color_bgr = tuple(
                    int(c) for c in cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]
                )
                radius = int(3 * alpha) + 1
                cv2.circle(result, (x, y), radius, color_bgr, -1)
                alive.append(p)

        self._particles = alive[-300:] if len(alive) > 300 else alive
        return result


class NeonGlowEffect(BaseEffect):
    """Glowing neon edges for the fingertip polygon."""

    def apply(
        self, frame: np.ndarray, arg: np.ndarray | list[tuple[int, int]]
    ) -> np.ndarray:
        points = cast(list[tuple[int, int]], arg)
        result = frame.copy()
        hull = convex_hull_of(points)
        if len(hull) < 2:
            return result
        overlay = result.copy()
        for i in range(len(hull)):
            a = hull[i]
            b = hull[(i + 1) % len(hull)]
            cv2.line(overlay, a, b, (255, 100, 0), 9)
            cv2.line(overlay, a, b, (0, 200, 255), 5)
            cv2.line(overlay, a, b, (180, 255, 255), 2)
        cv2.addWeighted(overlay, 0.6, result, 0.4, 0, dst=result)
        return result


class MagicPortalEffect(BaseEffect):
    """Darkened surroundings + glowing polygon interior with rotating gradient."""

    def __init__(self) -> None:
        self._angle = 0.0

    def apply(
        self, frame: np.ndarray, arg: np.ndarray | list[tuple[int, int]]
    ) -> np.ndarray:
        mask = cast(np.ndarray, arg)
        self._angle += 0.03
        h, w = frame.shape[:2]

        dark = (frame.astype(np.float32) * 0.25).astype(np.uint8)
        interior = frame.copy()
        result = np.where(mask[..., None] > 0, interior, dark)

        cx, cy = w // 2, h // 2
        Y, X = np.ogrid[:h, :w]
        dx = X - cx
        dy = Y - cy
        angle_map = (np.arctan2(dy, dx) + self._angle) % (2 * np.pi)
        hue = (angle_map / (2 * np.pi) * 180).astype(np.uint8)
        sat = np.full((h, w), 200, dtype=np.uint8)
        val = np.full((h, w), 255, dtype=np.uint8)
        gradient_hsv = np.stack([hue, sat, val], axis=-1)
        gradient_bgr = cv2.cvtColor(gradient_hsv, cv2.COLOR_HSV2BGR)

        alpha = 0.35
        roi = mask > 0
        result[roi] = cv2.addWeighted(
            result[roi], 1 - alpha, gradient_bgr[roi], alpha, 0
        )

        return result
