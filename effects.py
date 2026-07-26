from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeGuard

import cv2
import numpy as np

type Point = tuple[int, int]
type EffectArgument = np.ndarray | list[Point]


# ============================================================
# Mask and compositing helpers
# ============================================================


def _is_mask(arg: EffectArgument) -> TypeGuard[np.ndarray]:
    return isinstance(arg, np.ndarray)


def _is_fingertip_list(arg: EffectArgument) -> TypeGuard[list[Point]]:
    return isinstance(arg, list)


def polygon_mask(shape: tuple[int, int], points: list[Point]) -> np.ndarray:
    """Compatibility fallback for callers that only have fingertip positions."""
    mask = np.zeros(shape, dtype=np.uint8)
    if len(points) >= 3:
        hull = cv2.convexHull(np.array(points, dtype=np.int32))
        cv2.fillPoly(mask, [hull], 255)
    return mask


def convex_hull_of(points: list[Point]) -> list[Point]:
    if len(points) < 3:
        return points
    hull = cv2.convexHull(np.array(points, dtype=np.int32))
    return [_point_from_array(point[0]) for point in hull]


def _has_area(mask: np.ndarray) -> bool:
    return cv2.countNonZero(mask) >= 96


def _soft_mask(mask: np.ndarray, radius: float = 9.0) -> np.ndarray:
    blurred = cv2.GaussianBlur(mask, (0, 0), radius)
    return blurred.astype(np.float32) / 255.0


def _blend(
    base: np.ndarray, layer: np.ndarray, alpha: np.ndarray | float
) -> np.ndarray:
    normalized_alpha = alpha
    if isinstance(normalized_alpha, np.ndarray) and normalized_alpha.ndim == 2:
        normalized_alpha = np.expand_dims(normalized_alpha, axis=2)
    mixed = (
        base.astype(np.float32) * (1.0 - normalized_alpha)
        + layer.astype(np.float32) * normalized_alpha
    )
    return np.clip(mixed, 0, 255).astype(np.uint8)


def _add_light(
    base: np.ndarray, light: np.ndarray, strength: float = 1.0
) -> np.ndarray:
    return np.clip(
        base.astype(np.float32) + light.astype(np.float32) * strength, 0, 255
    ).astype(np.uint8)


def _mask_center(mask: np.ndarray) -> Point:
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        return mask.shape[1] // 2, mask.shape[0] // 2
    return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])


def _outline(mask: np.ndarray, width: int = 3) -> np.ndarray:
    size = width * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    outer = cv2.dilate(mask, kernel)
    inner = cv2.erode(mask, kernel)
    return cv2.subtract(outer, inner)


def _outline_glow(
    frame: np.ndarray,
    mask: np.ndarray,
    outer_color: tuple[int, int, int],
    core_color: tuple[int, int, int],
    pulse: float = 1.0,
) -> np.ndarray:
    edge = _outline(mask)
    glow = cv2.GaussianBlur(edge, (0, 0), 13)
    glow_layer = np.empty_like(frame)
    glow_layer[:] = outer_color
    result = _blend(frame, glow_layer, glow.astype(np.float32) / 255.0 * 0.48 * pulse)
    core_layer = np.empty_like(frame)
    core_layer[:] = core_color
    return _blend(result, core_layer, edge.astype(np.float32) / 255.0 * 0.78)


def _hsv_color(
    hue: int, saturation: int = 190, value: int = 255
) -> tuple[int, int, int]:
    pixel = np.array([[[hue % 180, saturation, value]]], dtype=np.uint8)
    bgr = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _point_from_array(values: np.ndarray) -> Point:
    return int(values[0]), int(values[1])


# ============================================================
# Visual effects
# ============================================================


class BaseEffect(ABC):
    @abstractmethod
    def apply(self, frame: np.ndarray, arg: EffectArgument) -> np.ndarray: ...


class StarfieldEffect(BaseEffect):
    """A drifting nebula and twinkling stars contained by the hand silhouette."""

    def __init__(self, num_stars: int = 110) -> None:
        self._num_stars = num_stars
        self._rng = np.random.default_rng(2026)
        self._stars = self._rng.random((num_stars, 4))
        self._time = 0.0

    def apply(self, frame: np.ndarray, arg: EffectArgument) -> np.ndarray:
        if not _is_mask(arg):
            return frame.copy()
        mask = arg
        if not _has_area(mask):
            return frame.copy()
        self._time += 0.045
        soft = _soft_mask(mask, 10)
        height, width = frame.shape[:2]
        yy, xx = np.mgrid[:height, :width]
        cx, cy = _mask_center(mask)
        wave = np.sin((xx - cx) * 0.035 + self._time) + np.cos(
            (yy - cy) * 0.045 - self._time * 0.8
        )
        nebula = np.empty_like(frame)
        nebula[..., 0] = np.clip(102 + 35 * wave, 35, 185)
        nebula[..., 1] = np.clip(24 + 16 * wave, 5, 88)
        nebula[..., 2] = np.clip(92 - 25 * wave, 28, 172)
        cosmic = _blend(frame, nebula, 0.76)
        result = _blend(frame, cosmic, soft * 0.84)

        rows, columns = np.where(mask > 0)
        if len(rows) == 0:
            return result
        x = int(columns.min())
        y = int(rows.min())
        box_width = int(columns.max()) - x + 1
        box_height = int(rows.max()) - y + 1
        self._stars[:, 1] = (self._stars[:, 1] - 0.0018) % 1.0
        self._stars[:, 3] += 0.08
        glow = np.zeros_like(frame)
        cores = np.zeros_like(frame)
        for normalized_x, normalized_y, size, phase in self._stars:
            star_x = int(x + normalized_x * max(box_width - 1, 1))
            star_y = int(y + normalized_y * max(box_height - 1, 1))
            if not (
                0 <= star_x < width and 0 <= star_y < height and mask[star_y, star_x]
            ):
                continue
            brightness = 0.38 + 0.62 * (np.sin(phase + self._time * 2.2) + 1.0) / 2.0
            radius = 1 + int(size * 2.4 * brightness)
            color = _hsv_color(102 + int(size * 34), 110, int(180 + 75 * brightness))
            cv2.circle(glow, (star_x, star_y), radius + 3, color, -1, cv2.LINE_AA)
            cv2.circle(
                cores, (star_x, star_y), radius, (255, 245, 230), -1, cv2.LINE_AA
            )
            if brightness > 0.9:
                cv2.line(
                    cores,
                    (star_x - 4, star_y),
                    (star_x + 4, star_y),
                    color,
                    1,
                    cv2.LINE_AA,
                )
                cv2.line(
                    cores,
                    (star_x, star_y - 4),
                    (star_x, star_y + 4),
                    color,
                    1,
                    cv2.LINE_AA,
                )
        result = _add_light(result, cv2.GaussianBlur(glow, (0, 0), 5), 0.55)
        result = _add_light(result, cores, 0.92)
        return _outline_glow(result, mask, (215, 68, 122), (255, 184, 168), 0.9)


class RainbowEffect(BaseEffect):
    """A smooth, pearlescent colour flow that retains the hand's texture."""

    def __init__(self) -> None:
        self._phase = 0.0

    def apply(self, frame: np.ndarray, arg: EffectArgument) -> np.ndarray:
        if not _is_mask(arg):
            return frame.copy()
        mask = arg
        if not _has_area(mask):
            return frame.copy()
        self._phase = (self._phase + 2.4) % 180
        height, width = frame.shape[:2]
        yy, xx = np.mgrid[:height, :width]
        ripple = 18 * np.sin(xx * 0.021 + yy * 0.012 + self._phase * 0.11)
        hue = np.mod(xx * 0.11 - yy * 0.055 + ripple + self._phase, 180).astype(
            np.uint8
        )
        saturation = np.full((height, width), 185, dtype=np.uint8)
        value = np.clip(
            220 + 26 * np.sin(yy * 0.035 - self._phase * 0.18), 0, 255
        ).astype(np.uint8)
        spectrum = cv2.cvtColor(np.dstack((hue, saturation, value)), cv2.COLOR_HSV2BGR)
        pearlescent = _blend(frame, spectrum, 0.66)
        soft = _soft_mask(mask, 9)
        result = _blend(frame, pearlescent, soft * 0.7)

        shimmer = np.maximum(0, np.sin((xx + yy * 0.42) * 0.075 - self._phase * 0.35))
        white = np.full_like(frame, 255)
        result = _blend(result, white, soft * shimmer * 0.14)
        edge_hue = int(self._phase) % 180
        return _outline_glow(
            result,
            mask,
            _hsv_color(edge_hue + 72, 200, 245),
            _hsv_color(edge_hue, 120, 255),
            0.82,
        )


@dataclass(slots=True)
class _Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    hue: int


class ParticleEffect(BaseEffect):
    """Soft, luminous sparks that trail from detected fingertips."""

    def __init__(self) -> None:
        self._particles: list[_Particle] = []
        self._previous_tips: list[Point] = []
        self._rng = np.random.default_rng(1138)

    def apply(self, frame: np.ndarray, arg: EffectArgument) -> np.ndarray:
        if not _is_fingertip_list(arg):
            return frame.copy()
        fingertips = arg
        for index, (finger_x, finger_y) in enumerate(fingertips):
            previous = (
                self._previous_tips[index]
                if index < len(self._previous_tips)
                else (finger_x, finger_y)
            )
            speed = float(np.hypot(finger_x - previous[0], finger_y - previous[1]))
            emissions = 2 + min(4, int(speed / 7))
            for _ in range(emissions):
                angle = self._rng.uniform(-2.75, -0.4)
                velocity = self._rng.uniform(1.2, 4.6) + speed * 0.045
                life = self._rng.uniform(0.55, 1.1)
                self._particles.append(
                    _Particle(
                        float(finger_x),
                        float(finger_y),
                        float(np.cos(angle) * velocity),
                        float(np.sin(angle) * velocity),
                        float(life),
                        float(life),
                        int(self._rng.integers(82, 164)),
                    )
                )
        self._previous_tips = fingertips.copy()

        glow = np.zeros_like(frame)
        cores = np.zeros_like(frame)
        alive: list[_Particle] = []
        height, width = frame.shape[:2]
        for particle in self._particles:
            particle.x += particle.vx
            particle.y += particle.vy
            particle.vx *= 0.985
            particle.vy += 0.055
            particle.life -= 0.026
            if particle.life <= 0:
                continue
            if not (0 <= particle.x < width and 0 <= particle.y < height):
                continue
            alive.append(particle)
            fade = particle.life / particle.max_life
            point = (int(particle.x), int(particle.y))
            color = _hsv_color(particle.hue, 190, int(145 + 110 * fade))
            tail = (
                int(particle.x - particle.vx * 4),
                int(particle.y - particle.vy * 4),
            )
            cv2.line(glow, tail, point, color, 2, cv2.LINE_AA)
            cv2.circle(glow, point, 3 + int(3 * fade), color, -1, cv2.LINE_AA)
            cv2.circle(
                cores, point, 1 + int(fade > 0.65), (255, 246, 225), -1, cv2.LINE_AA
            )
        self._particles = alive[-520:]
        result = _add_light(frame, cv2.GaussianBlur(glow, (0, 0), 7), 0.62)
        return _add_light(result, cores, 0.88)


class NeonGlowEffect(BaseEffect):
    """A dual-colour aura along the real hand outline, not the fingertip hull."""

    def __init__(self) -> None:
        self._phase = 0.0

    def apply(self, frame: np.ndarray, arg: EffectArgument) -> np.ndarray:
        if not _is_mask(arg):
            return frame.copy()
        mask = arg
        if not _has_area(mask):
            return frame.copy()
        self._phase += 0.12
        soft = _soft_mask(mask, 12)
        tint = np.empty_like(frame)
        tint[:] = (170, 42, 92)
        result = _blend(frame, tint, soft * 0.16)
        pulse = 0.82 + 0.18 * (np.sin(self._phase) + 1.0) / 2.0
        result = _outline_glow(result, mask, (255, 45, 122), (255, 228, 145), pulse)
        return _outline_glow(result, mask, (145, 30, 255), (240, 255, 225), 0.5)


class MagicPortalEffect(BaseEffect):
    """A dimensional portal with a soft background vignette and rotating plasma."""

    def __init__(self) -> None:
        self._angle = 0.0

    def apply(self, frame: np.ndarray, arg: EffectArgument) -> np.ndarray:
        if not _is_mask(arg):
            return frame.copy()
        mask = arg
        if not _has_area(mask):
            return frame.copy()
        self._angle += 0.045
        height, width = frame.shape[:2]
        center_x, center_y = _mask_center(mask)
        yy, xx = np.mgrid[:height, :width]
        dx = xx - center_x
        dy = yy - center_y
        distance = np.hypot(dx, dy)
        angle = np.arctan2(dy, dx)
        swirl = angle * 4.2 - distance * 0.085 + self._angle * 2.6
        hue = np.mod(138 + 38 * np.sin(swirl) + distance * 0.025, 180).astype(np.uint8)
        saturation = np.full((height, width), 220, dtype=np.uint8)
        value = np.clip(170 + 85 * np.sin(swirl * 1.5), 45, 255).astype(np.uint8)
        plasma = cv2.cvtColor(np.dstack((hue, saturation, value)), cv2.COLOR_HSV2BGR)

        soft = _soft_mask(mask, 11)
        darkened = (frame.astype(np.float32) * 0.56).astype(np.uint8)
        portal_interior = _blend(frame, plasma, 0.84)
        result = _blend(darkened, portal_interior, soft)
        result = _outline_glow(result, mask, (255, 42, 105), (255, 214, 128), 1.0)

        ring = _outline(mask, 8)
        ring_blur = cv2.GaussianBlur(ring, (0, 0), 4)
        ring_layer = np.empty_like(frame)
        ring_layer[:] = (255, 150, 44)
        return _blend(result, ring_layer, ring_blur.astype(np.float32) / 255.0 * 0.5)
