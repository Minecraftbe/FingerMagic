from dataclasses import dataclass
from itertools import pairwise
from typing import ClassVar

import cv2
import numpy as np

type Point = tuple[int, int]


@dataclass(slots=True)
class MagicPanel:
    corners: list[Point]
    mask: np.ndarray
    links: list[tuple[Point, Point]]
    fingertips: list[Point]

    @property
    def visible(self) -> bool:
        return len(self.corners) == 4 and cv2.countNonZero(self.mask) > 96


class MagicPanelTracker:
    def __init__(self) -> None:
        self._corners: np.ndarray | None = None
        self._last_shape = (0, 0)
        self._missed_frames = 0

    def update(self, shape: tuple[int, int], fingertips: list[Point]) -> MagicPanel:
        height, width = shape
        if len(fingertips) < 2:
            return self._hold_or_empty(shape)

        points = np.array(fingertips, dtype=np.float32)
        center = points.mean(axis=0)
        direction = self._primary_direction(points)
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        if normal[1] < 0:
            normal *= -1

        projected = (points - center) @ direction
        panel_width = max(150.0, float(projected.max() - projected.min()) + 64.0)
        panel_height = max(105.0, panel_width * 0.58)
        panel_center = center + normal * panel_height * 0.22
        target = np.array(
            [
                panel_center - direction * panel_width / 2 - normal * panel_height / 2,
                panel_center + direction * panel_width / 2 - normal * panel_height / 2,
                panel_center + direction * panel_width / 2 + normal * panel_height / 2,
                panel_center - direction * panel_width / 2 + normal * panel_height / 2,
            ],
            dtype=np.float32,
        )
        target[:, 0] = np.clip(target[:, 0], 0, width - 1)
        target[:, 1] = np.clip(target[:, 1], 0, height - 1)

        if self._corners is None or self._last_shape != shape:
            self._corners = target
        else:
            self._corners = self._corners * 0.58 + target * 0.42
        self._last_shape = shape
        self._missed_frames = 0

        corners = [(int(x), int(y)) for x, y in self._corners]
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.array(corners, dtype=np.int32), 255, cv2.LINE_AA)
        ordered = sorted(
            fingertips,
            key=lambda point: point[0] * direction[0] + point[1] * direction[1],
        )
        return MagicPanel(corners, mask, list(pairwise(ordered)), fingertips)

    def _hold_or_empty(self, shape: tuple[int, int]) -> MagicPanel:
        self._missed_frames += 1
        if self._corners is not None and self._missed_frames <= 3:
            corners = [(int(x), int(y)) for x, y in self._corners]
            mask = np.zeros(shape, dtype=np.uint8)
            cv2.fillConvexPoly(
                mask, np.array(corners, dtype=np.int32), 255, cv2.LINE_AA
            )
            return MagicPanel(corners, mask, [], [])
        self._corners = None
        return MagicPanel([], np.zeros(shape, dtype=np.uint8), [], [])

    @staticmethod
    def _primary_direction(points: np.ndarray) -> np.ndarray:
        if len(points) == 2:
            direction = points[1] - points[0]
        else:
            centered = points - points.mean(axis=0)
            _values, vectors = np.linalg.eigh(centered.T @ centered)
            direction = vectors[:, -1]
        length = float(np.linalg.norm(direction))
        if length < 1e-5:
            return np.array([1.0, 0.0], dtype=np.float32)
        direction = direction / length
        if direction[0] < 0:
            direction *= -1
        return direction.astype(np.float32)


class MagicPanelEffect:
    _PALETTES: ClassVar[
        dict[
            str, tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]
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

    def apply(self, frame: np.ndarray, panel: MagicPanel) -> np.ndarray:
        if not panel.visible:
            return frame.copy()
        self._phase += 0.11
        base_color, line_color, accent_color = self._PALETTES[self._style]
        x0, y0, x1, y1 = self._panel_bounds(panel.corners, frame.shape[:2])
        result = self._draw_fingertip_glows(frame, panel.fingertips, accent_color)
        roi = result[y0:y1, x0:x1]
        mask = panel.mask[y0:y1, x0:x1]
        local_corners = np.array(panel.corners, dtype=np.int32) - np.array([x0, y0])

        surface = np.empty_like(roi)
        surface[:] = base_color
        texture = surface.copy()
        self._draw_energy_grid(texture, line_color, accent_color)
        self._draw_runes(texture, line_color, accent_color)
        self._draw_panel_orbits(texture, line_color, accent_color)

        soft_mask = cv2.GaussianBlur(mask, (0, 0), 5).astype(np.float32) / 255.0
        panel_art = cv2.addWeighted(surface, 0.28, texture, 0.72, 0)
        blended = cv2.addWeighted(roi, 0.14, panel_art, 0.86, 0)
        roi[:] = self._composite(roi, blended, soft_mask * 0.9)

        edge = np.zeros_like(roi)
        cv2.polylines(edge, [local_corners], True, line_color, 2, cv2.LINE_AA)
        cv2.polylines(
            edge,
            [self._inset_corners(local_corners, 8)],
            True,
            accent_color,
            1,
            cv2.LINE_AA,
        )
        cv2.polylines(
            edge,
            [self._inset_corners(local_corners, 16)],
            True,
            line_color,
            1,
            cv2.LINE_AA,
        )
        edge_glow = cv2.GaussianBlur(edge, (0, 0), 8)
        roi[:] = self._add_light(roi, edge_glow, 0.65)
        roi[:] = self._add_light(roi, edge, 0.85)

        result = self._draw_energy_beams(result, panel, accent_color, line_color)

        for start, end in panel.links:
            cv2.line(result, start, end, line_color, 2, cv2.LINE_AA)
            cv2.circle(result, start, 6, accent_color, 1, cv2.LINE_AA)
        if panel.links:
            cv2.circle(result, panel.links[-1][1], 6, accent_color, 1, cv2.LINE_AA)

        return result

    def _draw_fingertip_glows(
        self,
        frame: np.ndarray,
        fingertips: list[Point],
        accent: tuple[int, int, int],
    ) -> np.ndarray:
        result = frame.copy()
        glow_layer = np.zeros_like(result)
        for x, y in fingertips:
            cv2.circle(glow_layer, (x, y), 18, accent, -1, cv2.LINE_AA)
            cv2.circle(glow_layer, (x, y), 28, accent, -1, cv2.LINE_AA)
        glow = cv2.GaussianBlur(glow_layer, (0, 0), 15)
        return self._add_light(result, glow, 0.55)

    def _draw_energy_beams(
        self,
        frame: np.ndarray,
        panel: MagicPanel,
        accent: tuple[int, int, int],
        line: tuple[int, int, int],
    ) -> np.ndarray:
        result = frame.copy()
        if len(panel.corners) != 4 or not panel.fingertips:
            return result

        panel_center = np.mean(panel.corners, axis=0).astype(np.int32)
        center = (int(panel_center[0]), int(panel_center[1]))
        beam_layer = np.zeros_like(result, dtype=np.float32)
        phase = self._phase

        for i, (fx, fy) in enumerate(panel.fingertips):
            offset = int(np.sin(phase * 3.7 + i * 1.2) * 12)
            mid_x = (fx + center[0]) // 2 + offset
            mid_y = (fy + center[1]) // 2 + offset

            pts = np.array([(fx, fy), (mid_x, mid_y), center], dtype=np.int32).reshape(
                (-1, 1, 2)
            )
            cv2.polylines(
                beam_layer.astype(np.uint8),
                [pts],
                False,
                accent,
                1,
                cv2.LINE_AA,
            )

            pulse = 0.35 + 0.25 * np.sin(phase * 2.5 + i)
            cv2.circle(
                beam_layer.astype(np.uint8),
                (mid_x, mid_y),
                int(4 + pulse * 6),
                line,
                -1,
                cv2.LINE_AA,
            )

        beam_blur = cv2.GaussianBlur(beam_layer, (0, 0), 6)
        return self._add_light(result, beam_blur.astype(np.uint8), 0.4)

    def _draw_energy_grid(
        self,
        texture: np.ndarray,
        line_color: tuple[int, int, int],
        accent_color: tuple[int, int, int],
    ) -> None:
        height, width = texture.shape[:2]
        offset = int((self._phase * 42) % 28)
        for x in range(-height + offset, width + height, 28):
            cv2.line(texture, (x, 0), (x + height, height), line_color, 1, cv2.LINE_AA)
        for y in range(offset - height, height + width, 30):
            cv2.line(
                texture,
                (0, y),
                (width, y - width),
                accent_color,
                1,
                cv2.LINE_AA,
            )

        center = (width // 2, height // 2)
        base_radius = max(26, min(width, height) // 4)
        for idx, mult in enumerate((1.0, 0.68, 0.36)):
            cv2.ellipse(
                texture,
                center,
                (int(base_radius * mult), int(base_radius * mult * 0.62)),
                int(self._phase * (36 + idx * 11)),
                0,
                360,
                line_color,
                1,
                cv2.LINE_AA,
            )

    def _draw_runes(
        self,
        texture: np.ndarray,
        line_color: tuple[int, int, int],
        accent_color: tuple[int, int, int],
    ) -> None:
        height, width = texture.shape[:2]
        center_x, center_y = width // 2, height // 2
        orbit = max(22, min(width, height) // 3)
        for index in range(9):
            angle = self._phase * 1.8 + index * (2 * np.pi / 9)
            x = int(center_x + np.cos(angle) * orbit)
            y = int(center_y + np.sin(angle) * orbit * 0.58)
            cv2.circle(texture, (x, y), 4, accent_color, 1, cv2.LINE_AA)
            cv2.line(texture, (x - 6, y), (x + 6, y), line_color, 1, cv2.LINE_AA)
            cv2.line(texture, (x, y - 6), (x, y + 6), line_color, 1, cv2.LINE_AA)

    def _draw_panel_orbits(
        self,
        texture: np.ndarray,
        line_color: tuple[int, int, int],
        accent_color: tuple[int, int, int],
    ) -> None:
        height, width = texture.shape[:2]
        center_x, center_y = width // 2, height // 2
        for layer in range(3):
            r = max(12, min(width, height) // 4 + layer * 16)
            angle_offset = self._phase * (24 + layer * 15)
            for i in range(2):
                a = np.radians(angle_offset + i * 180)
                x = int(center_x + np.cos(a) * r * 0.7)
                y = int(center_y + np.sin(a) * r * 0.45)
                cv2.circle(texture, (x, y), 3, line_color, -1, cv2.LINE_AA)

    @staticmethod
    def _panel_bounds(
        corners: list[Point], shape: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        height, width = shape
        xs = [point[0] for point in corners]
        ys = [point[1] for point in corners]
        padding = 24
        return (
            max(min(xs) - padding, 0),
            max(min(ys) - padding, 0),
            min(max(xs) + padding + 1, width),
            min(max(ys) + padding + 1, height),
        )

    @staticmethod
    def _inset_corners(corners: np.ndarray, pixels: int) -> np.ndarray:
        center = corners.mean(axis=0)
        vectors = corners.astype(np.float32) - center
        lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
        return (
            center + vectors * np.maximum(0, lengths - pixels) / np.maximum(lengths, 1)
        ).astype(np.int32)

    @staticmethod
    def _composite(
        base: np.ndarray, layer: np.ndarray, alpha: np.ndarray
    ) -> np.ndarray:
        mixed = (
            base.astype(np.float32) * (1 - alpha[..., None])
            + layer.astype(np.float32) * alpha[..., None]
        )
        return np.clip(mixed, 0, 255).astype(np.uint8)

    @staticmethod
    def _add_light(base: np.ndarray, light: np.ndarray, strength: float) -> np.ndarray:
        return np.clip(
            base.astype(np.float32) + light.astype(np.float32) * strength, 0, 255
        ).astype(np.uint8)
