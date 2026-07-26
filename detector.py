from dataclasses import dataclass

import cv2
import numpy as np

# ============================================================
# Hand / fingertip detector
# ============================================================

_SKIN_YCRCB_LOWER = np.array([0, 133, 77], dtype=np.uint8)
_SKIN_YCRCB_UPPER = np.array([255, 173, 127], dtype=np.uint8)
_SKIN_HSV_LOWER = np.array([0, 15, 50], dtype=np.uint8)
_SKIN_HSV_UPPER = np.array([30, 255, 255], dtype=np.uint8)

_MIN_AREA = 5_000
_MIN_DEFECT_DEPTH = 18
_MAX_DEFECT_ANGLE = 85
_CLUSTER_RADIUS = 48
_WARMUP_FRAMES = 0
_MAX_MISSED_FRAMES = 5

type Point = tuple[int, int]


@dataclass(slots=True)
class HandObservation:
    """The hand silhouette and the fingertip positions derived from it."""

    fingertips: list[Point]
    mask: np.ndarray
    contour: np.ndarray | None

    @property
    def found(self) -> bool:
        return self.contour is not None


class HandDetector:
    """Detect one moving hand and preserve its full skin-colour silhouette.

    Motion is used to discover a hand in a busy scene. Once found, overlap with
    the previous silhouette keeps the mask stable when the hand pauses, instead
    of cutting the effect away as soon as background subtraction settles.
    """

    def __init__(self) -> None:
        self._bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=36, detectShadows=False
        )
        self._kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self._prev_fingertips: list[Point] = []
        self._prev_mask: np.ndarray | None = None
        self._missed_frames = 0
        self._smooth_alpha = 0.38
        self._frame_count = 0

    def detect(self, frame: np.ndarray) -> HandObservation:
        """Return the most likely hand, or an empty observation during warm-up."""
        self._frame_count += 1
        fg_mask = self._bg_sub.apply(frame)
        skin = self._skin_mask(frame)
        skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, self._kernel_small)
        skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, self._kernel_large)

        if self._frame_count <= _WARMUP_FRAMES:
            return self._empty(frame.shape[:2])

        fg_mask = cv2.medianBlur(fg_mask, 5)
        fg_mask = cv2.dilate(fg_mask, self._kernel_small, iterations=1)
        contour = self._choose_hand(skin, fg_mask)
        if contour is None:
            return self._recover_or_empty(frame.shape[:2])

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        fingertips = self._smooth(_extract_fingertips(contour, frame.shape[0]))
        self._prev_mask = mask
        self._missed_frames = 0
        return HandObservation(fingertips, mask, contour)

    def find_fingertips(self, frame: np.ndarray) -> list[Point]:
        """Compatibility helper for callers that only need fingertip positions."""
        return self.detect(frame).fingertips

    @staticmethod
    def _skin_mask(frame: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return cv2.bitwise_or(
            cv2.inRange(ycrcb, _SKIN_YCRCB_LOWER, _SKIN_YCRCB_UPPER),
            cv2.inRange(hsv, _SKIN_HSV_LOWER, _SKIN_HSV_UPPER),
        )

    def _choose_hand(self, skin: np.ndarray, fg_mask: np.ndarray) -> np.ndarray | None:
        contours, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        frame_area = float(skin.shape[0] * skin.shape[1])
        best: np.ndarray | None = None
        best_score = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < _MIN_AREA or not _is_hand_shape(contour, area):
                continue

            candidate_mask = np.zeros_like(skin)
            cv2.drawContours(candidate_mask, [contour], -1, 255, thickness=cv2.FILLED)
            motion = cv2.countNonZero(cv2.bitwise_and(candidate_mask, fg_mask)) / max(
                area, 1.0
            )
            overlap = self._mask_overlap(candidate_mask)
            area_score = min(area / max(frame_area * 0.11, 1.0), 1.0)

            # A newly found hand must include some movement. After a hand has
            # been found, overlap keeps its silhouette visible while it rests.
            if self._prev_mask is None and motion < 0.045:
                continue
            score = 0.42 * area_score + 0.33 * min(motion, 1.0) + 0.25 * overlap
            if score > best_score:
                best = contour
                best_score = score

        return best

    def _mask_overlap(self, candidate: np.ndarray) -> float:
        if self._prev_mask is None:
            return 0.0
        intersection = cv2.countNonZero(cv2.bitwise_and(candidate, self._prev_mask))
        union = cv2.countNonZero(cv2.bitwise_or(candidate, self._prev_mask))
        return intersection / max(union, 1)

    def _recover_or_empty(self, shape: tuple[int, int]) -> HandObservation:
        self._missed_frames += 1
        if self._prev_mask is not None and self._missed_frames <= _MAX_MISSED_FRAMES:
            return HandObservation(self._prev_fingertips, self._prev_mask.copy(), None)
        self._prev_mask = None
        return self._empty(shape)

    def _empty(self, shape: tuple[int, int]) -> HandObservation:
        self._prev_fingertips = []
        return HandObservation([], np.zeros(shape, dtype=np.uint8), None)

    def _smooth(self, current: list[Point]) -> list[Point]:
        if not current:
            self._prev_fingertips = []
            return []
        if not self._prev_fingertips or len(self._prev_fingertips) != len(current):
            self._prev_fingertips = current
            return current

        remaining = current.copy()
        smoothed: list[Point] = []
        alpha = self._smooth_alpha
        for px, py in self._prev_fingertips:
            closest = min(
                remaining, key=lambda point: (point[0] - px) ** 2 + (point[1] - py) ** 2
            )
            remaining.remove(closest)
            cx, cy = closest
            smoothed.append(
                (int(px * (1 - alpha) + cx * alpha), int(py * (1 - alpha) + cy * alpha))
            )
        self._prev_fingertips = smoothed
        return smoothed


# ============================================================
# Helper functions
# ============================================================


def _is_hand_shape(contour: np.ndarray, area: float) -> bool:
    _x, _y, width, height = cv2.boundingRect(contour)
    aspect = width / max(height, 1)
    if aspect < 0.22 or aspect > 3.8:
        return False

    rect_area = width * height
    extent = area / max(rect_area, 1)
    if extent < 0.1 or extent > 0.93:
        return False

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / max(hull_area, 1)
    if solidity < 0.28 or solidity > 0.985:
        return False

    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / max(perimeter * perimeter, 1)
    return circularity <= 0.78


def _extract_fingertips(hand: np.ndarray, frame_height: int) -> list[Point]:
    hull_indices = cv2.convexHull(hand, returnPoints=False)
    candidates: list[Point] = []
    area = cv2.contourArea(hand)
    palm_radius = max(np.sqrt(area / np.pi), 24.0)

    if len(hull_indices) >= 3:
        defects = cv2.convexityDefects(hand, hull_indices)
        if defects is not None:  # pyright: ignore[reportUnnecessaryComparison] -- OpenCV can return None at runtime.
            for row in defects.reshape(-1, 4):
                start_index, end_index, far_index, depth_raw = (
                    int(value) for value in row
                )
                depth = depth_raw / 256.0
                if depth < _MIN_DEFECT_DEPTH:
                    continue
                start = _point_from_array(hand[start_index, 0])
                far = _point_from_array(hand[far_index, 0])
                end = _point_from_array(hand[end_index, 0])
                if _angle3(start, far, end) > _MAX_DEFECT_ANGLE:
                    continue
                candidates.extend((start, end))

    points = hand[:, 0, :]
    centroid = np.mean(points, axis=0)
    center_x, center_y = float(centroid[0]), float(centroid[1])
    top_index = int(points[:, 1].argmin())
    candidates.append(_point_from_array(points[top_index]))

    fingertips = _cluster_points(candidates)
    fingertips = [
        point
        for point in fingertips
        if point[1] < center_y + palm_radius * 0.3
        and np.hypot(point[0] - center_x, point[1] - center_y) > palm_radius * 0.7
    ]
    fingertips.sort(key=lambda point: point[0])
    return _filter_wrist(fingertips, frame_height)


def _cluster_points(points: list[Point]) -> list[Point]:
    if not points:
        return []

    unique = sorted(set(points))
    clusters: list[Point] = []
    unseen = set(range(len(unique)))
    while unseen:
        seed_index = unseen.pop()
        members = [seed_index]
        queue = [seed_index]
        while queue:
            index = queue.pop()
            x, y = unique[index]
            neighbours = [
                other
                for other in unseen
                if np.hypot(unique[other][0] - x, unique[other][1] - y)
                < _CLUSTER_RADIUS
            ]
            for other in neighbours:
                unseen.remove(other)
                queue.append(other)
                members.append(other)
        clusters.append(
            (
                int(np.mean([unique[index][0] for index in members])),
                int(np.mean([unique[index][1] for index in members])),
            )
        )
    return clusters


def _filter_wrist(points: list[Point], frame_height: int) -> list[Point]:
    cutoff = int(frame_height * 0.86)
    return [point for point in points if point[1] < cutoff]


def _point_from_array(values: np.ndarray) -> Point:
    return int(values[0]), int(values[1])


def _angle3(a: Point, b: Point, c: Point) -> float:
    first = np.array(a, dtype=np.float64)
    middle = np.array(b, dtype=np.float64)
    last = np.array(c, dtype=np.float64)
    left = first - middle
    right = last - middle
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator < 1e-8:
        return 0.0
    cosine = np.clip(np.dot(left, right) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))
