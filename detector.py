import cv2
import numpy as np

# ============================================================
# Hand / Fingertip Detector
# ============================================================

_SKIN_YCRCB_LOWER = np.array([0, 133, 77], dtype=np.uint8)
_SKIN_YCRCB_UPPER = np.array([255, 173, 127], dtype=np.uint8)
_SKIN_HSV_LOWER = np.array([0, 15, 50], dtype=np.uint8)
_SKIN_HSV_UPPER = np.array([30, 255, 255], dtype=np.uint8)

_MIN_AREA = 5000
_MIN_DEFECT_DEPTH = 18
_MAX_DEFECT_ANGLE = 85
_CLUSTER_RADIUS = 50


class HandDetector:
    """Fingertip detection using background subtraction + skin-color + geometry."""

    def __init__(self) -> None:
        self._bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=40, detectShadows=False
        )
        self._kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self._prev_fingertips: list[tuple[int, int]] = []
        self._smooth_alpha = 0.35
        self._frame_count = 0

    def find_fingertips(self, frame: np.ndarray) -> list[tuple[int, int]]:
        self._frame_count += 1

        # Motion mask from background subtraction (skip first few frames for bg init)
        fg_mask = self._bg_sub.apply(frame)
        if self._frame_count < 30:
            return self._smooth([])
        fg_mask = cv2.medianBlur(fg_mask, 5)

        # Skin mask
        skin = self._skin_mask(frame)

        # Combine: only moving skin regions
        combined = cv2.bitwise_and(fg_mask, skin)
        combined = cv2.dilate(combined, self._kernel_large, iterations=1)

        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return self._smooth([])

        valid: list[tuple[np.ndarray, float]] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < _MIN_AREA:
                continue
            if not _is_hand_shape(c, area):
                continue
            valid.append((c, area))

        if not valid:
            return self._smooth([])

        valid.sort(key=lambda x: x[1], reverse=True)
        hand = valid[0][0]
        return self._smooth(_extract_fingertips(hand, frame.shape[0]))

    @staticmethod
    def _skin_mask(frame: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(ycrcb, _SKIN_YCRCB_LOWER, _SKIN_YCRCB_UPPER),
            cv2.inRange(hsv, _SKIN_HSV_LOWER, _SKIN_HSV_UPPER),
        )
        return mask

    def _smooth(self, current: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not current:
            self._prev_fingertips = []
            return []
        if not self._prev_fingertips or len(self._prev_fingertips) != len(current):
            self._prev_fingertips = current
            return current
        a = self._smooth_alpha
        smoothed: list[tuple[int, int]] = []
        for (px, py), (cx, cy) in zip(self._prev_fingertips, current):
            smoothed.append((int(px * (1 - a) + cx * a), int(py * (1 - a) + cy * a)))
        self._prev_fingertips = smoothed
        return smoothed


# ============================================================
# Helper functions
# ============================================================


def _is_hand_shape(contour: np.ndarray, area: float) -> bool:
    _x, _y, w, h = cv2.boundingRect(contour)
    aspect = w / max(h, 1)
    if aspect < 0.25 or aspect > 3.5:
        return False

    rect_area = w * h
    extent = area / max(rect_area, 1)
    if extent < 0.12 or extent > 0.85:
        return False

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / max(hull_area, 1)
    if solidity < 0.3 or solidity > 0.95:
        return False

    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area / max(perimeter * perimeter, 1)
    if circularity > 0.55:
        return False

    return True


def _extract_fingertips(hand: np.ndarray, frame_h: int) -> list[tuple[int, int]]:
    hull_pts = cv2.convexHull(hand, returnPoints=False)
    candidates: list[tuple[int, int]] = []

    if len(hull_pts) >= 3:
        defects = cv2.convexityDefects(hand, hull_pts)
        if len(defects) > 0:
            has_extra_dim = defects.ndim == 3
            for i in range(defects.shape[0]):
                row = defects[i, 0] if has_extra_dim else defects[i]
                s, e, f, d = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                depth = d / 256.0
                if depth < _MIN_DEFECT_DEPTH:
                    continue
                start = tuple(hand[s][0])
                far = tuple(hand[f][0])
                end = tuple(hand[e][0])
                if _angle3(start, far, end) > _MAX_DEFECT_ANGLE:
                    continue
                candidates.append(start)
                candidates.append(end)

    pts = hand[:, 0, :]
    centroid = np.mean(pts, axis=0)
    cx, cy = float(centroid[0]), float(centroid[1])

    top_idx = int(pts[:, 1].argmin())
    candidates.append(tuple(pts[top_idx]))

    for p in pts:
        px, py = float(p[0]), float(p[1])
        if py < cy and np.hypot(px - cx, py - cy) > np.hypot(cx, cy) * 0.35:
            candidates.append((int(px), int(py)))

    fingertips = _cluster_points(candidates)
    return _filter_wrist(fingertips, frame_h)


def _cluster_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not points:
        return []
    unique = list(set(points))
    used: set[int] = set()
    clusters: list[tuple[int, int]] = []
    for i, p in enumerate(unique):
        if i in used:
            continue
        group = [p]
        used.add(i)
        for j, q in enumerate(unique):
            if j in used:
                continue
            if np.hypot(p[0] - q[0], p[1] - q[1]) < _CLUSTER_RADIUS:
                group.append(q)
                used.add(j)
        clusters.append(
            (
                int(np.mean([pt[0] for pt in group])),
                int(np.mean([pt[1] for pt in group])),
            )
        )
    return clusters


def _filter_wrist(points: list[tuple[int, int]], frame_h: int) -> list[tuple[int, int]]:
    cutoff = int(frame_h * 0.82)
    return [p for p in points if p[1] < cutoff]


def _angle3(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> float:
    va = np.array(a, dtype=np.float64)
    vb = np.array(b, dtype=np.float64)
    vc = np.array(c, dtype=np.float64)
    ba = va - vb
    bc = vc - vb
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-8:
        return 0.0
    cos_a = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))
