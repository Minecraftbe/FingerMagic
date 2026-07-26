import argparse
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypedDict, cast

import cv2
import numpy as np

# ============================================================
# Skin detection parameters
# ============================================================
SKIN_HSV_LOWER = np.array([0, 20, 70], dtype=np.uint8)
SKIN_HSV_UPPER = np.array([25, 255, 255], dtype=np.uint8)
SKIN_YCRCB_LOWER = np.array([0, 133, 77], dtype=np.uint8)
SKIN_YCRCB_UPPER = np.array([255, 173, 127], dtype=np.uint8)

MIN_CONTOUR_AREA = 4000
DEFECT_DEPTH_THRESHOLD = 30
FINGER_ANGLE_MAX = 80  # degrees, defect angle must be below this to be a finger gap


# ============================================================
# Hand / Fingertip Detector
# ============================================================


class HandDetector:
    """Detects hands and fingertips using skin-color segmentation + contour convexity."""

    def __init__(self) -> None:
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._prev_fingertips: list[tuple[int, int]] = []
        self._smooth_alpha = 0.4

    def detect_skin_mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask_hsv = cv2.inRange(hsv, SKIN_HSV_LOWER, SKIN_HSV_UPPER)
        mask_ycrcb = cv2.inRange(ycrcb, SKIN_YCRCB_LOWER, SKIN_YCRCB_UPPER)
        mask = cv2.bitwise_or(mask_hsv, mask_ycrcb)
        mask = cv2.erode(mask, self._kernel, iterations=1)
        mask = cv2.dilate(mask, self._kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        return mask

    def find_fingertips(self, frame: np.ndarray) -> list[tuple[int, int]]:
        mask = self.detect_skin_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._smooth([])

        hand = max(contours, key=cv2.contourArea)
        if cv2.contourArea(hand) < MIN_CONTOUR_AREA:
            return self._smooth([])

        hull = cv2.convexHull(hand, returnPoints=False)
        if len(hull) < 3:
            return self._smooth(_extreme_points(hand))

        defects = cv2.convexityDefects(hand, hull)
        candidates: list[tuple[int, int]] = []

        if len(defects) > 0:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                depth = d / 256.0
                if depth < DEFECT_DEPTH_THRESHOLD:
                    continue
                start = tuple(hand[s][0])
                far = tuple(hand[f][0])
                end = tuple(hand[e][0])
                angle = _angle_between(start, far, end)
                if angle > FINGER_ANGLE_MAX:
                    continue
                candidates.append(start)
                candidates.append(end)

        if len(candidates) < 2:
            candidates = _extreme_points(hand)

        fingertips = _cluster_points(candidates, threshold=45)
        fingertips = _filter_bottom(fingertips, frame.shape[0])

        return self._smooth(fingertips)

    def _smooth(self, current: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not current:
            self._prev_fingertips = []
            return []
        if not self._prev_fingertips or len(self._prev_fingertips) != len(current):
            self._prev_fingertips = current
            return current
        smoothed: list[tuple[int, int]] = []
        alpha = self._smooth_alpha
        for (px, py), (cx, cy) in zip(self._prev_fingertips, current):
            sx = int(px * (1 - alpha) + cx * alpha)
            sy = int(py * (1 - alpha) + cy * alpha)
            smoothed.append((sx, sy))
        self._prev_fingertips = smoothed
        return smoothed


def _angle_between(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> float:
    """Angle ABC in degrees."""
    va = np.array(a, dtype=np.float64)
    vb = np.array(b, dtype=np.float64)
    vc = np.array(c, dtype=np.float64)
    ba = va - vb
    bc = vc - vb
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm < 1e-6:
        return 0.0
    cos_a = np.clip(np.dot(ba, bc) / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def _extreme_points(contour: np.ndarray) -> list[tuple[int, int]]:
    pts = contour[:, 0, :]
    top = tuple(pts[pts[:, 1].argmin()])
    right = tuple(pts[pts[:, 0].argmax()])
    bottom = tuple(pts[pts[:, 1].argmax()])
    left = tuple(pts[pts[:, 0].argmin()])
    return [top, right, bottom, left]


def _cluster_points(
    points: list[tuple[int, int]], threshold: int = 40
) -> list[tuple[int, int]]:
    if not points:
        return []
    unique: list[tuple[int, int]] = list(set(points))
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
            if np.hypot(p[0] - q[0], p[1] - q[1]) < threshold:
                group.append(q)
                used.add(j)
        cx = int(np.mean([pt[0] for pt in group]))
        cy = int(np.mean([pt[1] for pt in group]))
        clusters.append((cx, cy))
    return clusters


def _filter_bottom(
    points: list[tuple[int, int]], frame_height: int
) -> list[tuple[int, int]]:
    """Remove points near the bottom edge (likely wrist, not finger)."""
    cutoff = int(frame_height * 0.85)
    return [p for p in points if p[1] < cutoff]


# ============================================================
# Polygon helpers
# ============================================================


def polygon_mask(shape: tuple[int, int], points: list[tuple[int, int]]) -> np.ndarray:
    """Binary mask for the convex hull of fingertip points."""
    mask = np.zeros(shape, dtype=np.uint8)
    if len(points) >= 3:
        pts = np.array(points, dtype=np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillPoly(mask, [hull], 255)
    return mask


def convex_hull_of(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return convex-hull ordered list of points."""
    if len(points) < 3:
        return points
    pts = np.array(points, dtype=np.int32)
    hull = cv2.convexHull(pts)
    return [tuple(h[0]) for h in hull]


# ============================================================
# Visual Effects
# ============================================================


class _BaseEffect(ABC):
    @abstractmethod
    def apply(
        self, frame: np.ndarray, arg: np.ndarray | list[tuple[int, int]]
    ) -> np.ndarray: ...


class StarfieldEffect(_BaseEffect):
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


class RainbowEffect(_BaseEffect):
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


class ParticleEffect(_BaseEffect):
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
                alpha = float(p["life"])
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


class NeonGlowEffect(_BaseEffect):
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


class MagicPortalEffect(_BaseEffect):
    """Darkened surroundings + glowing polygon interior with rotating gradient."""

    def __init__(self) -> None:
        self._angle = 0.0

    def apply(
        self, frame: np.ndarray, arg: np.ndarray | list[tuple[int, int]]
    ) -> np.ndarray:
        mask = cast(np.ndarray, arg)
        self._angle += 0.03
        h, w = frame.shape[:2]

        # Darken outside the polygon
        dark = (frame.astype(np.float32) * 0.25).astype(np.uint8)
        interior = frame.copy()
        result = np.where(mask[..., None] > 0, interior, dark)

        # Rotating radial gradient inside polygon
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


# ============================================================
# Main loop
# ============================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FingerMagic — fingertip detection with visual FX"
    )
    p.add_argument(
        "--input", "-i", default=None, help="Input video file (default: webcam)"
    )
    p.add_argument(
        "--output",
        "-o",
        default="output",
        help="Output directory for video (default: output/)",
    )
    p.add_argument(
        "--effect",
        "-e",
        default="starfield",
        choices=["starfield", "rainbow", "particle", "neon", "portal", "all"],
    )
    p.add_argument(
        "--camera", "-c", type=int, default=0, help="Camera device index (default: 0)"
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    # ---- input source ----
    out_path: Path | None = None
    if args.input:
        cap = cv2.VideoCapture(args.input)
        source_type = "video"
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        out_path = out_dir / "output.mp4"
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        print(f"[video]  {args.input}  ->  {out_path}")
    else:
        cap = cv2.VideoCapture(args.camera)
        source_type = "webcam"
        writer = None
        print(f"[webcam] device {args.camera}")

    if not cap.isOpened():
        print("Error: cannot open video source", file=sys.stderr)
        sys.exit(1)

    # ---- components ----
    detector = HandDetector()
    effect_map: dict[str, _BaseEffect] = {
        "starfield": StarfieldEffect(),
        "rainbow": RainbowEffect(),
        "particle": ParticleEffect(),
        "neon": NeonGlowEffect(),
        "portal": MagicPortalEffect(),
    }

    if args.effect == "all":
        active = list(effect_map.keys())
    else:
        active = [args.effect]

    paused = False
    frame_idx = 0
    t0 = time.time()

    print("\nKeys: 1-5 switch effect | a=all | SPACE=pause | q=quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            if source_type == "video":
                break
            continue

        if source_type == "webcam":
            frame = cv2.flip(frame, 1)

        if paused:
            cv2.imshow("FingerMagic", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                paused = False
            elif key == ord("q"):
                break
            continue

        # fingertip detection
        fingertips = detector.find_fingertips(frame)
        hull_pts = convex_hull_of(fingertips)
        pmask = polygon_mask(frame.shape[:2], hull_pts)

        # apply effects
        result = frame.copy()
        for name in active:
            e = effect_map[name]
            if name == "particle":
                result = e.apply(result, fingertips)
            elif name == "neon":
                result = e.apply(result, fingertips)
            else:
                result = e.apply(result, pmask)

        # draw fingertips
        for pt in fingertips:
            cv2.circle(result, pt, 9, (0, 255, 100), -1)
            cv2.circle(result, pt, 13, (0, 255, 100), 2)

        # draw connecting edges (convex hull)
        if len(hull_pts) >= 2:
            for i in range(len(hull_pts)):
                a = hull_pts[i]
                b = hull_pts[(i + 1) % len(hull_pts)]
                cv2.line(result, a, b, (0, 220, 255), 3)

        # HUD
        fps_now = 1.0 / max(time.time() - t0, 1e-3) if frame_idx > 0 else 0.0
        t0 = time.time()
        cv2.putText(
            result,
            f"FX: {'+'.join(active)}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            result,
            f"Fingers: {len(fingertips)}",
            (10, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            result,
            f"FPS: {fps_now:.1f}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        if writer:
            writer.write(result)

        cv2.imshow("FingerMagic", result)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
        if key == ord("1"):
            active = ["starfield"]
        if key == ord("2"):
            active = ["rainbow"]
        if key == ord("3"):
            active = ["particle"]
        if key == ord("4"):
            active = ["neon"]
        if key == ord("5"):
            active = ["portal"]
        if key == ord("a"):
            active = list(effect_map.keys())

        frame_idx += 1

    cap.release()
    if writer and out_path:
        writer.release()
        print(f"\nSaved -> {out_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
