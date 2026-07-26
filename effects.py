from dataclasses import dataclass
from itertools import pairwise
from math import cos, pi, sin, sqrt
from random import randint, random, uniform
from typing import ClassVar

import cv2
import numpy as np

type Point = tuple[int, int]

# =============================================================================
# Constants
# =============================================================================
_MAX_PARTICLES = 260
_MAX_TRAIL = 15
_MAX_RIPPLES = 8
_ORB_RING_LAYERS = 3
_RAY_COUNT = 5
_ARC_SEGMENTS = 24

# =============================================================================
# Data Classes
# =============================================================================


@dataclass(slots=True)
class _Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    size: float
    color: tuple[int, int, int]
    color_end: tuple[int, int, int]
    shape: int
    spawn_x: float
    spawn_y: float
    orbit: float
    orbit_speed: float


@dataclass(slots=True)
class _Ripple:
    x: float
    y: float
    radius: float
    max_radius: float
    life: float
    decay: float
    color: tuple[int, int, int]
    thickness: int


@dataclass(slots=True)
class _OrbState:
    x: float
    y: float
    phase: float
    base_r: float


# =============================================================================
# CyberEffect
# =============================================================================


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
        "arcane": ((220, 100, 40), (180, 40, 200), (255, 200, 50)),
        "plasma": ((255, 60, 180), (140, 40, 255), (255, 220, 80)),
        "matrix": ((40, 255, 100), (20, 200, 50), (160, 255, 100)),
        "cosmic": ((60, 160, 255), (120, 60, 255), (180, 240, 255)),
        "frost": ((80, 210, 255), (40, 160, 255), (200, 240, 255)),
        "neon": ((255, 20, 120), (255, 200, 0), (255, 80, 220)),
        "synthwave": ((255, 20, 147), (0, 255, 255), (255, 140, 0)),
        "hacker": ((0, 255, 65), (0, 160, 30), (180, 255, 80)),
        "ethereal": ((180, 140, 255), (255, 100, 200), (140, 200, 255)),
        "phoenix": ((255, 60, 20), (255, 160, 20), (255, 240, 140)),
        "aurora": ((60, 255, 160), (100, 180, 255), (200, 140, 255)),
        "void": ((120, 30, 200), (60, 10, 140), (220, 60, 255)),
        "prism": ((255, 100, 200), (100, 255, 200), (200, 150, 255)),
        "ember": ((255, 80, 30), (255, 40, 100), (255, 200, 80)),
        "ocean": ((30, 140, 255), (20, 220, 200), (140, 240, 255)),
    }

    def __init__(self, style: str = "ethereal") -> None:
        self._style = style
        self._phase = 0.0
        self._trails: dict[int, dict[int, list[Point]]] = {}
        self._trail_ages: dict[int, dict[int, int]] = {}
        self._particles: list[_Particle] = []
        self._ripples: list[_Ripple] = []
        self._orbs: dict[int, list[_OrbState]] = {}
        self._glitch_timer = 0.0
        self._glitch_active = False
        self._glitch_row = 0
        self._glitch_shift = 0
        self._glitch_height = 0
        self._glitch_boost_b = 0
        self._vignette_cache: dict[tuple[int, int], np.ndarray] = {}
        self._grid_cache: dict[tuple[int, int], np.ndarray] = {}
        self._arena_cache: dict[tuple[int, int], np.ndarray] = {}
        self._prev_hands = 0

    # ═══════════════════════════════════════════════════════════════════════
    # Pre-computed textures
    # ═══════════════════════════════════════════════════════════════════════

    def _get_vignette(self, h: int, w: int) -> np.ndarray:
        key = (h, w)
        cached = self._vignette_cache.get(key)
        if cached is not None:
            return cached
        y, x = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_dist = np.sqrt(cx**2 + cy**2)
        mask = np.clip(1.0 - (dist / max_dist) ** 1.5, 0.15, 1.0)
        vig = (mask * 255).astype(np.uint8)
        vig3: np.ndarray = np.dstack([vig, vig, vig])
        self._vignette_cache[key] = vig3
        return vig3

    def _get_grid(self, h: int, w: int) -> np.ndarray:
        key = (h, w)
        cached = self._grid_cache.get(key)
        if cached is not None:
            return cached
        grid = np.zeros((h, w, 3), dtype=np.uint8)
        step = 40
        dim_minor = (14, 14, 22)
        dim_major = (24, 24, 36)
        for y in range(0, h, step):
            cv2.line(grid, (0, y), (w, y), dim_minor, 1)
        for x in range(0, w, step):
            cv2.line(grid, (x, 0), (x, h), dim_minor, 1)
        big = step * 4
        for y in range(0, h, big):
            cv2.line(grid, (0, y), (w, y), dim_major, 1)
        for x in range(0, w, big):
            cv2.line(grid, (x, 0), (x, h), dim_major, 1)
        cv2.line(grid, (w // 2, 0), (w // 2, h), dim_major, 1)
        cv2.line(grid, (0, h // 2), (w, h // 2), dim_major, 1)
        self._grid_cache[key] = grid
        return grid

    def _get_arena(self, h: int, w: int) -> np.ndarray:
        """Circular arena border for wxll.hx-style framing."""
        key = (h, w)
        cached = self._arena_cache.get(key)
        if cached is not None:
            return cached
        arena = np.zeros((h, w, 3), dtype=np.uint8)
        cx, cy = w // 2, h // 2
        r = int(min(w, h) * 0.42)
        cv2.circle(arena, (cx, cy), r, (40, 40, 70), 2, cv2.LINE_AA)
        cv2.circle(arena, (cx, cy), r + 3, (20, 20, 45), 1, cv2.LINE_AA)
        # tick marks
        for a in np.linspace(0, 2 * pi, 24, endpoint=False):
            angle = float(a)
            x1 = int(cx + (r - 6) * cos(angle))
            y1 = int(cy + (r - 6) * sin(angle))
            x2 = int(cx + r * cos(angle))
            y2 = int(cy + r * sin(angle))
            cv2.line(arena, (x1, y1), (x2, y2), (30, 30, 55), 1, cv2.LINE_AA)
        self._arena_cache[key] = arena
        return arena

    # ═══════════════════════════════════════════════════════════════════════
    # HUD decorators
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _draw_hud_frame(canvas: np.ndarray, color: tuple[int, int, int]) -> None:
        h, w = canvas.shape[:2]
        L = 28
        t = 2
        m = 14
        cv2.line(canvas, (m, m + L), (m, m), color, t)
        cv2.line(canvas, (m, m), (m + L, m), color, t)
        cv2.line(canvas, (w - m - L, m), (w - m, m), color, t)
        cv2.line(canvas, (w - m, m), (w - m, m + L), color, t)
        cv2.line(canvas, (m, h - m - L), (m, h - m), color, t)
        cv2.line(canvas, (m, h - m), (m + L, h - m), color, t)
        cv2.line(canvas, (w - m - L, h - m), (w - m, h - m), color, t)
        cv2.line(canvas, (w - m, h - m), (w - m, h - m - L), color, t)
        cx, cy = w // 2, h // 2
        cv2.line(canvas, (cx - 18, cy), (cx + 18, cy), color, 1)
        cv2.line(canvas, (cx, cy - 18), (cx, cy + 18), color, 1)
        cv2.line(canvas, (m, 2), (w - m, 2), color, 1)
        cv2.line(canvas, (m, h - 2), (w - m, h - 2), color, 1)
        cv2.line(canvas, (2, m), (2, h - m), color, 1)
        cv2.line(canvas, (w - 2, m), (w - 2, h - m), color, 1)

    # ═══════════════════════════════════════════════════════════════════════
    # Module 1 — Energy Orbs
    # ═══════════════════════════════════════════════════════════════════════

    def _draw_energy_orbs(
        self,
        canvas: np.ndarray,
        fingertips: list[Point],
        hi: int,
        primary: tuple[int, int, int],
        secondary: tuple[int, int, int],
        accent: tuple[int, int, int],
    ) -> None:
        orbs = self._orbs.setdefault(hi, [])
        while len(orbs) < len(fingertips):
            orbs.append(_OrbState(0.0, 0.0, 0.0, 12.0))
        while len(orbs) > len(fingertips):
            orbs.pop()

        for idx, ((fx, fy), orb) in enumerate(zip(fingertips, orbs)):
            orb.x += (fx - orb.x) * 0.35
            orb.y += (fy - orb.y) * 0.35
            orb.phase += 0.08 + idx * 0.015
            orb.base_r = 10.0 + 4.0 * sin(orb.phase * 2.3 + idx * 0.8)

            ox, oy = int(orb.x), int(orb.y)
            br = orb.base_r

            # core — solid bright
            cv2.circle(canvas, (ox, oy), int(br * 0.4), accent, -1, cv2.LINE_AA)

            # inner glow
            cv2.circle(
                canvas,
                (ox, oy),
                int(br * 0.85),
                secondary,
                2 if br > 12 else 1,
                cv2.LINE_AA,
            )

            # outer aura rings
            for ring_i in range(_ORB_RING_LAYERS):
                r_phase = orb.phase + ring_i * 2.1
                r_radius = br * (1.4 + ring_i * 0.65)
                r_anim = r_radius + 4.0 * sin(r_phase * 1.7)
                alpha = 1.0 - ring_i * 0.32
                ring_color: tuple[int, int, int] = (
                    int(accent[0] * alpha),
                    int(accent[1] * alpha),
                    int(accent[2] * alpha),
                )
                cv2.circle(
                    canvas,
                    (ox, oy),
                    int(r_anim),
                    ring_color,
                    1,
                    cv2.LINE_AA,
                )

            # orbital glow — larger subtle outer ring
            orb_r = br * 2.6 + 6.0 * sin(orb.phase * 0.7)
            orb_color: tuple[int, int, int] = (
                int(primary[0] * 0.3),
                int(primary[1] * 0.3),
                int(primary[2] * 0.3),
            )
            cv2.circle(canvas, (ox, oy), int(orb_r), orb_color, 2, cv2.LINE_AA)

    # ═══════════════════════════════════════════════════════════════════════
    # Module 2 — Particle System v2
    # ═══════════════════════════════════════════════════════════════════════

    def _emit_particles(
        self,
        x: int,
        y: int,
        colors: tuple[
            tuple[int, int, int],
            tuple[int, int, int],
            tuple[int, int, int],
        ],
        count: int = 5,
    ) -> None:
        primary, secondary, accent = colors
        palette = (primary, secondary, accent)
        for _ in range(count):
            if len(self._particles) >= _MAX_PARTICLES:
                break
            angle = random() * 2 * pi
            speed = uniform(1.5, 5.5)
            c = palette[randint(0, 2)]
            c_end = palette[randint(0, 2)]
            self._particles.append(
                _Particle(
                    x=float(x),
                    y=float(y),
                    vx=cos(angle) * speed,
                    vy=sin(angle) * speed,
                    life=1.0,
                    max_life=uniform(0.5, 1.4),
                    size=uniform(1.5, 4.5),
                    color=c,
                    color_end=c_end,
                    shape=randint(0, 3),
                    spawn_x=float(x),
                    spawn_y=float(y),
                    orbit=random() * 2 * pi,
                    orbit_speed=uniform(0.03, 0.12),
                )
            )

    def _step_particles(self, h: int, w: int, cx: float, cy: float) -> None:
        alive: list[_Particle] = []
        for p in self._particles:
            p.orbit += p.orbit_speed
            p.life -= 0.018

            # organic motion: orbit around spawn + gravity toward centroid
            dist = sqrt((p.x - p.spawn_x) ** 2 + (p.y - p.spawn_y) ** 2)
            gx = (cx - p.x) * 0.002
            gy = (cy - p.y) * 0.002
            orbit_amp = 0.25 + 0.3 * sin(dist * 0.04)
            ox = cos(p.orbit) * orbit_amp
            oy = sin(p.orbit) * orbit_amp

            p.vx += gx + ox
            p.vy += gy + oy
            p.vx *= 0.96
            p.vy *= 0.96
            p.x += p.vx
            p.y += p.vy

            if p.life <= 0:
                continue
            if not (-20 <= p.x < w + 20 and -20 <= p.y < h + 20):
                continue
            alive.append(p)
        self._particles = alive

    def _draw_particles(self, canvas: np.ndarray) -> None:
        for p in self._particles:
            t = 1.0 - p.life / p.max_life
            cr = int(p.color[0] + (p.color_end[0] - p.color[0]) * t)
            cg = int(p.color[1] + (p.color_end[1] - p.color[1]) * t)
            cb = int(p.color[2] + (p.color_end[2] - p.color[2]) * t)
            alpha = p.life
            c: tuple[int, int, int] = (
                int(cr * alpha),
                int(cg * alpha),
                int(cb * alpha),
            )
            px, py = int(p.x), int(p.y)
            sz = max(1, int(p.size * alpha))

            if p.shape == 0:
                cv2.circle(canvas, (px, py), sz, c, -1, cv2.LINE_AA)
            elif p.shape == 1:
                # diamond
                pts = np.array(
                    [
                        [px, py - sz],
                        [px + sz, py],
                        [px, py + sz],
                        [px - sz, py],
                    ],
                    dtype=np.int32,
                )
                cv2.fillPoly(canvas, [pts], c, cv2.LINE_AA)
            elif p.shape == 2:
                # cross / plus
                cv2.line(
                    canvas,
                    (px - sz, py),
                    (px + sz, py),
                    c,
                    max(1, sz // 2),
                    cv2.LINE_AA,
                )
                cv2.line(
                    canvas,
                    (px, py - sz),
                    (px, py + sz),
                    c,
                    max(1, sz // 2),
                    cv2.LINE_AA,
                )
            else:
                # small square
                cv2.rectangle(
                    canvas,
                    (px - sz, py - sz),
                    (px + sz, py + sz),
                    c,
                    -1,
                    cv2.LINE_AA,
                )

    # ═══════════════════════════════════════════════════════════════════════
    # Module 3 — Arc Connections (Bezier between fingertips)
    # ═══════════════════════════════════════════════════════════════════════

    def _draw_arcs(
        self,
        canvas: np.ndarray,
        ordered: list[Point],
        primary: tuple[int, int, int],
        secondary: tuple[int, int, int],
    ) -> None:
        n = len(ordered)
        if n < 2:
            return
        for i in range(n):
            a = ordered[i]
            b = ordered[(i + 1) % n]
            # upward arch control point
            mx = (a[0] + b[0]) / 2
            my = (a[1] + b[1]) / 2
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            dist = sqrt(dx * dx + dy * dy)
            offset = min(dist * 0.55, 60.0)
            cpx = mx - dy / max(dist, 1) * offset
            cpy = my + dx / max(dist, 1) * offset

            prev: tuple[float, float] | None = None
            for seg in range(_ARC_SEGMENTS + 1):
                t = seg / _ARC_SEGMENTS
                t_inv = 1.0 - t
                px = t_inv * t_inv * a[0] + 2 * t_inv * t * cpx + t * t * b[0]
                py = t_inv * t_inv * a[1] + 2 * t_inv * t * cpy + t * t * b[1]
                if prev is not None:
                    cv2.line(
                        canvas,
                        (int(prev[0]), int(prev[1])),
                        (int(px), int(py)),
                        secondary,
                        1,
                        cv2.LINE_AA,
                    )
                prev = (px, py)

            # flow dots along the arc
            flow_t = (self._phase * 0.55) % 1.0
            for dot_i in range(3):
                dt = (flow_t + dot_i * 0.33) % 1.0
                t_inv = 1.0 - dt
                dx_p = t_inv * t_inv * a[0] + 2 * t_inv * dt * cpx + dt * dt * b[0]
                dy_p = t_inv * t_inv * a[1] + 2 * t_inv * dt * cpy + dt * dt * b[1]
                dot_alpha = abs(sin(dt * pi))
                dot_color: tuple[int, int, int] = (
                    int(primary[0] * dot_alpha),
                    int(primary[1] * dot_alpha),
                    int(primary[2] * dot_alpha),
                )
                cv2.circle(
                    canvas,
                    (int(dx_p), int(dy_p)),
                    3,
                    dot_color,
                    -1,
                    cv2.LINE_AA,
                )

    # ═══════════════════════════════════════════════════════════════════════
    # Module 4 — Ripple Waves
    # ═══════════════════════════════════════════════════════════════════════

    def _emit_ripple(
        self,
        x: float,
        y: float,
        color: tuple[int, int, int],
    ) -> None:
        if len(self._ripples) >= _MAX_RIPPLES:
            return
        self._ripples.append(
            _Ripple(
                x=x,
                y=y,
                radius=0.0,
                max_radius=uniform(120, 260),
                life=1.0,
                decay=uniform(0.008, 0.016),
                color=color,
                thickness=randint(1, 3),
            )
        )

    def _step_ripples(self, h: int, w: int) -> None:
        alive: list[_Ripple] = []
        for r in self._ripples:
            r.radius += (r.max_radius - r.radius) * 0.04 + 1.2
            r.life -= r.decay
            if r.life <= 0 or r.radius >= r.max_radius:
                continue
            alive.append(r)
        self._ripples = alive

    def _draw_ripples(self, canvas: np.ndarray) -> None:
        for r in self._ripples:
            alpha = r.life
            c: tuple[int, int, int] = (
                int(r.color[0] * alpha),
                int(r.color[1] * alpha),
                int(r.color[2] * alpha),
            )
            cv2.circle(
                canvas,
                (int(r.x), int(r.y)),
                int(r.radius),
                c,
                r.thickness,
                cv2.LINE_AA,
            )
            # inner echo
            if r.radius > 14:
                inner_r = int(r.radius * 0.55)
                inner_color: tuple[int, int, int] = (
                    int(r.color[0] * alpha * 0.45),
                    int(r.color[1] * alpha * 0.45),
                    int(r.color[2] * alpha * 0.45),
                )
                cv2.circle(
                    canvas,
                    (int(r.x), int(r.y)),
                    inner_r,
                    inner_color,
                    1,
                    cv2.LINE_AA,
                )

    # ═══════════════════════════════════════════════════════════════════════
    # Module 5 — Light Rays
    # ═══════════════════════════════════════════════════════════════════════

    def _draw_light_rays(
        self,
        canvas: np.ndarray,
        fingertips: list[Point],
        accent: tuple[int, int, int],
    ) -> None:
        for idx, (fx, fy) in enumerate(fingertips):
            base_angle = idx * 1.2 + self._phase * 0.3
            for ray_i in range(_RAY_COUNT):
                a = base_angle + ray_i * 2 * pi / _RAY_COUNT
                length = 18 + 14 * sin(self._phase * 2.5 + ray_i * 1.3 + idx)
                ex = int(fx + cos(a) * length)
                ey = int(fy + sin(a) * length)
                ray_alpha = 0.25 + 0.15 * sin(self._phase * 3 + ray_i)
                ray_color: tuple[int, int, int] = (
                    int(accent[0] * ray_alpha),
                    int(accent[1] * ray_alpha),
                    int(accent[2] * ray_alpha),
                )
                cv2.line(
                    canvas,
                    (fx, fy),
                    (ex, ey),
                    ray_color,
                    1,
                    cv2.LINE_AA,
                )

    # ═══════════════════════════════════════════════════════════════════════
    # Module 6 — Energy Field
    # ═══════════════════════════════════════════════════════════════════════

    def _draw_energy_field(
        self,
        canvas: np.ndarray,
        cx: int,
        cy: int,
        radius: float,
        primary: tuple[int, int, int],
        secondary: tuple[int, int, int],
        accent: tuple[int, int, int],
    ) -> None:
        h, w = canvas.shape[:2]
        r_int = int(radius)
        y0 = max(0, cy - r_int)
        y1 = min(h, cy + r_int)
        x0 = max(0, cx - r_int)
        x1 = min(w, cx + r_int)
        if y0 >= y1 or x0 >= x1:
            return

        # radial gradient within bounding box
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask = np.clip(1.0 - dist / max(radius, 1), 0.0, 1.0)

        # noise-like variation
        noise = 0.5 + 0.5 * np.sin(xx * 0.08 + self._phase * 2.0 + yy * 0.04) * np.cos(
            yy * 0.06 - self._phase * 1.4 + xx * 0.05
        )
        alpha = mask * noise * 0.10

        for ch in range(3):
            color_val = (primary[ch], secondary[ch], accent[ch])[(cx + ch) % 3]
            canvas[y0:y1, x0:x1, ch] = np.clip(
                canvas[y0:y1, x0:x1, ch].astype(np.float32) + alpha * color_val,
                0,
                255,
            ).astype(np.uint8)

    # ═══════════════════════════════════════════════════════════════════════
    # Trails
    # ═══════════════════════════════════════════════════════════════════════

    def _update_trails(self, hand_idx: int, fingertips: list[Point]) -> None:
        trails = self._trails.setdefault(hand_idx, {})
        ages = self._trail_ages.setdefault(hand_idx, {})
        for fi, pt in enumerate(fingertips):
            trail = trails.setdefault(fi, [])
            trail.append(pt)
            if len(trail) > _MAX_TRAIL:
                del trail[0]
            ages[fi] = 0

    def _age_trails(self) -> None:
        for hi in list(self._trails.keys()):
            ages = self._trail_ages.get(hi)
            if ages is None:
                continue
            trails = self._trails[hi]
            stale: list[int] = []
            for fi, age in ages.items():
                ages[fi] = age + 1
                if ages[fi] > 10:
                    stale.append(fi)
            for fi in stale:
                trails.pop(fi, None)
                ages.pop(fi, None)
            if not trails:
                self._trails.pop(hi, None)
                self._trail_ages.pop(hi, None)

    def _draw_trails(
        self,
        canvas: np.ndarray,
        primary: tuple[int, int, int],
        secondary: tuple[int, int, int],
    ) -> None:
        for trails in self._trails.values():
            for trail in trails.values():
                n = len(trail)
                if n < 2:
                    continue
                pts = np.array(trail, dtype=np.int32)
                for i in range(n - 1):
                    alpha = (i + 1) / n
                    mix_t = i / max(n - 1, 1)
                    cr = int(primary[0] * (1 - mix_t) + secondary[0] * mix_t)
                    cg = int(primary[1] * (1 - mix_t) + secondary[1] * mix_t)
                    cb = int(primary[2] * (1 - mix_t) + secondary[2] * mix_t)
                    c = (
                        int(cr * alpha * 0.7),
                        int(cg * alpha * 0.7),
                        int(cb * alpha * 0.7),
                    )
                    thick = int(alpha * 4) + 1
                    cv2.line(
                        canvas,
                        tuple(pts[i]),
                        tuple(pts[i + 1]),
                        c,
                        thick,
                        cv2.LINE_AA,
                    )

    # ═══════════════════════════════════════════════════════════════════════
    # Flow dots
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _draw_flow_dots(
        canvas: np.ndarray,
        a: Point,
        b: Point,
        color: tuple[int, int, int],
        offset: float,
    ) -> None:
        for i in range(4):
            t = (offset + i * 0.25) % 1.0
            x = int(a[0] + (b[0] - a[0]) * t)
            y = int(a[1] + (b[1] - a[1]) * t)
            cv2.circle(canvas, (x, y), 2, color, -1, cv2.LINE_AA)

    # ═══════════════════════════════════════════════════════════════════════
    # Glitch
    # ═══════════════════════════════════════════════════════════════════════

    def _tick_glitch(self, h: int) -> None:
        self._glitch_timer += 1.0
        if self._glitch_timer > 100.0:
            self._glitch_timer = 0.0
            if random() < 0.35:
                self._glitch_active = True
                self._glitch_row = randint(h // 6, 5 * h // 6)
                self._glitch_height = randint(6, 26)
                self._glitch_shift = randint(-32, 32)
                self._glitch_boost_b = randint(60, 150)
        if self._glitch_active:
            self._glitch_active = False

    def _apply_glitch(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        y0 = max(0, self._glitch_row)
        y1 = min(h, y0 + self._glitch_height)
        shift = self._glitch_shift
        if shift == 0 or y1 <= y0:
            return
        strip = frame[y0:y1].copy()
        if shift > 0:
            strip[:, shift:] = strip[:, : w - shift]
            strip[:, :shift] = 0
        else:
            s = -shift
            strip[:, : w - s] = strip[:, s:]
            strip[:, w - s :] = 0
        overlay = np.zeros_like(strip)
        overlay[:, :, 0] = self._glitch_boost_b
        strip = cv2.addWeighted(strip, 0.65, overlay, 0.35, 0)
        frame[y0:y1] = strip

    # ═══════════════════════════════════════════════════════════════════════
    # Background rendering
    # ═══════════════════════════════════════════════════════════════════════

    def _render_background(
        self,
        result: np.ndarray,
        h: int,
        w: int,
        accent: tuple[int, int, int],
    ) -> None:
        # scanlines
        scan = np.zeros((h, w, 3), dtype=np.uint8)
        offset = int(self._phase * 90) % 8
        for y in range(offset, h, 8):
            cv2.line(scan, (0, y), (w, y), (30, 30, 50), 1)
        h_offset = int(self._phase * 130) % 40
        for y in range(h_offset, h, 40):
            cv2.line(scan, (0, y), (w, y), (80, 80, 130), 1)
            cv2.line(scan, (0, y + 1), (w, y + 1), (50, 50, 85), 1)
        alpha_s = 0.06 + 0.025 * np.sin(self._phase * 1.6)
        cv2.addWeighted(result, 1.0, scan, float(alpha_s), 0, dst=result)

        # grid
        grid = self._get_grid(h, w)
        grid_alpha = 0.04 + 0.015 * np.sin(self._phase * 0.4)
        cv2.addWeighted(result, 1.0, grid, float(grid_alpha), 0, dst=result)

        # arena ring
        arena = self._get_arena(h, w)
        arena_alpha = 0.12 + 0.04 * np.sin(self._phase * 0.35)
        cv2.addWeighted(result, 1.0, arena, float(arena_alpha), 0, dst=result)

        # vignette
        vig = self._get_vignette(h, w)
        buf = cv2.multiply(
            result.astype(np.float32), vig.astype(np.float32) / 255.0
        ).astype(np.uint8)
        np.copyto(result, buf)

        # glitch
        self._apply_glitch(result)

        # HUD frame
        dim_color: tuple[int, int, int] = (
            int(accent[0] * 0.55),
            int(accent[1] * 0.55),
            int(accent[2] * 0.55),
        )
        hud = np.zeros((h, w, 3), dtype=np.uint8)
        self._draw_hud_frame(hud, dim_color)
        cv2.addWeighted(result, 1.0, hud, 0.55, 0, dst=result)

    # ═══════════════════════════════════════════════════════════════════════
    # Main compositing entry point
    # ═══════════════════════════════════════════════════════════════════════

    def apply(self, frame: np.ndarray, hands: list[list[Point]]) -> np.ndarray:
        self._phase += 0.11
        primary, secondary, accent = self._PALETTES[self._style]
        h, w = frame.shape[:2]

        # ---- layers ----
        fill_layer = np.zeros((h, w, 3), dtype=np.uint8)
        glow = np.zeros((h, w, 3), dtype=np.uint8)

        active: list[tuple[Point, list[Point]]] = []
        total_cx = 0.0
        total_cy = 0.0
        centroid_count = 0.0

        for hi, fingertips in enumerate(hands):
            if len(fingertips) < 2:
                continue
            ordered = sorted(fingertips, key=lambda p: p[0])
            n = len(ordered)

            cx = sum(p[0] for p in ordered) / n
            cy = sum(p[1] for p in ordered) / n
            centroid: Point = (int(cx), int(cy))
            active.append((centroid, ordered))

            total_cx += cx
            total_cy += cy
            centroid_count += 1.0

            self._update_trails(hi, ordered)

            pulse = 0.5 + 0.5 * np.sin(self._phase * 2.7)
            flow_off = (self._phase * 0.65) % 1.0

            # -- fill layer --
            poly_pts = np.array(ordered + [ordered[0]], dtype=np.int32)
            dim_primary: tuple[int, int, int] = (
                int(primary[0] * 0.55),
                int(primary[1] * 0.55),
                int(primary[2] * 0.55),
            )
            cv2.fillPoly(fill_layer, [poly_pts], dim_primary)

            dim_secondary: tuple[int, int, int] = (
                int(secondary[0] * 0.45),
                int(secondary[1] * 0.45),
                int(secondary[2] * 0.45),
            )
            for a, b in pairwise(ordered):
                tri = np.array([(centroid[0], centroid[1]), a, b], dtype=np.int32)
                cv2.fillPoly(fill_layer, [tri], dim_secondary)

            # -- glow layer: polygon outline --
            cv2.polylines(glow, [poly_pts], True, primary, 2, cv2.LINE_AA)
            cv2.polylines(
                glow,
                [poly_pts],
                True,
                (
                    int(primary[0] * pulse),
                    int(primary[1] * pulse),
                    int(primary[2] * pulse),
                ),
                1,
                cv2.LINE_AA,
            )

            # centroid spokes
            for pt in ordered:
                cv2.line(glow, centroid, pt, secondary, 1, cv2.LINE_AA)

            # -- arc connections (Module 3) --
            self._draw_arcs(glow, ordered, primary, secondary)

            # -- flow dots --
            for a, b in pairwise(ordered):
                self._draw_flow_dots(glow, a, b, accent, flow_off)
            self._draw_flow_dots(glow, ordered[-1], ordered[0], accent, flow_off)

            # -- energy orbs (Module 1) --
            self._draw_energy_orbs(glow, ordered, hi, primary, secondary, accent)

            # -- light rays (Module 5) --
            self._draw_light_rays(glow, ordered, accent)

            # fingertip markers (small core dots)
            for _i, (x, y) in enumerate(ordered):
                cv2.circle(glow, (x, y), 3, accent, -1, cv2.LINE_AA)

            # centroid marker
            cv2.circle(glow, centroid, 5, accent, -1, cv2.LINE_AA)

            # particles
            if random() < 0.18:
                for x, y in ordered:
                    self._emit_particles(x, y, (primary, secondary, accent), count=3)
            # centroid particle burst
            if random() < 0.07:
                self._emit_particles(
                    centroid[0],
                    centroid[1],
                    (primary, secondary, accent),
                    count=7,
                )

            # ripples (Module 4)
            if random() < 0.03:
                self._emit_ripple(float(cx), float(cy), accent)

        # -- no hands --
        if not active:
            self._trails.clear()
            self._trail_ages.clear()
            self._particles.clear()
            self._ripples.clear()
            self._orbs.clear()
            self._prev_hands = 0
            result = frame.copy()
            self._render_background(result, h, w, accent)
            return result

        # entry burst — when hands first appear
        if self._prev_hands == 0 and len(active) > 0:
            for cent, ordered in active:
                for x, y in ordered:
                    self._emit_particles(x, y, (primary, secondary, accent), count=12)
        self._prev_hands = len(active)

        # step simulations
        avg_cx = total_cx / centroid_count if centroid_count > 0 else w / 2
        avg_cy = total_cy / centroid_count if centroid_count > 0 else h / 2
        self._step_particles(h, w, avg_cx, avg_cy)
        self._step_ripples(h, w)
        self._tick_glitch(h)
        self._age_trails()

        # draw simulations onto glow
        self._draw_trails(glow, primary, secondary)
        self._draw_particles(glow)
        self._draw_ripples(glow)

        # energy field (Module 6)
        field = np.zeros((h, w, 3), dtype=np.uint8)
        for cent, _ in active:
            hand_radius = float(
                max(
                    sqrt((p[0] - cent[0]) ** 2 + (p[1] - cent[1]) ** 2)
                    for p in hands[active.index((cent, _))]
                    if hands[active.index((cent, _))]
                )
                or 100.0
            )
            self._draw_energy_field(
                field,
                cent[0],
                cent[1],
                hand_radius * 1.8,
                primary,
                secondary,
                accent,
            )

        # -- Module 7: enhanced bloom compositing --
        fill_blur = cv2.GaussianBlur(fill_layer, (0, 0), 20)
        glow_tight = cv2.GaussianBlur(glow, (0, 0), 3)
        glow_wide = cv2.GaussianBlur(glow, (0, 0), 12)
        glow_extra = cv2.GaussianBlur(glow, (0, 0), 28)
        field_blur = cv2.GaussianBlur(field, (0, 0), 8)

        # chromatic aberration on tight glow
        b_arr, g_arr, r_arr = cv2.split(glow_tight)
        s = 5
        M_r = np.array([[1, 0, s], [0, 1, 0]], dtype=np.float64)
        M_b = np.array([[1, 0, -s], [0, 1, 0]], dtype=np.float64)
        r_s = cv2.warpAffine(r_arr, M_r, (w, h))
        b_s = cv2.warpAffine(b_arr, M_b, (w, h))
        glow_ab = cv2.merge([b_s, g_arr, r_s])

        # additive compositing
        result = frame.astype(np.float32)
        result += fill_blur.astype(np.float32) * 0.18
        result += glow_extra.astype(np.float32) * 0.18
        result += glow_wide.astype(np.float32) * 0.25
        result += glow_ab.astype(np.float32) * 0.30
        result += glow.astype(np.float32) * 0.40
        result += field_blur.astype(np.float32) * 0.06
        np.clip(result, 0, 255, out=result)
        result_u8 = result.astype(np.uint8)

        self._render_background(result_u8, h, w, accent)

        return result_u8
