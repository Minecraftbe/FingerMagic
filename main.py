import argparse
import sys
import time
from pathlib import Path

import cv2

from detector import HandDetector
from effects import (
    BaseEffect,
    MagicPortalEffect,
    NeonGlowEffect,
    ParticleEffect,
    RainbowEffect,
    StarfieldEffect,
    convex_hull_of,
    polygon_mask,
)


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
    effect_map: dict[str, BaseEffect] = {
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

        fingertips = detector.find_fingertips(frame)
        hull_pts = convex_hull_of(fingertips)
        pmask = polygon_mask(frame.shape[:2], hull_pts)

        result = frame.copy()
        for name in active:
            e = effect_map[name]
            if name == "particle":
                result = e.apply(result, fingertips)
            elif name == "neon":
                result = e.apply(result, fingertips)
            else:
                result = e.apply(result, pmask)

        for pt in fingertips:
            cv2.circle(result, pt, 9, (0, 255, 100), -1)
            cv2.circle(result, pt, 13, (0, 255, 100), 2)

        if len(hull_pts) >= 2:
            for i in range(len(hull_pts)):
                a = hull_pts[i]
                b = hull_pts[(i + 1) % len(hull_pts)]
                cv2.line(result, a, b, (0, 220, 255), 3)

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
