import argparse
import sys
import time
from pathlib import Path

import cv2

from detector import HandDetector
from effects import CyberEffect


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
        default="ethereal",
        choices=[
            "arcane",
            "plasma",
            "matrix",
            "cosmic",
            "frost",
            "neon",
            "synthwave",
            "hacker",
            "ethereal",
            "phoenix",
            "aurora",
            "void",
            "prism",
            "ember",
            "ocean",
        ],
    )
    p.add_argument(
        "--camera", "-c", type=int, default=0, help="Camera device index (default: 0)"
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

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
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        source_type = "webcam"
        writer = None
        print(f"[webcam] device {args.camera}")

    if not cap.isOpened():
        print("Error: cannot open video source", file=sys.stderr)
        sys.exit(1)

    detector = HandDetector()
    effect_map = {
        "arcane": CyberEffect("arcane"),
        "plasma": CyberEffect("plasma"),
        "matrix": CyberEffect("matrix"),
        "cosmic": CyberEffect("cosmic"),
        "frost": CyberEffect("frost"),
        "neon": CyberEffect("neon"),
        "synthwave": CyberEffect("synthwave"),
        "hacker": CyberEffect("hacker"),
        "ethereal": CyberEffect("ethereal"),
        "phoenix": CyberEffect("phoenix"),
        "aurora": CyberEffect("aurora"),
        "void": CyberEffect("void"),
        "prism": CyberEffect("prism"),
        "ember": CyberEffect("ember"),
        "ocean": CyberEffect("ocean"),
    }
    active = args.effect

    paused = False
    frame_idx = 0
    t0 = time.time()

    print(
        "\nKeys: 1-8 classic | a=aurora e=ethereal m=ember o=ocean p=phoenix r=prism v=void | SPACE=pause | q=quit\n"
    )

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

        hands = detector.detect(frame)
        fingertips_list = [h.fingertips for h in hands]
        result = effect_map[active].apply(frame, fingertips_list)

        fps_now = 1.0 / max(time.time() - t0, 1e-3) if frame_idx > 0 else 0.0
        t0 = time.time()
        cv2.putText(
            result,
            f"Style: {active}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            result,
            f"Hands: {len(hands)}",
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
            active = "arcane"
        if key == ord("2"):
            active = "plasma"
        if key == ord("3"):
            active = "matrix"
        if key == ord("4"):
            active = "cosmic"
        if key == ord("5"):
            active = "frost"
        if key == ord("6"):
            active = "neon"
        if key == ord("7"):
            active = "synthwave"
        if key == ord("8"):
            active = "hacker"
        if key == ord("a"):
            active = "aurora"
        if key == ord("e"):
            active = "ethereal"
        if key == ord("m"):
            active = "ember"
        if key == ord("o"):
            active = "ocean"
        if key == ord("p"):
            active = "phoenix"
        if key == ord("r"):
            active = "prism"
        if key == ord("v"):
            active = "void"

        frame_idx += 1

    cap.release()
    if writer and out_path:
        writer.release()
        print(f"\nSaved -> {out_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
