from __future__ import annotations

import argparse
from pathlib import Path

from court_geometry import draw_court, write_landmark_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a top-down reference court image.")
    parser.add_argument("--out", required=True, help="PNG output path.")
    parser.add_argument("--landmarks-out", help="Optional JSON output with landmark names and feet coordinates.")
    parser.add_argument("--scale", type=float, default=20.0, help="Pixels per foot in the reference image.")
    parser.add_argument("--margin", type=int, default=40, help="Margin around the court in pixels.")
    parser.add_argument(
        "--no-landmark-labels",
        action="store_true",
        help="Draw the court without point labels.",
    )
    args = parser.parse_args()

    image = draw_court(
        scale_px_per_ft=args.scale,
        margin_px=args.margin,
        annotate_landmarks=not args.no_landmark_labels,
    )
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print(out_path)

    if args.landmarks_out:
        landmarks_out = Path(args.landmarks_out).expanduser().resolve()
        write_landmark_file(landmarks_out)
        print(landmarks_out)


if __name__ == "__main__":
    main()
