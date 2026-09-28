from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import ImageDraw

from court_geometry import draw_court, feet_to_pixels


def is_make(result: str) -> bool:
    normalized = result.strip().lower()
    return normalized in {"make", "made", "1", "true", "yes"}


def draw_shot(draw: ImageDraw.ImageDraw, x_px: float, y_px: float, made: bool) -> None:
    radius = 8
    if made:
        draw.ellipse([x_px - radius, y_px - radius, x_px + radius, y_px + radius], outline="#16803c", width=3)
        draw.ellipse([x_px - 3, y_px - 3, x_px + 3, y_px + 3], fill="#16803c")
        return

    draw.line([x_px - radius, y_px - radius, x_px + radius, y_px + radius], fill="#b42222", width=3)
    draw.line([x_px - radius, y_px + radius, x_px + radius, y_px - radius], fill="#b42222", width=3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a top-down shot chart from projected event coordinates.")
    parser.add_argument("--events", required=True, help="CSV from project_events.py.")
    parser.add_argument("--out", required=True, help="PNG output path.")
    parser.add_argument("--scale", type=float, default=20.0, help="Pixels per foot in the chart.")
    parser.add_argument("--margin", type=int, default=40, help="Margin around the court in pixels.")
    args = parser.parse_args()

    events_path = Path(args.events).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image = draw_court(scale_px_per_ft=args.scale, margin_px=args.margin, annotate_landmarks=False)
    draw = ImageDraw.Draw(image)

    makes = 0
    attempts = 0
    with events_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            x_ft = float(row["court_x_ft"])
            y_ft = float(row["court_y_ft"])
            x_px, y_px = feet_to_pixels((x_ft, y_ft), args.scale, args.margin)
            made = is_make(row.get("result", ""))
            makes += int(made)
            attempts += 1
            draw_shot(draw, x_px, y_px, made)

    draw.text((args.margin, 10), f"Attempts: {attempts}  Makes: {makes}", fill="#151515")
    image.save(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
