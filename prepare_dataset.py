"""Create a transparency-aware copy of the cleaned image dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def background_mask(image_bgr: np.ndarray, threshold: int) -> np.ndarray:
    """Return black regions connected to the image border."""

    black = np.all(image_bgr <= threshold, axis=2).astype(np.uint8)
    height, width = black.shape
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    connected = np.zeros_like(black)

    for x in range(width):
        if black[0, x]:
            cv2.floodFill(black, flood_mask, (x, 0), 2)
        if black[height - 1, x]:
            cv2.floodFill(black, flood_mask, (x, height - 1), 2)
    for y in range(height):
        if black[y, 0]:
            cv2.floodFill(black, flood_mask, (0, y), 2)
        if black[y, width - 1]:
            cv2.floodFill(black, flood_mask, (width - 1, y), 2)

    connected[black == 2] = 255
    return connected.astype(bool)


def convert_image(source_path: Path, destination_path: Path, threshold: int) -> None:
    """Convert one source image to RGBA while preserving its filename."""

    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {source_path}")

    alpha = np.full(image.shape[:2], 255, dtype=np.uint8)
    alpha[background_mask(image, threshold)] = 0
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination_path), rgba):
        raise OSError(f"Unable to write image: {destination_path}")


def convert_dataset(source_root: Path, destination_root: Path, threshold: int) -> tuple[int, int]:
    """Convert all supported images and return (converted, skipped)."""

    converted = 0
    skipped = 0
    for source_path in source_root.rglob("*"):
        if not source_path.is_file():
            continue
        relative_path = source_path.relative_to(source_root)
        destination_path = destination_root / relative_path
        if source_path.suffix.lower() not in IMAGE_EXTENSIONS:
            skipped += 1
            continue
        convert_image(source_path, destination_path.with_suffix(".png"), threshold)
        converted += 1
    return converted, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("dataset_clean"))
    parser.add_argument("--destination", type=Path, default=Path("dataset_new"))
    parser.add_argument(
        "--threshold",
        type=int,
        default=3,
        help="Maximum value for a pixel to be treated as black background.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {args.source}")
    if not 0 <= args.threshold <= 255:
        raise ValueError("--threshold must be between 0 and 255")

    converted, skipped = convert_dataset(args.source, args.destination, args.threshold)
    print(f"Converted {converted} images into {args.destination}")
    if skipped:
        print(f"Skipped {skipped} unsupported files")


if __name__ == "__main__":
    main()
