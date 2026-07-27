"""Image preprocessing utilities implemented with OpenCV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .config import PreprocessingConfig


@dataclass(slots=True)
class PreprocessResult:
    """Container for preprocessed images and their paths."""

    images: np.ndarray
    paths: list[str]


class ImagePreprocessor:
    """Apply configurable OpenCV preprocessing to images."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config

    def load_image(self, image_path: str | Path) -> np.ndarray:
        """Load an image from disk using OpenCV."""

        path = str(image_path)
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {path}")
        return image

    def preprocess_array(self, image: np.ndarray) -> np.ndarray:
        """Resize, convert to RGB, optionally enhance, and normalize an image."""

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image,
            self.config.image_size,
            interpolation=cv2.INTER_AREA,
        )

        if self.config.use_clahe:
            image = self._apply_clahe(image)

        if self.config.use_histogram_equalization:
            image = self._apply_histogram_equalization(image)

        image = image.astype(np.float32)
        if self.config.normalize:
            image /= 255.0
        return image

    def preprocess_path(self, image_path: str | Path) -> np.ndarray:
        """Load an image from disk and preprocess it."""

        return self.preprocess_array(self.load_image(image_path))

    def preprocess_batch(self, image_paths: Iterable[str | Path]) -> PreprocessResult:
        """Preprocess a batch of image paths deterministically."""

        processed_images: list[np.ndarray] = []
        processed_paths: list[str] = []
        for image_path in image_paths:
            processed_images.append(self.preprocess_path(image_path))
            processed_paths.append(str(image_path))
        if not processed_images:
            empty_shape = (*self.config.image_size, 3)
            return PreprocessResult(
                images=np.empty((0, *empty_shape), dtype=np.float32),
                paths=[],
            )
        return PreprocessResult(images=np.stack(processed_images), paths=processed_paths)

    def _apply_clahe(self, image_rgb: np.ndarray) -> np.ndarray:
        lab_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid_size,
        )
        l_channel = clahe.apply(l_channel)
        merged = cv2.merge((l_channel, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    def _apply_histogram_equalization(self, image_rgb: np.ndarray) -> np.ndarray:
        ycrcb_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
        y_channel, cr_channel, cb_channel = cv2.split(ycrcb_image)
        y_channel = cv2.equalizeHist(y_channel)
        merged = cv2.merge((y_channel, cr_channel, cb_channel))
        return cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB)
