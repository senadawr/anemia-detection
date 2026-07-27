"""Dataset discovery, statistics, and tf.data loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf

from .config import AppConfig, PreprocessingConfig
from .preprocessing import ImagePreprocessor

IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(slots=True)
class DatasetSplit:
    """Dataset paths and integer labels for a single split."""

    name: str
    paths: list[str]
    labels: np.ndarray
    class_names: tuple[str, ...]

    @property
    def sample_count(self) -> int:
        return len(self.paths)


@dataclass(slots=True)
class SplitStatistics:
    """Basic split statistics for reporting."""

    total_samples: int
    class_counts: dict[str, int]


@dataclass(slots=True)
class DatasetBundle:
    """Container for all dataset splits."""

    training: DatasetSplit
    validation: DatasetSplit
    testing: DatasetSplit

    def all_splits(self) -> dict[str, DatasetSplit]:
        return {
            "training": self.training,
            "validation": self.validation,
            "testing": self.testing,
        }


class DatasetLoader:
    """Load dataset splits from the cleaned folder structure."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.preprocessor = ImagePreprocessor(config.preprocessing)

    def discover_splits(self) -> DatasetBundle:
        """Discover paths and labels for all configured splits."""

        class_names = self._discover_class_names()
        training = self._discover_split("training", class_names)
        validation = self._discover_split("validation", class_names)
        testing = self._discover_split("testing", class_names)
        return DatasetBundle(training=training, validation=validation, testing=testing)

    def describe(self, bundle: DatasetBundle) -> str:
        """Return a human-readable dataset summary."""

        lines: list[str] = []
        for split_name, split in bundle.all_splits().items():
            stats = self.compute_statistics(split)
            lines.append(f"{split_name}: {stats.total_samples}")
            for class_name, count in stats.class_counts.items():
                lines.append(f"  {class_name}: {count}")
        return "\n".join(lines)

    def compute_statistics(self, split: DatasetSplit) -> SplitStatistics:
        """Compute class counts for a split."""

        class_counts = {
            class_name: int(np.sum(split.labels == index))
            for index, class_name in enumerate(split.class_names)
        }
        return SplitStatistics(total_samples=split.sample_count, class_counts=class_counts)

    def build_tf_dataset(
        self,
        split: DatasetSplit,
        batch_size: int,
        shuffle: bool = False,
    ) -> tf.data.Dataset:
        """Create a batched tf.data pipeline without shuffling."""

        paths = np.asarray(split.paths, dtype=np.str_)
        labels = np.asarray(split.labels, dtype=np.int64)
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=max(1, split.sample_count), seed=self.config.training.random_seed, reshuffle_each_iteration=False)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.map(self._batch_preprocess_map, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset.prefetch(tf.data.AUTOTUNE)

    def _discover_class_names(self) -> tuple[str, ...]:
        """Infer class names from the training split."""

        training_dir = self.config.dataset.root_dir / "training"
        class_names = [
            item.name
            for item in sorted(training_dir.iterdir())
            if item.is_dir()
        ]
        if not class_names:
            raise FileNotFoundError(f"No class folders found in {training_dir}")
        return tuple(class_names)

    def _discover_split(self, split_name: str, class_names: tuple[str, ...]) -> DatasetSplit:
        """Collect file paths and labels for a single split."""

        split_dir = self.config.dataset.root_dir / split_name
        if not split_dir.exists():
            raise FileNotFoundError(f"Split folder not found: {split_dir}")

        paths: list[str] = []
        labels: list[int] = []
        for class_index, class_name in enumerate(class_names):
            class_dir = split_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Class folder not found: {class_dir}")
            class_files = [
                path for path in sorted(class_dir.iterdir())
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
            paths.extend(str(path) for path in class_files)
            labels.extend([class_index] * len(class_files))

        if not paths:
            raise FileNotFoundError(f"No image files found in {split_dir}")

        return DatasetSplit(
            name=split_name,
            paths=paths,
            labels=np.asarray(labels, dtype=np.int64),
            class_names=class_names,
        )

    def _batch_preprocess_map(self, paths: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """TensorFlow wrapper around the OpenCV batch preprocessor."""

        images, labels = tf.py_function(
            func=self._load_and_preprocess_batch,
            inp=[paths, labels],
            Tout=[tf.float32, tf.int64],
        )
        image_height, image_width = self.config.preprocessing.image_size
        images.set_shape([None, image_height, image_width, 3])
        labels.set_shape([None])
        return images, labels

    def _load_and_preprocess_batch(self, paths: tf.Tensor, labels: tf.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Load and preprocess a batch of image paths on the Python side."""

        path_values = paths.numpy()
        label_values = labels.numpy().astype(np.int64)
        processed_images: list[np.ndarray] = []
        for path_value in path_values:
            image_path = path_value.decode("utf-8")
            processed_images.append(self.preprocessor.preprocess_path(image_path))
        return np.stack(processed_images).astype(np.float32), label_values
