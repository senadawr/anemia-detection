"""Model explanation utilities for investigation-only SHAP analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .config import AppConfig
from .dataset import DatasetSplit
from .models import ClassifierBundle, BundleStore


def explain_test_images(
    config: AppConfig,
    bundle: ClassifierBundle,
    test_split: DatasetSplit,
    output_dir: Path,
    sample_count: int,
    max_evals: int,
    progress_callback: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Create image-level SHAP heatmaps for a small test sample."""

    if sample_count <= 0:
        return []
    if max_evals < 20:
        raise ValueError("SHAP max_evals must be at least 20")

    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is not installed. Install dependencies from requirements.txt first.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_model = __import__("tensorflow").keras.models.load_model(bundle.feature_model_path)
    reducer = BundleStore.load_estimator(bundle.reducer_path)
    classifier = BundleStore.load_estimator(bundle.classifier_path)
    image_preprocessor = _image_preprocessor(config)

    selected = _select_samples(test_split, sample_count)
    images = np.stack([_read_rgb(Path(path)) for path in selected])
    masker = shap.maskers.Image("blur(32, 32)", images[0].shape)

    def predict_probability(batch: np.ndarray) -> np.ndarray:
        processed = np.stack([
            image_preprocessor.preprocess_array(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            for image in batch
        ])
        features = feature_model.predict(processed, verbose=0)
        reduced = reducer.transform(features)
        probabilities = classifier.predict_proba(reduced)
        class_one_index = int(np.flatnonzero(classifier.classes_ == 1)[0])
        return probabilities[:, class_one_index]

    explainer = shap.Explainer(predict_probability, masker, algorithm="partition")
    manifest: list[dict[str, Any]] = []
    manifest_path = output_dir / "manifest.json"
    for index, (path, image) in enumerate(zip(selected, images), start=1):
        if progress_callback:
            progress_callback(f"SHAP explanation {index}/{len(selected)} started: {Path(path).name}")
        explanation = explainer(
            image[np.newaxis, ...],
            max_evals=max_evals,
        )
        values = _extract_values(explanation)
        prediction = float(predict_probability(image[np.newaxis, ...])[0])
        predicted_class = int(prediction >= 0.5)
        predicted_class_name = test_split.class_names[predicted_class]
        class_relative_values = values if predicted_class == 1 else -values
        output_path = output_dir / f"explanation_{index:02d}.png"
        _save_heatmap(
            image,
            class_relative_values,
            output_path,
            Path(path).name,
            prediction,
            predicted_class_name,
        )
        item = {
            "image": Path(path).name,
            "actual_class": int(test_split.labels[test_split.paths.index(path)]),
            "predicted_class": predicted_class,
            "predicted_class_name": predicted_class_name,
            "predicted_probability_class_1": prediction,
            "path": f"explainability/{output_path.name}",
        }
        manifest.append(item)
        manifest_path.write_text(
            __import__("json").dumps(manifest, indent=2),
            encoding="utf-8",
        )
        if progress_callback:
            progress_callback(f"SHAP explanation {index}/{len(selected)} complete: {output_path.name}")

    return manifest


def _image_preprocessor(config: AppConfig) -> Any:
    from .preprocessing import ImagePreprocessor

    return ImagePreprocessor(config.preprocessing)


def _select_samples(split: DatasetSplit, sample_count: int) -> list[str]:
    indices = np.linspace(0, len(split.paths) - 1, min(sample_count, len(split.paths)), dtype=int)
    return [split.paths[index] for index in indices]


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        bgr = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        neutral_background = np.full_like(bgr, 127.5)
        image = np.round(bgr * alpha + neutral_background * (1.0 - alpha)).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _extract_values(explanation: Any) -> np.ndarray:
    values = np.asarray(explanation.values)
    if values.ndim == 5:
        values = values[0, ..., 0]
    elif values.ndim == 4:
        values = values[0]
    return values


def _save_heatmap(
    image: np.ndarray,
    values: np.ndarray,
    output_path: Path,
    name: str,
    probability: float,
    predicted_class_name: str,
) -> None:
    importance = np.sum(values, axis=2)
    limit = float(np.max(np.abs(importance))) or 1.0
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].imshow(image)
    axes[0].set_title(name)
    axes[1].imshow(image)
    axes[1].imshow(importance, cmap="coolwarm", vmin=-limit, vmax=limit, alpha=0.65)
    axes[1].set_title(f"SHAP overlay | predicted {predicted_class_name} | p(class 1) {probability:.3f}")
    for axis in axes:
        axis.axis("off")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
