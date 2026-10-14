"""Training orchestration for the anemia detection pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from collections.abc import Callable

import numpy as np
import tensorflow as tf

from .config import AppConfig, DEFAULT_DATASET_ROOT
from .dataset import DatasetBundle, DatasetLoader
from .dimensionality_reduction import ReducerFactory, ReducerStore
from .evaluation import EvaluationBundle, Evaluator
from .explainability import explain_test_images
from .feature_extraction import FeatureMatrix, MobileNetFeatureExtractor
from .models import BundleStore, ClassifierBundle, ClassifierFactory, extract_positive_probability
from .runtime import configure_accelerator
from .utils.io_utils import read_json, write_json
from .utils.logging_utils import setup_logging


@dataclass(slots=True)
class PredictionResult:
    """Predicted class information for a single image."""

    class_index: int
    class_name: str
    confidence: float
    probabilities: np.ndarray


@dataclass(slots=True)
class TrainingResult:
    """Artifacts returned after training a pipeline."""

    bundle: ClassifierBundle
    history: dict[str, list[float]]
    evaluation: EvaluationBundle
    metadata: dict[str, Any]


def restore_saved_artifacts(
    bundle: ClassifierBundle,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Recreate evaluation and explainability artifacts without training."""

    metadata = bundle.load_metadata()
    config = AppConfig.from_dict(metadata["config"])
    if not config.dataset.root_dir.exists():
        config.dataset.root_dir = DEFAULT_DATASET_ROOT
    config.output.results_dir = bundle.metrics_path.parent if bundle.metrics_path else (
        bundle.root_dir.parent.parent / "results" / bundle.root_dir.name
    )

    loader = DatasetLoader(config)
    dataset = loader.discover_splits()
    feature_model = tf.keras.models.load_model(bundle.feature_model_path)
    reducer = BundleStore.load_estimator(bundle.reducer_path)
    classifier = BundleStore.load_estimator(bundle.classifier_path)
    result_dir = config.output.results_dir
    result_dir.mkdir(parents=True, exist_ok=True)

    evaluations: dict[str, Any] = {}
    for split_name, split in dataset.all_splits().items():
        if progress_callback:
            progress_callback(f"Rebuilding {split_name} evaluation artifacts...")
        images = np.stack([
            loader.preprocessor.preprocess_path(path)
            for path in split.paths
        ])
        features = feature_model.predict(images, batch_size=config.training.batch_size, verbose=0)
        reduced = reducer.transform(features)
        predictions = classifier.predict(reduced)
        scores = extract_positive_probability(classifier, reduced)
        evaluation = Evaluator.evaluate_split(split.labels, predictions, scores)
        evaluations[split_name] = evaluation
        split_dir = result_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        write_json(split_dir / "classification_report.json", evaluation.report)
        Evaluator.plot_confusion_matrix(
            evaluation.confusion_matrix,
            split.class_names,
            split_dir / "confusion_matrix.png",
            f"{split_name.title()} Confusion Matrix",
        )
        Evaluator.plot_roc_curve(
            split.labels,
            scores,
            split_dir / "roc_curve.png",
            f"{split_name.title()} ROC Curve",
        )
        write_json(
            split_dir / "chart_data.json",
            {
                "class_names": list(split.class_names),
                "confusion_matrix": evaluation.confusion_matrix.tolist(),
                "roc": Evaluator.roc_points(split.labels, scores),
            },
        )

    evaluation_bundle = EvaluationBundle(
        training=evaluations["training"],
        validation=evaluations["validation"],
        testing=evaluations["testing"],
    )
    Evaluator.save_metrics(evaluation_bundle, result_dir)
    if bundle.history_path and bundle.history_path.exists():
        Evaluator.plot_history(read_json(bundle.history_path), result_dir / "training_curves")

    shap_manifest = result_dir / "explainability" / "manifest.json"
    if not shap_manifest.exists():
        if progress_callback:
            progress_callback("Rebuilding SHAP explanations from the saved model...")
        explain_test_images(
            config,
            bundle,
            dataset.testing,
            result_dir / "explainability",
            config.training.shap_samples,
            config.training.shap_max_evals,
            progress_callback=progress_callback,
        )
    return result_dir


class PipelineTrainer:
    """Train, evaluate, and persist a complete pipeline."""

    def __init__(
        self,
        config: AppConfig,
        progress_callback: Callable[[str], None] | None = None,
        result_dir_callback: Callable[[Path], None] | None = None,
    ) -> None:
        self.config = config
        self.progress_callback = progress_callback
        self.result_dir_callback = result_dir_callback
        self.config.ensure_directories()
        self.logger = setup_logging(config.output.logs_dir)
        self.accelerator = configure_accelerator()
        if self.accelerator["available"]:
            self.logger.info("Using TensorFlow GPU(s): %s", ", ".join(self.accelerator["devices"]))
        else:
            self.logger.info("No TensorFlow GPU is visible; training will use the CPU.")
        tf.keras.utils.set_random_seed(self.config.training.random_seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

    def train_pipeline(
        self,
        reducer_name: str | None = None,
        classifier_name: str | None = None,
    ) -> TrainingResult:
        """Run the full training and evaluation workflow."""

        run_config = AppConfig.from_dict(self.config.to_dict())
        if reducer_name is not None:
            run_config.reduction.reducer = reducer_name  # type: ignore[assignment]
        if classifier_name is not None:
            run_config.classifier.classifier = classifier_name  # type: ignore[assignment]

        loader = DatasetLoader(run_config)
        dataset_bundle = loader.discover_splits()
        self.logger.info("Dataset summary:\n%s", loader.describe(dataset_bundle))

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        pipeline_name = f"mobilenetv2_{run_config.reduction.reducer}_{run_config.classifier.classifier}_{run_id}"
        model_dir = run_config.output.models_dir / pipeline_name
        result_dir = run_config.output.results_dir / pipeline_name
        cache_dir = result_dir / "feature_cache"
        model_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        if self.result_dir_callback:
            self.result_dir_callback(result_dir)

        train_dataset = loader.build_tf_dataset(dataset_bundle.training, batch_size=run_config.training.batch_size, shuffle=True)
        validation_dataset = loader.build_tf_dataset(dataset_bundle.validation, batch_size=run_config.training.batch_size, shuffle=False)

        extractor = MobileNetFeatureExtractor(run_config)
        history = extractor.fine_tune(
            train_dataset,
            validation_dataset,
            model_dir,
            progress_callback=self.progress_callback,
        )

        train_features = extractor.extract_split_features(dataset_bundle.training, run_config.training.batch_size, cache_dir)
        validation_features = extractor.extract_split_features(dataset_bundle.validation, run_config.training.batch_size, cache_dir)
        testing_features = extractor.extract_split_features(dataset_bundle.testing, run_config.training.batch_size, cache_dir)

        reducer = ReducerFactory.create(run_config.reduction)
        reduced_train = reducer.fit_transform(train_features.features, train_features.labels)
        reduced_validation = reducer.transform(validation_features.features)
        reduced_testing = reducer.transform(testing_features.features)

        classifier = ClassifierFactory.create(run_config.classifier)
        classifier.fit(reduced_train, train_features.labels)

        training_eval = self._evaluate_split(classifier, reduced_train, train_features.labels)
        validation_eval = self._evaluate_split(classifier, reduced_validation, validation_features.labels)
        testing_eval = self._evaluate_split(classifier, reduced_testing, testing_features.labels)
        evaluation = EvaluationBundle(
            training=training_eval,
            validation=validation_eval,
            testing=testing_eval,
        )

        feature_model_path = model_dir / "feature_extractor.keras"
        reducer_path = model_dir / "reducer.joblib"
        classifier_path = model_dir / "classifier.joblib"
        metadata_path = model_dir / "metadata.json"
        history_path = model_dir / "history.json"
        metrics_path = result_dir / "metrics.json"

        extractor.save_feature_model(feature_model_path)
        ReducerStore.save(reducer, reducer_path)
        BundleStore.save_estimator(classifier_path, classifier)

        metadata = {
            "pipeline_name": pipeline_name,
            "config": run_config.to_dict(),
            "class_names": list(dataset_bundle.training.class_names),
            "split_sizes": {
                "training": dataset_bundle.training.sample_count,
                "validation": dataset_bundle.validation.sample_count,
                "testing": dataset_bundle.testing.sample_count,
            },
            "feature_shape": list(train_features.features.shape[1:]),
            "reducer": run_config.reduction.reducer,
            "classifier": run_config.classifier.classifier,
            "accelerator": self.accelerator,
        }
        write_json(metadata_path, metadata)
        if history:
            write_json(history_path, history)
        Evaluator.save_metrics(evaluation, result_dir)
        Evaluator.plot_history(history, result_dir / "training_curves")
        self._save_split_artifacts("training", train_features.labels, training_eval, result_dir, dataset_bundle.training.class_names, reduced_train, classifier)
        self._save_split_artifacts("validation", validation_features.labels, validation_eval, result_dir, dataset_bundle.validation.class_names, reduced_validation, classifier)
        self._save_split_artifacts("testing", testing_features.labels, testing_eval, result_dir, dataset_bundle.testing.class_names, reduced_testing, classifier)

        if run_config.training.shap_enabled:
            self.progress_callback and self.progress_callback(
                f"Starting SHAP investigation on {run_config.training.shap_samples} test images..."
            )
            explain_test_images(
                run_config,
                ClassifierBundle(
                    root_dir=model_dir,
                    feature_model_path=feature_model_path,
                    reducer_path=reducer_path,
                    classifier_path=classifier_path,
                    metadata_path=metadata_path,
                ),
                dataset_bundle.testing,
                result_dir / "explainability",
                run_config.training.shap_samples,
                run_config.training.shap_max_evals,
                progress_callback=self.progress_callback,
            )
            self.progress_callback and self.progress_callback("SHAP investigation complete.")

        bundle = ClassifierBundle(
            root_dir=model_dir,
            feature_model_path=feature_model_path,
            reducer_path=reducer_path,
            classifier_path=classifier_path,
            metadata_path=metadata_path,
            history_path=history_path if history else None,
            metrics_path=metrics_path,
        )
        return TrainingResult(bundle=bundle, history=history, evaluation=evaluation, metadata=metadata)

    def _evaluate_split(self, classifier: Any, features: np.ndarray, labels: np.ndarray):
        predictions = classifier.predict(features)
        scores = None
        if hasattr(classifier, "predict_proba"):
            scores = extract_positive_probability(classifier, features)
        return Evaluator.evaluate_split(labels, predictions, scores)

    def _save_split_artifacts(
        self,
        split_name: str,
        labels: np.ndarray,
        evaluation,
        result_dir: Path,
        class_names: tuple[str, ...],
        features: np.ndarray,
        classifier: Any,
    ) -> None:
        split_dir = result_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        write_json(split_dir / "classification_report.json", evaluation.report)
        Evaluator.plot_confusion_matrix(
            evaluation.confusion_matrix,
            tuple(class_names),
            split_dir / "confusion_matrix.png",
            f"{split_name.title()} Confusion Matrix",
        )
        chart_payload: dict[str, Any] = {
            "class_names": list(class_names),
            "confusion_matrix": evaluation.confusion_matrix.tolist(),
            "roc": [],
        }
        if hasattr(classifier, "predict_proba") and len(np.unique(labels)) > 1:
            scores = extract_positive_probability(classifier, features)
            Evaluator.plot_roc_curve(
                labels,
                scores,
                split_dir / "roc_curve.png",
                f"{split_name.title()} ROC Curve",
            )
            chart_payload["roc"] = Evaluator.roc_points(labels, scores)
        write_json(split_dir / "chart_data.json", chart_payload)


class PipelinePredictor:
    """Load a saved bundle and predict labels for new images."""

    def __init__(self, bundle: ClassifierBundle, config: AppConfig | None = None) -> None:
        self.bundle = bundle
        metadata = bundle.load_metadata()
        self.config = config or AppConfig.from_dict(metadata["config"])
        self.class_names = tuple(metadata["class_names"])
        self.preprocessor = DatasetLoader(self.config).preprocessor
        self.feature_model = tf.keras.models.load_model(bundle.feature_model_path)
        self.reducer = BundleStore.load_estimator(bundle.reducer_path)
        self.classifier = BundleStore.load_estimator(bundle.classifier_path)

    def predict_image(self, image_path: str | Path) -> PredictionResult:
        """Predict the class of one image."""

        image = self.preprocessor.preprocess_path(image_path)
        feature_vector = self.feature_model.predict(np.expand_dims(image, axis=0), verbose=0)
        reduced_vector = self.reducer.transform(feature_vector)
        probabilities = self.classifier.predict_proba(reduced_vector)[0]
        class_index = int(np.argmax(probabilities))
        predicted_label = int(self.classifier.classes_[class_index])
        class_name = self.class_names[predicted_label]
        confidence = float(probabilities[class_index])
        return PredictionResult(
            class_index=predicted_label,
            class_name=class_name,
            confidence=confidence,
            probabilities=probabilities,
        )

    def predict_paths(self, image_paths: list[str]) -> np.ndarray:
        """Predict a collection of images in one feature-extraction batch."""

        _, probabilities = self.predict_paths_with_probabilities(image_paths)
        return np.asarray(self.classifier.classes_[
            np.argmax(probabilities, axis=1)
        ], dtype=np.int64)

    def predict_paths_with_probabilities(
        self,
        image_paths: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return hard predictions and class probabilities for image paths."""

        if not image_paths:
            return np.asarray([], dtype=np.int64), np.empty((0, 0))
        images = np.stack([
            self.preprocessor.preprocess_path(path)
            for path in image_paths
        ])
        features = self.feature_model.predict(
            images,
            batch_size=self.config.training.batch_size,
            verbose=0,
        )
        reduced = self.reducer.transform(features)
        probabilities = np.asarray(self.classifier.predict_proba(reduced))
        predictions = np.asarray(
            self.classifier.classes_[np.argmax(probabilities, axis=1)],
            dtype=np.int64,
        )
        return predictions, probabilities

    def positive_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        """Extract the probability for binary class label 1."""

        class_indices = np.flatnonzero(np.asarray(self.classifier.classes_) == 1)
        if len(class_indices) != 1:
            raise ValueError("The loaded classifier does not expose binary class label 1.")
        return probabilities[:, class_indices[0]]
