"""Training orchestration for the anemia detection pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from .config import AppConfig
from .dataset import DatasetBundle, DatasetLoader
from .dimensionality_reduction import ReducerFactory, ReducerStore
from .evaluation import EvaluationBundle, Evaluator
from .feature_extraction import FeatureMatrix, MobileNetFeatureExtractor
from .models import BundleStore, ClassifierBundle, ClassifierFactory, extract_positive_probability
from .utils.io_utils import write_json
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


class PipelineTrainer:
    """Train, evaluate, and persist a complete pipeline."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.config.ensure_directories()
        self.logger = setup_logging(config.output.logs_dir)
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

        train_dataset = loader.build_tf_dataset(dataset_bundle.training, batch_size=run_config.training.batch_size, shuffle=False)
        validation_dataset = loader.build_tf_dataset(dataset_bundle.validation, batch_size=run_config.training.batch_size, shuffle=False)

        extractor = MobileNetFeatureExtractor(run_config)
        history = extractor.fine_tune(train_dataset, validation_dataset, model_dir)

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
        }
        write_json(metadata_path, metadata)
        if history:
            write_json(history_path, history)
        Evaluator.save_metrics(evaluation, result_dir)
        Evaluator.plot_history(history, result_dir / "training_curves")
        self._save_split_artifacts("training", train_features.labels, training_eval, result_dir, dataset_bundle.training.class_names, reduced_train, classifier)
        self._save_split_artifacts("validation", validation_features.labels, validation_eval, result_dir, dataset_bundle.validation.class_names, reduced_validation, classifier)
        self._save_split_artifacts("testing", testing_features.labels, testing_eval, result_dir, dataset_bundle.testing.class_names, reduced_testing, classifier)

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
            probabilities = classifier.predict_proba(features)
            scores = probabilities[:, 1] if probabilities.ndim == 2 and probabilities.shape[1] > 1 else probabilities.ravel()
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
        if hasattr(classifier, "predict_proba") and len(np.unique(labels)) > 1:
            probabilities = classifier.predict_proba(features)
            scores = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities.ravel()
            Evaluator.plot_roc_curve(
                labels,
                scores,
                split_dir / "roc_curve.png",
                f"{split_name.title()} ROC Curve",
            )


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
