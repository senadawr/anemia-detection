"""Classical classifier wrappers and pipeline bundle persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import ClassifierConfig
from .utils.io_utils import ensure_parent_dir, read_json, write_json


@dataclass(slots=True)
class ClassifierBundle:
    """Saved artifacts for a trained pipeline."""

    root_dir: Path
    feature_model_path: Path
    reducer_path: Path
    classifier_path: Path
    metadata_path: Path
    history_path: Path | None = None
    metrics_path: Path | None = None

    @classmethod
    def from_root_dir(cls, root_dir: Path) -> "ClassifierBundle":
        """Build a bundle from a saved model directory."""

        return cls(
            root_dir=root_dir,
            feature_model_path=root_dir / "feature_extractor.keras",
            reducer_path=root_dir / "reducer.joblib",
            classifier_path=root_dir / "classifier.joblib",
            metadata_path=root_dir / "metadata.json",
            history_path=root_dir / "history.json",
            metrics_path=root_dir.parent.parent / "results" / root_dir.name / "metrics.json",
        )

    def load_classifier(self) -> Any:
        return joblib.load(self.classifier_path)

    def load_reducer(self) -> Any:
        return joblib.load(self.reducer_path)

    def load_metadata(self) -> dict[str, Any]:
        return read_json(self.metadata_path)


class ClassifierFactory:
    """Create scikit-learn classifiers used by the thesis pipelines."""

    @staticmethod
    def create(config: ClassifierConfig) -> Any:
        if config.classifier == "logistic_regression":
            estimator = LogisticRegression(
                C=config.logistic_c,
                max_iter=config.logistic_max_iter,
                class_weight=config.class_weight,
                solver="lbfgs",
            )
            return make_pipeline(StandardScaler(), estimator)
        if config.classifier == "random_forest":
            return RandomForestClassifier(
                n_estimators=config.random_forest_n_estimators,
                max_depth=config.random_forest_max_depth,
                class_weight=config.class_weight,
                random_state=42,
            )
        estimator = SVC(
            kernel="rbf",
            C=config.svm_c,
            gamma=config.svm_gamma,
            probability=True,
            class_weight=config.class_weight,
        )
        return make_pipeline(StandardScaler(), estimator)


class BundleStore:
    """Persist and restore model bundles."""

    @staticmethod
    def save_bundle(bundle: ClassifierBundle, metadata: dict[str, Any]) -> None:
        bundle.root_dir.mkdir(parents=True, exist_ok=True)
        write_json(bundle.metadata_path, metadata)

    @staticmethod
    def save_estimator(path: Path, estimator: Any) -> None:
        ensure_parent_dir(path)
        joblib.dump(estimator, path)

    @staticmethod
    def load_estimator(path: Path) -> Any:
        return joblib.load(path)


def extract_positive_probability(estimator: Any, features: np.ndarray) -> np.ndarray:
    """Return the positive-class probability for binary classifiers."""

    probabilities = estimator.predict_proba(features)
    if probabilities.shape[1] == 1:
        return probabilities[:, 0]
    return probabilities[:, 1]
