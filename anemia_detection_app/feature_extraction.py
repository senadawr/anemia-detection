"""MobileNetV2 feature extraction and optional fine-tuning."""

from __future__ import annotations

import hashlib
from importlib.util import find_spec
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

from .config import AppConfig
from .dataset import DatasetLoader, DatasetSplit
from .utils.io_utils import ensure_parent_dir, write_json


@dataclass(slots=True)
class FeatureMatrix:
    """Feature matrix together with labels and source paths."""

    features: np.ndarray
    labels: np.ndarray
    paths: list[str]


class FeatureCache:
    """Disk cache for extracted features."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.joblib"

    def load(self, cache_key: str) -> FeatureMatrix | None:
        path = self.cache_path(cache_key)
        if not path.exists():
            return None
        payload = joblib.load(path)
        return FeatureMatrix(
            features=payload["features"],
            labels=payload["labels"],
            paths=payload["paths"],
        )

    def save(self, cache_key: str, matrix: FeatureMatrix) -> None:
        path = self.cache_path(cache_key)
        ensure_parent_dir(path)
        joblib.dump(
            {
                "features": matrix.features,
                "labels": matrix.labels,
                "paths": matrix.paths,
            },
            path,
        )


class MobileNetFeatureExtractor:
    """Build and use MobileNetV2 as a feature extractor."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.input_shape = (*config.preprocessing.image_size, 3)
        self.backbone = self._build_backbone()
        self.feature_model = self._build_feature_model()
        self.classifier_model = self._build_classifier_model()

    def _build_backbone(self) -> tf.keras.Model:
        backbone = MobileNetV2(
            include_top=False,
            weights=self.config.feature_extractor.weights,
            input_shape=self.input_shape,
        )
        backbone.trainable = not self.config.feature_extractor.freeze_backbone
        return backbone

    def _build_feature_model(self) -> tf.keras.Model:
        inputs = Input(shape=self.input_shape)
        x = self.backbone(inputs, training=False)
        x = GlobalAveragePooling2D(name="global_average_pooling")(x)
        if self.config.feature_extractor.dense_feature_dim:
            x = Dense(
                self.config.feature_extractor.dense_feature_dim,
                activation="relu",
                name="feature_projection",
            )(x)
        return Model(inputs=inputs, outputs=x, name="mobilenetv2_feature_extractor")

    def _build_classifier_model(self) -> tf.keras.Model:
        inputs = Input(shape=self.input_shape)
        x = self.backbone(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1, activation="sigmoid")(x)
        return Model(inputs=inputs, outputs=outputs, name="mobilenetv2_binary_classifier")

    def fine_tune(
        self,
        train_dataset: tf.data.Dataset,
        validation_dataset: tf.data.Dataset,
        output_dir: Path,
    ) -> dict[str, list[float]]:
        """Train a lightweight binary head with the MobileNetV2 backbone."""

        if not self.config.feature_extractor.fine_tune:
            return {}

        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "best_feature_extractor.keras"
        log_dir = output_dir / self.config.training.tensorboard_log_dir

        callbacks: list[tf.keras.callbacks.Callback] = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.training.patience,
                restore_best_weights=True,
                min_delta=self.config.training.min_delta,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                patience=self.config.training.reduce_lr_patience,
                factor=0.5,
                min_lr=1e-7,
            ),
        ]

        if find_spec("tensorboard") is not None:
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=str(log_dir)))

        self.classifier_model.compile(
            optimizer=tf.keras.optimizers.Adam(self.config.training.fine_tune_learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )
        history = self.classifier_model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=self.config.training.fine_tune_epochs,
            callbacks=callbacks,
            verbose=1,
        )

        if checkpoint_path.exists():
            self.classifier_model = tf.keras.models.load_model(checkpoint_path)
            self.backbone = self.classifier_model.get_layer(index=1)
            self.feature_model = self._build_feature_model()

        return history.history

    def extract_split_features(
        self,
        split: DatasetSplit,
        batch_size: int,
        cache_dir: Path,
        use_cache: bool = True,
    ) -> FeatureMatrix:
        """Extract MobileNetV2 features for a split, using a cache when possible."""

        cache = FeatureCache(cache_dir)
        cache_key = self._build_cache_key(split)
        if use_cache:
            cached = cache.load(cache_key)
            if cached is not None:
                return cached

        loader = DatasetLoader(self.config)
        dataset = loader.build_tf_dataset(split, batch_size=batch_size, shuffle=False)
        feature_dataset = dataset.map(lambda images, labels: images)
        features = self.feature_model.predict(feature_dataset, verbose=0)
        labels = split.labels.astype(np.int64)
        matrix = FeatureMatrix(features=features, labels=labels, paths=list(split.paths))
        if use_cache:
            cache.save(cache_key, matrix)
        return matrix

    def save_feature_model(self, model_path: Path) -> None:
        """Persist the feature extractor model to disk."""

        ensure_parent_dir(model_path)
        self.feature_model.save(model_path)

    def save_metadata(self, metadata_path: Path, metadata: dict[str, Any]) -> None:
        """Persist extractor metadata as JSON."""

        write_json(metadata_path, metadata)

    def _build_cache_key(self, split: DatasetSplit) -> str:
        payload = {
            "split": split.name,
            "paths": split.paths,
            "image_size": self.config.preprocessing.image_size,
            "normalize": self.config.preprocessing.normalize,
            "use_clahe": self.config.preprocessing.use_clahe,
            "use_histogram_equalization": self.config.preprocessing.use_histogram_equalization,
            "dense_feature_dim": self.config.feature_extractor.dense_feature_dim,
            "weights": self.config.feature_extractor.weights,
            "fine_tune": self.config.feature_extractor.fine_tune,
        }
        digest = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()
        return digest[:24]
