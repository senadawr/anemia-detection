"""Dimensionality reduction for extracted MobileNetV2 features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
from sklearn.manifold import Isomap, LocallyLinearEmbedding

from .config import ReductionConfig
from .utils.io_utils import ensure_parent_dir


class Reducer(Protocol):
    """Common reducer interface."""

    def fit(self, features: np.ndarray, labels: np.ndarray | None = None) -> "Reducer":
        ...

    def transform(self, features: np.ndarray) -> np.ndarray:
        ...

    def fit_transform(self, features: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
        ...


@dataclass(slots=True)
class IdentityReducer:
    """No-op dimensionality reducer."""

    n_components: int | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray | None = None) -> "IdentityReducer":
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features)

    def fit_transform(self, features: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
        return np.asarray(features)


class LLEReducer:
    """Locally Linear Embedding wrapper."""

    def __init__(self, config: ReductionConfig) -> None:
        self.config = config
        self.model = LocallyLinearEmbedding(
            n_components=config.n_components,
            n_neighbors=config.n_neighbors,
            method=config.lle_method,
            eigen_solver=config.lle_eigen_solver,
        )

    def fit(self, features: np.ndarray, labels: np.ndarray | None = None) -> "LLEReducer":
        self.model.fit(features)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        return self.model.transform(features)

    def fit_transform(self, features: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
        return self.model.fit_transform(features)


class IsomapReducer:
    """ISOMAP wrapper."""

    def __init__(self, config: ReductionConfig) -> None:
        self.config = config
        self.model = Isomap(
            n_components=config.n_components,
            n_neighbors=config.n_neighbors,
            eigen_solver=config.isomap_eigen_solver,
            path_method=config.isomap_path_method,
        )

    def fit(self, features: np.ndarray, labels: np.ndarray | None = None) -> "IsomapReducer":
        self.model.fit(features)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        return self.model.transform(features)

    def fit_transform(self, features: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
        return self.model.fit_transform(features)


class ReducerFactory:
    """Create the requested reducer implementation."""

    @staticmethod
    def create(config: ReductionConfig) -> Reducer:
        if config.reducer == "lle":
            return LLEReducer(config)
        if config.reducer == "isomap":
            return IsomapReducer(config)
        return IdentityReducer(n_components=None)


class ReducerStore:
    """Persist reducers with joblib."""

    @staticmethod
    def save(reducer: Reducer, path: Path) -> None:
        ensure_parent_dir(path)
        joblib.dump(reducer, path)

    @staticmethod
    def load(path: Path) -> Reducer:
        return joblib.load(path)
