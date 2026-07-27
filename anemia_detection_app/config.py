"""Central configuration for the anemia detection application."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset_clean"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "saved_models"
DEFAULT_LOGS_DIR = PROJECT_ROOT / "logs"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_ASSETS_DIR = PROJECT_ROOT / "assets"

SplitName = Literal["training", "validation", "testing"]
ReducerName = Literal["none", "lle", "isomap"]
ClassifierName = Literal["svm", "logistic_regression", "random_forest"]


@dataclass(slots=True)
class DatasetConfig:
    """Dataset paths and class metadata."""

    root_dir: Path = DEFAULT_DATASET_ROOT
    class_names: tuple[str, str] = ("anemic", "non_anemic")
    split_names: tuple[SplitName, SplitName, SplitName] = (
        "training",
        "validation",
        "testing",
    )


@dataclass(slots=True)
class PreprocessingConfig:
    """OpenCV preprocessing configuration."""

    image_size: tuple[int, int] = (224, 224)
    normalize: bool = True
    use_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    use_histogram_equalization: bool = False


@dataclass(slots=True)
class FeatureExtractorConfig:
    """MobileNetV2 feature extraction settings."""

    backbone_name: str = "MobileNetV2"
    include_top: bool = False
    weights: str = "imagenet"
    freeze_backbone: bool = True
    fine_tune: bool = False
    fine_tune_unfreeze_last_n_layers: int = 30
    dense_feature_dim: int | None = None
    cache_features: bool = True


@dataclass(slots=True)
class ReductionConfig:
    """Dimensionality reduction settings."""

    reducer: ReducerName = "none"
    n_components: int = 64
    n_neighbors: int = 10
    lle_method: str = "standard"
    lle_eigen_solver: str = "auto"
    isomap_eigen_solver: str = "auto"
    isomap_path_method: str = "auto"


@dataclass(slots=True)
class ClassifierConfig:
    """Classical classifier settings."""

    classifier: ClassifierName = "svm"
    svm_c: float = 1.0
    svm_gamma: str = "scale"
    logistic_c: float = 1.0
    logistic_max_iter: int = 2000
    random_forest_n_estimators: int = 300
    random_forest_max_depth: int | None = None
    class_weight: str | dict[str, float] | None = "balanced"


@dataclass(slots=True)
class TrainingConfig:
    """Fine-tuning and pipeline training settings."""

    batch_size: int = 32
    epochs: int = 20
    fine_tune_epochs: int = 8
    learning_rate: float = 1e-4
    fine_tune_learning_rate: float = 1e-5
    validation_split_name: SplitName = "validation"
    patience: int = 5
    reduce_lr_patience: int = 3
    min_delta: float = 1e-4
    tensorboard_log_dir: str = "tensorboard"
    random_seed: int = 42
    workers: int = 1


@dataclass(slots=True)
class OutputConfig:
    """Generated artifact directories."""

    models_dir: Path = DEFAULT_MODELS_DIR
    logs_dir: Path = DEFAULT_LOGS_DIR
    results_dir: Path = DEFAULT_RESULTS_DIR
    assets_dir: Path = DEFAULT_ASSETS_DIR


@dataclass(slots=True)
class AppConfig:
    """Root application configuration."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    feature_extractor: FeatureExtractorConfig = field(default_factory=FeatureExtractorConfig)
    reduction: ReductionConfig = field(default_factory=ReductionConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def ensure_directories(self) -> None:
        """Create all runtime directories if they do not exist."""

        for directory in (
            self.output.models_dir,
            self.output.logs_dir,
            self.output.results_dir,
            self.output.assets_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert the configuration tree to plain Python objects."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AppConfig":
        """Reconstruct an application configuration from a dictionary."""

        dataset = DatasetConfig(
            root_dir=Path(payload["dataset"]["root_dir"]),
            class_names=tuple(payload["dataset"]["class_names"]),
            split_names=tuple(payload["dataset"]["split_names"]),
        )
        preprocessing = PreprocessingConfig(**payload["preprocessing"])
        feature_extractor = FeatureExtractorConfig(**payload["feature_extractor"])
        reduction = ReductionConfig(**payload["reduction"])
        classifier = ClassifierConfig(**payload["classifier"])
        training = TrainingConfig(**payload["training"])
        output = OutputConfig(
            models_dir=Path(payload["output"]["models_dir"]),
            logs_dir=Path(payload["output"]["logs_dir"]),
            results_dir=Path(payload["output"]["results_dir"]),
            assets_dir=Path(payload["output"]["assets_dir"]),
        )
        return cls(
            dataset=dataset,
            preprocessing=preprocessing,
            feature_extractor=feature_extractor,
            reduction=reduction,
            classifier=classifier,
            training=training,
            output=output,
        )


def build_default_config() -> AppConfig:
    """Return the default application configuration."""

    config = AppConfig()
    config.ensure_directories()
    return config
