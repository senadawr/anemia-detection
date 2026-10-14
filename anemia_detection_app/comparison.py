"""Pairwise statistical comparison of saved classifier bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Callable

from sklearn.metrics import roc_auc_score

from .config import AppConfig
from .dataset import DatasetLoader
from .evaluation import Evaluator, run_delong_tests, run_mcnemar_tests
from .models import ClassifierBundle
from .training import PipelinePredictor


def compare_saved_models(
    model_directories: list[str | Path],
    dataset_root: str | Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Evaluate three saved models on one test split and compare their errors."""

    if len(model_directories) != 3:
        raise ValueError("Select exactly three trained model directories.")
    if any(not isinstance(directory, (str, Path)) for directory in model_directories):
        raise ValueError("Model directories must be filesystem paths.")
    directories = [Path(directory).expanduser().resolve() for directory in model_directories]
    if len(set(directories)) != 3:
        raise ValueError("Select three different trained model directories.")
    if any(not directory.is_dir() for directory in directories):
        raise ValueError("Every selected model directory must exist.")

    bundles = [ClassifierBundle.from_root_dir(directory) for directory in directories]
    metadata = [bundle.load_metadata() for bundle in bundles]
    class_names = tuple(metadata[0]["class_names"])
    if any(tuple(item["class_names"]) != class_names for item in metadata[1:]):
        raise ValueError("All selected models must use the same class names.")

    config = AppConfig.from_dict(metadata[0]["config"])
    if dataset_root is not None and str(dataset_root).strip():
        config.dataset.root_dir = Path(dataset_root).expanduser().resolve()
    dataset = DatasetLoader(config).discover_splits()
    test_split = dataset.testing
    predictors = []
    for index, bundle in enumerate(bundles, 1):
        if progress_callback:
            progress_callback(f"Loading comparison model {index}/3: {bundle.root_dir.name}")
        predictors.append(PipelinePredictor(bundle))
    predictions = {}
    probabilities = {}
    for index, (bundle, predictor) in enumerate(zip(bundles, predictors, strict=True), 1):
        if progress_callback:
            progress_callback(
                f"Predicting test images with model {index}/3 "
                f"({test_split.sample_count} images)..."
            )
        predicted_labels, class_probabilities = predictor.predict_paths_with_probabilities(
            test_split.paths
        )
        model_name = bundle.root_dir.name
        predictions[model_name] = predicted_labels
        probabilities[model_name] = predictor.positive_probabilities(class_probabilities)

    metrics = []
    for name, model_predictions in predictions.items():
        evaluation = Evaluator.evaluate_split(test_split.labels, model_predictions)
        metrics.append({
            "model": name,
            "accuracy": evaluation.accuracy,
            "precision": evaluation.precision,
            "recall": evaluation.recall,
            "f1_score": evaluation.f1_score,
        })

    tests = [
        {
            "model_a": result.model_a,
            "model_b": result.model_b,
            "contingency_matrix": result.contingency_matrix,
            "chi_squared": result.chi_squared,
            "p_value": result.p_value,
            "corrected_p_value": result.corrected_p_value,
        }
        for result in run_mcnemar_tests(test_split.labels, predictions)
    ]
    delong_tests = [
        {
            "model_a": result.model_a,
            "model_b": result.model_b,
            "auc_a": result.auc_a,
            "auc_b": result.auc_b,
            "z_score": result.z_score,
            "p_value": result.p_value,
            "significant": result.significant,
        }
        for result in run_delong_tests(test_split.labels, probabilities)
    ]
    roc_curves = [
        {
            "model": name,
            "auc": float(roc_auc_score(test_split.labels, scores)),
            "points": Evaluator.roc_points(test_split.labels, scores),
        }
        for name, scores in probabilities.items()
    ]
    return {
        "models": metrics,
        "tests": tests,
        "bonferroni_comparisons": len(tests),
        "test_sample_count": test_split.sample_count,
        "class_names": list(class_names),
        "delong": {
            "alpha": 0.05,
            "bonferroni_alpha": 0.05 / 3,
            "tests": delong_tests,
            "roc_curves": roc_curves,
        },
    }
