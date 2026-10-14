"""Evaluation and visualization utilities for the anemia pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2, norm
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .utils.io_utils import ensure_parent_dir, write_json


@dataclass(slots=True)
class SplitEvaluation:
    """Metrics for a single dataset split."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float | None
    confusion_matrix: np.ndarray
    report: dict[str, Any]


@dataclass(slots=True)
class EvaluationBundle:
    """Metrics for train, validation, and test splits."""

    training: SplitEvaluation
    validation: SplitEvaluation
    testing: SplitEvaluation


@dataclass(slots=True)
class McNemarResult:
    """A pairwise McNemar test result."""

    model_a: str
    model_b: str
    contingency_matrix: list[list[int]]
    chi_squared: float
    p_value: float
    corrected_p_value: float


@dataclass(slots=True)
class DeLongResult:
    """A pairwise correlated ROC-AUC comparison."""

    model_a: str
    model_b: str
    auc_a: float
    auc_b: float
    z_score: float
    p_value: float
    significant: bool


def run_delong_tests(
    y_true: np.ndarray,
    probabilities: dict[str, np.ndarray],
    alpha: float = 0.05,
) -> list[DeLongResult]:
    """Run all three pairwise DeLong tests with a Bonferroni threshold."""

    if len(probabilities) != 3:
        raise ValueError("DeLong comparison requires exactly three models.")
    labels = np.asarray(y_true)
    if len(labels) == 0 or set(np.unique(labels)) != {0, 1}:
        raise ValueError("DeLong testing requires both binary target classes.")
    names = list(probabilities)
    scores = np.asarray([np.asarray(probabilities[name]) for name in names], dtype=float)
    if scores.ndim != 2 or scores.shape[1] != len(labels):
        raise ValueError("All model probabilities must match the test labels.")
    if not np.isfinite(scores).all():
        raise ValueError("Model probabilities must be finite.")

    aucs, covariance = _delong_auc_covariance(labels, scores)
    threshold = alpha / 3
    results: list[DeLongResult] = []
    for first_index, second_index in ((0, 1), (1, 2), (0, 2)):
        first_name = names[first_index]
        variance = (
            covariance[first_index, first_index]
            + covariance[second_index, second_index]
            - 2 * covariance[first_index, second_index]
        )
        z_score = (
            float((aucs[first_index] - aucs[second_index]) / np.sqrt(variance))
            if variance > 0
            else 0.0
        )
        p_value = float(2 * norm.sf(abs(z_score)))
        results.append(DeLongResult(
            model_a=first_name,
            model_b=names[second_index],
            auc_a=float(aucs[first_index]),
            auc_b=float(aucs[second_index]),
            z_score=z_score,
            p_value=p_value,
            significant=p_value < threshold,
        ))
    return results


def _delong_auc_covariance(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate correlated AUC estimates and their DeLong covariance."""

    order = np.argsort(-y_true)
    sorted_scores = probabilities[:, order]
    positive_count = int(np.sum(y_true))
    negative_count = len(y_true) - positive_count
    positive_ranks = np.asarray([
        _compute_midrank(row[:positive_count]) for row in sorted_scores
    ])
    negative_ranks = np.asarray([
        _compute_midrank(row[positive_count:]) for row in sorted_scores
    ])
    all_ranks = np.asarray([_compute_midrank(row) for row in sorted_scores])
    aucs = (
        all_ranks[:, :positive_count].sum(axis=1) / positive_count / negative_count
        - (positive_count + 1) / (2 * negative_count)
    )
    v01 = (all_ranks[:, :positive_count] - positive_ranks) / negative_count
    v10 = 1 - (all_ranks[:, positive_count:] - negative_ranks) / positive_count
    covariance = np.atleast_2d(np.cov(v01, bias=False) / positive_count
                               + np.cov(v10, bias=False) / negative_count)
    return aucs, covariance


def _compute_midrank(values: np.ndarray) -> np.ndarray:
    """Return average ranks for tied values."""

    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and sorted_values[end] == sorted_values[index]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + end - 1) + 1
        index = end
    return ranks


def run_mcnemar_tests(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> list[McNemarResult]:
    """Run all three Bonferroni-corrected pairwise McNemar tests."""

    if len(predictions) != 3:
        raise ValueError("McNemar comparison requires exactly three models.")
    model_names = list(predictions)
    if len(y_true) == 0:
        raise ValueError("The test dataset must contain at least one sample.")
    normalized_predictions = {
        name: np.asarray(values) for name, values in predictions.items()
    }
    if any(len(values) != len(y_true) for values in normalized_predictions.values()):
        raise ValueError("All model predictions must match the test labels.")

    results: list[McNemarResult] = []
    for first_index, first_name in enumerate(model_names):
        for second_name in model_names[first_index + 1:]:
            first_correct = normalized_predictions[first_name] == y_true
            second_correct = normalized_predictions[second_name] == y_true
            both_correct = int(np.count_nonzero(first_correct & second_correct))
            first_correct_second_wrong = int(np.count_nonzero(first_correct & ~second_correct))
            first_wrong_second_correct = int(np.count_nonzero(~first_correct & second_correct))
            both_wrong = int(np.count_nonzero(~first_correct & ~second_correct))
            discordant = first_correct_second_wrong + first_wrong_second_correct
            chi_squared = (
                (abs(first_correct_second_wrong - first_wrong_second_correct) - 1) ** 2 / discordant
                if discordant
                else 0.0
            )
            results.append(
                McNemarResult(
                    model_a=first_name,
                    model_b=second_name,
                    contingency_matrix=[
                        [both_correct, first_correct_second_wrong],
                        [first_wrong_second_correct, both_wrong],
                    ],
                    chi_squared=float(chi_squared),
                    p_value=float(chi2.sf(chi_squared, 1)),
                    corrected_p_value=0.0,
                )
            )

    for result in results:
        result.corrected_p_value = min(1.0, result.p_value * len(results))
    return results


class Evaluator:
    """Compute metrics and save evaluation artifacts."""

    @staticmethod
    def evaluate_split(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> SplitEvaluation:
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        roc_auc = None
        if y_score is not None and len(np.unique(y_true)) > 1:
            roc_auc = float(roc_auc_score(y_true, y_score))
        return SplitEvaluation(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1_score=float(f1_score(y_true, y_pred, zero_division=0)),
            roc_auc=roc_auc,
            confusion_matrix=confusion_matrix(y_true, y_pred),
            report=report,
        )

    @staticmethod
    def save_metrics(bundle: EvaluationBundle, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "metrics.json", _bundle_to_dict(bundle))

    @staticmethod
    def save_classification_reports(reports: dict[str, dict[str, Any]], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for split_name, report in reports.items():
            (output_dir / f"classification_report_{split_name}.json").write_text(
                str(report),
                encoding="utf-8",
            )

    @staticmethod
    def plot_confusion_matrix(confusion: np.ndarray, class_names: tuple[str, str], output_path: Path, title: str) -> None:
        ensure_parent_dir(output_path)
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(confusion, interpolation="nearest", cmap="Blues")
        fig.colorbar(image, ax=ax)
        tick_positions = np.arange(len(class_names))
        ax.set_xticks(tick_positions, class_names, rotation=45, ha="right")
        ax.set_yticks(tick_positions, class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)
        for row_index in range(confusion.shape[0]):
            for col_index in range(confusion.shape[1]):
                ax.text(col_index, row_index, int(confusion[row_index, col_index]), ha="center", va="center", color="black")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, output_path: Path, title: str) -> None:
        ensure_parent_dir(output_path)
        fig, ax = plt.subplots(figsize=(6, 5))
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            ax.plot(fpr, tpr, label="ROC curve")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

    @staticmethod
    def roc_points(y_true: np.ndarray, y_score: np.ndarray) -> list[dict[str, float]]:
        """Return ROC coordinates for the interactive web chart."""

        if len(np.unique(y_true)) < 2:
            return []
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_score)
        return [
            {"false_positive_rate": float(fpr), "true_positive_rate": float(tpr)}
            for fpr, tpr in zip(false_positive_rate, true_positive_rate)
        ]

    @staticmethod
    def plot_history(history: dict[str, list[float]], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not history:
            return

        if "accuracy" in history or "val_accuracy" in history:
            fig, ax = plt.subplots(figsize=(7, 5))
            if "accuracy" in history:
                ax.plot(history["accuracy"], label="Train Accuracy")
            if "val_accuracy" in history:
                ax.plot(history["val_accuracy"], label="Validation Accuracy")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.set_title("Training Accuracy")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / "accuracy_curve.png", dpi=200)
            plt.close(fig)

        if "loss" in history or "val_loss" in history:
            fig, ax = plt.subplots(figsize=(7, 5))
            if "loss" in history:
                ax.plot(history["loss"], label="Train Loss")
            if "val_loss" in history:
                ax.plot(history["val_loss"], label="Validation Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Training Loss")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / "loss_curve.png", dpi=200)
            plt.close(fig)


def _bundle_to_dict(bundle: EvaluationBundle) -> dict[str, Any]:
    return {
        "training": _split_to_dict(bundle.training),
        "validation": _split_to_dict(bundle.validation),
        "testing": _split_to_dict(bundle.testing),
    }


def _split_to_dict(split: SplitEvaluation) -> dict[str, Any]:
    return {
        "accuracy": split.accuracy,
        "precision": split.precision,
        "recall": split.recall,
        "f1_score": split.f1_score,
        "roc_auc": split.roc_auc,
        "confusion_matrix": split.confusion_matrix.tolist(),
        "report": split.report,
    }
