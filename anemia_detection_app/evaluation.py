"""Evaluation and visualization utilities for the anemia pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
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
