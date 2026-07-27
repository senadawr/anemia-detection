"""PyQt6 desktop application for training and inference."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QComboBox,
    QCheckBox,
    QSplitter,
    QTextBrowser,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .config import AppConfig, build_default_config
from .models import ClassifierBundle
from .training import PipelinePredictor, PipelineTrainer, TrainingResult


class TrainingWorker(QThread):
    """Background worker that trains a pipeline without freezing the UI."""

    log_message = pyqtSignal(str)
    finished_training = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config

    def run(self) -> None:
        try:
            trainer = PipelineTrainer(self.config)
            result = trainer.train_pipeline(
                reducer_name=self.config.reduction.reducer,
                classifier_name=self.config.classifier.classifier,
            )
            self.finished_training.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


class ScaledImageLabel(QLabel):
    """Label that keeps the whole pixmap visible while the widget resizes."""

    def __init__(self, text: str = "") -> None:
        super().__init__(text)
        self._original_pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("border: 1px solid #999; background: #fafafa;")
        self.setMinimumHeight(260)

    def set_original_pixmap(self, pixmap: QPixmap) -> None:
        self._original_pixmap = pixmap
        self._apply_scaled_pixmap()

    def clear_image(self, text: str = "No image loaded") -> None:
        self._original_pixmap = None
        self.setText(text)
        self.setPixmap(QPixmap())

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._apply_scaled_pixmap()

    def _apply_scaled_pixmap(self) -> None:
        if self._original_pixmap is None or self._original_pixmap.isNull():
            return
        target_size = self.contentsRect().size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return
        scaled = self._original_pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        super().setPixmap(scaled)


class AnemiaDetectionWindow(QMainWindow):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Anemia Detection Thesis Application")
        self.resize(1400, 900)
        self.config = build_default_config()
        self.bundle: ClassifierBundle | None = None
        self.predictor: PipelinePredictor | None = None
        self.worker: TrainingWorker | None = None
        self.result_dir: Path | None = None
        self._build_ui()
        self._refresh_dataset_path()

    def _build_ui(self) -> None:
        central = QWidget()
        root_layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter)
        self.setCentralWidget(central)

        controls = QWidget()
        control_layout = QVBoxLayout(controls)
        control_layout.setSpacing(10)

        dataset_box = QGroupBox("Dataset")
        dataset_layout = QFormLayout(dataset_box)
        self.dataset_path_edit = QLineEdit(str(self.config.dataset.root_dir))
        browse_dataset_button = QPushButton("Browse")
        browse_dataset_button.clicked.connect(self._browse_dataset)
        dataset_row = QWidget()
        dataset_row_layout = QHBoxLayout(dataset_row)
        dataset_row_layout.setContentsMargins(0, 0, 0, 0)
        dataset_row_layout.addWidget(self.dataset_path_edit)
        dataset_row_layout.addWidget(browse_dataset_button)
        dataset_layout.addRow("Root", dataset_row)

        pipeline_box = QGroupBox("Pipeline")
        pipeline_layout = QFormLayout(pipeline_box)
        self.reducer_combo = QComboBox()
        self.reducer_combo.addItems(["none", "lle", "isomap"])
        self.classifier_combo = QComboBox()
        self.classifier_combo.addItems(["svm", "logistic_regression", "random_forest"])
        self.fine_tune_check = QCheckBox("Enable fine-tuning")
        self.fine_tune_check.setChecked(self.config.feature_extractor.fine_tune)
        self.fine_tune_check.stateChanged.connect(self._toggle_fine_tuning)
        pipeline_layout.addRow("Reducer", self.reducer_combo)
        pipeline_layout.addRow("Classifier", self.classifier_combo)
        pipeline_layout.addRow("Backbone", self.fine_tune_check)

        actions_box = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_box)
        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self._train_model)
        self.load_button = QPushButton("Load Saved Model")
        self.load_button.clicked.connect(self._load_model)
        self.predict_button = QPushButton("Predict Image")
        self.predict_button.clicked.connect(self._predict_image)
        self.predict_button.setEnabled(False)
        actions_layout.addWidget(self.train_button)
        actions_layout.addWidget(self.load_button)
        actions_layout.addWidget(self.predict_button)

        prediction_box = QGroupBox("Prediction")
        prediction_layout = QVBoxLayout(prediction_box)
        self.prediction_label = QLabel("Class: -")
        self.confidence_label = QLabel("Confidence: -")
        self.image_preview = ScaledImageLabel("Image preview")
        prediction_layout.addWidget(self.prediction_label)
        prediction_layout.addWidget(self.confidence_label)
        prediction_layout.addWidget(self.image_preview)

        metrics_box = QGroupBox("Evaluation")
        metrics_layout = QVBoxLayout(metrics_box)
        self.metrics_text = QTextBrowser()
        self.metrics_text.setOpenExternalLinks(False)
        self.metrics_text.setStyleSheet(
            "QTextBrowser { background: #111; color: #e8e8e8; border: 1px solid #444; padding: 8px; }"
        )
        metrics_layout.addWidget(self.metrics_text)

        control_layout.addWidget(dataset_box)
        control_layout.addWidget(pipeline_box)
        control_layout.addWidget(actions_box)
        control_layout.addWidget(prediction_box)
        control_layout.addWidget(metrics_box)
        control_layout.addStretch(1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.tabs = QTabWidget()
        self.graph_tabs: dict[str, ScaledImageLabel] = {}
        for name in ("Training Curves", "Training Confusion", "Validation Confusion", "Testing Confusion", "Testing ROC"):
            tab = QWidget()
            layout = QVBoxLayout(tab)
            label = ScaledImageLabel("No graph loaded")
            label.setStyleSheet("border: 1px solid #999; background: #ffffff;")
            label.setMinimumHeight(320)
            layout.addWidget(label)
            self.tabs.addTab(tab, name)
            self.graph_tabs[name] = label

        self.log_box = QTextBrowser()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Training and prediction logs appear here.")

        right_layout.addWidget(self.tabs, stretch=3)
        right_layout.addWidget(QLabel("Logs"))
        right_layout.addWidget(self.log_box, stretch=2)

        splitter.addWidget(controls)
        splitter.addWidget(right_panel)
        splitter.setSizes([450, 950])

    def _browse_dataset(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select Dataset Root", str(self.config.dataset.root_dir))
        if not directory:
            return
        self.dataset_path_edit.setText(directory)
        self._refresh_dataset_path()

    def _refresh_dataset_path(self) -> None:
        self.config.dataset.root_dir = Path(self.dataset_path_edit.text()).expanduser().resolve()
        self._log(f"Dataset root set to {self.config.dataset.root_dir}")

    def _toggle_fine_tuning(self) -> None:
        self.config.feature_extractor.fine_tune = self.fine_tune_check.isChecked()

    def _build_runtime_config(self) -> AppConfig:
        runtime_config = AppConfig.from_dict(self.config.to_dict())
        runtime_config.dataset.root_dir = Path(self.dataset_path_edit.text()).expanduser().resolve()
        runtime_config.reduction.reducer = self.reducer_combo.currentText()  # type: ignore[assignment]
        runtime_config.classifier.classifier = self.classifier_combo.currentText()  # type: ignore[assignment]
        runtime_config.feature_extractor.fine_tune = self.fine_tune_check.isChecked()
        return runtime_config

    def _train_model(self) -> None:
        self._refresh_dataset_path()
        runtime_config = self._build_runtime_config()
        self.train_button.setEnabled(False)
        self.predict_button.setEnabled(False)
        self._log("Starting training run...")
        self.worker = TrainingWorker(runtime_config)
        self.worker.finished_training.connect(self._on_training_finished)
        self.worker.failed.connect(self._on_training_failed)
        self.worker.start()

    def _on_training_finished(self, result: object) -> None:
        training_result = result  # keep a readable name
        if not isinstance(training_result, TrainingResult):
            self._on_training_failed("Training finished with an unexpected result object.")
            return
        self.bundle = training_result.bundle
        self.predictor = PipelinePredictor(self.bundle)
        self.result_dir = self.bundle.root_dir.parent.parent / "results" / self.bundle.root_dir.name
        self._populate_metrics(training_result)
        self._populate_graphs(self.result_dir)
        self.train_button.setEnabled(True)
        self.predict_button.setEnabled(True)
        self._log(f"Training complete. Artifacts saved under {self.bundle.root_dir}")

    def _on_training_failed(self, message: str) -> None:
        self.train_button.setEnabled(True)
        self.predict_button.setEnabled(self.predictor is not None)
        self._log(message)
        QMessageBox.critical(self, "Training Failed", message)

    def _load_model(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select Saved Model Directory", str(self.config.output.models_dir))
        if not directory:
            return
        bundle = ClassifierBundle.from_root_dir(Path(directory))
        try:
            self.bundle = bundle
            self.predictor = PipelinePredictor(bundle)
            self.result_dir = bundle.metrics_path.parent if bundle.metrics_path else bundle.root_dir.parent.parent / "results" / bundle.root_dir.name
            self.predict_button.setEnabled(True)
            self._populate_saved_metrics(bundle)
            self._populate_graphs(self.result_dir)
            self._log(f"Loaded saved model from {directory}")
        except Exception as exc:
            QMessageBox.critical(self, "Load Failed", str(exc))
            self._log(f"Failed to load model: {exc}")

    def _predict_image(self) -> None:
        if self.predictor is None:
            QMessageBox.warning(self, "No Model", "Train or load a model first.")
            return
        image_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            str(self.config.dataset.root_dir),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)",
        )
        if not image_path:
            return
        try:
            result = self.predictor.predict_image(image_path)
            self.prediction_label.setText(f"Class: {result.class_name}")
            self.confidence_label.setText(f"Confidence: {result.confidence:.4f}")
            self._show_image(image_path)
            self._log(f"Predicted {result.class_name} with confidence {result.confidence:.4f}")
        except Exception as exc:
            QMessageBox.critical(self, "Prediction Failed", str(exc))
            self._log(f"Prediction failed: {exc}")

    def _populate_metrics(self, result: TrainingResult) -> None:
        self.metrics_text.setHtml(self._format_metrics_html(result.evaluation, result.metadata))

    def _populate_saved_metrics(self, bundle: ClassifierBundle) -> None:
        if not bundle.metrics_path or not bundle.metrics_path.exists():
            self.metrics_text.setHtml("<p>Saved model loaded. No metrics file found.</p>")
            return
        try:
            import json

            payload = json.loads(bundle.metrics_path.read_text(encoding="utf-8"))
            self.metrics_text.setHtml(self._format_saved_metrics_html(payload))
        except Exception:
            self.metrics_text.setPlainText(bundle.metrics_path.read_text(encoding="utf-8"))

    def _populate_graphs(self, result_dir: Path | None) -> None:
        if result_dir is None or not result_dir.exists():
            return
        mapping = {
            "Training Curves": result_dir / "training_curves" / "accuracy_curve.png",
            "Training Confusion": result_dir / "training" / "confusion_matrix.png",
            "Validation Confusion": result_dir / "validation" / "confusion_matrix.png",
            "Testing Confusion": result_dir / "testing" / "confusion_matrix.png",
            "Testing ROC": result_dir / "testing" / "roc_curve.png",
        }
        for tab_name, image_path in mapping.items():
            label = self.graph_tabs[tab_name]
            if image_path.exists():
                pixmap = QPixmap(str(image_path))
                label.set_original_pixmap(pixmap)
            else:
                label.clear_image("No graph loaded")

    def _show_image(self, image_path: str) -> None:
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.image_preview.clear_image("Unable to preview image")
            return
        self.image_preview.set_original_pixmap(pixmap)

    def _format_metrics_html(self, evaluation: Any, metadata: dict[str, Any]) -> str:
        def metric_card(title: str, split: Any) -> str:
            roc_auc = f"{split.roc_auc:.4f}" if split.roc_auc is not None else "n/a"
            return (
                "<div style='background:#1a1a1a;border:1px solid #333;border-radius:10px;padding:12px;margin-bottom:10px;'>"
                f"<div style='font-size:16px;font-weight:700;color:#ffffff;margin-bottom:8px;'>{title}</div>"
                "<table style='width:100%;border-collapse:collapse;color:#e8e8e8;'>"
                f"<tr><td><b>Accuracy</b></td><td>{split.accuracy:.4f}</td></tr>"
                f"<tr><td><b>Precision</b></td><td>{split.precision:.4f}</td></tr>"
                f"<tr><td><b>Recall</b></td><td>{split.recall:.4f}</td></tr>"
                f"<tr><td><b>F1-score</b></td><td>{split.f1_score:.4f}</td></tr>"
                f"<tr><td><b>ROC-AUC</b></td><td>{roc_auc}</td></tr>"
                "</table>"
                "</div>"
            )

        def confusion_box(split: Any) -> str:
            matrix = split.confusion_matrix.tolist()
            return (
                "<div style='margin-top:10px;'>"
                "<div style='font-weight:700;margin-bottom:6px;color:#ffffff;'>Confusion Matrix</div>"
                "<table style='border-collapse:collapse;'>"
                + "".join(
                    "<tr>"
                    + "".join(
                        f"<td style='border:1px solid #555;padding:8px 12px;text-align:center;background:#222;color:#fff;'>{value}</td>"
                        for value in row
                    )
                    + "</tr>"
                    for row in matrix
                )
                + "</table></div>"
            )

        html = [
            "<html><body style='font-family:Segoe UI,Arial,sans-serif;background:#111;color:#e8e8e8;'>",
            f"<div style='font-size:18px;font-weight:700;margin-bottom:12px;'>Pipeline: {metadata['pipeline_name']}</div>",
            f"<div style='margin-bottom:12px;'>Feature shape: {metadata['feature_shape']}</div>",
        ]
        for split_name in ("training", "validation", "testing"):
            split = getattr(evaluation, split_name)
            html.append(metric_card(split_name.title(), split))
            html.append(confusion_box(split))
        html.append("</body></html>")
        return "".join(html)

    def _format_saved_metrics_html(self, payload: dict[str, Any]) -> str:
        html = [
            "<html><body style='font-family:Segoe UI,Arial,sans-serif;background:#111;color:#e8e8e8;'>",
            "<div style='font-size:18px;font-weight:700;margin-bottom:12px;'>Saved Metrics</div>",
        ]
        for split_name in ("training", "validation", "testing"):
            split = payload.get(split_name, {})
            html.append(
                "<div style='background:#1a1a1a;border:1px solid #333;border-radius:10px;padding:12px;margin-bottom:10px;'>"
                f"<div style='font-size:16px;font-weight:700;color:#ffffff;margin-bottom:8px;'>{split_name.title()}</div>"
                "<table style='width:100%;border-collapse:collapse;color:#e8e8e8;'>"
                f"<tr><td><b>Accuracy</b></td><td>{split.get('accuracy', 'n/a')}</td></tr>"
                f"<tr><td><b>Precision</b></td><td>{split.get('precision', 'n/a')}</td></tr>"
                f"<tr><td><b>Recall</b></td><td>{split.get('recall', 'n/a')}</td></tr>"
                f"<tr><td><b>F1-score</b></td><td>{split.get('f1_score', 'n/a')}</td></tr>"
                f"<tr><td><b>ROC-AUC</b></td><td>{split.get('roc_auc', 'n/a')}</td></tr>"
                "</table>"
                "</div>"
            )
        html.append("</body></html>")
        return "".join(html)

    def _log(self, message: str) -> None:
        self.log_box.append(message)


def run_application() -> int:
    """Launch the PyQt6 application."""

    app = QApplication([])
    window = AnemiaDetectionWindow()
    window.show()
    return app.exec()
