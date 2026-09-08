"""Local web interface for training and using anemia classifiers.

The HTTP layer only coordinates user actions.  All training and inference still
go through the same PipelineTrainer and PipelinePredictor classes used by CLI.
"""

from __future__ import annotations

import json
import threading
import traceback
import webbrowser
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from flask import Flask, jsonify, request

from .config import AppConfig, build_default_config
from .models import ClassifierBundle
from .runtime import configure_accelerator
from .training import PipelinePredictor, PipelineTrainer, TrainingResult


class ApplicationState:
    """Keep the long-running ML objects separate from request handlers."""

    def __init__(self) -> None:
        self.config = build_default_config()
        self.accelerator = configure_accelerator()
        self.bundle: ClassifierBundle | None = None
        self.predictor: PipelinePredictor | None = None
        self.result_dir: Path | None = None
        self.training = False
        self.logs: list[str] = ["Ready. Configure a pipeline or load a saved model."]
        self.error: str | None = None
        self.lock = threading.Lock()

    def log(self, message: str) -> None:
        with self.lock:
            self.logs.append(message)
            self.logs = self.logs[-150:]


def _result_directory(bundle: ClassifierBundle) -> Path:
    return bundle.metrics_path.parent if bundle.metrics_path else bundle.root_dir.parent.parent / "results" / bundle.root_dir.name


def _runtime_config(payload: dict[str, Any]) -> AppConfig:
    config = build_default_config()
    dataset_root = str(payload.get("dataset_root", "")).strip()
    if dataset_root:
        config.dataset.root_dir = Path(dataset_root).expanduser().resolve()
    config.reduction.reducer = payload.get("reducer", "none")  # type: ignore[assignment]
    config.classifier.classifier = payload.get("classifier", "svm")  # type: ignore[assignment]
    config.feature_extractor.fine_tune = bool(payload.get("fine_tune", False))
    return config


def create_app() -> Flask:
    """Create the local dashboard and its small JSON API."""

    app = Flask(__name__, static_folder="web", static_url_path="")
    state = ApplicationState()
    app.config["STATE"] = state

    @app.get("/")
    def dashboard():
        return app.send_static_file("index.html")

    @app.get("/api/status")
    def status():
        with state.lock:
            return jsonify({
                "training": state.training,
                "model_loaded": state.predictor is not None,
                "result_dir": str(state.result_dir) if state.result_dir else None,
                "logs": state.logs,
                "error": state.error,
                "default_dataset": str(state.config.dataset.root_dir),
                "accelerator": state.accelerator,
            })

    @app.post("/api/train")
    def train():
        payload = request.get_json(silent=True) or {}
        with state.lock:
            if state.training:
                return jsonify({"error": "A training run is already in progress."}), 409
            state.training = True
            state.error = None
        config = _runtime_config(payload)
        state.log(f"Training requested: {config.reduction.reducer} + {config.classifier.classifier}.")

        def run_training() -> None:
            try:
                result = PipelineTrainer(config).train_pipeline(
                    reducer_name=config.reduction.reducer,
                    classifier_name=config.classifier.classifier,
                )
                assert isinstance(result, TrainingResult)
                with state.lock:
                    state.bundle = result.bundle
                    state.predictor = PipelinePredictor(result.bundle)
                    state.result_dir = _result_directory(result.bundle)
                state.log(f"Training complete. Saved model: {result.bundle.root_dir}")
            except Exception:
                message = traceback.format_exc()
                with state.lock:
                    state.error = message
                state.log("Training failed. See the dashboard log for details.")
            finally:
                with state.lock:
                    state.training = False

        threading.Thread(target=run_training, name="anemia-training", daemon=True).start()
        return jsonify({"ok": True})

    @app.post("/api/load-model")
    def load_model():
        payload = request.get_json(silent=True) or {}
        directory = Path(str(payload.get("model_directory", "")).strip()).expanduser()
        if not directory.is_dir():
            return jsonify({"error": "Enter a valid saved-model directory."}), 400
        try:
            bundle = ClassifierBundle.from_root_dir(directory.resolve())
            predictor = PipelinePredictor(bundle)
            with state.lock:
                state.bundle, state.predictor = bundle, predictor
                state.result_dir = _result_directory(bundle)
                state.error = None
            state.log(f"Loaded model from {directory.resolve()}")
            return jsonify({"ok": True})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/predict")
    def predict():
        upload = request.files.get("image")
        if upload is None or not upload.filename:
            return jsonify({"error": "Choose an image before predicting."}), 400
        with state.lock:
            predictor = state.predictor
        if predictor is None:
            return jsonify({"error": "Train or load a model first."}), 400
        suffix = Path(upload.filename).suffix or ".png"
        try:
            with TemporaryDirectory() as temporary_directory:
                image_path = Path(temporary_directory) / f"prediction{suffix}"
                upload.save(image_path)
                result = predictor.predict_image(image_path)
            state.log(f"Predicted {result.class_name} at {result.confidence:.2%} confidence.")
            return jsonify({
                "class_name": result.class_name,
                "confidence": result.confidence,
                "probabilities": result.probabilities.tolist(),
            })
        except Exception as exc:
            state.log(f"Prediction failed: {exc}")
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/results")
    def results():
        with state.lock:
            result_dir = state.result_dir
            bundle = state.bundle
        if result_dir is None and bundle is not None:
            result_dir = _result_directory(bundle)
        if result_dir is None:
            return jsonify({"metrics": None, "history": {}, "charts": {}})
        metrics_path = result_dir / "metrics.json"
        history_path = bundle.history_path if bundle else None
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path and history_path.exists() else {}
        charts: dict[str, Any] = {}
        for split in ("training", "validation", "testing"):
            path = result_dir / split / "chart_data.json"
            if path.exists():
                charts[split] = json.loads(path.read_text(encoding="utf-8"))
        return jsonify({"metrics": metrics, "history": history, "charts": charts, "result_dir": str(result_dir)})

    return app


def run_application() -> int:
    """Launch the dashboard in the user's browser."""

    app = create_app()
    url = "http://127.0.0.1:8765"
    threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=8765, debug=False, threaded=True, use_reloader=False)
    return 0
