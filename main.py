"""Application entry point."""

from __future__ import annotations

import argparse
import sys

from anemia_detection_app.config import build_default_config
from anemia_detection_app.gui import run_application
from anemia_detection_app.training import PipelineTrainer


def main() -> int:
    """Run the GUI or a command-line training job."""

    parser = argparse.ArgumentParser(description="Anemia detection thesis application")
    parser.add_argument("--train", action="store_true", help="Run a training job without opening the GUI")
    parser.add_argument("--reducer", choices=["none", "lle", "isomap"], default="none")
    parser.add_argument("--classifier", choices=["svm", "logistic_regression", "random_forest"], default="svm")
    parser.add_argument("--fine-tune", action="store_true", help="Enable MobileNetV2 fine-tuning")
    args = parser.parse_args()

    if args.train:
        config = build_default_config()
        config.reduction.reducer = args.reducer  # type: ignore[assignment]
        config.classifier.classifier = args.classifier  # type: ignore[assignment]
        config.feature_extractor.fine_tune = args.fine_tune
        trainer = PipelineTrainer(config)
        trainer.train_pipeline(reducer_name=args.reducer, classifier_name=args.classifier)
        return 0

    return run_application()


if __name__ == "__main__":
    raise SystemExit(main())
