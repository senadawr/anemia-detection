# Anemia Detection Thesis Application

This project implements MobileNetV2-based anemia classification pipelines for the cleaned dataset in `dataset_clean/`.

## Included modules
- Configuration
- Dataset discovery and loading
- OpenCV preprocessing
- MobileNetV2 feature extraction
- LLE and ISOMAP reduction
- Classical classifiers
- Training and evaluation
- PyQt6 GUI

## Run

```bash
python main.py
```

To run a headless training job:

```bash
python main.py --train --reducer lle --classifier svm --fine-tune
```

## Output folders

- `saved_models/` for trained bundles
- `logs/` for application logs
- `results/` for metrics and plots
- `assets/` for GUI assets
