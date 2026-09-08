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
- Modern local web dashboard (HTML, CSS, and JavaScript)

## Run

```bash
python main.py
```

This opens a local dashboard at `http://127.0.0.1:8765`. The dashboard starts
training in a background thread, so its controls and logs remain responsive.

## GPU training on Windows

The app automatically trains on any GPU TensorFlow exposes and reports the
active mode in the dashboard. Current TensorFlow releases do not support CUDA
GPU training directly on native Windows; use the project from WSL2 with an
NVIDIA GPU driver to enable it. Run `python -c "import tensorflow as tf;
print(tf.config.list_physical_devices('GPU'))"` there before training.

### AMD Radeon / DirectML

For an AMD Radeon GPU on native Windows, use the separate DirectML environment
instead of the standard `.venv`. It intentionally uses Python 3.10,
TensorFlow 2.10, and NumPy 1.23.5 because those are the versions supported by
Microsoft's DirectML plugin.

```powershell
./run-directml.ps1
```

The dashboard should show `GPU ready` once DirectML detects the adapter. The
standard `python main.py` command continues to use the current CPU-focused
environment.

To run a headless training job:

```bash
python main.py --train --reducer lle --classifier svm --fine-tune
```

## Output folders

- `saved_models/` for trained bundles
- `logs/` for application logs
- `results/` for metrics and plots
- `assets/` for GUI assets
