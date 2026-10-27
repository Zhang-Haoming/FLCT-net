"""User-editable FLCT settings, preserved from run_comparison.py."""

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "results"

# d_model is the number of samples per subsegment; latent width is d_model * 4.
MODEL_CONFIG = {
    "d_model": 256,
    "nhead": 16,
    "num_layers": 4,
}

TRAIN_CONFIG = {
    "batch_size": 256,
    "epochs": 300,
    "lr": 0.0005,
    "loss": "mse",  # "mse" or "rrmset"; omitting this key also selects MSE.
}

DATA_CONFIG = {
    "seed": 12,
    "train_combin_num": 15,
    "val_combin_num": 1,
    "train_per": 0.8,
    "train_db": [-7, 5],
    "test_db": [-7, 5, 13],
    "clean_EEG": 0,
    "val_clean_EEG": 0,
}

EVALUATION_CONFIG = {
    "sampling_rate_hz": 512,
    "rrmsef_max_frequency_hz": 120,
    "latency_warmup": 10,
    "latency_runs": 100,
}

# Preserve the currently selected condition in the original runner.
DEFAULT_NOISE_TYPES = ["ecg"]
DATA_NOISE_TYPES = {
    "eog": "EOG",
    "emg": "EMG",
    "ecg": "ECG",
    "eog_emg": "EOG+EMG",
    "eog_ecg": "EOG+ECG",
    "emg_ecg": "EMG+ECG",
    "all": "all",  # The triple-artifact condition, not all seven experiments.
}

DATA_FILES = {
    "EEG_all": "EEG_all_epochs_512hz.npy",
    "EOG_noise_all": "EOG_all_epochs_512hz.npy",
    "EMG_noise_all": "EMG_all_epochs_512hz.npy",
    "ECG_noise_all": "ECG_all_epochs_512hz.npy",
}
