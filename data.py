"""Load EEG/artifact arrays and call the unchanged data preparation module."""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import DATA_FILES, DATA_NOISE_TYPES
from data_prepare_new_3noise import prepare_data_new


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_noise_type(name):
    value = str(name).lower().replace(" ", "").replace("_", "+")
    if value == "all":
        return "all"
    parts = value.split("+")
    if len(parts) != len(set(parts)):
        raise ValueError(f"Repeated artifact in noise condition: {name}")
    for key, label in DATA_NOISE_TYPES.items():
        if key != "all" and set(parts) == set(label.lower().split("+")):
            return key
    if set(parts) == {"eog", "emg", "ecg"}:
        return "all"
    raise ValueError(f"Unknown noise condition: {name}")


class EEGDataset(Dataset):
    def __init__(self, inputs, targets):
        inputs = np.asarray(inputs)
        targets = np.asarray(targets)
        if inputs.ndim != 2 or inputs.shape != targets.shape or not len(inputs):
            raise ValueError("Expected nonempty, matching (samples, time_points) arrays.")
        self.X = torch.tensor(inputs[:, np.newaxis, :], dtype=torch.float32)
        self.Y = torch.tensor(targets[:, np.newaxis, :], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


def _load_arrays(data_dir, subsegment_samples, smoke_test):
    data_dir = Path(data_dir)
    missing = [str(data_dir / name) for name in DATA_FILES.values()
               if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing input data:\n" + "\n".join(missing))

    arrays = {}
    for key, name in DATA_FILES.items():
        array = np.load(data_dir / name, allow_pickle=False)
        if array.ndim != 2 or array.shape[0] < 10:
            raise ValueError(f"{name}: expected at least 10 rows of 2D EEG segments.")
        if not np.isfinite(array).all():
            raise ValueError(f"{name}: input contains NaN or infinity.")
        if np.any(np.std(array, axis=1) == 0):
            raise ValueError(f"{name}: constant segments cannot be normalized.")
        arrays[key] = array[:64] if smoke_test else array

    lengths = {array.shape[1] for array in arrays.values()}
    if len(lengths) != 1 or next(iter(lengths)) % subsegment_samples:
        raise ValueError("All input arrays must have matching lengths divisible by d_model.")
    return arrays


def _limit_snr_blocks(inputs, targets, count, per_snr=2):
    if len(inputs) % count:
        raise ValueError("Validation/test samples do not form equal SNR blocks.")
    shape = (count, -1, inputs.shape[-1])
    return (
        inputs.reshape(shape)[:, :per_snr].reshape(-1, inputs.shape[-1]),
        targets.reshape(shape)[:, :per_snr].reshape(-1, targets.shape[-1]),
    )


def prepare_datasets(data_dir, noise_type, data_config, model_config, smoke_test=False):
    """Return train/validation/test sets in the original SNR-major order.

    Smoke tests use real data from the first 64 rows of each source, eight
    training samples, and two validation/test samples at every SNR level.
    Normal runs always use the full source arrays and original settings.
    """
    set_seed(data_config["seed"])
    arrays = _load_arrays(data_dir, model_config["d_model"], smoke_test)
    settings = {key: value for key, value in data_config.items() if key != "seed"}
    result = prepare_data_new(
        **arrays,
        **settings,
        noise_type=DATA_NOISE_TYPES[normalize_noise_type(noise_type)],
    )
    train_x, train_y, val_x, val_y, test_x, test_y, _ = result
    if smoke_test:
        if data_config["val_clean_EEG"]:
            raise ValueError("Smoke tests require val_clean_EEG=0.")
        train_x, train_y = train_x[:8], train_y[:8]
        count = int(data_config["test_db"][2])
        val_x, val_y = _limit_snr_blocks(val_x, val_y, count)
        test_x, test_y = _limit_snr_blocks(test_x, test_y, count)

    return (
        EEGDataset(train_x, train_y),
        EEGDataset(val_x, val_y),
        EEGDataset(test_x, test_y),
    )


def make_dataloaders(datasets, batch_size):
    # The original runner uses shuffle=False, including for training.
    return tuple(DataLoader(dataset, batch_size=batch_size, shuffle=False)
                 for dataset in datasets)
