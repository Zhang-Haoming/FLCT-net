"""Original temporal RRMSE, 0-120 Hz spectral RRMSE, and correlation metrics."""

import math
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SAMPLING_RATE_HZ = 512
DEFAULT_RRMSEF_MAX_FREQUENCY_HZ = 120


def get_rms(records):
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))


def rrmset(eeg, deeg, eog=None, i=0, print_max_min=False):
    values = []
    for clean, denoised in zip(eeg, deeg):
        values.append(get_rms(denoised - clean) / get_rms(clean))
    mean, std = np.mean(values), np.std(values)
    if print_max_min and i == 1:
        maximum, minimum = values.index(max(values)), values.index(min(values))
        print("RRMSET max/index/min/index:", max(values), maximum, min(values), minimum)
        return mean, std, maximum, minimum
    return mean, std


def rrmsef(eeg, deeg, eog=None, sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ,
           max_frequency_hz=DEFAULT_RRMSEF_MAX_FREQUENCY_HZ):
    num_fft = eeg.shape[1]
    frequencies = np.fft.rfftfreq(num_fft, d=1 / sampling_rate_hz)
    frequency_mask = frequencies <= max_frequency_hz
    clean_psd = np.abs(np.fft.rfft(eeg, n=num_fft, axis=1)) ** 2 / num_fft
    denoised_psd = np.abs(np.fft.rfft(deeg, n=num_fft, axis=1)) ** 2 / num_fft
    clean_psd = clean_psd[:, frequency_mask]
    denoised_psd = denoised_psd[:, frequency_mask]
    numerator = np.sqrt(np.mean((denoised_psd - clean_psd) ** 2, axis=1))
    denominator = np.sqrt(np.mean(clean_psd ** 2, axis=1))
    values = numerator / denominator
    return np.mean(values), np.std(values)


def acc(eeg, deeg, eog=None):
    values = [np.corrcoef(denoised, clean)[0, 1]
              for clean, denoised in zip(eeg, deeg)]
    return np.mean(values), np.std(values)


def quantify_with_test_db(EEG, DEEG, noiseEEG, targetlocation, test_db=(-7, 5, 13),
                          sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ,
                          rrmsef_max_frequency_hz=DEFAULT_RRMSEF_MAX_FREQUENCY_HZ,
                          clean_samples=0):
    """Evaluate equal-sized SNR blocks; preserve the original CSV columns.

    If preparation appended clean-only samples, they are saved in the output
    arrays but excluded from SNR-specific metrics using clean_samples.
    """
    arrays = [np.asarray(array) for array in (EEG, DEEG, noiseEEG)]
    if any(array.ndim != 2 or array.shape != arrays[0].shape for array in arrays):
        raise ValueError("Metrics require matching (samples, time_points) arrays.")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("Metrics received NaN or infinity.")
    if clean_samples:
        if clean_samples < 0 or clean_samples >= len(arrays[0]):
            raise ValueError("Invalid number of appended clean-only samples.")
        arrays = [array[:-clean_samples] for array in arrays]
    snrs = np.linspace(test_db[0], test_db[1], num=int(test_db[2]))
    count = len(snrs)
    if not len(arrays[0]) or len(arrays[0]) % count:
        raise ValueError("Test samples must form equal-sized blocks for each SNR.")
    points = arrays[0].shape[1]
    clean, denoised, noisy = [array.reshape(count, -1, points) for array in arrays]
    if np.any(np.mean(clean ** 2, axis=2) == 0):
        raise ValueError("RRMSE is undefined for a zero-energy clean segment.")
    rows = []
    for index, snr in enumerate(snrs):
        artifact = noisy[index] - clean[index]
        mean_t, std_t = rrmset(clean[index], denoised[index], artifact, index, False)
        mean_f, std_f = rrmsef(
            clean[index], denoised[index], artifact,
            sampling_rate_hz=sampling_rate_hz,
            max_frequency_hz=rrmsef_max_frequency_hz,
        )
        mean_cc, std_cc = acc(clean[index], denoised[index], artifact)
        rows.append({
            "SNR_test_dB": snr,
            "RRMSET_mean": mean_t, "RRMSET_std": std_t,
            "RRMSEF_mean": mean_f, "RRMSEF_std": std_f,
            "ACC_mean": mean_cc, "ACC_std": std_cc,
        })
    result = pd.DataFrame(rows)
    targetlocation = Path(targetlocation)
    targetlocation.mkdir(parents=True, exist_ok=True)
    result.to_csv(targetlocation / "quantify.csv", index=False)
    return result
