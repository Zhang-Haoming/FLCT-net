import os

import numpy as np
import pandas as pd


DEFAULT_SAMPLING_RATE_HZ = 512
POWER_BAND_REFERENCE_HZ = (1, 80)
POWER_BANDS_HZ = (
    ("Delta", 1, 4),
    ("Theta", 4, 8),
    ("Alpha", 8, 13),
    ("Beta", 13, 30),
    ("Gamma", 30, 80),
)


def _relative_band_power(signal, sampling_rate_hz):
    num_fft = signal.shape[0]
    frequencies = np.fft.rfftfreq(num_fft, d=1 / sampling_rate_hz)
    power = np.abs(np.fft.rfft(signal, n=num_fft)) ** 2 / num_fft

    reference_lower, reference_upper = POWER_BAND_REFERENCE_HZ
    total_mask = (frequencies >= reference_lower) & (
        frequencies <= reference_upper
    )
    total_power = np.sum(power[total_mask])

    relative_power = []
    for _, lower, upper in POWER_BANDS_HZ:
        if upper == reference_upper:
            band_mask = (frequencies >= lower) & (frequencies <= upper)
        else:
            band_mask = (frequencies >= lower) & (frequencies < upper)
        relative_power.append(np.sum(power[band_mask]) / total_power)

    return tuple(relative_power)


def powerbandmain(
    EEG_test,
    Denoiseoutput_test_end,
    noiseEEG_test,
    save_power_band_to_csv,
    targetlocation,
    sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ,
):
    clean_power = np.array(
        [_relative_band_power(signal, sampling_rate_hz) for signal in EEG_test]
    ).mean(axis=0)
    denoised_power = np.array(
        [
            _relative_band_power(signal, sampling_rate_hz)
            for signal in Denoiseoutput_test_end
        ]
    ).mean(axis=0)
    noisy_power = np.array(
        [_relative_band_power(signal, sampling_rate_hz) for signal in noiseEEG_test]
    ).mean(axis=0)

    result = pd.DataFrame(
        {
            "powerband": [name for name, _, _ in POWER_BANDS_HZ],
            "Noise_EEG": noisy_power,
            "clean_EEG": clean_power,
            "Denoised_EEG": denoised_power,
        }
    )
    print(result)

    if save_power_band_to_csv:
        os.makedirs(targetlocation, exist_ok=True)
        result.to_csv(os.path.join(targetlocation, "powerband.csv"), index=False)
