# FLCT
## Description
Code and data for: 'FLCT: Subsegment-Based CNN–Transformer with Rotary Position Embedding for Flexible-Length EEG Denoising'.
PyTorch implementation of FLCT, a subsegment-based CNN–Transformer framework for flexible-length EEG denoising.

FLCT encodes 0.5 s EEG subsegments using a shared residual CNN, integrates their representations through a Transformer with rotary position embedding (RoPE), and reconstructs the denoised waveform using a residual decoder.

![Overview of FLCT](xxxx.png)

## Repository Structure

```text
FLCT-net/
├── flct.py                       # FLCT architecture
├── config.py                     # Model, training, and evaluation settings
├── data.py                       # Data loading and dataset construction
├── data_prepare_new_3noise.py     # Original artifact mixing and normalization
├── train.py                      # Training and validation
├── test.py                       # Checkpoint loading, inference, and export
├── main.py                       # Training and testing entry point
├── quantify.py                   # Temporal RRMSE, spectral RRMSE, and CC
├── power_band.py                 # Relative EEG band-power analysis
├── requirements.txt

```


## Data Preparation

Prepare the following NumPy arrays at 512 Hz, with 1024 samples per segment:

```text
data/
├── EEG_all_epochs_512hz.npy
├── EOG_all_epochs_512hz.npy
├── EMG_all_epochs_512hz.npy
└── ECG_all_epochs_512hz.npy
```

Each array must have shape `(number_of_segments, 1024)`. All four arrays are required by the original data preparation procedure.

Clean EEG, EOG, and EMG signals are obtained from [EEGdenoiseNet](https://github.com/ncclabsustech/EEGdenoiseNet). ECG signals are obtained from the [MIT–BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) and must be preprocessed into the required NumPy format.

The preparation procedure retains the original data splitting, artifact pairing, SNR scaling, and normalization. For mixed artifacts, individual components are first scaled to the same SNR relative to the clean EEG. Their sum is then rescaled to the target total SNR.

## Training

Train and evaluate FLCT under the combined EOG+EMG+ECG condition:

```bash
python main.py --data-dir ./data --noise-types all
```

Run all seven artifact conditions:

```bash
python main.py --data-dir ./data --all-noise-types
```

Supported conditions are `eog`, `emg`, `ecg`, `eog_emg`, `eog_ecg`, `emg_ecg`, and `all`. Here, `all` denotes EOG+EMG+ECG. A separate model is trained for each selected condition.

Default settings are defined in `config.py`:

| Setting | Default |
|---|---|
| Subsegment size | 256 samples |
| Transformer layers | 4 |
| Attention heads | 16 |
| FFN dimension | 2048 |
| Optimizer | Adam |
| Learning rate | 0.0005 |
| Batch size | 256 |
| Epochs | 300 |
| Loss | MSE |
| Random seed | 12 |
