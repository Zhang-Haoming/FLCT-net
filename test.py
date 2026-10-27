"""Load the best FLCT checkpoint, run inference, and export test metrics."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from power_band import powerbandmain
from quantify import quantify_with_test_db
from train import model_output_to_target_shape, sync_if_cuda


def load_checkpoint(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model


def predict(dataloader, model, device):
    model.eval()
    predictions, noisy_inputs, clean_targets = [], [], []
    sync_if_cuda(device)
    start = time.perf_counter()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            prediction = model_output_to_target_shape(model(inputs))
            predictions.append(prediction.cpu())
            noisy_inputs.append(inputs.cpu())
            clean_targets.append(targets.cpu())
    sync_if_cuda(device)
    elapsed = time.perf_counter() - start
    if not predictions:
        raise ValueError("Test dataloader is empty.")
    # Only remove the channel dimension; preserve the sample axis for batch size 1.
    arrays = [torch.cat(items, dim=0).squeeze(1).numpy()
              for items in (predictions, noisy_inputs, clean_targets)]
    return (*arrays, elapsed)


def measure_single_sample_latency(model, dataset, device, warmup=10, runs=100):
    if warmup < 0 or runs < 1:
        raise ValueError("Latency measurement requires warmup >= 0 and runs >= 1.")
    sample, _ = dataset[0]
    sample = sample.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model_output_to_target_shape(model(sample))
        sync_if_cuda(device)
        start = time.perf_counter()
        for _ in range(runs):
            model_output_to_target_shape(model(sample))
        sync_if_cuda(device)
    return (time.perf_counter() - start) / runs


def _training_summary(checkpoint_path):
    log_path = Path(checkpoint_path).parent / "training_log.csv"
    if not log_path.is_file():
        return {}
    history = pd.read_csv(log_path)
    return {
        "best_val_loss": float(history["val_loss"].min()),
        "best_epoch": int(history.loc[history["val_loss"].idxmin(), "epoch"]),
        "completed_epochs": len(history),
        "total_train_time_sec": float(history["epoch_time_sec"].sum()),
        "training_log_path": str(log_path.resolve()),
    }


def evaluate(model, dataloader, device, checkpoint_path, save_dir, run_config,
             training_summary=None):
    load_checkpoint(model, checkpoint_path, device)
    print(f"[FLCT] Loaded checkpoint: {checkpoint_path}")
    predictions, noisy, clean, inference_time = predict(dataloader, model, device)
    evaluation = run_config["evaluation_config"]
    latency = measure_single_sample_latency(
        model, dataloader.dataset, device,
        warmup=evaluation["latency_warmup"], runs=evaluation["latency_runs"],
    )
    output_dir = Path(save_dir) / "nn_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "noiseEEG_test.npy", noisy)
    np.save(output_dir / "EEG_test.npy", clean)
    np.save(output_dir / "denoise_output_test.npy", predictions)
    powerbandmain(
        clean, predictions, noisy, True, str(output_dir),
        sampling_rate_hz=evaluation["sampling_rate_hz"],
    )
    data_config = run_config["data_config"]
    snr_count = int(data_config["test_db"][2])
    appended_clean = 0
    if data_config["val_clean_EEG"]:
        if len(clean) % (snr_count + 1):
            raise ValueError("Cannot identify appended clean-only test samples.")
        appended_clean = len(clean) // (snr_count + 1)
    metrics = quantify_with_test_db(
        clean, predictions, noisy, output_dir,
        test_db=data_config["test_db"],
        sampling_rate_hz=evaluation["sampling_rate_hz"],
        rrmsef_max_frequency_hz=evaluation["rrmsef_max_frequency_hz"],
        clean_samples=appended_clean,
    )
    settings = run_config["train_config"]
    summary = {
        **(_training_summary(checkpoint_path) if training_summary is None
           else training_summary),
        "model": "flct", "noise_type": run_config["noise_type"],
        "batch_size": dataloader.batch_size,
        "epochs": settings["epochs"], "lr": settings["lr"],
        "loss": settings.get("loss", "mse"),
        "test_inference_time_sec": inference_time,
        "single_sample_latency_sec": latency,
        "single_sample_latency_ms": latency * 1000,
        "model_path": str(Path(checkpoint_path).resolve()),
        "nn_output_dir": str(output_dir.resolve()),
        "mean_metrics": {
            column: float(metrics[column].mean())
            for column in ("RRMSET_mean", "RRMSEF_mean", "ACC_mean")
        },
        "evaluation_mode": "saved_checkpoint",
        "smoke_test": run_config["smoke_test"],
    }
    with (Path(save_dir) / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(metrics.to_string(index=False))
    print(f"[FLCT] Test outputs saved to: {output_dir}")
    return summary
