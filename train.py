"""FLCT optimization and selection of the lowest-validation-loss checkpoint."""

import math
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm


def get_rms(x, dim=-1):
    if x.dim() == 3:
        x = x.squeeze(1)
    return torch.sqrt(torch.mean(torch.square(x), dim=dim))


class RRMSETLoss(nn.Module):
    def forward(self, eeg_pred, eeg_true):
        if eeg_pred.ndim != 3 or eeg_pred.shape != eeg_true.shape:
            raise ValueError("Expected matching (batch, 1, time_points) tensors.")
        if eeg_pred.shape[1] != 1:
            raise ValueError("RRMSETLoss expects a single EEG channel.")
        residual = eeg_pred.squeeze(1) - eeg_true.squeeze(1)
        numerator = get_rms(residual, dim=1)
        denominator = torch.clamp(get_rms(eeg_true.squeeze(1), dim=1), min=1e-8)
        values = numerator / denominator
        return values.mean(), values.std(unbiased=False)


def make_loss_fn(settings):
    name = str(settings.get("loss", "mse")).strip().lower()
    if name == "rrmset":
        return RRMSETLoss()
    if name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unsupported loss: {name}; choose 'mse' or 'rrmset'.")


def model_output_to_target_shape(prediction):
    if prediction.ndim == 1:
        prediction = prediction.unsqueeze(0)
    if prediction.ndim == 2:
        prediction = prediction.unsqueeze(1)
    return prediction


def compute_loss(loss_fn, prediction, target):
    if isinstance(loss_fn, RRMSETLoss):
        return loss_fn(
            model_output_to_target_shape(prediction),
            model_output_to_target_shape(target),
        )[0]
    return loss_fn(prediction, target)


def sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_epoch(dataloader, model, loss_fn, optimizer, device):
    start = time.time()
    total_loss, sample_count = 0.0, 0
    model.train()
    for inputs, targets in tqdm(dataloader, desc="Training", leave=True):
        inputs, targets = inputs.to(device), targets.to(device)
        prediction = model_output_to_target_shape(model(inputs))
        loss = compute_loss(loss_fn, prediction, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss; inspect the input data.")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(inputs)
        sample_count += len(inputs)
    sync_if_cuda(device)
    if not sample_count:
        raise ValueError("Training dataloader is empty.")
    return total_loss / sample_count, time.time() - start


def validate(dataloader, model, loss_fn, device):
    start = time.time()
    total_loss, sample_count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            prediction = model_output_to_target_shape(model(inputs))
            loss = compute_loss(loss_fn, prediction, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite validation loss; inspect the input data.")
            total_loss += loss.item() * len(inputs)
            sample_count += len(inputs)
    sync_if_cuda(device)
    if not sample_count:
        raise ValueError("Validation dataloader is empty.")
    return total_loss / sample_count, time.time() - start


def fit(model, train_loader, val_loader, settings, save_dir, device):
    from torch.utils.tensorboard import SummaryWriter

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / "denoise_model.pth"
    loss_fn = make_loss_fn(settings)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])
    history, best_loss, best_epoch = [], math.inf, None
    start = time.time()

    with SummaryWriter(str(save_dir)) as writer:
        for epoch in range(1, settings["epochs"] + 1):
            print(f"\n[FLCT] Epoch {epoch}/{settings['epochs']}")
            train_loss, train_time = train_epoch(
                train_loader, model, loss_fn, optimizer, device
            )
            val_loss, val_time = validate(val_loader, model, loss_fn, device)
            epoch_time = train_time + val_time
            for tag, value in {
                "Loss/train": train_loss, "Loss/val": val_loss,
                "time/train": train_time, "time/val": val_time,
                "time/epoch": epoch_time,
            }.items():
                writer.add_scalar(tag, value, epoch - 1)
            history.append({
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                "train_time_sec": train_time, "val_time_sec": val_time,
                "epoch_time_sec": epoch_time,
            })
            pd.DataFrame(history).to_csv(save_dir / "training_log.csv", index=False)
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
                # Keep the original plain state_dict checkpoint format.
                torch.save(model.state_dict(), checkpoint_path)
                print(f"[FLCT] Best model saved, val_loss={best_loss:.8f}")
            print(f"[FLCT] train_loss={train_loss:.8f}, val_loss={val_loss:.8f}, "
                  f"epoch_time={epoch_time:.2f}s")

    if best_epoch is None:
        raise RuntimeError("No valid model checkpoint was saved.")
    return {
        "best_val_loss": best_loss,
        "best_epoch": best_epoch,
        "completed_epochs": len(history),
        "total_train_time_sec": time.time() - start,
        "model_path": str(checkpoint_path.resolve()),
        "training_log_path": str((save_dir / "training_log.csv").resolve()),
    }
