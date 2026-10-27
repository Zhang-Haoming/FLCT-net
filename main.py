"""FLCT training/testing entry point; see --help for runtime options."""

import argparse
import json
import platform
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

import config
from data import make_dataloaders, normalize_noise_type, prepare_datasets, set_seed
from flct import FLCT
from test import evaluate
from train import fit


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train or test FLCT EEG denoising.")
    parser.add_argument("--mode", choices=("train", "test"), default="train")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    noise = parser.add_mutually_exclusive_group()
    noise.add_argument("--noise-types", nargs="+", help="eog emg ecg eog_emg eog_ecg emg_ecg all")
    noise.add_argument("--all-noise-types", action="store_true", help="Run all seven conditions.")
    parser.add_argument("--checkpoint", type=Path, help="Test one existing plain state_dict checkpoint.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--epochs", type=int, help="Override config.py for this run.")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--loss", choices=("mse", "rrmset"))
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="One epoch on a small real-data subset; results go to _smoke_test/.",
    )
    args = parser.parse_args(argv)
    if args.checkpoint and args.mode != "test":
        parser.error("--checkpoint requires --mode test.")
    if args.mode == "test" and any(value is not None
                                   for value in (args.epochs, args.lr, args.loss, args.seed)):
        parser.error("Test mode restores training settings from run_config.json.")
    return args


def _device(value):
    selected = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if value == "auto" else torch.device(value)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select --device cpu.")
    return selected


def _default_run(noise_type, args):
    run = {
        "model": "flct", "noise_type": noise_type,
        "model_config": deepcopy(config.MODEL_CONFIG),
        "train_config": deepcopy(config.TRAIN_CONFIG),
        "data_config": deepcopy(config.DATA_CONFIG),
        "evaluation_config": deepcopy(config.EVALUATION_CONFIG),
        "smoke_test": args.smoke_test,
    }
    if args.smoke_test:
        run["train_config"].update(epochs=1, batch_size=4)
    for argument, key in (("epochs", "epochs"), ("batch_size", "batch_size"),
                          ("lr", "lr"), ("loss", "loss")):
        value = getattr(args, argument)
        if value is not None:
            run["train_config"][key] = value
    if args.seed is not None:
        run["data_config"]["seed"] = args.seed
    return run


def _restore_run(checkpoint_path, run, args):
    path = checkpoint_path.parent / "run_config.json"
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            saved = json.load(handle)
        saved_noise = normalize_noise_type(saved.get("noise_type", run["noise_type"]))
        if saved_noise != run["noise_type"]:
            raise ValueError(f"Checkpoint condition is {saved_noise}, not {run['noise_type']}.")
        if "model_config" in saved:
            for key in ("model_config", "train_config", "data_config", "evaluation_config"):
                run[key].update(saved[key])
            run["smoke_test"] = saved.get("smoke_test", False)
        else:
            # Existing checkpoints from run_comparison.py use a flat JSON config.
            for key in ("train_config", "data_config", "evaluation_config"):
                run[key].update({name: saved[name] for name in run[key] if name in saved})
            run["smoke_test"] = False
        if args.smoke_test and not run["smoke_test"]:
            raise ValueError("A full-data checkpoint must be evaluated on its full test set.")
    else:
        print("No run_config.json next to the checkpoint; using config.py settings.")
    if args.batch_size is not None:
        run["train_config"]["batch_size"] = args.batch_size
    return run


def _validate_run(run):
    training, data = run["train_config"], run["data_config"]
    for name in ("epochs", "batch_size"):
        if training[name] < 1:
            raise ValueError(f"{name} must be positive.")
    if not np.isfinite(training["lr"]) or training["lr"] <= 0:
        raise ValueError("Learning rate must be positive and finite.")
    training["loss"] = str(training.get("loss", "mse")).strip().lower()
    if training["loss"] not in ("mse", "rrmset"):
        raise ValueError("Loss must be mse or rrmset.")
    if not 0 < data["train_per"] < 1:
        raise ValueError("train_per must lie between 0 and 1.")
    if data["train_combin_num"] < 1 or data["val_combin_num"] < 1:
        raise ValueError("Data pairing multipliers must be positive.")
    if len(data["test_db"]) != 3 or data["test_db"][2] < 1:
        raise ValueError("test_db must be [minimum_SNR, maximum_SNR, count].")


def main(argv=None):
    args = parse_args(argv)
    device = _device(args.device)
    noise_types = (list(config.DATA_NOISE_TYPES) if args.all_noise_types
                   else args.noise_types or config.DEFAULT_NOISE_TYPES)
    if args.checkpoint and not args.noise_types and not args.all_noise_types:
        saved_path = args.checkpoint.parent / "run_config.json"
        if saved_path.is_file():
            with saved_path.open(encoding="utf-8") as handle:
                noise_types = [json.load(handle)["noise_type"]]
    noise_types = list(dict.fromkeys(normalize_noise_type(name) for name in noise_types))
    if args.checkpoint and len(noise_types) != 1:
        raise ValueError("--checkpoint requires exactly one noise condition.")
    print(f"Device: {device}\nData directory: {args.data_dir}")

    summaries = []
    for noise_type in noise_types:
        run = _default_run(noise_type, args)
        output_root = args.output_dir / "_smoke_test" if args.smoke_test else args.output_dir
        save_dir = output_root / f"flct_{noise_type}"
        checkpoint = args.checkpoint or save_dir / "denoise_model.pth"
        if args.mode == "test":
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
            run = _restore_run(checkpoint, run, args)
            # An explicit smoke checkpoint keeps its exports out of normal results.
            if run["smoke_test"] and not args.smoke_test:
                save_dir = args.output_dir / "_smoke_test" / f"flct_{noise_type}"
        _validate_run(run)
        print(f"\nNoise condition: {noise_type}; smoke_test={run['smoke_test']}")
        datasets = prepare_datasets(
            args.data_dir, noise_type, run["data_config"], run["model_config"],
            smoke_test=run["smoke_test"],
        )
        # Reset after data preparation, matching the original training runner.
        set_seed(run["data_config"]["seed"])
        model = FLCT(**run["model_config"]).to(device)
        train_loader, val_loader, test_loader = make_dataloaders(
            datasets, run["train_config"]["batch_size"]
        )
        run.update({
            "data_directory": str(args.data_dir.resolve()),
            "output_directory": str(save_dir.resolve()),
            "device": str(device),
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "numpy": np.__version__},
            "snr_definition": "20 * log10(RMS(signal) / RMS(total_noise))",
            "mixed_noise_scaling": "equal_component_SNR_then_rescale_summed_artifact",
            "epoch_loss_aggregation": "sample_weighted_mean",
        })
        save_dir.mkdir(parents=True, exist_ok=True)
        training_summary = None
        if args.mode == "train":
            with (save_dir / "run_config.json").open("w", encoding="utf-8") as handle:
                json.dump(run, handle, indent=2)
            training_summary = fit(
                model, train_loader, val_loader, run["train_config"], save_dir, device
            )
        summary = evaluate(
            model, test_loader, device, checkpoint, save_dir, run, training_summary
        )
        summaries.append(summary)
        del model, datasets, train_loader, val_loader, test_loader
    return summaries


if __name__ == "__main__":
    main()
