"""Shared utilities for UDL Assignment 2 (Group 91).

Imported identically by Part A and Part B notebooks. Cross-platform
(Windows/Linux/macOS): only pathlib paths, no OS-specific shell calls.
"""

from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid, save_image

SEED = 42


# --------------------------------------------------------------------------- #
# Reproducibility / device
# --------------------------------------------------------------------------- #

def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_env_banner(device: torch.device) -> None:
    print(f"Chosen device: {device}")
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        print(f"GPU name: {torch.cuda.get_device_name(idx)}")
        print(f"CUDA version: {torch.version.cuda}")
        total_mem = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        print(f"Total GPU memory: {total_mem:.2f} GB")
    elif device.type == "mps":
        print("Apple MPS backend in use (no CUDA memory stats available).")
    else:
        print("Running on CPU.")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def get_loaders(batch_size: int, data_dir: Optional[Path] = None):
    """Official CIFAR-10 train/test split, pixels in [0, 1], no augmentation."""
    data_dir = data_dir or Path(os.environ.get("CIFAR10_DIR", "./data"))
    data_dir.mkdir(parents=True, exist_ok=True)

    tfm = transforms.ToTensor()  # scales to [0, 1], no mean/std normalisation

    train_set = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=True, download=True, transform=tfm
    )
    test_set = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=False, download=True, transform=tfm
    )

    num_workers = 0 if os.name == "nt" else min(4, os.cpu_count() or 1)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )
    return train_loader, test_loader


def infinite_loader(loader: DataLoader):
    """Yields batches forever by re-iterating the DataLoader."""
    while True:
        for batch in loader:
            yield batch


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    """Per-image PSNR in dB for tensors in [0, 1] (MAX = 1), averaged over the batch."""
    x = x.clamp(0.0, 1.0)
    y = y.clamp(0.0, 1.0)
    mse = torch.mean((x - y) ** 2, dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10)
    psnr_per_image = 10.0 * torch.log10(1.0 / mse)
    return psnr_per_image.mean().item()


class FIDHelper:
    """Wraps torchmetrics FrechetInceptionDistance with a fixed protocol.

    feature=2048, normalize=True (expects float [0,1] input), the real-image
    statistics (10,000 CIFAR-10 test images) are cached once per notebook via
    reset_real_features=False, so repeated fake-side updates don't recompute them.
    """

    def __init__(self, device: torch.device, real_loader: Iterable, n_real: int = 10000, batch_size: int = 100):
        from torchmetrics.image.fid import FrechetInceptionDistance

        # torchmetrics' FID keeps some accumulators in float64, which the MPS backend does
        # not support; fall back to CPU for the metric only (model training stays on `device`).
        self.metric_device = torch.device("cpu") if device.type == "mps" else device
        self.device = device
        self.batch_size = batch_size
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True, reset_real_features=False).to(self.metric_device)

        seen = 0
        with torch.no_grad():
            for imgs, _ in real_loader:
                imgs = imgs.to(self.metric_device)
                self.fid.update(imgs, real=True)
                seen += imgs.shape[0]
                if seen >= n_real:
                    break
        self._real_ready = True

    def _update_fake(self, images: torch.Tensor) -> None:
        images = images.clamp(0.0, 1.0).to(self.metric_device)
        self.fid.update(images, real=False)

    def fid_of_images(self, images_iterable) -> float:
        """images_iterable: tensor (N,3,H,W) in [0,1] or an iterable of such batches."""
        if isinstance(images_iterable, torch.Tensor):
            n = images_iterable.shape[0]
            for i in range(0, n, self.batch_size):
                self._update_fake(images_iterable[i:i + self.batch_size])
        else:
            for batch in images_iterable:
                imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
                self._update_fake(imgs)
        value = self.fid.compute().item()
        self.fid.reset()  # reset_real_features=False keeps the cached real stats
        return value

    def fid_of_generator(self, sample_fn: Callable[[int], torch.Tensor], n: int = 10000) -> float:
        with torch.no_grad():
            produced = 0
            while produced < n:
                bs = min(self.batch_size, n - produced)
                imgs = sample_fn(bs)
                self._update_fake(imgs)
                produced += bs
        value = self.fid.compute().item()
        self.fid.reset()
        return value


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #

def save_grid(images: torch.Tensor, path: Path, nrow: int = 10) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = images.clamp(0.0, 1.0)
    save_image(images, str(path), nrow=nrow)
    try:
        import matplotlib.pyplot as plt

        grid = make_grid(images, nrow=nrow).permute(1, 2, 0).cpu().numpy()
        plt.figure(figsize=(8, 8))
        plt.imshow(grid)
        plt.axis("off")
        plt.title(path.stem)
        plt.show()
    except Exception as exc:  # pragma: no cover - display only
        print(f"(inline display skipped: {exc})")


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: Path):
    with open(Path(path), "r") as f:
        return json.load(f)


@contextmanager
def Timer():
    state = {"start": time.perf_counter(), "elapsed": None}
    try:
        yield state
    finally:
        state["elapsed"] = time.perf_counter() - state["start"]


def plot_curves(x, ys: dict, xlabel: str, ylabel: str, title: str, path: Optional[Path] = None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))
    for label, y in ys.items():
        plt.plot(x[: len(y)] if len(x) >= len(y) else list(range(len(y))), y, label=str(label), marker="o", markersize=3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.show()
