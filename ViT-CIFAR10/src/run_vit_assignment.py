#!/usr/bin/env python3
"""Run CIFAR-10 ViT assignment experiments.

This script is intentionally self-contained so the submitted code can be
re-run with one command. It trains a compact ViT, a CNN baseline, patch-size
variants, capacity variants, and a student twist model, then saves all metrics
and figures required by the assignment guideline.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import tarfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


@dataclass
class ModelConfig:
    name: str
    family: str
    patch_size: int | None = None
    embed_dim: int | None = None
    depth: int | None = None
    heads: int | None = None
    mlp_dim: int | None = None
    dropout: float = 0.1
    pool: str = "cls"
    use_positional_encoding: bool = True
    augmentation: str = "standard"
    lr: float = 3e-4


class Cutout:
    def __init__(self, size: int = 8) -> None:
        self.size = size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, h, w = x.shape
        cy = random.randint(0, h - 1)
        cx = random.randint(0, w - 1)
        y1 = max(0, cy - self.size // 2)
        y2 = min(h, cy + self.size // 2)
        x1 = max(0, cx - self.size // 2)
        x2 = min(w, cx + self.size // 2)
        x[:, y1:y2, x1:x2] = 0.0
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_chans: int, embed_dim: int) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TinyViT(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 96,
        depth: int = 3,
        heads: int = 4,
        mlp_dim: int = 192,
        num_classes: int = 10,
        dropout: float = 0.1,
        pool: str = "cls",
        use_positional_encoding: bool = True,
    ) -> None:
        super().__init__()
        self.pool = pool
        self.use_positional_encoding = use_positional_encoding
        self.patch_embed = PatchEmbedding(image_size, patch_size, 3, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.use_cls = pool == "cls"
        token_count = self.num_patches + (1 if self.use_cls else 0)
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None
        self.pos_embed = nn.Parameter(torch.zeros(1, token_count, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.use_cls:
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls, x), dim=1)
        if self.use_positional_encoding:
            x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x)
        if self.pool == "cls":
            x = x[:, 0]
        elif self.pool == "mean":
            x = x.mean(dim=1)
        else:
            raise ValueError(f"unknown pool: {self.pool}")
        return self.head(x)


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_cifar(data_dir: Path) -> None:
    cifar_dir = data_dir / "cifar-10-batches-py"
    if cifar_dir.exists():
        return
    tar_path = data_dir / "cifar-10-python.tar.gz"
    if not tar_path.exists():
        raise FileNotFoundError(
            f"{tar_path} is missing. Download it from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        )
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(data_dir)


def make_transforms(augmentation: str) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    if augmentation == "cutout":
        train_tf = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                Cutout(size=8),
            ]
        )
    elif augmentation == "standard":
        train_tf = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        train_tf = transforms.Compose([transforms.ToTensor(), normalize])
    eval_tf = transforms.Compose([transforms.ToTensor(), normalize])
    return train_tf, eval_tf


def balanced_indices(labels: list[int], per_class: int, offset_per_class: int = 0) -> list[int]:
    labels_np = np.array(labels)
    out: list[int] = []
    for cls in range(10):
        idx = np.where(labels_np == cls)[0]
        start = offset_per_class
        end = offset_per_class + per_class
        out.extend(idx[start:end].tolist())
    return out


def make_loaders(
    data_dir: Path,
    batch_size: int,
    train_per_class: int,
    val_per_class: int,
    test_per_class: int,
    augmentation: str,
) -> tuple[DataLoader, DataLoader, DataLoader, datasets.CIFAR10]:
    train_tf, eval_tf = make_transforms(augmentation)
    train_aug = datasets.CIFAR10(data_dir, train=True, download=False, transform=train_tf)
    train_eval = datasets.CIFAR10(data_dir, train=True, download=False, transform=eval_tf)
    test_eval = datasets.CIFAR10(data_dir, train=False, download=False, transform=eval_tf)
    raw_train = datasets.CIFAR10(data_dir, train=True, download=False, transform=None)

    train_idx = balanced_indices(train_aug.targets, train_per_class, offset_per_class=0)
    val_idx = balanced_indices(train_aug.targets, val_per_class, offset_per_class=train_per_class)
    test_idx = balanced_indices(test_eval.targets, test_per_class, offset_per_class=0)

    train_loader = DataLoader(
        Subset(train_aug, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        Subset(train_eval, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        Subset(test_eval, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader, test_loader, raw_train


def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.family == "cnn":
        return SmallCNN()
    if cfg.family == "vit":
        assert cfg.patch_size is not None
        assert cfg.embed_dim is not None
        assert cfg.depth is not None
        assert cfg.heads is not None
        assert cfg.mlp_dim is not None
        return TinyViT(
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            heads=cfg.heads,
            mlp_dim=cfg.mlp_dim,
            dropout=cfg.dropout,
            pool=cfg.pool,
            use_positional_encoding=cfg.use_positional_encoding,
        )
    raise ValueError(f"unknown family: {cfg.family}")


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    all_preds: list[int] = []
    all_targets: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            batch = y.size(0)
            total_loss += loss.item() * batch
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total += batch
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_targets.extend(y.cpu().tolist())
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(10)))
    class_acc = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    return {
        "loss": total_loss / total,
        "acc": total_correct / total,
        "preds": all_preds,
        "targets": all_targets,
        "cm": cm,
        "class_acc": class_acc,
    }


def train_model(
    cfg: ModelConfig,
    loaders: tuple[DataLoader, DataLoader, DataLoader],
    device: torch.device,
    epochs: int,
) -> tuple[pd.DataFrame, dict[str, object], nn.Module]:
    train_loader, val_loader, test_loader = loaders
    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    history: list[dict[str, float | int | str]] = []
    total_train_time = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        start = time.perf_counter()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            batch = y.size(0)
            epoch_loss += loss.item() * batch
            epoch_correct += (logits.argmax(dim=1) == y).sum().item()
            epoch_total += batch
        epoch_time = time.perf_counter() - start
        total_train_time += epoch_time
        val = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "model": cfg.name,
                "epoch": epoch,
                "train_loss": epoch_loss / epoch_total,
                "train_acc": epoch_correct / epoch_total,
                "val_loss": float(val["loss"]),
                "val_acc": float(val["acc"]),
                "epoch_time_sec": epoch_time,
            }
        )
        print(
            f"[{cfg.name}] epoch {epoch:02d}/{epochs} "
            f"train_acc={history[-1]['train_acc']:.3f} val_acc={history[-1]['val_acc']:.3f} "
            f"time={epoch_time:.1f}s",
            flush=True,
        )

    test = evaluate(model, test_loader, criterion, device)
    hist_df = pd.DataFrame(history)
    params = count_params(model)
    token_count = None
    num_patches = None
    if cfg.family == "vit" and cfg.patch_size is not None:
        num_patches = (32 // cfg.patch_size) ** 2
        token_count = num_patches + (1 if cfg.pool == "cls" else 0)
    final = {
        **asdict(cfg),
        "params": params,
        "num_patches": num_patches,
        "token_count": token_count,
        "train_time_sec": total_train_time,
        "avg_epoch_time_sec": total_train_time / epochs,
        "final_train_acc": float(hist_df.iloc[-1]["train_acc"]),
        "final_val_acc": float(hist_df.iloc[-1]["val_acc"]),
        "best_val_acc": float(hist_df["val_acc"].max()),
        "final_val_loss": float(hist_df.iloc[-1]["val_loss"]),
        "test_acc": float(test["acc"]),
        "test_loss": float(test["loss"]),
        "train_val_gap": float(hist_df.iloc[-1]["train_acc"] - hist_df.iloc[-1]["val_acc"]),
        "confusion_matrix": np.asarray(test["cm"]).tolist(),
        "class_acc": np.asarray(test["class_acc"]).tolist(),
    }
    return hist_df, final, model


def unnormalize(img: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.4914, 0.4822, 0.4465])[:, None, None]
    std = torch.tensor([0.2470, 0.2435, 0.2616])[:, None, None]
    x = (img.cpu() * std + mean).clamp(0, 1)
    return x.permute(1, 2, 0).numpy()


def save_dataset_samples(raw_train: datasets.CIFAR10, out: Path) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.2))
    shown = {c: False for c in range(10)}
    picked: list[int] = []
    for idx, (_, label) in enumerate(raw_train):
        if not shown[label]:
            picked.append(idx)
            shown[label] = True
        if len(picked) == 10:
            break
    for ax, idx in zip(axes.flat, picked):
        img, label = raw_train[idx]
        ax.imshow(img)
        ax.set_title(CLASSES[label], fontsize=10)
        ax.axis("off")
    fig.suptitle("Figure 1. CIFAR-10 class sample visualization", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def save_patch_visualization(raw_train: datasets.CIFAR10, out: Path) -> None:
    img, label = raw_train[0]
    panels: list[Image.Image] = []
    for patch in [4, 8, 16]:
        copy = img.copy().resize((256, 256), resample=Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(copy)
        step = 256 // (32 // patch)
        for x in range(0, 257, step):
            draw.line([(x, 0), (x, 256)], fill=(255, 50, 50), width=2)
        for y in range(0, 257, step):
            draw.line([(0, y), (256, y)], fill=(255, 50, 50), width=2)
        panels.append(copy)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, patch, panel in zip(axes, [4, 8, 16], panels):
        token_count = (32 // patch) ** 2
        ax.imshow(panel)
        ax.set_title(f"patch={patch}, tokens={token_count}")
        ax.axis("off")
    fig.suptitle(f"Figure 2. Patch split visualization on a '{CLASSES[label]}' image", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_learning_curve(history: pd.DataFrame, model_name: str, out: Path, title: str) -> None:
    data = history[history["model"] == model_name]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(data["epoch"], data["train_loss"], marker="o", label="train loss")
    axes[0].plot(data["epoch"], data["val_loss"], marker="o", label="val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(data["epoch"], data["train_acc"], marker="o", label="train acc")
    axes[1].plot(data["epoch"], data["val_acc"], marker="o", label="val acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle(title, fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_bar(df: pd.DataFrame, names: list[str], value_col: str, out: Path, title: str, ylabel: str) -> None:
    sub = df.set_index("name").loc[names].reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(sub["name"], sub[value_col], color=["#2f6fbb", "#db7c26", "#3f9b60", "#9467bd"][: len(sub)])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, weight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=15)
    for bar in bars:
        val = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_patch_comparison(df: pd.DataFrame, history: pd.DataFrame, out_a: Path, out_b: Path) -> None:
    names = ["vit_patch4_baseline", "vit_patch8", "vit_patch16"]
    sub = df.set_index("name").loc[names].reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    columns = [
        ("test_acc", "Test accuracy"),
        ("train_time_sec", "Training time (sec)"),
        ("num_patches", "Patch token count"),
    ]
    for ax, (col, label) in zip(axes, columns):
        ax.bar(sub["patch_size"].astype(str), sub[col], color="#3572a5")
        ax.set_xlabel("Patch size")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)
        for i, value in enumerate(sub[col]):
            text = f"{value:.3f}" if col == "test_acc" else f"{value:.0f}"
            ax.text(i, value, text, ha="center", va="bottom", fontsize=9)
    fig.suptitle("Figure 6. Patch size vs accuracy, training time, token count", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(out_a, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for name in names:
        data = history[history["model"] == name]
        patch = df[df["name"] == name].iloc[0]["patch_size"]
        ax.plot(data["epoch"], data["val_acc"], marker="o", label=f"patch {int(patch)}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title("Figure 6b. Patch size validation learning curves", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(out_b, dpi=180)
    plt.close(fig)


def plot_capacity(df: pd.DataFrame, history: pd.DataFrame, out: Path) -> None:
    names = ["vit_capacity_small", "vit_patch4_baseline", "vit_capacity_large"]
    sub = df.set_index("name").loc[names].reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    axes[0].scatter(sub["params"], sub["test_acc"], s=95, color="#375a7f")
    for _, row in sub.iterrows():
        axes[0].annotate(row["name"].replace("vit_", ""), (row["params"], row["test_acc"]), fontsize=8)
    axes[0].set_xlabel("Parameter count")
    axes[0].set_ylabel("Test accuracy")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(sub["train_time_sec"], sub["test_acc"], s=95, color="#c46a2b")
    for _, row in sub.iterrows():
        axes[1].annotate(row["name"].replace("vit_", ""), (row["train_time_sec"], row["test_acc"]), fontsize=8)
    axes[1].set_xlabel("Training time (sec)")
    axes[1].set_ylabel("Test accuracy")
    axes[1].grid(alpha=0.25)

    axes[2].bar(sub["name"].str.replace("vit_", "", regex=False), sub["train_val_gap"], color="#6b8e23")
    axes[2].set_ylabel("Train - validation accuracy")
    axes[2].tick_params(axis="x", rotation=15)
    axes[2].grid(axis="y", alpha=0.25)
    for i, value in enumerate(sub["train_val_gap"]):
        axes[2].text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Figure 7. Model capacity trade-off: params, time, and overfitting gap", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_confusion(df: pd.DataFrame, names: list[str], out: Path, title: str) -> None:
    fig, axes = plt.subplots(1, len(names), figsize=(6.4 * len(names), 5.3))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        row = df[df["name"] == name].iloc[0]
        cm = np.array(row["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(10), CLASSES, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(10), CLASSES, fontsize=8)
        for i in range(10):
            for j in range(10):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def collect_examples(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out: Path,
    title: str,
    max_correct: int = 5,
    max_wrong: int = 5,
) -> None:
    model.eval()
    correct: list[tuple[np.ndarray, int, int]] = []
    wrong: list[tuple[np.ndarray, int, int]] = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            preds = logits.argmax(dim=1).cpu()
            for img, target, pred in zip(x, y, preds):
                item = (unnormalize(img), int(target), int(pred))
                if target == pred and len(correct) < max_correct:
                    correct.append(item)
                elif target != pred and len(wrong) < max_wrong:
                    wrong.append(item)
            if len(correct) >= max_correct and len(wrong) >= max_wrong:
                break
    examples = correct + wrong
    labels = ["Correct"] * len(correct) + ["Wrong"] * len(wrong)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.2))
    for ax, item, tag in zip(axes.flat, examples, labels):
        img, target, pred = item
        ax.imshow(img)
        ax.set_title(f"{tag}\nT:{CLASSES[target]} / P:{CLASSES[pred]}", fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_twist(df: pd.DataFrame, history: pd.DataFrame, out: Path) -> None:
    names = ["vit_patch4_baseline", "vit_twist_mean_cutout"]
    sub = df.set_index("name").loc[names].reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].bar(["Baseline CLS", "Mean+Cutout"], sub["test_acc"], color=["#2f6fbb", "#c44e52"])
    axes[0].set_ylabel("Test accuracy")
    axes[0].grid(axis="y", alpha=0.25)
    for i, value in enumerate(sub["test_acc"]):
        axes[0].text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    for name, label in zip(names, ["Baseline CLS", "Mean+Cutout"]):
        data = history[history["model"] == name]
        axes[1].plot(data["epoch"], data["val_acc"], marker="o", label=label)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].bar(["Baseline CLS", "Mean+Cutout"], sub["train_val_gap"], color=["#2f6fbb", "#c44e52"])
    axes[2].set_ylabel("Train - validation gap")
    axes[2].grid(axis="y", alpha=0.25)
    for i, value in enumerate(sub["train_val_gap"]):
        axes[2].text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Figure 10. Student twist: mean pooling + Cutout regularization", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_classwise(df: pd.DataFrame, out: Path) -> None:
    names = ["cnn_baseline", "vit_patch4_baseline", "vit_twist_mean_cutout"]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(10)
    width = 0.25
    for k, name in enumerate(names):
        row = df[df["name"] == name].iloc[0]
        vals = np.array(row["class_acc"])
        ax.bar(x + (k - 1) * width, vals, width=width, label=name)
    ax.set_xticks(x, CLASSES, rotation=30, ha="right")
    ax.set_ylabel("Class-wise accuracy")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ax.set_title("Figure 11. Class-wise accuracy comparison", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def make_configs() -> list[ModelConfig]:
    return [
        ModelConfig(name="cnn_baseline", family="cnn", lr=1e-3),
        ModelConfig(name="vit_patch4_baseline", family="vit", patch_size=4, embed_dim=96, depth=3, heads=4, mlp_dim=192),
        ModelConfig(name="vit_patch8", family="vit", patch_size=8, embed_dim=96, depth=3, heads=4, mlp_dim=192),
        ModelConfig(name="vit_patch16", family="vit", patch_size=16, embed_dim=96, depth=3, heads=4, mlp_dim=192),
        ModelConfig(name="vit_capacity_small", family="vit", patch_size=4, embed_dim=64, depth=2, heads=2, mlp_dim=128),
        ModelConfig(name="vit_capacity_large", family="vit", patch_size=4, embed_dim=128, depth=4, heads=4, mlp_dim=256),
        ModelConfig(
            name="vit_twist_mean_cutout",
            family="vit",
            patch_size=4,
            embed_dim=96,
            depth=3,
            heads=4,
            mlp_dim=192,
            pool="mean",
            augmentation="cutout",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-per-class", type=int, default=500)
    parser.add_argument("--val-per-class", type=int, default=150)
    parser.add_argument("--test-per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", nargs="*", default=None, help="optional model names to run")
    args = parser.parse_args()

    root = project_root()
    data_dir = root / "data"
    fig_dir = root / "figures"
    run_dir = root / "runs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    ensure_cifar(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"device={device} torch={torch.__version__}")

    configs = make_configs()
    if args.only:
        names = set(args.only)
        configs = [cfg for cfg in configs if cfg.name in names]
        missing = names - {cfg.name for cfg in configs}
        if missing:
            raise ValueError(f"unknown configs: {sorted(missing)}")

    all_histories: list[pd.DataFrame] = []
    all_results: list[dict[str, object]] = []
    trained_models: dict[str, nn.Module] = {}

    raw_train_for_figures = None
    for cfg in configs:
        loaders = make_loaders(
            data_dir=data_dir,
            batch_size=args.batch_size,
            train_per_class=args.train_per_class,
            val_per_class=args.val_per_class,
            test_per_class=args.test_per_class,
            augmentation=cfg.augmentation,
        )
        train_loader, val_loader, test_loader, raw_train = loaders
        raw_train_for_figures = raw_train
        print(f"\n=== training {cfg.name} ===", flush=True)
        hist, final, model = train_model(cfg, (train_loader, val_loader, test_loader), device, args.epochs)
        all_histories.append(hist)
        all_results.append(final)
        if cfg.name in {"vit_patch4_baseline", "cnn_baseline", "vit_twist_mean_cutout"}:
            trained_models[cfg.name] = model.cpu()

    history_df = pd.concat(all_histories, ignore_index=True)
    results_df = pd.DataFrame(all_results)
    history_df.to_csv(run_dir / "history.csv", index=False)
    results_df.to_csv(run_dir / "metrics.csv", index=False)
    with (run_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "train_per_class": args.train_per_class,
                "val_per_class": args.val_per_class,
                "test_per_class": args.test_per_class,
                "seed": args.seed,
                "device": str(device),
                "torch": torch.__version__,
            },
            f,
            indent=2,
        )

    if raw_train_for_figures is not None:
        save_dataset_samples(raw_train_for_figures, fig_dir / "fig01_dataset_samples.png")
        save_patch_visualization(raw_train_for_figures, fig_dir / "fig02_patch_split.png")

    available = set(results_df["name"])
    if "vit_patch4_baseline" in available:
        plot_learning_curve(
            history_df,
            "vit_patch4_baseline",
            fig_dir / "fig03_vit_baseline_learning_curve.png",
            "Figure 3. ViT baseline learning curve",
        )
    if "cnn_baseline" in available:
        plot_learning_curve(
            history_df,
            "cnn_baseline",
            fig_dir / "fig04_cnn_baseline_learning_curve.png",
            "Figure 4. CNN baseline learning curve",
        )
    if {"cnn_baseline", "vit_patch4_baseline"}.issubset(available):
        plot_bar(
            results_df,
            ["cnn_baseline", "vit_patch4_baseline"],
            "test_acc",
            fig_dir / "fig05_cnn_vs_vit_accuracy.png",
            "Figure 5. CNN vs ViT test accuracy",
            "Test accuracy",
        )
    if {"vit_patch4_baseline", "vit_patch8", "vit_patch16"}.issubset(available):
        plot_patch_comparison(
            results_df,
            history_df,
            fig_dir / "fig06_patch_size_comparison.png",
            fig_dir / "fig06b_patch_learning_curves.png",
        )
    if {"vit_capacity_small", "vit_patch4_baseline", "vit_capacity_large"}.issubset(available):
        plot_capacity(results_df, history_df, fig_dir / "fig07_capacity_tradeoff.png")
    if {"cnn_baseline", "vit_patch4_baseline"}.issubset(available):
        plot_confusion(
            results_df,
            ["cnn_baseline", "vit_patch4_baseline"],
            fig_dir / "fig08_confusion_matrices.png",
            "Figure 8. CNN vs ViT confusion matrices",
        )
    if {"vit_patch4_baseline", "vit_twist_mean_cutout"}.issubset(available):
        plot_twist(results_df, history_df, fig_dir / "fig10_twist_mean_cutout.png")
        plot_confusion(
            results_df,
            ["vit_patch4_baseline", "vit_twist_mean_cutout"],
            fig_dir / "fig10b_twist_confusion_matrices.png",
            "Figure 10b. Baseline ViT vs twist confusion matrices",
        )
    if {"cnn_baseline", "vit_patch4_baseline", "vit_twist_mean_cutout"}.issubset(available):
        plot_classwise(results_df, fig_dir / "fig11_classwise_accuracy.png")

    # Recreate eval loader once for qualitative examples using the baseline model.
    if "vit_patch4_baseline" in trained_models:
        _, _, test_loader, _ = make_loaders(
            data_dir=data_dir,
            batch_size=args.batch_size,
            train_per_class=args.train_per_class,
            val_per_class=args.val_per_class,
            test_per_class=args.test_per_class,
            augmentation="standard",
        )
        collect_examples(
            trained_models["vit_patch4_baseline"],
            test_loader,
            torch.device("cpu"),
            fig_dir / "fig09_correct_wrong_examples.png",
            "Figure 9. ViT baseline correct and wrong prediction examples",
        )

    print("\nSaved:")
    print(f"- {run_dir / 'metrics.csv'}")
    print(f"- {run_dir / 'history.csv'}")
    print(f"- {fig_dir}")


if __name__ == "__main__":
    main()
