"""
PyTorch Reference Implementation - Training Pipeline with Pseudo-Labeling

This module implements the complete training pipeline for semi-supervised learning
on the PlantVillage dataset, matching the Rust/Burn implementation for fair comparison.

Author: Warre Snaet
"""

import json
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import (
    PlantClassifier,
    PlantClassifierLite,
    count_parameters,
    get_model_size_mb,
)
from PIL import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

# ==============================================================================
# Configuration Classes
# ==============================================================================


@dataclass
class TrainingConfig:
    """Training configuration matching Rust implementation."""

    data_dir: str = "data/plantvillage"
    output_dir: str = "output/pytorch"
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    labeled_ratio: float = 0.1  # Only 10% labeled data
    image_size: int = 128  # Match Rust implementation
    num_workers: int = 4
    seed: int = 42
    use_lite_model: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Learning rate scheduler
    scheduler_type: str = "cosine"  # "cosine" or "step"
    step_size: int = 10
    gamma: float = 0.1


@dataclass
class PseudoLabelConfig:
    """Pseudo-labeling configuration for semi-supervised learning."""

    confidence_threshold: float = 0.9
    max_per_class: Optional[int] = 500
    retrain_threshold: int = 200

    # Curriculum learning settings
    curriculum_learning: bool = False
    curriculum_initial_threshold: float = 0.95
    curriculum_final_threshold: float = 0.8
    curriculum_epochs: int = 20

    def get_threshold(self, current_epoch: int) -> float:
        """Get effective threshold based on curriculum learning."""
        if not self.curriculum_learning:
            return self.confidence_threshold

        if current_epoch >= self.curriculum_epochs:
            return self.curriculum_final_threshold

        # Linear interpolation
        progress = current_epoch / self.curriculum_epochs
        threshold = self.curriculum_initial_threshold - progress * (
            self.curriculum_initial_threshold - self.curriculum_final_threshold
        )
        return threshold


@dataclass
class PseudoLabelStats:
    """Statistics for pseudo-labeling quality tracking."""

    total_processed: int = 0
    total_accepted: int = 0
    rejected_low_confidence: int = 0
    rejected_class_limit: int = 0
    correct_predictions: int = 0
    incorrect_predictions: int = 0

    def acceptance_rate(self) -> float:
        if self.total_processed == 0:
            return 0.0
        return self.total_accepted / self.total_processed

    def accuracy(self) -> float:
        total = self.correct_predictions + self.incorrect_predictions
        if total == 0:
            return 0.0
        return self.correct_predictions / total

    def __str__(self) -> str:
        return (
            f"PseudoLabelStats:\n"
            f"  Total processed: {self.total_processed}\n"
            f"  Total accepted: {self.total_accepted}\n"
            f"  Rejected (low confidence): {self.rejected_low_confidence}\n"
            f"  Rejected (class limit): {self.rejected_class_limit}\n"
            f"  Correct predictions: {self.correct_predictions}\n"
            f"  Incorrect predictions: {self.incorrect_predictions}\n"
            f"  Acceptance rate: {self.acceptance_rate():.2%}\n"
            f"  Accuracy: {self.accuracy():.2%}"
        )


@dataclass
class PseudoLabeledSample:
    """A pseudo-labeled sample."""

    image_path: str
    predicted_label: int
    confidence: float
    ground_truth: int
    is_correct: bool
    assigned_epoch: int = 0


# ==============================================================================
# Dataset Classes
# ==============================================================================


class PlantVillageDataset(Dataset):
    """PlantVillage dataset loader."""

    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        image_size: int = 128,
    ):
        self.root_dir = Path(root_dir)
        self.image_size = image_size

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transform

        # Find all classes (subdirectories)
        self.classes = sorted(
            [
                d.name
                for d in self.root_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]
        )
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Collect all samples
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            class_idx = self.class_to_idx[class_name]

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.samples.append((str(img_path), class_idx))

        print(f"Loaded {len(self.samples)} images from {len(self.classes)} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image on error
            image = torch.zeros(3, self.image_size, self.image_size)

        return image, label, img_path


class PseudoLabeledDataset(Dataset):
    """Dataset that combines labeled and pseudo-labeled samples."""

    def __init__(
        self,
        labeled_samples: list[tuple[str, int]],
        pseudo_samples: list[PseudoLabeledSample],
        transform: transforms.Compose,
        image_size: int = 128,
    ):
        self.labeled_samples = labeled_samples
        self.pseudo_samples = pseudo_samples
        self.transform = transform
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.labeled_samples) + len(self.pseudo_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        if idx < len(self.labeled_samples):
            img_path, label = self.labeled_samples[idx]
        else:
            pseudo_idx = idx - len(self.labeled_samples)
            pseudo = self.pseudo_samples[pseudo_idx]
            img_path, label = pseudo.image_path, pseudo.predicted_label

        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = torch.zeros(3, self.image_size, self.image_size)

        return image, label


# ==============================================================================
# Pseudo-Labeler
# ==============================================================================


class PseudoLabeler:
    """Pseudo-labeling for semi-supervised learning."""

    def __init__(self, config: PseudoLabelConfig):
        self.config = config
        self.pseudo_labels: list[PseudoLabeledSample] = []
        self.class_counts: dict[int, int] = {}
        self.stats = PseudoLabelStats()
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def current_threshold(self) -> float:
        return self.config.get_threshold(self.current_epoch)

    def process_predictions(
        self,
        model: nn.Module,
        unlabeled_loader: DataLoader,
        device: torch.device,
    ) -> list[PseudoLabeledSample]:
        """
        Run inference on unlabeled data and generate pseudo-labels.
        """
        model.eval()
        threshold = self.current_threshold()
        new_pseudo_labels = []

        with torch.no_grad():
            for images, labels, paths in tqdm(
                unlabeled_loader, desc="Generating pseudo-labels"
            ):
                images = images.to(device)
                labels = labels.numpy()  # Ground truth for evaluation

                # Forward pass
                logits = model(images)
                probs = F.softmax(logits, dim=1)
                confidences, predictions = probs.max(dim=1)

                confidences = confidences.cpu().numpy()
                predictions = predictions.cpu().numpy()

                # Process each prediction
                for i in range(len(images)):
                    self.stats.total_processed += 1

                    conf = confidences[i]
                    pred = predictions[i]
                    gt = labels[i]
                    path = paths[i]

                    # Check confidence threshold
                    if conf < threshold:
                        self.stats.rejected_low_confidence += 1
                        continue

                    # Check class limit
                    if self.config.max_per_class is not None:
                        current_count = self.class_counts.get(pred, 0)
                        if current_count >= self.config.max_per_class:
                            self.stats.rejected_class_limit += 1
                            continue

                    # Create pseudo-label
                    is_correct = pred == gt
                    pseudo = PseudoLabeledSample(
                        image_path=path,
                        predicted_label=int(pred),
                        confidence=float(conf),
                        ground_truth=int(gt),
                        is_correct=is_correct,
                        assigned_epoch=self.current_epoch,
                    )

                    # Update statistics
                    self.stats.total_accepted += 1
                    if is_correct:
                        self.stats.correct_predictions += 1
                    else:
                        self.stats.incorrect_predictions += 1

                    # Update class counts
                    self.class_counts[pred] = self.class_counts.get(pred, 0) + 1

                    new_pseudo_labels.append(pseudo)
                    self.pseudo_labels.append(pseudo)

        print(
            f"Generated {len(new_pseudo_labels)} new pseudo-labels "
            f"(threshold: {threshold:.2f}, accuracy: {self.stats.accuracy():.2%})"
        )

        return new_pseudo_labels

    def should_retrain(self) -> bool:
        return len(self.pseudo_labels) >= self.config.retrain_threshold

    def get_pseudo_labels(self) -> list[PseudoLabeledSample]:
        return self.pseudo_labels.copy()

    def clear(self):
        self.pseudo_labels.clear()
        self.class_counts.clear()
        self.stats = PseudoLabelStats()


# ==============================================================================
# Trainer
# ==============================================================================


class SemiSupervisedTrainer:
    """
    Semi-supervised trainer with pseudo-labeling.

    Implements the same training strategy as the Rust/Burn implementation.
    """

    def __init__(
        self,
        training_config: TrainingConfig,
        pseudo_config: PseudoLabelConfig,
    ):
        self.config = training_config
        self.pseudo_config = pseudo_config
        self.device = torch.device(training_config.device)

        # Set random seeds for reproducibility
        self._set_seeds(training_config.seed)

        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.pseudo_labeler = PseudoLabeler(pseudo_config)

        # Metrics tracking
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_acc": [],
            "pseudo_label_count": [],
            "pseudo_label_accuracy": [],
            "epoch_times": [],
        }

        # Create output directory
        os.makedirs(training_config.output_dir, exist_ok=True)

    def _set_seeds(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _create_model(self, num_classes: int) -> nn.Module:
        if self.config.use_lite_model:
            model = PlantClassifierLite(num_classes=num_classes)
        else:
            model = PlantClassifier(num_classes=num_classes)

        model = model.to(self.device)
        print(
            f"Created model with {count_parameters(model):,} parameters "
            f"({get_model_size_mb(model):.2f} MB)"
        )

        return model

    def _create_optimizer(self, model: nn.Module) -> Adam:
        return Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _create_scheduler(self, optimizer: Adam, epochs: int):
        if self.config.scheduler_type == "cosine":
            return CosineAnnealingLR(optimizer, T_max=epochs)
        else:
            return StepLR(
                optimizer,
                step_size=self.config.step_size,
                gamma=self.config.gamma,
            )

    def _create_transforms(self, training: bool = True) -> transforms.Compose:
        if training:
            return transforms.Compose(
                [
                    transforms.Resize((self.config.image_size, self.config.image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize((self.config.image_size, self.config.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def _split_dataset(self, dataset: PlantVillageDataset) -> tuple[list, list, list]:
        """
        Split dataset into labeled, unlabeled, and validation sets.

        Returns:
            (labeled_samples, unlabeled_samples, val_samples)
        """
        # Shuffle samples
        samples = dataset.samples.copy()
        random.shuffle(samples)

        n_total = len(samples)
        n_val = int(n_total * 0.1)  # 10% validation
        n_labeled = int((n_total - n_val) * self.config.labeled_ratio)

        val_samples = samples[:n_val]
        remaining = samples[n_val:]
        labeled_samples = remaining[:n_labeled]
        unlabeled_samples = remaining[n_labeled:]

        print(f"Dataset split:")
        print(f"  Labeled:   {len(labeled_samples)}")
        print(f"  Unlabeled: {len(unlabeled_samples)}")
        print(f"  Validation: {len(val_samples)}")

        return labeled_samples, unlabeled_samples, val_samples

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: Adam,
        epoch: int,
    ) -> tuple[float, float]:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for images, labels in progress:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            _, predicted = logits.max(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            progress.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{100 * correct / total:.2f}%"}
            )

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def _evaluate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
    ) -> float:
        """Evaluate model on validation set."""
        model.eval()
        correct = 0
        total = 0

        for images, labels, _ in val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = model(images)
            _, predicted = logits.max(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        return 100 * correct / total

    def train_supervised_only(self) -> nn.Module:
        """
        Train with only labeled data (baseline for comparison).
        """
        print("=" * 60)
        print("SUPERVISED TRAINING (Baseline)")
        print("=" * 60)

        # Load dataset
        dataset = PlantVillageDataset(
            self.config.data_dir,
            transform=self._create_transforms(training=False),
            image_size=self.config.image_size,
        )

        if len(dataset) == 0:
            print("ERROR: No images found in dataset!")
            print(f"Please download PlantVillage dataset to: {self.config.data_dir}")
            return None

        # Split dataset
        labeled_samples, _, val_samples = self._split_dataset(dataset)

        # Create data loaders
        train_transform = self._create_transforms(training=True)
        val_transform = self._create_transforms(training=False)

        class SimpleDataset(Dataset):
            def __init__(self, samples, transform, image_size):
                self.samples = samples
                self.transform = transform
                self.image_size = image_size

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                path, label = self.samples[idx]
                try:
                    image = Image.open(path).convert("RGB")
                    image = self.transform(image)
                except:
                    image = torch.zeros(3, self.image_size, self.image_size)
                return image, label

        train_dataset = SimpleDataset(
            labeled_samples, train_transform, self.config.image_size
        )
        val_dataset = PlantVillageDataset(
            self.config.data_dir,
            transform=val_transform,
            image_size=self.config.image_size,
        )
        # Use only validation samples
        val_indices = [i for i, s in enumerate(val_dataset.samples) if s in val_samples]
        val_subset = Subset(val_dataset, val_indices[: len(val_samples)])

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        # Create model and optimizer
        num_classes = len(dataset.classes)
        self.model = self._create_model(num_classes)
        self.optimizer = self._create_optimizer(self.model)
        self.scheduler = self._create_scheduler(self.optimizer, self.config.epochs)

        # Training loop
        best_val_acc = 0.0

        for epoch in range(self.config.epochs):
            start_time = time.time()

            # Train
            train_loss, train_acc = self._train_epoch(
                self.model, train_loader, self.optimizer, epoch
            )

            # Evaluate
            val_acc = self._evaluate(self.model, val_loader)

            # Step scheduler
            self.scheduler.step()

            epoch_time = time.time() - start_time

            # Track history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["epoch_times"].append(epoch_time)

            # Check for best model
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                self._save_checkpoint("best_model.pth")

            print(
                f"Epoch {epoch + 1}/{self.config.epochs}: "
                f"Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                f"Val Acc={val_acc:.2f}%{' ★' if is_best else ''} "
                f"({epoch_time:.1f}s)"
            )

        # Save final model
        self._save_checkpoint("final_model.pth")
        self._save_history("supervised")

        print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")

        return self.model

    def train_semi_supervised(self) -> nn.Module:
        """
        Train with semi-supervised learning using pseudo-labeling.
        """
        print("=" * 60)
        print("SEMI-SUPERVISED TRAINING (Pseudo-Labeling)")
        print("=" * 60)

        # Load dataset
        dataset = PlantVillageDataset(
            self.config.data_dir,
            transform=self._create_transforms(training=False),
            image_size=self.config.image_size,
        )

        if len(dataset) == 0:
            print("ERROR: No images found in dataset!")
            return None

        # Split dataset
        labeled_samples, unlabeled_samples, val_samples = self._split_dataset(dataset)

        # Create validation loader
        val_transform = self._create_transforms(training=False)
        val_dataset = PlantVillageDataset(
            self.config.data_dir,
            transform=val_transform,
            image_size=self.config.image_size,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        # Create unlabeled loader (with paths for pseudo-labeling)
        unlabeled_dataset = PlantVillageDataset(
            self.config.data_dir,
            transform=val_transform,
            image_size=self.config.image_size,
        )
        unlabeled_dataset.samples = unlabeled_samples
        unlabeled_loader = DataLoader(
            unlabeled_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        # Create model and optimizer
        num_classes = len(dataset.classes)
        self.model = self._create_model(num_classes)
        self.optimizer = self._create_optimizer(self.model)
        self.scheduler = self._create_scheduler(self.optimizer, self.config.epochs)

        # Initial training on labeled data
        train_transform = self._create_transforms(training=True)

        best_val_acc = 0.0

        for epoch in range(self.config.epochs):
            start_time = time.time()
            self.pseudo_labeler.set_epoch(epoch)

            # Combine labeled + pseudo-labeled data
            current_pseudo_labels = self.pseudo_labeler.get_pseudo_labels()

            combined_dataset = PseudoLabeledDataset(
                labeled_samples=labeled_samples,
                pseudo_samples=current_pseudo_labels,
                transform=train_transform,
                image_size=self.config.image_size,
            )

            train_loader = DataLoader(
                combined_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=True,
            )

            # Train epoch
            train_loss, train_acc = self._train_epoch(
                self.model, train_loader, self.optimizer, epoch
            )

            # Generate new pseudo-labels every 5 epochs
            if (epoch + 1) % 5 == 0 and epoch < self.config.epochs - 1:
                print(f"\n>>> Generating pseudo-labels (epoch {epoch + 1})...")
                self.pseudo_labeler.process_predictions(
                    self.model, unlabeled_loader, self.device
                )

            # Evaluate
            val_acc = self._evaluate(self.model, val_loader)

            # Step scheduler
            self.scheduler.step()

            epoch_time = time.time() - start_time

            # Track history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["pseudo_label_count"].append(len(current_pseudo_labels))
            self.history["pseudo_label_accuracy"].append(
                self.pseudo_labeler.stats.accuracy()
            )
            self.history["epoch_times"].append(epoch_time)

            # Check for best model
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                self._save_checkpoint("best_model_ssl.pth")

            print(
                f"Epoch {epoch + 1}/{self.config.epochs}: "
                f"Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                f"Val Acc={val_acc:.2f}%{' ★' if is_best else ''}, "
                f"Pseudo-labels={len(current_pseudo_labels)} "
                f"({epoch_time:.1f}s)"
            )

        # Final statistics
        print("\n" + "=" * 60)
        print("PSEUDO-LABELING STATISTICS")
        print("=" * 60)
        print(self.pseudo_labeler.stats)

        # Save final model and history
        self._save_checkpoint("final_model_ssl.pth")
        self._save_history("semi_supervised")

        print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")

        return self.model

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = Path(self.config.output_dir) / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": vars(self.config),
            },
            path,
        )
        print(f"  Saved checkpoint: {path}")

    def _save_history(self, prefix: str):
        """Save training history to JSON."""
        path = Path(self.config.output_dir) / f"{prefix}_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"  Saved history: {path}")


# ==============================================================================
# Benchmarking
# ==============================================================================


def benchmark_inference(
    model: nn.Module,
    device: torch.device,
    image_size: int = 128,
    num_iterations: int = 100,
    warmup: int = 10,
    batch_size: int = 1,
) -> dict:
    """
    Benchmark inference latency.

    Returns dict with timing statistics.
    """
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, image_size, image_size).to(device)

    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # Synchronize if using CUDA
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({num_iterations} iterations)...")
    times = []

    with torch.no_grad():
        for _ in tqdm(range(num_iterations)):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(dummy_input)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

    times = np.array(times)

    results = {
        "device": str(device),
        "batch_size": batch_size,
        "image_size": image_size,
        "num_iterations": num_iterations,
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "throughput_fps": float(batch_size / (np.mean(times) / 1000)),
    }

    print("\nBenchmark Results:")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Mean latency: {results['mean_ms']:.2f} ± {results['std_ms']:.2f} ms")
    print(
        f"  P50/P95/P99: {results['p50_ms']:.2f}/{results['p95_ms']:.2f}/{results['p99_ms']:.2f} ms"
    )
    print(f"  Throughput: {results['throughput_fps']:.1f} FPS")

    return results


# ==============================================================================
# Main
# ==============================================================================


def main():
    """Main entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch PlantVillage Training")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/plantvillage",
        help="Path to PlantVillage dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/pytorch",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--labeled-ratio",
        type=float,
        default=0.1,
        help="Ratio of labeled data (0.0-1.0)",
    )
    parser.add_argument("--image-size", type=int, default=128, help="Image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["supervised", "semi", "both", "benchmark"],
        default="both",
        help="Training mode",
    )
    parser.add_argument("--lite", action="store_true", help="Use lightweight model")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.9,
        help="Confidence threshold for pseudo-labeling",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Use curriculum learning for pseudo-labeling",
    )

    args = parser.parse_args()

    # Create configs
    training_config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        labeled_ratio=args.labeled_ratio,
        image_size=args.image_size,
        seed=args.seed,
        use_lite_model=args.lite,
    )

    pseudo_config = PseudoLabelConfig(
        confidence_threshold=args.confidence_threshold,
        curriculum_learning=args.curriculum,
    )

    # Print configuration
    print("=" * 60)
    print("PlantVillage Semi-Supervised Learning (PyTorch Reference)")
    print("=" * 60)
    print(f"Data directory: {training_config.data_dir}")
    print(f"Output directory: {training_config.output_dir}")
    print(f"Device: {training_config.device}")
    print(f"Epochs: {training_config.epochs}")
    print(f"Batch size: {training_config.batch_size}")
    print(f"Learning rate: {training_config.learning_rate}")
    print(f"Labeled ratio: {training_config.labeled_ratio:.0%}")
    print(f"Model: {'Lite' if training_config.use_lite_model else 'Full'}")
    print(f"Mode: {args.mode}")
    print()

    # Create trainer
    trainer = SemiSupervisedTrainer(training_config, pseudo_config)

    if args.mode == "benchmark":
        # Just run benchmark
        model = trainer._create_model(38)  # PlantVillage has 38 classes
        device = torch.device(training_config.device)
        benchmark_inference(model, device, training_config.image_size)
        return

    if args.mode in ["supervised", "both"]:
        print("\n" + "=" * 60)
        trainer.train_supervised_only()

        # Reset for next training
        trainer.history = {k: [] for k in trainer.history}

    if args.mode in ["semi", "both"]:
        print("\n" + "=" * 60)
        trainer.train_semi_supervised()

    # Final benchmark
    if trainer.model is not None:
        print("\n" + "=" * 60)
        print("INFERENCE BENCHMARK")
        print("=" * 60)
        device = torch.device(training_config.device)
        results = benchmark_inference(trainer.model, device, training_config.image_size)

        # Save benchmark results
        with open(Path(args.output_dir) / "benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
