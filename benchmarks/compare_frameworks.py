#!/usr/bin/env python3
"""
Burn vs PyTorch Benchmark Comparison

This script compares the performance of the Rust/Burn and Python/PyTorch
implementations for the PlantVillage semi-supervised learning task.

Metrics compared:
- Training time (wall-clock)
- Inference latency (GPU and CPU)
- Model size
- Memory usage
- Final accuracy

Author: Warre Snaet
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "pytorch_reference"))


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    framework: str
    device: str
    batch_size: int
    image_size: int

    # Latency metrics (ms)
    mean_latency_ms: float = 0.0
    std_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Throughput
    throughput_fps: float = 0.0

    # Training metrics
    training_time_s: float = 0.0
    epochs: int = 0
    final_train_acc: float = 0.0
    final_val_acc: float = 0.0

    # Model metrics
    model_size_mb: float = 0.0
    num_parameters: int = 0

    # Memory metrics
    peak_memory_mb: float = 0.0

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""


@dataclass
class ComparisonReport:
    """Comparison report between frameworks."""

    burn_result: Optional[BenchmarkResult] = None
    pytorch_result: Optional[BenchmarkResult] = None

    def latency_speedup(self) -> float:
        """Calculate Burn latency speedup over PyTorch."""
        if self.burn_result and self.pytorch_result:
            if self.burn_result.mean_latency_ms > 0:
                return (
                    self.pytorch_result.mean_latency_ms
                    / self.burn_result.mean_latency_ms
                )
        return 0.0

    def throughput_improvement(self) -> float:
        """Calculate Burn throughput improvement over PyTorch."""
        if self.burn_result and self.pytorch_result:
            if self.pytorch_result.throughput_fps > 0:
                return (
                    self.burn_result.throughput_fps / self.pytorch_result.throughput_fps
                )
        return 0.0

    def training_speedup(self) -> float:
        """Calculate Burn training speedup over PyTorch."""
        if self.burn_result and self.pytorch_result:
            if self.burn_result.training_time_s > 0:
                return (
                    self.pytorch_result.training_time_s
                    / self.burn_result.training_time_s
                )
        return 0.0

    def size_reduction(self) -> float:
        """Calculate model size reduction (Burn vs PyTorch)."""
        if self.burn_result and self.pytorch_result:
            if self.pytorch_result.model_size_mb > 0:
                return 1 - (
                    self.burn_result.model_size_mb / self.pytorch_result.model_size_mb
                )
        return 0.0


def run_pytorch_benchmark(
    data_dir: str,
    output_dir: str,
    epochs: int = 10,
    batch_size: int = 32,
    image_size: int = 128,
    device: str = "cuda",
) -> BenchmarkResult:
    """Run PyTorch benchmark."""
    print("\n" + "=" * 60)
    print("Running PyTorch Benchmark")
    print("=" * 60)

    result = BenchmarkResult(
        framework="PyTorch",
        device=device,
        batch_size=batch_size,
        image_size=image_size,
    )

    try:
        import torch
        from model import PlantClassifier, count_parameters, get_model_size_mb
        from trainer import (
            PseudoLabelConfig,
            SemiSupervisedTrainer,
            TrainingConfig,
            benchmark_inference,
        )

        # Model metrics
        model = PlantClassifier(num_classes=38)
        result.num_parameters = count_parameters(model)
        result.model_size_mb = get_model_size_mb(model)

        print(f"Model parameters: {result.num_parameters:,}")
        print(f"Model size: {result.model_size_mb:.2f} MB")

        # Training benchmark
        if Path(data_dir).exists() and any(Path(data_dir).iterdir()):
            print("\nRunning training benchmark...")
            config = TrainingConfig(
                data_dir=data_dir,
                output_dir=output_dir,
                epochs=epochs,
                batch_size=batch_size,
                image_size=image_size,
                device=device,
            )
            pseudo_config = PseudoLabelConfig()
            trainer = SemiSupervisedTrainer(config, pseudo_config)

            start_time = time.time()
            trainer.train_supervised_only()
            result.training_time_s = time.time() - start_time
            result.epochs = epochs

            if trainer.history["train_acc"]:
                result.final_train_acc = trainer.history["train_acc"][-1]
            if trainer.history["val_acc"]:
                result.final_val_acc = trainer.history["val_acc"][-1]
        else:
            print(f"Dataset not found at {data_dir}, skipping training benchmark")

        # Inference benchmark
        print("\nRunning inference benchmark...")
        torch_device = torch.device(device)
        model = model.to(torch_device)

        inference_results = benchmark_inference(
            model=model,
            device=torch_device,
            image_size=image_size,
            num_iterations=100,
            warmup=10,
            batch_size=batch_size,
        )

        result.mean_latency_ms = inference_results["mean_ms"]
        result.std_latency_ms = inference_results["std_ms"]
        result.min_latency_ms = inference_results["min_ms"]
        result.max_latency_ms = inference_results["max_ms"]
        result.p50_latency_ms = inference_results["p50_ms"]
        result.p95_latency_ms = inference_results["p95_ms"]
        result.p99_latency_ms = inference_results["p99_ms"]
        result.throughput_fps = inference_results["throughput_fps"]

        # Memory usage
        if device == "cuda" and torch.cuda.is_available():
            result.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    except ImportError as e:
        print(f"Error importing PyTorch modules: {e}")
        print(
            "Make sure to install requirements: pip install -r pytorch_reference/requirements.txt"
        )
    except Exception as e:
        print(f"Error running PyTorch benchmark: {e}")
        import traceback

        traceback.print_exc()

    return result


def run_burn_benchmark(
    data_dir: str,
    output_dir: str,
    epochs: int = 10,
    batch_size: int = 32,
    image_size: int = 128,
    device: str = "cuda",
) -> BenchmarkResult:
    """Run Burn/Rust benchmark."""
    print("\n" + "=" * 60)
    print("Running Burn/Rust Benchmark")
    print("=" * 60)

    result = BenchmarkResult(
        framework="Burn (Rust)",
        device=device,
        batch_size=batch_size,
        image_size=image_size,
    )

    project_root = Path(__file__).parent.parent / "plantvillage_ssl"

    try:
        # Check if Rust project is built
        binary_path = project_root / "target" / "release" / "plantvillage_ssl"
        if not binary_path.exists():
            print("Building Rust project...")
            build_result = subprocess.run(
                ["cargo", "build", "--release"],
                cwd=project_root,
                capture_output=True,
                text=True,
            )
            if build_result.returncode != 0:
                print(f"Build failed: {build_result.stderr}")
                return result

        # Run inference benchmark
        print("\nRunning Burn inference benchmark...")
        benchmark_cmd = [
            str(binary_path),
            "benchmark",
            "--batch-size",
            str(batch_size),
            "--iterations",
            "100",
        ]

        # Add output file for results
        benchmark_output = Path(output_dir) / "burn_benchmark.json"
        os.makedirs(output_dir, exist_ok=True)

        start_time = time.time()
        bench_result = subprocess.run(
            benchmark_cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        bench_time = time.time() - start_time

        if bench_result.returncode == 0:
            # Try to parse output for metrics
            output = bench_result.stdout
            print(output)

            # Parse latency from output (format depends on actual implementation)
            # This is a fallback if JSON output isn't available
            for line in output.split("\n"):
                if "mean" in line.lower() and "ms" in line.lower():
                    try:
                        # Try to extract mean latency
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if "ms" in part:
                                value = float(parts[i - 1].replace(",", ""))
                                result.mean_latency_ms = value
                                break
                    except (ValueError, IndexError):
                        pass
                if "throughput" in line.lower() or "fps" in line.lower():
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if "fps" in part.lower():
                                value = float(parts[i - 1].replace(",", ""))
                                result.throughput_fps = value
                                break
                    except (ValueError, IndexError):
                        pass
        else:
            print(f"Benchmark failed: {bench_result.stderr}")

        # Run training benchmark if dataset exists
        if Path(data_dir).exists():
            print("\nRunning Burn training benchmark...")
            train_cmd = [
                str(binary_path),
                "train",
                "--data-dir",
                data_dir,
                "--epochs",
                str(epochs),
                "--batch-size",
                str(batch_size),
                "--output-dir",
                output_dir,
            ]

            start_time = time.time()
            train_result = subprocess.run(
                train_cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )
            result.training_time_s = time.time() - start_time
            result.epochs = epochs

            if train_result.returncode == 0:
                print(
                    train_result.stdout[-2000:]
                    if len(train_result.stdout) > 2000
                    else train_result.stdout
                )
                # Try to parse accuracy from output
                for line in train_result.stdout.split("\n"):
                    if "val" in line.lower() and "acc" in line.lower():
                        try:
                            # Extract accuracy percentage
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if "%" in part:
                                    value = float(part.replace("%", ""))
                                    result.final_val_acc = value
                                    break
                        except (ValueError, IndexError):
                            pass
            else:
                print(f"Training failed: {train_result.stderr}")

        # Get model size from saved checkpoint (find latest plant_classifier_*.mpk)
        model_files = list(Path(output_dir).glob("plant_classifier_*.mpk"))
        if model_files:
            model_path = max(model_files, key=lambda p: p.stat().st_mtime)
            result.model_size_mb = model_path.stat().st_size / (1024 * 1024)

    except subprocess.TimeoutExpired:
        print("Benchmark timed out!")
    except FileNotFoundError:
        print("Rust binary not found. Make sure to build with: cargo build --release")
    except Exception as e:
        print(f"Error running Burn benchmark: {e}")
        import traceback

        traceback.print_exc()

    return result


def generate_comparison_charts(report: ComparisonReport, output_dir: str):
    """Generate comparison charts."""
    print("\n" + "=" * 60)
    print("Generating Comparison Charts")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Set up matplotlib style
    plt.style.use("seaborn-v0_8-darkgrid")
    colors = {"Burn": "#E74C3C", "PyTorch": "#3498DB"}

    # 1. Latency Comparison
    if report.burn_result and report.pytorch_result:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart for mean latency
        ax1 = axes[0]
        frameworks = ["Burn (Rust)", "PyTorch"]
        latencies = [
            report.burn_result.mean_latency_ms,
            report.pytorch_result.mean_latency_ms,
        ]
        errors = [
            report.burn_result.std_latency_ms,
            report.pytorch_result.std_latency_ms,
        ]

        bars = ax1.bar(
            frameworks,
            latencies,
            yerr=errors,
            capsize=5,
            color=[colors["Burn"], colors["PyTorch"]],
            alpha=0.8,
        )
        ax1.set_ylabel("Latency (ms)")
        ax1.set_title("Inference Latency Comparison")

        # Add value labels
        for bar, lat in zip(bars, latencies):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{lat:.2f} ms",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # Throughput comparison
        ax2 = axes[1]
        throughputs = [
            report.burn_result.throughput_fps,
            report.pytorch_result.throughput_fps,
        ]

        bars = ax2.bar(
            frameworks,
            throughputs,
            color=[colors["Burn"], colors["PyTorch"]],
            alpha=0.8,
        )
        ax2.set_ylabel("Throughput (FPS)")
        ax2.set_title("Inference Throughput Comparison")

        for bar, tp in zip(bars, throughputs):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{tp:.1f} FPS",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.tight_layout()
        plt.savefig(Path(output_dir) / "latency_comparison.png", dpi=150)
        plt.close()
        print("  Saved: latency_comparison.png")

        # 2. Percentile Latency Distribution
        fig, ax = plt.subplots(figsize=(10, 6))

        percentiles = ["P50", "P95", "P99", "Max"]
        burn_values = [
            report.burn_result.p50_latency_ms,
            report.burn_result.p95_latency_ms,
            report.burn_result.p99_latency_ms,
            report.burn_result.max_latency_ms,
        ]
        pytorch_values = [
            report.pytorch_result.p50_latency_ms,
            report.pytorch_result.p95_latency_ms,
            report.pytorch_result.p99_latency_ms,
            report.pytorch_result.max_latency_ms,
        ]

        x = np.arange(len(percentiles))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2,
            burn_values,
            width,
            label="Burn (Rust)",
            color=colors["Burn"],
            alpha=0.8,
        )
        bars2 = ax.bar(
            x + width / 2,
            pytorch_values,
            width,
            label="PyTorch",
            color=colors["PyTorch"],
            alpha=0.8,
        )

        ax.set_ylabel("Latency (ms)")
        ax.set_title("Latency Distribution (Percentiles)")
        ax.set_xticks(x)
        ax.set_xticklabels(percentiles)
        ax.legend()

        plt.tight_layout()
        plt.savefig(Path(output_dir) / "latency_percentiles.png", dpi=150)
        plt.close()
        print("  Saved: latency_percentiles.png")

        # 3. Training Time Comparison
        if (
            report.burn_result.training_time_s > 0
            and report.pytorch_result.training_time_s > 0
        ):
            fig, ax = plt.subplots(figsize=(8, 6))

            training_times = [
                report.burn_result.training_time_s,
                report.pytorch_result.training_time_s,
            ]

            bars = ax.bar(
                frameworks,
                training_times,
                color=[colors["Burn"], colors["PyTorch"]],
                alpha=0.8,
            )
            ax.set_ylabel("Time (seconds)")
            ax.set_title(
                f"Training Time Comparison ({report.burn_result.epochs} epochs)"
            )

            for bar, t in zip(bars, training_times):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{t:.1f}s",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

            plt.tight_layout()
            plt.savefig(Path(output_dir) / "training_time.png", dpi=150)
            plt.close()
            print("  Saved: training_time.png")

        # 4. Summary Dashboard
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Speedup metrics
        ax = axes[0, 0]
        metrics = ["Latency\nSpeedup", "Throughput\nImprovement", "Training\nSpeedup"]
        values = [
            report.latency_speedup(),
            report.throughput_improvement(),
            report.training_speedup(),
        ]
        colors_bar = ["#27AE60" if v > 1 else "#E74C3C" for v in values]

        bars = ax.bar(metrics, values, color=colors_bar, alpha=0.8)
        ax.axhline(y=1, color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel("Speedup Factor (Burn/PyTorch)")
        ax.set_title("Burn vs PyTorch Speedup")

        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{v:.2f}x",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        # Model size comparison
        ax = axes[0, 1]
        sizes = [
            report.burn_result.model_size_mb if report.burn_result else 0,
            report.pytorch_result.model_size_mb if report.pytorch_result else 0,
        ]
        bars = ax.bar(
            frameworks, sizes, color=[colors["Burn"], colors["PyTorch"]], alpha=0.8
        )
        ax.set_ylabel("Size (MB)")
        ax.set_title("Model Size Comparison")

        for bar, s in zip(bars, sizes):
            if s > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    f"{s:.2f} MB",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        # Accuracy comparison
        ax = axes[1, 0]
        accuracies = [
            report.burn_result.final_val_acc if report.burn_result else 0,
            report.pytorch_result.final_val_acc if report.pytorch_result else 0,
        ]
        bars = ax.bar(
            frameworks, accuracies, color=[colors["Burn"], colors["PyTorch"]], alpha=0.8
        )
        ax.set_ylabel("Validation Accuracy (%)")
        ax.set_title("Final Accuracy Comparison")
        ax.set_ylim(0, 100)

        for bar, acc in zip(bars, accuracies):
            if acc > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{acc:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        # Summary text
        ax = axes[1, 1]
        ax.axis("off")

        summary_text = (
            f"""
        BENCHMARK SUMMARY
        ─────────────────────────────────

        Device: {report.burn_result.device if report.burn_result else "N/A"}
        Image Size: {report.burn_result.image_size if report.burn_result else 128}x{report.burn_result.image_size if report.burn_result else 128}
        Batch Size: {report.burn_result.batch_size if report.burn_result else 1}

        BURN (Rust):
          • Latency: {report.burn_result.mean_latency_ms:.2f} ms
          • Throughput: {report.burn_result.throughput_fps:.1f} FPS
          • Accuracy: {report.burn_result.final_val_acc:.1f}%

        PyTorch:
          • Latency: {report.pytorch_result.mean_latency_ms:.2f} ms
          • Throughput: {report.pytorch_result.throughput_fps:.1f} FPS
          • Accuracy: {report.pytorch_result.final_val_acc:.1f}%

        VERDICT:
          Latency: {report.latency_speedup():.2f}x {"faster" if report.latency_speedup() > 1 else "slower"}
          Throughput: {report.throughput_improvement():.2f}x {"higher" if report.throughput_improvement() > 1 else "lower"}
        """
            if report.burn_result and report.pytorch_result
            else "Incomplete benchmark data"
        )

        ax.text(
            0.1,
            0.9,
            summary_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        plt.savefig(Path(output_dir) / "benchmark_summary.png", dpi=150)
        plt.close()
        print("  Saved: benchmark_summary.png")


def save_results(report: ComparisonReport, output_dir: str):
    """Save benchmark results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "burn": vars(report.burn_result) if report.burn_result else None,
        "pytorch": vars(report.pytorch_result) if report.pytorch_result else None,
        "comparison": {
            "latency_speedup": report.latency_speedup(),
            "throughput_improvement": report.throughput_improvement(),
            "training_speedup": report.training_speedup(),
            "size_reduction": report.size_reduction(),
        },
    }

    output_path = Path(output_dir) / "benchmark_comparison.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved results to: {output_path}")


def print_comparison_table(report: ComparisonReport):
    """Print a formatted comparison table."""
    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON: BURN (RUST) vs PYTORCH")
    print("=" * 70)

    if not report.burn_result or not report.pytorch_result:
        print("Incomplete benchmark data!")
        return

    b = report.burn_result
    p = report.pytorch_result

    print(f"\n{'Metric':<30} {'Burn (Rust)':<18} {'PyTorch':<18} {'Speedup':<12}")
    print("-" * 70)

    # Latency
    speedup = p.mean_latency_ms / b.mean_latency_ms if b.mean_latency_ms > 0 else 0
    print(
        f"{'Mean Latency':<30} {b.mean_latency_ms:>15.2f} ms {p.mean_latency_ms:>15.2f} ms {speedup:>10.2f}x"
    )

    print(
        f"{'P50 Latency':<30} {b.p50_latency_ms:>15.2f} ms {p.p50_latency_ms:>15.2f} ms"
    )
    print(
        f"{'P95 Latency':<30} {b.p95_latency_ms:>15.2f} ms {p.p95_latency_ms:>15.2f} ms"
    )
    print(
        f"{'P99 Latency':<30} {b.p99_latency_ms:>15.2f} ms {p.p99_latency_ms:>15.2f} ms"
    )

    # Throughput
    improvement = b.throughput_fps / p.throughput_fps if p.throughput_fps > 0 else 0
    print(
        f"{'Throughput':<30} {b.throughput_fps:>14.1f} FPS {p.throughput_fps:>14.1f} FPS {improvement:>10.2f}x"
    )

    # Training
    if b.training_time_s > 0 and p.training_time_s > 0:
        speedup = p.training_time_s / b.training_time_s
        print(
            f"{'Training Time ({b.epochs} epochs)':<30} {b.training_time_s:>15.1f} s {p.training_time_s:>15.1f} s {speedup:>10.2f}x"
        )

    # Accuracy
    print(
        f"{'Final Val Accuracy':<30} {b.final_val_acc:>15.1f} % {p.final_val_acc:>15.1f} %"
    )

    # Model
    print(f"{'Model Size':<30} {b.model_size_mb:>14.2f} MB {p.model_size_mb:>14.2f} MB")

    print("-" * 70)
    print("\n📊 Summary:")
    print(
        f"   • Burn is {report.latency_speedup():.2f}x {'faster' if report.latency_speedup() > 1 else 'slower'} in inference latency"
    )
    print(
        f"   • Burn has {report.throughput_improvement():.2f}x {'higher' if report.throughput_improvement() > 1 else 'lower'} throughput"
    )
    if report.training_speedup() > 0:
        print(
            f"   • Burn trains {report.training_speedup():.2f}x {'faster' if report.training_speedup() > 1 else 'slower'}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compare Burn (Rust) and PyTorch implementations"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/plantvillage",
        help="Path to PlantVillage dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/benchmarks",
        help="Output directory for results",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs for benchmark",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Image size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run benchmarks on",
    )
    parser.add_argument(
        "--skip-pytorch",
        action="store_true",
        help="Skip PyTorch benchmark",
    )
    parser.add_argument(
        "--skip-burn",
        action="store_true",
        help="Skip Burn benchmark",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip chart generation",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("BURN vs PYTORCH BENCHMARK COMPARISON")
    print("=" * 70)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.image_size}")

    report = ComparisonReport()

    # Run PyTorch benchmark
    if not args.skip_pytorch:
        report.pytorch_result = run_pytorch_benchmark(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            device=args.device,
        )

    # Run Burn benchmark
    if not args.skip_burn:
        report.burn_result = run_burn_benchmark(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            device=args.device,
        )

    # Print comparison
    print_comparison_table(report)

    # Generate charts
    if not args.no_charts:
        try:
            generate_comparison_charts(report, args.output_dir)
        except Exception as e:
            print(f"Warning: Could not generate charts: {e}")

    # Save results
    save_results(report, args.output_dir)

    print("\n✅ Benchmark comparison complete!")


if __name__ == "__main__":
    main()
