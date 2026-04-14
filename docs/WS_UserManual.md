# User Manual: Plant Disease Detection System

**Author:** Warre Snaet  
**Project:** Research Project 2025-2026  
**Version:** 1.0

---

## 1. Getting Started (Hoe start een gebruiker)

This guide explains how to install and launch the Plant Disease Detection application. The system is designed to run locally on your machine, supporting both desktop (Linux/Windows/macOS) and edge devices (NVIDIA Jetson).

### Prerequisites

* **Operating System**: Linux (recommended), Windows, or macOS.
* **Drivers**: NVIDIA CUDA drivers (if using GPU acceleration).
* **Software**: Rust (Cargo) and Bun (for the GUI).

### Installation & Launch

You can run the application in two modes: the **Graphical User Interface (GUI)** for ease of use, or the **Command Line Interface (CLI)** for automation and headless environments.

#### Option A: Graphical User Interface (Recommended)

The GUI provides a modern, visual dashboard for all functionalities.

1. **Open a terminal** in the project directory.
2. **Navigate** to the GUI folder:

    ```bash
    cd plantvillage_ssl/gui
    ```

3. **Install dependencies** (first time only):

    ```bash
    bun install
    ```

4. **Start the application**:

    ```bash
    bun run tauri:dev
    ```

    *The application window will open automatically.*

#### Option B: Command Line Interface

Best for scripting or server environments.

1. **Navigate** to the core library folder:

    ```bash
    cd plantvillage_ssl
    ```

2. **Display help** to see available commands:

    ```bash
    cargo run --release --bin plantvillage_ssl -- --help
    ```

---

## 2. Capabilities & Functionalities (Mogelijkheden)

This section describes what you can do with the application.

### 🍃 A. Real-time Disease Detection (Inference)

**Goal**: Identify plant diseases from leaf images instantly.

* **How to use**:
    1. Go to the **"Inference"** tab in the sidebar.
    2. **Drag & Drop** an image of a plant leaf into the upload zone (or click to browse).
    3. The system analyzes the image and displays:
        * **Predicted Disease**: The most likely class.
        * **Confidence**: How certain the model is (Green = High, Red = Low).
        * **Top 5**: A chart showing other potential matches.

### 🔄 B. Semi-Supervised Learning Simulation

**Goal**: Visualize how the model improves by learning from unlabeled data without human intervention.

* **How to use**:
    1. Go to the **"SSL Demo"** tab.
    2. Set the **Daily Batch Size** (e.g., 100 images/day).
    3. Click **"Start Stream"**.
    4. **Observe**:
        * The system processes "unlabeled" images.
        * High-confidence predictions become "pseudo-labels".
        * The model **automatically retrains** when enough new data is collected.

### 🧪 C. Experiments

**Goal**: Run specific experimental scenarios to validate learning performance.

* **How to use**:
    1. Go to the **"Experiments"** tab.
    2. Select an experiment type (e.g., Class Incremental Learning).
    3. Monitor the metrics as the experiment runs.

### 🏋️ D. Train New Models

**Goal**: Create a custom classification model from a dataset.

* **How to use**:
    1. Go to the **"Training"** (or command line) to initiate a full training run.
    2. Select your dataset folder (must follow the PlantVillage structure).
    3. Adjust parameters:
        * **Labeled Ratio**: How much data is "known" (e.g., 0.2 for 20%).
        * **Epochs**: Training duration (recommended: 30).
    4. The dashboard updates with real-time loss and accuracy graphs.

---

## 3. CLI Reference

For advanced users, the command line offers direct control:

| Command | Description |
|---------|-------------|
| `train` | Train a new model with semi-supervised learning |
| `infer` | Run inference on a single image |
| `simulate` | Run SSL simulation pipeline |
| `stats` | Display dataset statistics |
| `export` | Export model metrics to CSV/JSON |

---

## 4. Troubleshooting

| Issue | Solution |
|-------|----------|
| **App won't start** | Ensure `bun` and `cargo` are installed and in your PATH. |
| **"CUDA not found"** | Install NVIDIA Toolkit or run without `--cuda` (slower). |
| **Low Accuracy** | Check image lighting or try training with more epochs. |
| **Out of Memory** | Close other GPU-heavy apps or reduce batch size in training. |
