# PlantVillage SSL - User Manual
**Version:** 1.0
**Author:** Warre Snaet

This manual provides instructions for using the PlantVillage Semi-Supervised Learning application, specifically the Graphical User Interface (GUI) developed for easy interaction on Edge devices.

## 1. Getting Started

### Launching the Application
On your Jetson Orin Nano (or desktop), launch the application from the terminal:

```bash
cd plantvillage_ssl/gui
bun run tauri:dev
```
*(Or use the desktop shortcut if installed)*

### The Dashboard
Upon startup, you are greeted by the **Dashboard**. This gives you an at-a-glance view of the system status.
- **Model Status:** Shows if a model is loaded (Red/Green indicator).
- **System Stats:** Current GPU VRAM usage and CPU load.
- **Activity Log:** Recent actions (training, inference, errors) are logged at the bottom.

---

## 2. Live Inference (Demo Mode)

This is the main feature for demonstrating the application's capabilities.

1.  **Navigate:** Click the **"Inference"** tab on the left sidebar.
2.  **Load Model:** Ensure a model is loaded. If not, go to the **Settings** or **Training** tab to load/train one.
3.  **Upload Image:**
    *   Click the upload area or drag-and-drop a leaf image.
    *   *Supported formats:* JPG, PNG.
4.  **View Results:**
    *   **Main Prediction:** The predicted disease class is shown prominently (e.g., "Tomato Early Blight").
    *   **Confidence Score:** A percentage (0-100%).
        *   **Green (>90%):** High confidence (Safe to auto-label).
        *   **Yellow (70-90%):** Medium confidence.
        *   **Red (<70%):** Low confidence (Manual review recommended).
    *   **Latency:** The time taken (ms) for the inference is displayed below the score.
    *   **Top-5 List:** A bar chart showing the top 5 most likely classes. This helps understand if the model was "confused" between two similar diseases.

---

## 3. Semi-Supervised Simulation

To demonstrate how the model learns from unlabeled data:

1.  **Navigate:** Click the **"Simulation"** tab.
2.  **Configuration:**
    *   **Daily Batch:** Set how many "new" images arrive per day (e.g., 50).
    *   **Threshold:** Set the confidence threshold (default: 0.9).
3.  **Start:** Click **"Start Stream"**.
4.  **Monitor:**
    *   Watch the **"Pseudo-labels Generated"** counter increase.
    *   The chart shows the ratio of **Accepted** vs **Rejected** images.
    *   When the "Retrain Threshold" (e.g., 200 images) is reached, the system will automatically trigger a background retraining session.

---

## 4. Benchmarking

To prove the speed of the application on your hardware:

1.  **Navigate:** Click the **"Benchmark"** tab.
2.  **Run:** Click **"Run Benchmark"**.
3.  **Results:**
    *   **Latency:** Average time per image (e.g., 1.3ms).
    *   **Throughput:** Images per second (FPS).
    *   **Comparison:** A chart comparing your current run against a reference PyTorch baseline.

---

## 5. Troubleshooting

*   **"No Model Loaded"**: You must train a model first using the **Training** tab or the CLI.
*   **"Inference Failed"**: Check if the image is valid. If running on Jetson, check `dmesg` for "Out of Memory" errors.
*   **Slow Interface**: If the VRAM usage is near 100%, close other applications or reduce the batch size in the code.
