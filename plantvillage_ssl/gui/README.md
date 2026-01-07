# PlantVillage SSL Dashboard

A Tauri + Svelte GUI for visualizing and interacting with the PlantVillage Semi-Supervised Learning pipeline.

## Features

- **Dashboard**: Overview of model status, dataset statistics, and recent activity
- **Training**: Real-time training visualization with loss/accuracy charts
- **Inference**: Drag-and-drop image classification with confidence visualization
- **Pseudo-Labeling Demo**: Interactive exploration of confidence thresholds
- **Simulation**: Stream simulation showing SSL improvement over time
- **Benchmark**: Inference latency and throughput measurement

## Prerequisites

- Node.js 18+ and npm
- Rust toolchain (stable)
- NVIDIA CUDA toolkit (for GPU support)
- System dependencies for Tauri:
  - Linux: `webkit2gtk-4.1`, `libgtk-3-dev`, etc.

## Development

```bash
# Install npm dependencies
npm install

# Run in development mode
npm run tauri dev
```

## Building

```bash
# Build for production
npm run tauri build
```

The built application will be in `src-tauri/target/release/`.

## Architecture

```
gui/
├── src/                    # Svelte frontend
│   ├── lib/
│   │   ├── components/     # Reusable UI components
│   │   ├── pages/          # Page components
│   │   └── stores/         # Svelte stores for state
│   └── routes/             # SvelteKit routes
├── src-tauri/              # Tauri (Rust) backend
│   └── src/
│       ├── commands/       # Tauri command handlers
│       ├── state.rs        # Application state
│       └── lib.rs          # Main entry point
└── static/                 # Static assets
```

## Usage

1. **Load Dataset**: Click "Load Dataset" on the Dashboard and select your PlantVillage dataset directory
2. **Load Model**: Click "Load Model" and select a trained `.mpk` model file
3. **Run Inference**: Go to the Inference page, drag/drop an image to classify
4. **Train Model**: Go to Training, configure parameters, and click "Start Training"
5. **Run Simulation**: Go to Simulation to see SSL improvement over time

## Tech Stack

- **Frontend**: Svelte 5, SvelteKit, Tailwind CSS, Chart.js
- **Backend**: Tauri 2.0, Rust
- **ML Framework**: Burn (with CUDA backend)
