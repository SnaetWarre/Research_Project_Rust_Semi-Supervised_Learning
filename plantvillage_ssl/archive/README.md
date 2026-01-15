# Archive - Jetson Development Code

This folder contains archived code from the Jetson Orin Nano development phase.
This code is kept for reference and demonstration purposes but is no longer compiled or maintained.

## Reason for Archiving

The NVIDIA Jetson Orin Nano with 8GB of unified memory proved insufficient for training
neural networks efficiently. Training was moved to a more powerful laptop with dedicated
GPU memory (CUDA 13).

## Archived Components

### `server/`
HTTP API server for running PlantVillage SSL training on Jetson devices remotely.
- Used Axum web framework
- Provided REST API for starting/stopping training
- SSE streaming for real-time training output
- File management endpoints for models and experiments

### `inference/jetson.rs`
Jetson-specific inference code including:
- Power monitoring via tegrastats
- Jetson device detection
- Power mode and GPU frequency reporting
- JetsonDeviceInfo and JetsonPowerStats structures

### `scripts/`
Jetson setup scripts:
- `setup-jetson-pytorch.sh` - PyTorch installation for JetPack 6.x
- `setup_jetson.sh` - General Jetson environment setup

### `gui-jetson-client/`
GUI components for remote Jetson connectivity:
- `client/mod.rs` - HTTP client for Jetson API
- `connection.rs` - Tauri commands for connection management  
- `remote.rs` - Remote training commands
- `ConnectionStatus.svelte` - Frontend connection status component
- `connection.ts` - Svelte store for connection state

## Configuration Changes Made

When archiving this code, the following changes were made to the main codebase:

1. **Batch size**: Reverted from 4 to 32 (standard GPU training)
2. **Cargo.toml**: Removed wgpu, jetson features, and release-jetson profile
3. **Backend**: Simplified to CUDA-only without Jetson-specific comments
4. **Documentation**: Updated to reference "GPU" instead of "Jetson Orin Nano"

## Date Archived

January 2026
