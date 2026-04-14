# PlantVillage Mobile Export - iPhone PWA Demo

This folder contains the complete pipeline to export Burn model weights to ONNX format and create a Progressive Web App (PWA) for offline plant disease detection on iPhone.

## Overview

The pipeline works as follows:

1. **Burn → JSON**: Extract weights from trained `.mpk` model using Rust
2. **JSON → PyTorch**: Load weights into matching PyTorch architecture
3. **PyTorch → ONNX**: Export to ONNX format for web inference
4. **ONNX → Browser**: Load ONNX model in browser using ONNX Runtime Web

## Quick Start

### Step 1: Export Burn Weights

```bash
cd plantvillage_ssl
cargo run --release --bin export_weights -- --model best_model.mpk --output mobile_export/weights
```

### Step 2: Convert to ONNX

```bash
cd mobile_export
source .venv/bin/activate  # or create venv: python3 -m venv .venv && pip install torch onnx numpy
python load_weights.py --weights weights/weights.json --output model.onnx --num-classes 38 --verify
```

### Step 3: Copy to PWA folder

```bash
cp model.onnx pwa/
```

### Step 4: Test Locally

PWAs require HTTPS in production, but for local testing you can use Python's HTTP server:

```bash
cd pwa
python3 -m http.server 8080
```

Then open http://localhost:8080 in your browser.

## iPhone Deployment Options

### Option A: GitHub Pages (Recommended for Demo)

1. Create a GitHub repository
2. Push the `pwa/` folder contents to the repo
3. Enable GitHub Pages in Settings → Pages
4. Access via `https://yourusername.github.io/repo-name`

On iPhone:
- Open the URL in Safari
- Tap Share → "Add to Home Screen"
- The app will work offline after first load

### Option B: Netlify/Vercel (Free Hosting)

1. Create account on Netlify or Vercel
2. Drag and drop the `pwa/` folder
3. Get your free HTTPS URL

### Option C: Local Network Testing

For testing on iPhone without deploying:

1. Install `mkcert` for local HTTPS:
   ```bash
   # macOS/Linux
   brew install mkcert  # or use your package manager
   mkcert -install
   mkcert localhost 127.0.0.1
   ```

2. Run HTTPS server:
   ```bash
   # Using Python with ssl
   python3 -c "
   import http.server, ssl
   server = http.server.HTTPServer(('0.0.0.0', 8443), http.server.SimpleHTTPRequestHandler)
   server.socket = ssl.wrap_socket(server.socket, certfile='./localhost+1.pem', keyfile='./localhost+1-key.pem', server_side=True)
   print('Server running at https://YOUR_IP:8443')
   server.serve_forever()
   "
   ```

3. Open `https://YOUR_COMPUTER_IP:8443` on iPhone (must be on same WiFi)

## PWA Features

- **Offline Support**: Model and assets are cached by service worker
- **Camera Access**: Take photos directly or upload from gallery  
- **Responsive Design**: Optimized for mobile screens
- **No Installation**: Add to Home Screen for app-like experience

## Model Architecture

The model is a CNN with:
- 4 convolutional blocks (32→64→128→256 filters)
- BatchNorm after each conv layer
- MaxPooling (2x2) after each block
- Global Average Pooling
- FC layers: 256→256→38 classes

## Troubleshooting

### "Model failed to load"
- Check console for detailed error
- Ensure `model.onnx` is in the same folder as `index.html`
- Verify the model exports correctly with `--verify` flag

### "Camera not working"
- PWAs require HTTPS for camera access
- On localhost, use https://localhost or file://

### "App not installing on Home Screen"
- Ensure you're accessing via HTTPS
- Wait for service worker to cache assets
- Try refreshing the page first

## Files

| File | Description |
|------|-------------|
| `load_weights.py` | Python script to load Burn weights and export ONNX |
| `weights/weights.json` | Exported weights from Burn model |
| `model.onnx` | ONNX model for browser inference |
| `pwa/` | Complete PWA ready for deployment |
| `pwa/index.html` | Main app interface |
| `pwa/sw.js` | Service worker for offline support |
| `pwa/manifest.json` | PWA manifest |
