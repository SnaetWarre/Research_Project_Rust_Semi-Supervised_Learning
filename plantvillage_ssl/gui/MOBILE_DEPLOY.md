# Mobile Build & Deployment Guide

## Overview

This guide explains how to **build the Tauri app for your iPhone** on your laptop, then install it on your phone. You're right - you don't compile on the phone itself! You compile on your laptop (with Xcode), and the result is an `.ipa` file that gets installed on your iPhone.

---

## Prerequisites

### On Your Laptop (macOS Required for iOS)

1. **Xcode** (from Mac App Store)
   ```bash
   xcode-select --install
   ```

2. **Xcode Command Line Tools**
   ```bash
   sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
   ```

3. **iOS Development Certificate** (free with Apple ID)
   - Open Xcode → Preferences → Accounts
   - Add your Apple ID
   - Download certificates

4. **Node.js/Bun** (already have this)
   ```bash
   bun --version  # Should work
   ```

5. **Rust** (already have this)
   ```bash
   rustc --version  # Should work
   ```

6. **Tauri CLI**
   ```bash
   cargo install tauri-cli --version "^2.0.0"
   ```

### On Your iPhone

1. **Enable Developer Mode**
   - Settings → Privacy & Security → Developer Mode → ON
   - Restart phone

2. **Trust Your Computer**
   - Connect iPhone to laptop via USB
   - Tap "Trust" on phone

---

## Step 1: Initialize Tauri for iOS (First Time Only)

```bash
cd plantvillage_ssl/gui

# Initialize iOS project
bun run tauri ios init
```

This creates: `src-tauri/gen/apple/` directory with Xcode project.

---

## Step 2: Configure for Mobile

Tauri automatically detects iOS target and uses `mobile` feature, but let's verify the configuration:

### Edit `src-tauri/tauri.conf.json` (Add iOS settings)

```json
{
  "$schema": "https://schema.tauri.app/config/2",
  "productName": "PlantVillage SSL",
  "version": "0.1.0",
  "identifier": "com.plantvillage.ssl",
  "build": {
    "beforeDevCommand": "bun run dev",
    "devUrl": "http://localhost:1420",
    "beforeBuildCommand": "bun run build",
    "frontendDist": "../build"
  },
  "bundle": {
    "active": true,
    "targets": ["app", "dmg", "deb"],
    "iOS": {
      "minimumSystemVersion": "13.0"
    },
    "icon": [
      "icons/32x32.png",
      "icons/128x128.png",
      "icons/icon.icns",
      "icons/icon.ico"
    ],
    "resources": [
      "mobile_dataset/**/*"
    ]
  }
}
```

---

## Step 3: Build the Dataset Bundle for Mobile

Before building for mobile, create a small dataset bundle (can't fit 87K images on phone!):

```bash
cd plantvillage_ssl/gui

# Option A: From Rust (command line)
cd ../
cargo run --bin plantvillage_ssl -- --help  # Check if there's a bundle command

# Option B: From GUI (easier)
# 1. Run the desktop app
cd gui
bun run tauri:dev

# 2. In the app, use the dataset bundling command:
#    invoke('create_dataset_bundle', {
#      images_per_class: 50,
#      source_dir: '../data/plantvillage',
#      output_dir: './mobile_dataset'
#    });

# Option C: Manually copy a subset
mkdir -p gui/mobile_dataset
# Copy ~50 images per class from data/plantvillage to gui/mobile_dataset
```

This creates a ~200MB dataset suitable for mobile.

---

## Step 4: Build for iOS

### Development Build (with your Apple ID - Free)

```bash
cd plantvillage_ssl/gui

# Build and run on connected iPhone
bun run tauri ios dev --open
```

This:
1. Compiles Rust with `--target aarch64-apple-ios` and `--features mobile,ndarray`
2. Compiles frontend (Svelte)
3. Opens Xcode
4. Builds the app
5. Installs on your connected iPhone
6. Runs it!

### Production Build

```bash
cd plantvillage_ssl/gui

# Build release IPA
bun run tauri ios build --release

# Output: src-tauri/gen/apple/build/arm64/PlantVillage SSL.ipa
```

---

## Step 5: Install on Your iPhone

### Method 1: Direct Install (Dev Build - Easiest)

```bash
# With iPhone connected via USB
bun run tauri ios dev --open
```

Xcode automatically installs and launches the app on your phone!

### Method 2: Xcode Manual Install

1. Open the Xcode project:
   ```bash
   open src-tauri/gen/apple/plantvillage_ssl.xcodeproj
   ```

2. In Xcode:
   - Select your iPhone from device dropdown
   - Select "Any iOS Device (arm64)" for build
   - Click "Run" (▶️ button)

3. First time: Xcode asks for signing
   - Team: Select your Apple ID
   - Signing Certificate: Automatic

4. App installs and runs on your phone!

### Method 3: Install IPA via Finder (Release Build)

1. Build the IPA:
   ```bash
   bun run tauri ios build --release
   ```

2. Connect iPhone to laptop

3. Open Finder → Click on your iPhone in sidebar

4. Drag `src-tauri/gen/apple/build/arm64/PlantVillage SSL.ipa` to the apps section

5. Click "Sync"

6. On iPhone: Settings → General → VPN & Device Management
   - Trust the developer certificate
   - Tap "Trust" again

7. Launch the app from home screen!

---

## Step 6: Run the App on iPhone

Once installed:

1. **Open the app** from home screen

2. **The app automatically:**
   - Detects device = Mobile
   - Uses NdArray backend (CPU)
   - Applies mobile config (batch=8, epochs=2)

3. **Take photos** of plant leaves or use the bundled dataset

4. **Run inference** (50-100ms per image)

5. **Accumulate 200+ pseudo-labels**

6. **Tap "Retrain Model"** → Retraining runs in background (15-30 min)

7. **Model improves!** Next inferences use the updated model

---

## Troubleshooting

### "No developer certificate found"

**Solution:** Add your Apple ID to Xcode
```bash
# Open Xcode
open /Applications/Xcode.app

# Xcode → Settings → Accounts → Add Apple ID
# Then build again
```

### "Untrusted Enterprise Developer"

**On iPhone:**
1. Settings → General → VPN & Device Management
2. Tap on developer profile
3. Tap "Trust"

### "iPhone is not available"

**Solution:** Enable Developer Mode on iPhone
1. Settings → Privacy & Security → Developer Mode → ON
2. Restart iPhone
3. Try building again

### "Build failed: Rust compilation error"

**Check Rust iOS target:**
```bash
rustup target add aarch64-apple-ios
rustup target add aarch64-apple-ios-sim  # For simulator
```

### "App crashes on launch"

**Check logs:**
```bash
# In Xcode:
Window → Devices and Simulators → Select your iPhone → View Console
# Look for crash logs
```

---

## Android Build (Bonus)

If you want to build for Android too:

### Prerequisites

1. **Android Studio** (download from android.com/studio)
2. **Android NDK** (install via Android Studio SDK Manager)
3. **Java JDK 17+**

### Build

```bash
cd plantvillage_ssl/gui

# Initialize Android project (first time)
bun run tauri android init

# Build and run
bun run tauri android dev

# Or build APK
bun run tauri android build --release
# Output: src-tauri/gen/android/app/build/outputs/apk/universal/release/app-universal-release.apk
```

### Install APK on Android

```bash
# With phone connected via USB (USB debugging enabled)
adb install src-tauri/gen/android/app/build/outputs/apk/universal/release/app-universal-release.apk

# Or just copy the APK file to your phone and tap to install
```

---

## What Happens Under the Hood

When you run `bun run tauri ios build`:

1. **Rust compilation:**
   ```bash
   cargo build --release \
     --target aarch64-apple-ios \
     --no-default-features \
     --features mobile,ndarray
   ```

   This:
   - Targets iOS ARM64 (iPhone CPU)
   - Disables CUDA (mobile doesn't have NVIDIA GPUs!)
   - Enables NdArray (CPU backend)
   - Compiles to native iOS library

2. **Frontend build:**
   ```bash
   bun run build  # Builds Svelte to static HTML/JS/CSS
   ```

3. **Xcode build:**
   - Creates iOS app bundle
   - Embeds Rust library
   - Embeds frontend assets
   - Embeds dataset bundle (if configured)
   - Signs with your certificate
   - Packages as `.ipa` file

4. **Install:**
   - Xcode transfers `.ipa` to iPhone
   - iOS unpacks and installs the app
   - Done!

---

## File Structure After iOS Init

```
plantvillage_ssl/gui/
├── src/                          # Svelte frontend
├── src-tauri/
│   ├── src/                      # Rust backend (adaptive!)
│   ├── Cargo.toml                # With mobile features
│   ├── tauri.conf.json           # Tauri config
│   └── gen/
│       └── apple/                # iOS project (created by init)
│           ├── plantvillage_ssl.xcodeproj
│           ├── Sources/          # iOS Swift wrapper
│           ├── Assets.xcassets/  # iOS icons
│           └── build/            # Build output
│               └── arm64/
│                   └── PlantVillage SSL.ipa  # The file!
└── mobile_dataset/               # Bundled dataset (~200MB)
    ├── Apple___Apple_scab/       (50 images)
    ├── Apple___Black_rot/        (50 images)
    └── ... (36 more classes)
```

---

## Quick Command Reference

```bash
# First time setup
cd plantvillage_ssl/gui
bun install
cargo install tauri-cli --version "^2.0.0"
bun run tauri ios init

# Development (with iPhone connected)
bun run tauri ios dev

# Production build
bun run tauri ios build --release

# Open in Xcode
open src-tauri/gen/apple/plantvillage_ssl.xcodeproj

# Check Rust iOS target
rustup target list | grep ios
rustup target add aarch64-apple-ios

# View iPhone logs
# Xcode → Window → Devices and Simulators → Console
```

---

## Performance Expectations on iPhone

| Task | Time | Notes |
|------|------|-------|
| **App launch** | 2-3 seconds | Cold start |
| **Model load** | 500ms-1s | 1.8MB model |
| **Inference (1 image)** | 50-100ms | NdArray CPU backend |
| **Inference (100 images)** | 5-10 seconds | Batch processing |
| **SSL Retraining (200 pseudo + 17K labeled, 2 epochs)** | 15-30 minutes | Background task |
| **Memory usage** | ~300MB | Model + activations |
| **Battery drain (retraining)** | ~10-15% | Plug in while retraining! |

---

## Tips for Demo

1. **Show device detection:**
   - Open app on laptop → Shows "CUDA" backend
   - Open app on iPhone → Shows "NdArray" backend

2. **Show same code, different performance:**
   - Desktop: SSL retrain in 1-2 minutes
   - iPhone: SSL retrain in 15-30 minutes
   - Same button, same command!

3. **Explain to teacher:**
   - "We compile on laptop with Xcode"
   - "Xcode creates .ipa file for iPhone"
   - "Install via USB or Xcode"
   - "App runs natively on iPhone with NdArray"
   - "No cloud needed - true on-device learning!"

---

## Summary

**You build on laptop:** `bun run tauri ios build`
**Xcode creates:** `PlantVillage SSL.ipa`
**Install via:** Xcode, Finder, or USB
**Runs on iPhone:** Automatically uses NdArray (CPU) backend
**Retraining works:** 15-30 minutes for SSL (totally practical!)

Ready to try? Start with `bun run tauri ios init` 🚀📱
