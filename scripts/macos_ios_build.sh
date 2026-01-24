#!/bin/bash
#
# macOS iOS Build Script for PlantVillage SSL Demo
#
# This script:
# 1. Checks for required tools (installs only if missing)
# 2. Copies the codebase from external SSD to internal drive (to avoid metadata issues)
# 3. Builds the iOS app
# 4. Opens Xcode for deploying to iPhone
#
# Usage: Run this script on your MacBook with the T7 SSD plugged in
#   bash /Volumes/T7/Documents/howest/Semester_5/Research_Project_Source/scripts/macos_ios_build.sh
#

set -e

echo "=============================================="
echo "  PlantVillage SSL - iOS Build Script"
echo "=============================================="
echo ""

# Configuration
SSD_PATH="/Volumes/T7/Documents/howest/Semester_5/Research_Project_Source"
LOCAL_PATH="$HOME/PlantVillage_iOS_Build"
PROJECT_NAME="plantvillage_ssl"

# Check if running on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "ERROR: This script must be run on macOS!"
    exit 1
fi

# Check if SSD is mounted
if [ ! -d "$SSD_PATH" ]; then
    echo "ERROR: External SSD not found at $SSD_PATH"
    echo "Please make sure the T7 SSD is connected and mounted."
    exit 1
fi

echo "Step 1: Checking required tools..."
echo "--------------------------------------"

# Check/Install Homebrew
if command -v brew &> /dev/null; then
    echo "✓ Homebrew already installed: $(brew --version | head -1)"
else
    echo "✗ Homebrew not found. Installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add to PATH for Apple Silicon
    if [[ $(uname -m) == "arm64" ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
    echo "✓ Homebrew installed"
fi

# Check/Install Rust
if command -v rustc &> /dev/null; then
    echo "✓ Rust already installed: $(rustc --version)"
else
    echo "✗ Rust not found. Installing..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    echo "✓ Rust installed"
fi

# Ensure cargo is in PATH
source "$HOME/.cargo/env" 2>/dev/null || true

# Check iOS targets
echo "  Checking iOS targets..."
if rustup target list --installed | grep -q "aarch64-apple-ios"; then
    echo "✓ iOS targets already installed"
else
    echo "  Adding iOS targets..."
    rustup target add aarch64-apple-ios
    rustup target add aarch64-apple-ios-sim
    rustup target add x86_64-apple-ios
    echo "✓ iOS targets added"
fi

# Check/Install Bun
if command -v bun &> /dev/null; then
    echo "✓ Bun already installed: $(bun --version)"
else
    echo "✗ Bun not found. Installing..."
    curl -fsSL https://bun.sh/install | bash
    export PATH="$HOME/.bun/bin:$PATH"
    echo "✓ Bun installed"
fi

# Ensure bun is in PATH
export PATH="$HOME/.bun/bin:$PATH"

# Check Xcode
if command -v xcodebuild &> /dev/null; then
    echo "✓ Xcode already installed: $(xcodebuild -version | head -1)"
else
    echo ""
    echo "ERROR: Xcode is not installed!"
    echo "Please install Xcode from the App Store and run:"
    echo "  xcode-select --install"
    echo "  sudo xcodebuild -license accept"
    exit 1
fi

# Check/Install Tauri CLI
if cargo install --list | grep -q "tauri-cli"; then
    echo "✓ Tauri CLI already installed"
else
    echo "  Installing Tauri CLI..."
    cargo install tauri-cli --version "^2.0.0"
    echo "✓ Tauri CLI installed"
fi

echo ""
echo "Step 2: Copying project to internal drive..."
echo "----------------------------------------------"
echo "Source: $SSD_PATH"
echo "Destination: $LOCAL_PATH"

# Remove old build directory if it exists
if [ -d "$LOCAL_PATH" ]; then
    echo "Removing old build directory..."
    rm -rf "$LOCAL_PATH"
fi

# Create destination directory
mkdir -p "$LOCAL_PATH"

# Copy files (excluding large/unnecessary directories)
echo "Copying files (this may take a few minutes)..."
rsync -av --progress \
    --exclude 'target' \
    --exclude 'node_modules' \
    --exclude '.git' \
    --exclude '*.o' \
    --exclude '*.a' \
    --exclude '*.dylib' \
    --exclude '.DS_Store' \
    "$SSD_PATH/" "$LOCAL_PATH/"

echo ""
echo "Step 3: Installing frontend dependencies..."
echo "--------------------------------------------"
cd "$LOCAL_PATH/gui"
bun install

echo ""
echo "Step 4: Building iOS app..."
echo "----------------------------"

# Initialize iOS project if needed
if [ ! -d "$LOCAL_PATH/gui/src-tauri/gen/apple" ]; then
    echo "Initializing Tauri iOS project..."
    cd "$LOCAL_PATH/gui"
    bun run tauri ios init
fi

# Build the iOS app
echo "Building iOS release..."
cd "$LOCAL_PATH/gui"
bun run tauri ios build --release

echo ""
echo "=============================================="
echo "  BUILD COMPLETE!"
echo "=============================================="
echo ""
echo "Next steps to deploy to your iPhone:"
echo ""
echo "1. Open Xcode with the project:"
echo "   open '$LOCAL_PATH/gui/src-tauri/gen/apple/plantvillage-gui.xcodeproj'"
echo ""
echo "2. In Xcode:"
echo "   - Select your iPhone from the device dropdown (top left)"
echo "   - Click the 'plantvillage-gui' target in the sidebar"
echo "   - Go to 'Signing & Capabilities' tab"
echo "   - Select your Apple Developer Team"
echo "   - Change Bundle Identifier if needed (e.g., com.yourname.plantvillage)"
echo ""
echo "3. Connect your iPhone 12 via USB"
echo ""
echo "4. Click the Play button (or Cmd+R) to build and run on device"
echo ""
echo "NOTE: You may need to trust the developer certificate on your iPhone:"
echo "   Settings > General > VPN & Device Management > Developer App"
echo ""

# Open Xcode automatically
echo "Opening Xcode..."
open "$LOCAL_PATH/gui/src-tauri/gen/apple/plantvillage-gui.xcodeproj"

echo ""
echo "Build files location: $LOCAL_PATH"
echo ""
