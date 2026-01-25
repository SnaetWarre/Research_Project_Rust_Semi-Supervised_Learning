#!/bin/bash
#
# System Cleanup Script for PlantVillage Project
#
# This script removes:
# 1. The local build directory ($HOME/PlantVillage_iOS_Build)
# 2. Bun runtime
# 3. Rust & Rustup
# 4. Cursor Editor
# 5. Xcode (Application and Developer files)
# 6. Opencode CLI
#
# Usage: sudo bash /Volumes/T7/Documents/howest/Semester_5/Research_Project_Source/scripts/cleanup_system.sh
#

set -e

# Configuration
LOCAL_PATH="$HOME/PlantVillage_iOS_Build"

echo "=============================================="
echo "  PlantVillage System Cleanup Script"
echo "=============================================="
echo "WARNING: This script will uninstall development tools and delete data."
echo "Affected items: Bun, Rust, Cursor, Xcode, Opencode, and project temp files."
echo ""
read -p "Are you sure you want to proceed? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# 1. Remove Local Build Directory
if [ -d "$LOCAL_PATH" ]; then
    echo "[1/6] Removing local build directory ($LOCAL_PATH)..."
    rm -rf "$LOCAL_PATH"
    echo "✓ Removed $LOCAL_PATH"
else
    echo "[1/6] Local build directory not found (skipped)."
fi

# 2. Remove Bun
if [ -d "$HOME/.bun" ]; then
    echo "[2/6] Removing Bun..."
    rm -rf "$HOME/.bun"
    # Attempt to remove from rc files is complex, leaving that to user usually, 
    # but we can try to warn or clean specific lines if we knew them.
    echo "✓ Removed ~/.bun"
else
    echo "[2/6] Bun not found (skipped)."
fi

# 3. Remove Rust/Rustup
if command -v rustup &> /dev/null; then
    echo "[3/6] Removing Rust and Rustup..."
    # rustup self uninstall -y is the clean way, but if it fails we force delete
    rustup self uninstall -y || rm -rf "$HOME/.cargo" "$HOME/.rustup"
    echo "✓ Removed Rust"
elif [ -d "$HOME/.cargo" ]; then
    echo "[3/6] Removing .cargo directory..."
    rm -rf "$HOME/.cargo"
    rm -rf "$HOME/.rustup"
    echo "✓ Removed Rust directories"
else
    echo "[3/6] Rust not found (skipped)."
fi

# 4. Remove Cursor
echo "[4/6] Removing Cursor..."
FOUND_CURSOR=0
if [ -d "/Applications/Cursor.app" ]; then
    echo "  - Removing /Applications/Cursor.app"
    rm -rf "/Applications/Cursor.app"
    FOUND_CURSOR=1
fi
if [ -d "$HOME/Library/Application Support/Cursor" ]; then
    echo "  - Removing Application Support/Cursor"
    rm -rf "$HOME/Library/Application Support/Cursor"
    FOUND_CURSOR=1
fi
if [ -d "$HOME/.cursor" ]; then
    echo "  - Removing ~/.cursor"
    rm -rf "$HOME/.cursor"
    FOUND_CURSOR=1
fi

if [ $FOUND_CURSOR -eq 1 ]; then
    echo "✓ Removed Cursor"
else
    echo "Cursor not found (skipped)."
fi

# 5. Remove Xcode
echo "[5/6] Removing Xcode..."
echo "WARNING: This will delete /Applications/Xcode.app and ~/Library/Developer."
read -p "Really remove Xcode? (y/N) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ -d "/Applications/Xcode.app" ]; then
        echo "  - Removing Xcode.app (this may take a while)..."
        rm -rf "/Applications/Xcode.app"
    fi
    
    if [ -d "$HOME/Library/Developer" ]; then
        echo "  - Removing ~/Library/Developer..."
        rm -rf "$HOME/Library/Developer"
    fi
    
    # Optional: Remove command line tools if strictly requested, but often shared
    # rm -rf /Library/Developer/CommandLineTools
    
    echo "✓ Removed Xcode application and developer data"
else
    echo "Skipped Xcode removal."
fi

# 6. Remove Opencode
echo "[6/6] Removing Opencode..."
# Assuming opencode is a binary in /usr/local/bin or similar, or an app
if command -v opencode &> /dev/null; then
    OPENCODE_PATH=$(which opencode)
    echo "  - Removing binary at $OPENCODE_PATH"
    rm -f "$OPENCODE_PATH"
    echo "✓ Removed opencode binary"
else
    echo "Opencode binary not found in PATH."
fi

# Check for common app locations if it's an app
if [ -d "/Applications/Opencode.app" ]; then
    rm -rf "/Applications/Opencode.app"
    echo "✓ Removed Opencode.app"
fi

echo ""
echo "=============================================="
echo "  Cleanup Complete"
echo "=============================================="
echo "Note: You may still have lines in your shell profile (~/.zshrc or ~/.bash_profile)"
echo "referencing 'bun', 'cargo', or 'rust'. You can manually remove them."
