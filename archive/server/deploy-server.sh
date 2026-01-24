#!/bin/bash
# deploy-server.sh - Deploy the PlantVillage server to Jetson
#
# Usage: ./deploy-server.sh [--build-remote]
#   --build-remote: Build on Jetson instead of copying pre-built binary

set -e

JETSON_HOST="jetson"
JETSON_USER="nvidia-user"
REMOTE_DIR="/home/nvidia-user/plantvillage_ssl"
LOCAL_SERVER_DIR="$(dirname "$0")"

echo "=========================================="
echo "PlantVillage Server Deployment"
echo "=========================================="

# Check SSH connection
echo "Checking SSH connection to $JETSON_HOST..."
if ! ssh -q "$JETSON_HOST" exit; then
    echo "Error: Cannot connect to $JETSON_HOST. Check your SSH config."
    exit 1
fi
echo "✓ SSH connection OK"

# Parse arguments
BUILD_REMOTE=false
if [[ "$1" == "--build-remote" ]]; then
    BUILD_REMOTE=true
fi

if $BUILD_REMOTE; then
    echo ""
    echo "Building server on Jetson (recommended for ARM64)..."
    
    # Sync the server source code
    echo "Syncing source code..."
    rsync -avz --progress \
        --exclude 'target' \
        --exclude '.git' \
        "$LOCAL_SERVER_DIR/" \
        "$JETSON_HOST:$REMOTE_DIR/server/"
    
    # Build on Jetson
    echo "Building on Jetson (this may take a few minutes)..."
    ssh "$JETSON_HOST" "source ~/.cargo/env && cd $REMOTE_DIR/server && cargo build --release"
    
else
    echo ""
    echo "Note: The server needs to be built on Jetson (ARM64 architecture)."
    echo "Use --build-remote to build on the Jetson."
    echo ""
    echo "For now, syncing source code and building remotely..."
    
    # Sync the server source code
    echo "Syncing source code..."
    rsync -avz --progress \
        --exclude 'target' \
        --exclude '.git' \
        "$LOCAL_SERVER_DIR/" \
        "$JETSON_HOST:$REMOTE_DIR/server/"
    
    # Build on Jetson
    echo "Building on Jetson (this may take a few minutes)..."
    ssh "$JETSON_HOST" "source ~/.cargo/env && cd $REMOTE_DIR/server && cargo build --release"
fi

echo ""
echo "=========================================="
echo "Installing systemd service..."
echo "=========================================="

# Copy the service file
scp "$LOCAL_SERVER_DIR/plantvillage-server.service" \
    "$JETSON_HOST:/tmp/plantvillage-server.service"

# Install and enable the service
ssh "$JETSON_HOST" bash << 'EOF'
    # Move service file (requires sudo)
    sudo mv /tmp/plantvillage-server.service /etc/systemd/system/
    sudo chmod 644 /etc/systemd/system/plantvillage-server.service
    
    # Reload systemd
    sudo systemctl daemon-reload
    
    # Enable the service (start on boot)
    sudo systemctl enable plantvillage-server
    
    # Start/restart the service
    sudo systemctl restart plantvillage-server
    
    # Check status
    sleep 2
    sudo systemctl status plantvillage-server --no-pager
EOF

echo ""
echo "=========================================="
echo "Deployment complete!"
echo "=========================================="
echo ""
echo "Server is now running on http://10.42.0.10:8080"
echo ""
echo "Useful commands:"
echo "  Check status:  ssh jetson 'sudo systemctl status plantvillage-server'"
echo "  View logs:     ssh jetson 'sudo journalctl -u plantvillage-server -f'"
echo "  Restart:       ssh jetson 'sudo systemctl restart plantvillage-server'"
echo "  Stop:          ssh jetson 'sudo systemctl stop plantvillage-server'"
echo ""
echo "Test the server:"
echo "  curl http://10.42.0.10:8080/health"
echo ""
