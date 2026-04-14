#!/bin/bash
# Stress Test Script for Jetson Orin Nano
# Runs continuous inference for 1 hour to check for memory leaks and thermal throttling.

DURATION_MINUTES=60
LOG_FILE="output/stress_test_metrics.csv"
MODEL_PATH="output/models/best_model.mpk" # Update this to your real model path

# Create output dir
mkdir -p output

# Initialize CSV
echo "timestamp,gpu_util,vram_used,cpu_util,ram_used" > $LOG_FILE

echo "Starting Stress Test for $DURATION_MINUTES minutes..."
echo "Logging metrics to $LOG_FILE"

# Start the benchmark in the background (infinite loop of inference)
# We use 'benchmark' command with a huge number of iterations
# 100M iterations is effectively infinite for 1 hour
./plantvillage_ssl/target/release/plantvillage_ssl benchmark \
    --iterations 100000000 \
    --batch-size 1 \
    --warmup 100 \
    > output/stress_test_log.txt 2>&1 &

PID=$!
echo "Benchmark running with PID: $PID"

# Monitor loop
START_TIME=$(date +%s)
END_TIME=$((START_TIME + DURATION_MINUTES * 60))

# CSV Header
echo "timestamp,gpu_util_total,vram_total_used,system_cpu,system_ram_used,app_cpu,app_ram_mb" > $LOG_FILE

while [ $(date +%s) -lt $END_TIME ]; do
    CURRENT_TIME=$(date +%H:%M:%S)
    
    # 1. System GPU Stats (NVIDIA)
    if command -v nvidia-smi &> /dev/null; then
        GPU_STATS=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ' ')
        GPU_UTIL=$(echo $GPU_STATS | cut -d',' -f1)
        VRAM_USED=$(echo $GPU_STATS | cut -d',' -f2)
    else
        GPU_UTIL="0"
        VRAM_USED="0"
    fi

    # 2. System CPU/RAM Stats
    SYS_CPU=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}')
    SYS_RAM=$(free -m | grep Mem | awk '{print $3}')

    # 3. Application-Specific Stats (The "Burn" process)
    # rss = Resident Set Size (RAM in KB)
    # %cpu = CPU usage percent
    if ps -p $PID > /dev/null; then
        APP_STATS=$(ps -p $PID -o %cpu=,rss= | tr -s ' ')
        APP_CPU=$(echo $APP_STATS | awk '{print $1}')
        APP_RAM_KB=$(echo $APP_STATS | awk '{print $2}')
        APP_RAM_MB=$((APP_RAM_KB / 1024))
    else
        APP_CPU="0.0"
        APP_RAM_MB="0"
    fi

    # Log to CSV
    echo "$CURRENT_TIME,$GPU_UTIL,$VRAM_USED,$SYS_CPU,$SYS_RAM,$APP_CPU,$APP_RAM_MB" >> $LOG_FILE
    
    # Print to console
    echo "[$CURRENT_TIME] App RAM: ${APP_RAM_MB}MB | System RAM: ${SYS_RAM}MB | VRAM: ${VRAM_USED}MB"
    
    sleep 5
done

# Stop the benchmark
echo "Time's up! Stopping benchmark..."
kill $PID
echo "Stress test complete."
