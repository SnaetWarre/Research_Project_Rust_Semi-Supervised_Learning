#!/usr/bin/env bash
# Runs the full experiment: split, supervised baseline, SSL, the fully supervised upper bound,
# and a test-pool evaluation of all three. Extra arguments go to every command after `split`,
# so only the flags they share work: --device, --batch-size and --seed.
# Example: scripts/reproduce.sh --device cpu
set -euo pipefail
cd "$(dirname "$0")/.."

cargo build --release -p plantssl
plantssl() { target/release/plantssl "$@"; }

plantssl split
plantssl train --out runs/supervised "$@"
plantssl ssl --init runs/supervised --out runs/ssl "$@"
plantssl train --with-stream-labels --out runs/upper-bound "$@"

for run in supervised ssl upper-bound; do
    plantssl eval --model "runs/$run" "$@"
done
