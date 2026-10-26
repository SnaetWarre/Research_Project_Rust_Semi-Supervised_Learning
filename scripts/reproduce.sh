#!/usr/bin/env bash
# Runs the full experiment: split, supervised baseline, SSL, the fully supervised upper bound,
# and a test-pool evaluation of all three. Steps whose results already exist are skipped, so the
# script can be rerun after an interruption; delete runs/ to start over.
# Extra arguments go to every command after `split`, so only the flags they share work:
# --device, --batch-size and --seed. Example: scripts/reproduce.sh --device cpu
set -euo pipefail
cd "$(dirname "$0")/.."

cargo build --release -p plantssl
plantssl() { target/release/plantssl "$@"; }

[[ -f runs/split.json ]] || plantssl split
[[ -f runs/supervised/train.json ]] || plantssl train --out runs/supervised "$@"
[[ -f runs/ssl/ssl.json ]] || plantssl ssl --init runs/supervised --out runs/ssl "$@"
[[ -f runs/upper-bound/train.json ]] || plantssl train --with-stream-labels --out runs/upper-bound "$@"

for run in supervised ssl upper-bound; do
    plantssl eval --model "runs/$run" "$@"
done
