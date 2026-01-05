#!/usr/bin/env bash
set -euo pipefail

# PlantVillage dataset bootstrapper (bash-only, no Python)
# - Downloads from Kaggle
# - Normalizes into data/plantvillage/organized/{class}/image.jpg
# - Writes deterministic split manifests (test/val/labeled/stream/future)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET="abdallahalidev/plantvillage-dataset"
DEFAULT_OUTPUT="$PROJECT_ROOT/data/plantvillage"
OUTPUT_DIR="$DEFAULT_OUTPUT"
FORCE=0
SKIP_DOWNLOAD=0
SEED=42

usage() {
  cat <<EOF
Usage: $0 [--output-dir PATH] [--force] [--skip-download]

Options:
  --output-dir PATH   Target dataset root (default: $DEFAULT_OUTPUT)
  --force             Redownload/rebuild even if organized data exists
  --skip-download     Reuse existing raw data but rebuild manifests
  -h, --help          Show this help

Outputs (under --output-dir):
  raw/         - Raw Kaggle contents (zip + extracted)
  organized/   - Normalized class-per-directory structure
  splits/      - Manifest files for test/val/labeled/stream/future
EOF
}

log() { printf "==> %s\n" "$*"; }
err() { printf "ERROR: %s\n" "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir) OUTPUT_DIR="$(realpath "$2")"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) err "Unknown option: $1" ;;
  esac
done

RAW_DIR="$OUTPUT_DIR/raw"
ORG_DIR="$OUTPUT_DIR/organized"
SPLIT_DIR="$OUTPUT_DIR/splits"
ZIP_PATH="$RAW_DIR/plantvillage-dataset.zip"

require_kaggle() {
  command -v kaggle >/dev/null 2>&1 || err "kaggle CLI not found. Install with: pip install kaggle"
  [[ -f "$HOME/.kaggle/kaggle.json" || -f "$HOME/.config/kaggle/kaggle.json" ]] || err "Kaggle API key missing. Download from Kaggle account > API and place in ~/.config/kaggle/kaggle.json (chmod 600)."
}

ceil_pct() {
  local count="$1" pct="$2"
  printf '%s\n' $(( (count * pct + 99) / 100 ))
}

detect_class_root() {
  local candidate
  while IFS= read -r candidate; do
    local subdirs images
    subdirs=$(find "$candidate" -maxdepth 1 -mindepth 1 -type d | wc -l | tr -d ' ')
    images=$(find "$candidate" -maxdepth 2 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' \) -print -quit)
    if [[ "$subdirs" -ge 30 && -n "$images" ]]; then
      echo "$candidate"
      return 0
    fi
  done < <(find "$RAW_DIR" -maxdepth 3 -type d)
  # Fallback: raw dir itself
  echo "$RAW_DIR"
}

download_dataset() {
  [[ "$SKIP_DOWNLOAD" -eq 1 ]] && return 0

  require_kaggle

  rm -rf "$RAW_DIR"
  mkdir -p "$RAW_DIR"

  log "Downloading PlantVillage dataset via Kaggle CLI..."
  kaggle datasets download -d "$DATASET" -p "$RAW_DIR" --force

  if [[ -f "$ZIP_PATH" ]]; then
    log "Unzipping archive..."
    unzip -q "$ZIP_PATH" -d "$RAW_DIR"
  fi
}

organize_dataset() {
  mkdir -p "$ORG_DIR"

  local class_root
  class_root="$(detect_class_root)"
  log "Detected class root: $class_root"

  if [[ "$FORCE" -eq 1 ]]; then
    rm -rf "$ORG_DIR"
    mkdir -p "$ORG_DIR"
  fi

  if [[ -n "$(find "$ORG_DIR" -maxdepth 1 -mindepth 1 -type d 2>/dev/null)" && "$FORCE" -eq 0 ]]; then
    log "Organized data already present at $ORG_DIR (use --force to rebuild)"
    return
  fi

  log "Copying normalized class folders into $ORG_DIR ..."
  rsync -a --delete "$class_root"/ "$ORG_DIR"/

  # Fix folder names with spaces (replace spaces with underscores)
  log "Normalizing folder names (replacing spaces with underscores)..."
  find "$ORG_DIR" -maxdepth 1 -type d -name "* *" | while IFS= read -r dir; do
    local newname
    newname="$(echo "$dir" | tr ' ' '_')"
    mv "$dir" "$newname"
    log "  Renamed: $(basename "$dir") -> $(basename "$newname")"
  done
}

create_manifests() {
  rm -rf "$SPLIT_DIR"
  mkdir -p "$SPLIT_DIR"

  local test_file="$SPLIT_DIR/test.txt"
  local val_file="$SPLIT_DIR/validation.txt"
  local labeled_file="$SPLIT_DIR/labeled_pool.txt"
  local stream_file="$SPLIT_DIR/stream_pool.txt"

  : > "$test_file"
  : > "$val_file"
  : > "$labeled_file"
  : > "$stream_file"

  local total=0 test_total=0 val_total=0 labeled_total=0 stream_total=0

  for class_dir in "$ORG_DIR"/*/; do
    [[ -d "$class_dir" ]] || continue
    local class_name
    class_name="$(basename "$class_dir")"

    mapfile -t files < <(find "$class_dir" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' \) | LC_ALL=C sort | shuf --random-source <(yes "$SEED"))
    local count="${#files[@]}"
    (( total += count ))
    if [[ "$count" -eq 0 ]]; then
      log "Skipping empty class: $class_name"
      continue
    fi

    # Split: 10% test, 10% val, 30% labeled (for CNN), 50% stream (for SSL)
    local n_test n_val remaining n_labeled n_stream
    n_test=$(ceil_pct "$count" 10)
    n_val=$(ceil_pct "$count" 10)
    remaining=$((count - n_test - n_val))
    [[ $remaining -lt 0 ]] && remaining=0
    n_labeled=$(ceil_pct "$remaining" 30)
    n_stream=$((remaining - n_labeled))  # All remaining goes to SSL stream
    [[ $n_stream -lt 0 ]] && n_stream=0

    test_total=$((test_total + n_test))
    val_total=$((val_total + n_val))
    labeled_total=$((labeled_total + n_labeled))
    stream_total=$((stream_total + n_stream))

    local idx=0
    for f in "${files[@]}"; do
      rel="${f#$ORG_DIR/}"
      if [[ $idx -lt $n_test ]]; then
        printf '%s\n' "$rel" >> "$test_file"
      elif [[ $idx -lt $((n_test + n_val)) ]]; then
        printf '%s\n' "$rel" >> "$val_file"
      elif [[ $idx -lt $((n_test + n_val + n_labeled)) ]]; then
        printf '%s\n' "$rel" >> "$labeled_file"
      else
        printf '%s\n' "$rel" >> "$stream_file"
      fi
      idx=$((idx + 1))
    done
  done

  cat > "$SPLIT_DIR/split_config.json" <<EOF
{
  "seed": $SEED,
  "totals": {
    "all": $total,
    "test": $test_total,
    "validation": $val_total,
    "labeled_pool": $labeled_total,
    "stream_pool": $stream_total
  },
  "fractions": {
    "test": 0.10,
    "validation": 0.10,
    "labeled_of_remaining": 0.30,
    "stream_of_remaining": 0.70
  },
  "notes": "Manifests list paths relative to organized/; splits are stratified per-class with deterministic shuf seed. All unlabeled data goes to stream_pool for SSL."
}
EOF

  log "Split manifests written to $SPLIT_DIR"
  log "Totals -> test: $test_total, val: $val_total, labeled: $labeled_total, stream: $stream_total"
}

main() {
  log "PlantVillage dataset setup"
  log "Output root: $OUTPUT_DIR"

  mkdir -p "$OUTPUT_DIR"

  if [[ "$FORCE" -eq 0 && -d "$ORG_DIR" && -s "$ORG_DIR" ]]; then
    log "Organized data exists at $ORG_DIR (using existing files; --force to redownload)"
  else
    download_dataset
    organize_dataset
  fi

  create_manifests

  log "✅ Dataset ready."
  log "  Organized data : $ORG_DIR"
  log "  Split manifests: $SPLIT_DIR"
}

main "$@"
