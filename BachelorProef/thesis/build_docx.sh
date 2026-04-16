#!/usr/bin/env bash
# Build the bachelor thesis as Word (.docx) from Markdown in this directory.
#
# Dependencies: pandoc, python3 (strip step). Example: sudo pacman -S pandoc
#
# Pipeline:
#   • --reference-doc → Howest MCT template (Heading 1/2, Normal, …).
#   • pandoc/docx_polish.lua → page break before each main # heading; default figure width.
#   • pandoc/strip_word_heading_list_numbering.py → strips Word list numbering (w:numPr) from
#     Heading 1/2/4–9 in the output only. The template binds those styles to a multilevel list,
#     which would otherwise add 1., 2., … before headings that already include chapter numbers.
#
# Table of contents: Pandoc TOC is off by default (PANDOC_TOC=1 for a quick draft). For the final
# document, in Word: cursor after front matter → References → Table of Contents → Custom.
#
# Optional environment (combine as needed):
#   THESIS_GDRIVE_SYNC_DIR WSL path for the Google Drive desktop sync copy (default below).
#   PANDOC_TOC=1             Insert Pandoc TOC at the top.
#   BUILD_DOCX_VERBOSE=1     Bash xtrace, verbose strip script.
#   RCLONE_REMOTE, RCLONE_PATH   Upload via rclone after build (remote name from rclone config;
#                            RCLONE_PATH = folder on that remote). OAuth from WSL can be awkward;
#                            Drive desktop sync + THESIS_GDRIVE_SYNC_DIR is often simpler.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/../template/Template_Bachelorproef_MCT.docx"
OUT_DIR="${SCRIPT_DIR}/build"
OUT_FILE="${OUT_DIR}/Bachelorproef_Snaet_2026.docx"
# After each successful build, the .docx is copied here (mkdir -p, then cp). Override with THESIS_GDRIVE_SYNC_DIR.
GDRIVE_SYNC_DEFAULT="/mnt/c/Users/G513/Desktop/ThesisConnection"
LUA_FILTER="${SCRIPT_DIR}/pandoc/docx_polish.lua"
STRIP_HEADING_NUM="${SCRIPT_DIR}/pandoc/strip_word_heading_list_numbering.py"

# Repo root (plantvillage_ssl/... paths in Markdown resolve from BachelorProef/thesis)
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

log() { echo "[build_docx $(date -Iseconds)] $*" >&2; }

if [[ "${BUILD_DOCX_VERBOSE:-}" == "1" ]]; then
  set -x
fi

log "starting (SCRIPT_DIR=${SCRIPT_DIR})"
log "pandoc: $(command -v pandoc) $(pandoc --version | head -1)"
log "python3: $(command -v python3) $(python3 --version 2>&1)"
log "REPO_ROOT=${REPO_ROOT} (resource-path for figures)"

if ! command -v pandoc >/dev/null 2>&1; then
  echo "pandoc not found. Install with: sudo pacman -S pandoc" >&2
  exit 1
fi

if [[ ! -f "${TEMPLATE}" ]]; then
  echo "Template not found: ${TEMPLATE}" >&2
  exit 1
fi

if [[ ! -f "${LUA_FILTER}" ]]; then
  echo "Lua filter not found: ${LUA_FILTER}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

TOC_ARGS=()
if [[ "${PANDOC_TOC:-}" == "1" ]]; then
  TOC_ARGS=(--toc --toc-depth=3)
  log "Pandoc TOC enabled (--toc --toc-depth=3)"
else
  log "Pandoc TOC disabled (set PANDOC_TOC=1 to enable)"
fi

# Title page + abstract only (drop the Markdown “Table of Contents”; Word can add a real TOC).
TITLE_FRONT="$(mktemp)"
trap 'rm -f "${TITLE_FRONT}"' EXIT
head -n 29 "${SCRIPT_DIR}/00_title_and_abstract.md" > "${TITLE_FRONT}"
log "title front matter: first29 lines of 00_title_and_abstract.md → ${TITLE_FRONT}"

log "running pandoc → ${OUT_FILE}"
pandoc \
  "${TITLE_FRONT}" \
  "${SCRIPT_DIR}/00a_foreword.md" \
  "${SCRIPT_DIR}/00b_list_of_figures.md" \
  "${SCRIPT_DIR}/00c_abbreviations.md" \
  "${SCRIPT_DIR}/00d_glossary.md" \
  "${SCRIPT_DIR}/01_introduction.md" \
  "${SCRIPT_DIR}/02_research.md" \
  "${SCRIPT_DIR}/03_results.md" \
  "${SCRIPT_DIR}/04_reflection.md" \
  "${SCRIPT_DIR}/05_advice.md" \
  "${SCRIPT_DIR}/06_conclusion.md" \
  "${SCRIPT_DIR}/07_references.md" \
  "${SCRIPT_DIR}/appendices/A_installation_guide.md" \
  "${SCRIPT_DIR}/appendices/B_interview_template.md" \
  "${SCRIPT_DIR}/appendices/C_guest_session_nviso.md" \
  "${SCRIPT_DIR}/appendices/D_guest_session_2.md" \
  --output="${OUT_FILE}" \
  --from=markdown+pipe_tables+table_captions \
  --to=docx \
  --reference-doc="${TEMPLATE}" \
  --resource-path="${SCRIPT_DIR}:${REPO_ROOT}" \
  --lua-filter="${LUA_FILTER}" \
  --no-highlight \
  --metadata=author:"Warre Snaet" \
  "${TOC_ARGS[@]}"

if [[ ! -f "${OUT_FILE}" ]]; then
  log "error: pandoc did not create ${OUT_FILE}" >&2
  exit 1
fi
SZ="$(stat -c '%s' "${OUT_FILE}" 2>/dev/null || wc -c < "${OUT_FILE}")"
log "pandoc finished: ${OUT_FILE} (${SZ} bytes)"

log "stripping Word heading list numbering (template Heading1/2 multilevel conflict)"
STRIP_FLAGS=()
if [[ "${BUILD_DOCX_VERBOSE:-}" == "1" ]]; then
  STRIP_FLAGS=(-v)
fi
python3 "${STRIP_HEADING_NUM}" "${STRIP_FLAGS[@]}" "${OUT_FILE}"

SZ2="$(stat -c '%s' "${OUT_FILE}" 2>/dev/null || wc -c < "${OUT_FILE}")"
log "final: ${OUT_FILE} (${SZ2} bytes)"

# Always copy the finished .docx into the Google Drive desktop sync folder (WSL path).
if [[ -n "${THESIS_GDRIVE_SYNC_DIR:-}" ]]; then
  SYNC_DIR="${THESIS_GDRIVE_SYNC_DIR%/}"
  log "Google Drive sync: using THESIS_GDRIVE_SYNC_DIR → ${SYNC_DIR}"
else
  SYNC_DIR="${GDRIVE_SYNC_DEFAULT%/}"
  log "Google Drive sync: using default folder → ${SYNC_DIR}"
fi
DEST_SYNC="${SYNC_DIR}/$(basename "${OUT_FILE}")"
log "Google Drive sync: mkdir -p ${SYNC_DIR}"
mkdir -p "${SYNC_DIR}"
log "Google Drive sync: cp -f ${OUT_FILE} → ${DEST_SYNC}"
cp -f "${OUT_FILE}" "${DEST_SYNC}"
SYNC_SZ="$(stat -c '%s' "${DEST_SYNC}" 2>/dev/null || wc -c < "${DEST_SYNC}")"
log "Google Drive sync: finished (${DEST_SYNC}, ${SYNC_SZ} bytes)"

if [[ -n "${RCLONE_REMOTE:-}" ]]; then
  if ! command -v rclone >/dev/null 2>&1; then
    log "warning: RCLONE_REMOTE is set but rclone not in PATH; skipping upload"
  else
    # Remote name only (e.g. mydrive), no trailing colon — we add colon below.
    RC="${RCLONE_REMOTE%%:}"
    RPATH="${RC}:${RCLONE_PATH:-}"
    RPATH="${RPATH%/}"
    log "rclone copyto → ${RPATH}/$(basename "${OUT_FILE}")"
    rclone copyto "${OUT_FILE}" "${RPATH}/$(basename "${OUT_FILE}")" -v --stats=1s
    log "rclone upload finished"
  fi
else
  log "rclone upload skipped (set RCLONE_REMOTE and optional RCLONE_PATH to upload)"
fi

log "done."
echo "Wrote: ${OUT_FILE}"
