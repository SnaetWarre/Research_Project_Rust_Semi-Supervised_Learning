#!/usr/bin/env bash
# Build the bachelor thesis as Word (.docx) from Markdown in this directory.
#
# Dependencies: pandoc, python3, libreoffice, poppler (pdftoppm).
# Example: sudo pacman -S pandoc libreoffice-fresh poppler
#
# Pipeline:
#   • --reference-doc → Howest MCT template (Heading 1/2, Normal, …).
#   • cover/Kaft_bachelorproef_2025_2026_EN.docx → prepared thesis cover.
#   • pandoc/docx_polish.lua → page break before each main # heading; default figure width.
#   • pandoc/polish_docx_layout.py → box code snippets and fit the long title line.
#   • pandoc/prepend_cover_docx.py → render and prepend the prepared cover page.
#   • pandoc/strip_word_heading_list_numbering.py → strips Word list numbering (w:numPr) from
#     Heading 1/2/4–9 in the output only. The template binds those styles to a multilevel list,
#     which would otherwise add 1., 2., … before headings that already include chapter numbers.
#
# Table of contents: the build inserts only a plain “Table of Contents” page.
# Fill the actual TOC manually in Google Docs/Word so later manual edits remain simple.
#
# Optional environment (combine as needed):
#   BUILD_DOCX_OUT_DIR       Output directory for the generated docx
#                            (default: /home/warre/ThesisConnectionv2).
#   BUILD_DOCX_VERBOSE=1     Bash xtrace, verbose strip script.
#   RCLONE_REMOTE, RCLONE_PATH   Upload via rclone after build (remote name from rclone config;
#                            RCLONE_PATH = folder on that remote).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/../template/Template_Bachelorproef_MCT.docx"
OUT_DIR="${BUILD_DOCX_OUT_DIR:-/home/warre/ThesisConnectionv2}"
OUT_FILE="${OUT_DIR}/Bachelorproef_Snaet_2026.docx"
LOCAL_BUILD_DIR="${SCRIPT_DIR}/build"
LOCAL_OUT_FILE="${LOCAL_BUILD_DIR}/Bachelorproef_Snaet_2026.docx"
COVER_FILE="${SCRIPT_DIR}/cover/Kaft_bachelorproef_2025_2026_EN.docx"
LUA_FILTER="${SCRIPT_DIR}/pandoc/docx_polish.lua"
POLISH_LAYOUT="${SCRIPT_DIR}/pandoc/polish_docx_layout.py"
STRIP_HEADING_NUM="${SCRIPT_DIR}/pandoc/strip_word_heading_list_numbering.py"
PREPEND_COVER="${SCRIPT_DIR}/pandoc/prepend_cover_docx.py"

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

if [[ ! -f "${POLISH_LAYOUT}" ]]; then
  echo "DOCX layout polish script not found: ${POLISH_LAYOUT}" >&2
  exit 1
fi
if [[ ! -f "${PREPEND_COVER}" ]]; then
  echo "DOCX cover prepend script not found: ${PREPEND_COVER}" >&2
  exit 1
fi
if [[ ! -f "${COVER_FILE}" ]]; then
  echo "Cover DOCX not found: ${COVER_FILE}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}" "${LOCAL_BUILD_DIR}"

log "manual TOC placeholder enabled (plain heading only; no generated Word TOC field)"

# Keep the abstract, but drop the old Markdown title/metadata page. The prepared
# kaft DOCX is prepended after the body has been polished.
ABSTRACT_FRONT="$(mktemp)"
BODY_FILE="$(mktemp --suffix=.docx)"
trap 'rm -f "${ABSTRACT_FRONT}" "${BODY_FILE}"' EXIT
awk 'found || /^# Abstract$/ { found = 1; print }' "${SCRIPT_DIR}/00_title_and_abstract.md" > "${ABSTRACT_FRONT}"
if [[ ! -s "${ABSTRACT_FRONT}" ]]; then
  echo "Could not extract abstract from ${SCRIPT_DIR}/00_title_and_abstract.md" >&2
  exit 1
fi
log "abstract front matter without Markdown title page → ${ABSTRACT_FRONT}"

log "running pandoc body build → ${BODY_FILE}"
pandoc \
  "${SCRIPT_DIR}/00a_foreword.md" \
  "${ABSTRACT_FRONT}" \
  "${SCRIPT_DIR}/00e_table_of_contents.md" \
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
  "${SCRIPT_DIR}/appendices/B_pedro_morais.md" \
  "${SCRIPT_DIR}/appendices/C_helena_torres.md" \
  "${SCRIPT_DIR}/appendices/D_guest_session_nviso.md" \
  "${SCRIPT_DIR}/appendices/E_guest_session_2.md" \
  --output="${BODY_FILE}" \
  --from=markdown+pipe_tables+table_captions \
  --to=docx \
  --reference-doc="${TEMPLATE}" \
  --resource-path="${SCRIPT_DIR}:${REPO_ROOT}" \
  --lua-filter="${LUA_FILTER}" \
  --highlight-style=tango

if [[ ! -f "${BODY_FILE}" ]]; then
  log "error: pandoc did not create ${BODY_FILE}" >&2
  exit 1
fi
SZ="$(stat -c '%s' "${BODY_FILE}" 2>/dev/null || wc -c < "${BODY_FILE}")"
log "pandoc body finished: ${BODY_FILE} (${SZ} bytes)"

log "stripping Word heading list numbering (template Heading1/2 multilevel conflict)"
STRIP_FLAGS=()
if [[ "${BUILD_DOCX_VERBOSE:-}" == "1" ]]; then
  STRIP_FLAGS=(-v)
fi
python3 "${STRIP_HEADING_NUM}" "${STRIP_FLAGS[@]}" "${BODY_FILE}"

log "applying DOCX layout polish (code boxes, fitted title)"
python3 "${POLISH_LAYOUT}" "${BODY_FILE}"

log "prepending cover ${COVER_FILE} → ${OUT_FILE}"
python3 "${PREPEND_COVER}" "${COVER_FILE}" "${BODY_FILE}" "${OUT_FILE}"

SZ2="$(stat -c '%s' "${OUT_FILE}" 2>/dev/null || wc -c < "${OUT_FILE}")"
log "final: ${OUT_FILE} (${SZ2} bytes)"

cp -f "${OUT_FILE}" "${LOCAL_OUT_FILE}"
SZ3="$(stat -c '%s' "${LOCAL_OUT_FILE}" 2>/dev/null || wc -c < "${LOCAL_OUT_FILE}")"
log "local build copy: ${LOCAL_OUT_FILE} (${SZ3} bytes)"



if [[ -n "${RCLONE_REMOTE:-}" ]]; then
  if ! command -v rclone >/dev/null 2>&1; then
    log "warning: RCLONE_REMOTE is set but rclone not in PATH; skipping upload"
  else
    # Remote name only (e.g. mydrive), no trailing colon - we add colon below.
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
echo "Wrote: ${LOCAL_OUT_FILE}"
