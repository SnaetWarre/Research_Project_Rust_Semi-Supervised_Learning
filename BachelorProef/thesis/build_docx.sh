#!/usr/bin/env bash
# Build the full bachelor thesis as Word (.docx) from Markdown sources.
# Requires: pandoc (pacman -S pandoc)
#
# Uses the Howest MCT template styles via --reference-doc (keeps Heading 1/2, Normal, etc.).
# Run from anywhere; paths are resolved relative to this script.
#
# Layout polish (see pandoc/docx_polish.lua): page breaks before each main chapter heading,
# sensible default image width. No auto-TOC here (it tends to land wrong and fight pagination);
# in Word: place cursor after front matter → References → Table of Contents → Custom.
#
# Optional: PANDOC_TOC=1 ./build_docx.sh  — inserts Pandoc’s TOC at the top (quick draft only).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/../template/Template_Bachelorproef_MCT.docx"
OUT_DIR="${SCRIPT_DIR}/build"
OUT_FILE="${OUT_DIR}/Bachelorproef_Snaet_2026.docx"
LUA_FILTER="${SCRIPT_DIR}/pandoc/docx_polish.lua"

# Repo root (plantvillage_ssl/... paths in Markdown resolve from BachelorProef/thesis)
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

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
fi

# Title page + abstract only (drop the Markdown “Table of Contents”; Word can add a real TOC).
TITLE_FRONT="$(mktemp)"
trap 'rm -f "${TITLE_FRONT}"' EXIT
head -n 29 "${SCRIPT_DIR}/00_title_and_abstract.md" > "${TITLE_FRONT}"

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

echo "Wrote: ${OUT_FILE}"
