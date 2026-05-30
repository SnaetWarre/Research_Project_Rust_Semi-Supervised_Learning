#!/usr/bin/env python3
"""Remove multilevel list numbering from Word heading styles in a .docx.

The Howest MCT reference template links Heading 1/2/4–9 to numbering definition1.
That makes Word print automatic numbers (1., 2., … 9.) in front of heading text that
already contains manual chapter numbers (e.g. \"3. Research Results\").

This script edits only the generated output file (not the template): it drops w:numPr
from the affected heading styles inside word/styles.xml.

Usage:
  strip_word_heading_list_numbering.py [-q] [-v] <file.docx>

  -q, --quiet   only errors
  -v, --verbose per-style details and byte sizes
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from io import BytesIO
import xml.etree.ElementTree as ET

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
STYLES_PATH = "word/styles.xml"

STRIP_STYLE_IDS = frozenset(
    {
        "Heading1",
        "Heading2",
        "Heading4",
        "Heading5",
        "Heading6",
        "Heading7",
        "Heading8",
        "Heading9",
    }
)


def _strip_numpr_from_styles(xml_bytes: bytes) -> tuple[bytes, list[str]]:
    ET.register_namespace("w", W_NS)
    root = ET.fromstring(xml_bytes)
    removed_from: list[str] = []

    for style in root.findall(f".//{{{W_NS}}}style"):
        sid = style.get(f"{{{W_NS}}}styleId")
        if sid not in STRIP_STYLE_IDS:
            continue
        ppr = style.find(f"{{{W_NS}}}pPr")
        if ppr is None:
            continue
        numpr = ppr.find(f"{{{W_NS}}}numPr")
        if numpr is not None:
            ppr.remove(numpr)
            if sid:
                removed_from.append(sid)

    out = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return out, removed_from


def _log(msg: str, *, quiet: bool) -> None:
    if not quiet:
        print(msg, file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("docx", help="Path to .docx (modified in place)")
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress informational logs")
    parser.add_argument("-v", "--verbose", action="store_true", help="extra detail")
    args = parser.parse_args()

    docx_path = args.docx
    quiet = args.quiet
    verbose = args.verbose

    if not os.path.isfile(docx_path):
        print(f"error: not a file: {docx_path}", file=sys.stderr)
        sys.exit(1)

    size_before = os.path.getsize(docx_path)
    _log(f"[strip_heading_num] input: {docx_path}", quiet=quiet)
    _log(f"[strip_heading_num] size before: {size_before:,} bytes", quiet=quiet)

    buf = BytesIO()
    with zipfile.ZipFile(docx_path, "r") as zin:
        names = zin.namelist()
        if STYLES_PATH not in names:
            print(f"[strip_heading_num] skip: {STYLES_PATH} missing in archive", file=sys.stderr)
            return
        raw_styles = zin.read(STYLES_PATH)
        if verbose:
            _log(f"[strip_heading_num] read {STYLES_PATH}: {len(raw_styles):,} bytes", quiet=quiet)
        new_styles, removed = _strip_numpr_from_styles(raw_styles)
        if not removed:
            _log(
                "[strip_heading_num] no w:numPr found on targeted heading styles "
                f"({', '.join(sorted(STRIP_STYLE_IDS))}); template may already be clean or differ",
                quiet=quiet,
            )
        else:
            _log(
                "[strip_heading_num] removed list numbering (w:numPr) from styles: "
                + ", ".join(removed),
                quiet=quiet,
            )
            if verbose:
                for sid in removed:
                    _log(f"[strip_heading_num]   - {sid}", quiet=quiet)

        n_members = len(zin.infolist())
        with zipfile.ZipFile(buf, "w") as zout:
            for info in zin.infolist():
                data = zin.read(info.filename)
                if info.filename == STYLES_PATH:
                    data = new_styles
                zout.writestr(info, data)
        if verbose:
            _log(f"[strip_heading_num] repacked archive members: {n_members}", quiet=quiet)

    out_bytes = buf.getvalue()
    with open(docx_path, "wb") as f:
        f.write(out_bytes)

    size_after = len(out_bytes)
    _log(f"[strip_heading_num] size after:  {size_after:,} bytes", quiet=quiet)
    delta = size_after - size_before
    if delta != 0 and verbose:
        _log(f"[strip_heading_num] delta: {delta:+,} bytes (XML rewrite)", quiet=quiet)
    _log("[strip_heading_num] done (in-place update)", quiet=quiet)


if __name__ == "__main__":
    main()
