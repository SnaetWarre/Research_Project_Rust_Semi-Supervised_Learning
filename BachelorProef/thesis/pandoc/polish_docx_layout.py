#!/usr/bin/env python3
"""Apply final Word layout fixes to the generated thesis .docx.

This script edits only the generated output file. It keeps the Markdown source
clean while handling Word-specific layout details that Pandoc cannot express
reliably:

- put SourceCode paragraphs in visible boxed snippets;
- center figure images and their captions;
- split image and caption content into separate Word paragraphs;
- keep headings, table captions and figures with the following paragraph;
- add subtle internal grid lines to generated Word tables;
- keep regular chapter tables on one page when they fit;
- make regular chapter tables use the full printable width;
- make the long thesis title fit on one centered line.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import zipfile
from io import BytesIO
import xml.etree.ElementTree as ET

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
DOCUMENT_PATH = "word/document.xml"

ET.register_namespace("w", W_NS)


def _w(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def _attr(name: str) -> str:
    return f"{{{W_NS}}}{name}"


def _child(parent: ET.Element, tag: str) -> ET.Element | None:
    return parent.find(_w(tag))


def _ensure_child(parent: ET.Element, tag: str, index: int | None = None) -> ET.Element:
    child = _child(parent, tag)
    if child is None:
        child = ET.Element(_w(tag))
        if index is None:
            parent.append(child)
        else:
            parent.insert(index, child)
    return child


def _paragraph_text(p: ET.Element) -> str:
    return "".join(t.text or "" for t in p.findall(f".//{_w('t')}"))


def _paragraph_style(p: ET.Element) -> str | None:
    ppr = _child(p, "pPr")
    if ppr is None:
        return None
    style = _child(ppr, "pStyle")
    if style is None:
        return None
    return style.get(_attr("val"))


def _ensure_ppr(p: ET.Element) -> ET.Element:
    ppr = _child(p, "pPr")
    if ppr is None:
        ppr = ET.Element(_w("pPr"))
        p.insert(0, ppr)
    return ppr


def _ensure_tbl_pr(tbl: ET.Element) -> ET.Element:
    tbl_pr = _child(tbl, "tblPr")
    if tbl_pr is None:
        tbl_pr = ET.Element(_w("tblPr"))
        tbl.insert(0, tbl_pr)
    return tbl_pr


def _set_or_replace(parent: ET.Element, child: ET.Element) -> None:
    existing = _child(parent, child.tag.rsplit("}", 1)[-1])
    if existing is not None:
        parent.remove(existing)
    parent.append(child)


def _is_caption_text(text: str) -> bool:
    return (text.startswith("Figure ") or text.startswith("Appendix Figure ")) and ":" in text


def _is_table_caption_text(text: str) -> bool:
    return text.startswith("Table ") and ":" in text


def _keep_with_next(p: ET.Element) -> None:
    ppr = _ensure_ppr(p)
    keep_next = ET.Element(_w("keepNext"))
    _set_or_replace(ppr, keep_next)


def _keep_lines_together(p: ET.Element) -> None:
    ppr = _ensure_ppr(p)
    keep_lines = ET.Element(_w("keepLines"))
    _set_or_replace(ppr, keep_lines)


def _table_column_count(tbl: ET.Element) -> int:
    grid_cols = tbl.findall(f"./{_w('tblGrid')}/{_w('gridCol')}")
    if grid_cols:
        return len(grid_cols)

    first_row = tbl.find(_w("tr"))
    if first_row is None:
        return 0
    return len(first_row.findall(_w("tc")))


def _table_row_count(tbl: ET.Element) -> int:
    return len(tbl.findall(_w("tr")))


def _table_rows(tbl: ET.Element) -> list[ET.Element]:
    return tbl.findall(_w("tr"))


def _table_cell_paragraphs(row: ET.Element) -> list[ET.Element]:
    return row.findall(f".//{_w('tc')}/{_w('p')}")


def _set_full_width_table(tbl: ET.Element) -> None:
    tbl_pr = _ensure_tbl_pr(tbl)

    tbl_w = ET.Element(_w("tblW"))
    tbl_w.set(_attr("w"), "5000")
    tbl_w.set(_attr("type"), "pct")
    _set_or_replace(tbl_pr, tbl_w)

    jc = ET.Element(_w("jc"))
    jc.set(_attr("val"), "center")
    _set_or_replace(tbl_pr, jc)


def _set_row_cannot_split(row: ET.Element) -> None:
    tr_pr = _child(row, "trPr")
    if tr_pr is None:
        tr_pr = ET.Element(_w("trPr"))
        row.insert(0, tr_pr)

    cant_split = ET.Element(_w("cantSplit"))
    _set_or_replace(tr_pr, cant_split)


def _keep_table_together(tbl: ET.Element) -> int:
    rows = _table_rows(tbl)
    if len(rows) <= 1:
        return 0

    kept = 0
    for row_index, row in enumerate(rows):
        _set_row_cannot_split(row)
        paragraphs = _table_cell_paragraphs(row)
        for p in paragraphs:
            _keep_lines_together(p)
            if row_index < len(rows) - 1:
                _keep_with_next(p)
                kept += 1
    return kept


def _is_content_chapter_heading(text: str) -> bool:
    return re.match(r"^[1-6]\.\s+", text) is not None


def _apply_content_table_layout(root: ET.Element) -> tuple[int, int, int]:
    body = root.find(f".//{_w('body')}")
    if body is None:
        return 0, 0, 0

    in_content_chapter = False
    widened_tables = 0
    kept_tables = 0
    kept_cell_paragraphs = 0

    for child in body:
        if child.tag == _w("p"):
            style = _paragraph_style(child)
            text = _paragraph_text(child).strip()
            if style == "Heading1":
                in_content_chapter = _is_content_chapter_heading(text)
            continue

        if child.tag != _w("tbl") or not in_content_chapter:
            continue

        rows = _table_row_count(child)
        # Large lists are allowed to flow over pages. Regular content tables in
        # the thesis chapters are short enough to keep together.
        if rows > 12:
            continue

        _set_full_width_table(child)
        widened_tables += 1
        kept_cell_paragraphs += _keep_table_together(child)
        kept_tables += 1

    return widened_tables, kept_tables, kept_cell_paragraphs


def _apply_internal_table_grid(tbl: ET.Element) -> bool:
    if _table_row_count(tbl) < 2 and _table_column_count(tbl) < 2:
        return False

    tbl_pr = _ensure_tbl_pr(tbl)
    borders = ET.Element(_w("tblBorders"))

    # Keep the outside edge clean and only draw the internal grid.
    for side in ("top", "left", "bottom", "right"):
        border = ET.SubElement(borders, _w(side))
        border.set(_attr("val"), "nil")

    for side in ("insideH", "insideV"):
        border = ET.SubElement(borders, _w(side))
        border.set(_attr("val"), "single")
        border.set(_attr("sz"), "4")
        border.set(_attr("space"), "0")
        border.set(_attr("color"), "D9DEE7")

    _set_or_replace(tbl_pr, borders)
    return True


def _box_source_code_paragraph(p: ET.Element) -> None:
    ppr = _ensure_ppr(p)

    p_bdr = ET.Element(_w("pBdr"))
    for side in ("top", "left", "bottom", "right"):
        border = ET.SubElement(p_bdr, _w(side))
        border.set(_attr("val"), "single")
        border.set(_attr("sz"), "8")
        border.set(_attr("space"), "4")
        border.set(_attr("color"), "B7C0CC")
    _set_or_replace(ppr, p_bdr)

    shd = ET.Element(_w("shd"))
    shd.set(_attr("val"), "clear")
    shd.set(_attr("color"), "auto")
    shd.set(_attr("fill"), "F5F7FA")
    _set_or_replace(ppr, shd)

    spacing = ET.Element(_w("spacing"))
    spacing.set(_attr("before"), "120")
    spacing.set(_attr("after"), "120")
    _set_or_replace(ppr, spacing)

    ind = ET.Element(_w("ind"))
    ind.set(_attr("left"), "180")
    ind.set(_attr("right"), "180")
    _set_or_replace(ppr, ind)


def _center_paragraph(p: ET.Element) -> None:
    ppr = _ensure_ppr(p)
    jc = ET.Element(_w("jc"))
    jc.set(_attr("val"), "center")
    _set_or_replace(ppr, jc)


def _is_figure_paragraph(p: ET.Element, text: str) -> bool:
    has_drawing = p.find(f".//{_w('drawing')}") is not None
    has_figure_caption = text.startswith("Figure ") or text.startswith("Appendix Figure ")
    return has_drawing or has_figure_caption


def _copy_paragraph_properties(p: ET.Element) -> ET.Element:
    ppr = _child(p, "pPr")
    if ppr is not None:
        return ET.fromstring(ET.tostring(ppr, encoding="utf-8"))

    copied = ET.Element(_w("pPr"))
    style = ET.SubElement(copied, _w("pStyle"))
    style.set(_attr("val"), "BodyText")
    return copied


def _split_image_caption_paragraph(parent: ET.Element, index: int, p: ET.Element) -> bool:
    if p.find(f".//{_w('drawing')}") is None:
        return False

    runs = list(p.findall(_w("r")))
    caption_start = None
    for run_index, run in enumerate(runs):
        text = "".join(t.text or "" for t in run.findall(f".//{_w('t')}")).strip()
        if _is_caption_text(text):
            caption_start = run_index
            break

    if caption_start is None:
        return False

    caption_runs = runs[caption_start:]
    for run in caption_runs:
        p.remove(run)

    # Remove whitespace-only runs left after the drawing so the image paragraph
    # contains only the image. This prevents Word from treating the caption as
    # trailing inline content beside the figure.
    for run in list(p.findall(_w("r"))):
        if run.find(f".//{_w('drawing')}") is not None:
            continue
        text = "".join(t.text or "" for t in run.findall(f".//{_w('t')}"))
        if text.strip() == "":
            p.remove(run)

    caption_p = ET.Element(_w("p"))
    caption_p.append(_copy_paragraph_properties(p))
    for run in caption_runs:
        caption_p.append(run)
    _center_paragraph(caption_p)
    _keep_with_next(p)

    parent.insert(index + 1, caption_p)
    return True


def _fit_title_paragraph(p: ET.Element) -> None:
    ppr = _ensure_ppr(p)

    jc = ET.Element(_w("jc"))
    jc.set(_attr("val"), "center")
    _set_or_replace(ppr, jc)

    keep_lines = ET.Element(_w("keepLines"))
    _set_or_replace(ppr, keep_lines)

    spacing = ET.Element(_w("spacing"))
    spacing.set(_attr("before"), "0")
    spacing.set(_attr("after"), "240")
    _set_or_replace(ppr, spacing)

    # Fit the official long research-question title to the printable width.
    # 9000 twips is roughly the available line width in the Howest template.
    for run in p.findall(_w("r")):
        rpr = _child(run, "rPr")
        if rpr is None:
            rpr = ET.Element(_w("rPr"))
            run.insert(0, rpr)

        fit_text = ET.Element(_w("fitText"))
        fit_text.set(_attr("val"), "9000")
        _set_or_replace(rpr, fit_text)

        sz = ET.Element(_w("sz"))
        sz.set(_attr("val"), "28")
        _set_or_replace(rpr, sz)

        sz_cs = ET.Element(_w("szCs"))
        sz_cs.set(_attr("val"), "28")
        _set_or_replace(rpr, sz_cs)


def _split_inline_figure_captions(root: ET.Element) -> int:
    split_count = 0
    for parent in root.iter():
        children = list(parent)
        inserted_for_parent = 0
        for index, child in enumerate(children):
            if child.tag != _w("p"):
                continue
            if _split_image_caption_paragraph(parent, index + inserted_for_parent, child):
                split_count += 1
                inserted_for_parent += 1
    return split_count


def _polish_document(xml_bytes: bytes) -> tuple[bytes, int, int, int, int, int, int, int, int, bool]:
    root = ET.fromstring(xml_bytes)
    split_captions = _split_inline_figure_captions(root)
    boxed_code_blocks = 0
    centered_figures = 0
    keep_next_paragraphs = 0
    gridded_tables = 0
    title_fitted = False
    widened_tables, kept_tables, kept_table_cell_paragraphs = _apply_content_table_layout(root)

    for tbl in root.findall(f".//{_w('tbl')}"):
        if _apply_internal_table_grid(tbl):
            gridded_tables += 1

    for p in root.findall(f".//{_w('p')}"):
        style = _paragraph_style(p)
        text = _paragraph_text(p).strip()

        if style == "SourceCode":
            _box_source_code_paragraph(p)
            boxed_code_blocks += 1

        if _is_figure_paragraph(p, text):
            _center_paragraph(p)
            centered_figures += 1
            if p.find(f".//{_w('drawing')}") is not None:
                _keep_with_next(p)
                keep_next_paragraphs += 1

        if (style or "").startswith("Heading") and not style == "Heading1":
            _keep_with_next(p)
            _keep_lines_together(p)
            keep_next_paragraphs += 1

        if _is_table_caption_text(text):
            _keep_with_next(p)
            keep_next_paragraphs += 1

        if (
            not title_fitted
            and style == "Heading1"
            and text.startswith("How Can a Semi-Supervised Neural Network")
        ):
            _fit_title_paragraph(p)
            title_fitted = True

    out = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return (
        out,
        boxed_code_blocks,
        centered_figures,
        split_captions,
        keep_next_paragraphs,
        gridded_tables,
        widened_tables,
        kept_tables,
        kept_table_cell_paragraphs,
        title_fitted,
    )


def _log(msg: str, *, quiet: bool) -> None:
    if not quiet:
        print(msg, file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("docx", help="Path to .docx (modified in place)")
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress informational logs")
    args = parser.parse_args()

    docx_path = args.docx
    quiet = args.quiet

    if not os.path.isfile(docx_path):
        print(f"error: not a file: {docx_path}", file=sys.stderr)
        sys.exit(1)

    _log(f"[polish_docx_layout] input: {docx_path}", quiet=quiet)

    buf = BytesIO()
    with zipfile.ZipFile(docx_path, "r") as zin:
        if DOCUMENT_PATH not in zin.namelist():
            print(f"[polish_docx_layout] skip: {DOCUMENT_PATH} missing in archive", file=sys.stderr)
            return

        (
            document_xml,
            boxed_code_blocks,
            centered_figures,
            split_captions,
            keep_next_paragraphs,
            gridded_tables,
            widened_tables,
            kept_tables,
            kept_table_cell_paragraphs,
            title_fitted,
        ) = _polish_document(
            zin.read(DOCUMENT_PATH)
        )

        with zipfile.ZipFile(buf, "w") as zout:
            for info in zin.infolist():
                data = document_xml if info.filename == DOCUMENT_PATH else zin.read(info.filename)
                zout.writestr(info, data)

    with open(docx_path, "wb") as f:
        f.write(buf.getvalue())

    _log(
        f"[polish_docx_layout] boxed SourceCode paragraphs: {boxed_code_blocks}",
        quiet=quiet,
    )
    _log(f"[polish_docx_layout] centered figure paragraphs: {centered_figures}", quiet=quiet)
    _log(f"[polish_docx_layout] split inline figure captions: {split_captions}", quiet=quiet)
    _log(f"[polish_docx_layout] keep-with-next paragraphs: {keep_next_paragraphs}", quiet=quiet)
    _log(f"[polish_docx_layout] internal-grid tables: {gridded_tables}", quiet=quiet)
    _log(f"[polish_docx_layout] full-width content tables: {widened_tables}", quiet=quiet)
    _log(f"[polish_docx_layout] keep-together content tables: {kept_tables}", quiet=quiet)
    _log(
        f"[polish_docx_layout] keep-together table cell paragraphs: {kept_table_cell_paragraphs}",
        quiet=quiet,
    )
    _log(f"[polish_docx_layout] fitted title paragraph: {title_fitted}", quiet=quiet)
    _log("[polish_docx_layout] done (in-place update)", quiet=quiet)


if __name__ == "__main__":
    main()
