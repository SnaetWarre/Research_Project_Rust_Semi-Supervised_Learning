#!/usr/bin/env python3
"""Apply final Word layout fixes to the generated thesis .docx.

This script edits only the generated output file. It keeps the Markdown source
clean while handling Word-specific layout details that Pandoc cannot express
reliably:

- put SourceCode paragraphs in visible boxed snippets;
- center figure images and their captions;
- split image and caption content into separate Word paragraphs;
- keep headings, table captions and figures with the following paragraph;
- keep key-finding labels with the first finding;
- make generated tables use explicit cell widths for stable DOCX import;
- keep non-numbered front matter and appendix headings out of the native TOC;
- make the long thesis title fit on one centered line.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import zipfile
from copy import deepcopy
from io import BytesIO
import xml.etree.ElementTree as ET

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
DOCUMENT_PATH = "word/document.xml"
STYLES_PATH = "word/styles.xml"

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


def _set_paragraph_style(p: ET.Element, style_id: str) -> None:
    ppr = _ensure_ppr(p)
    style = _child(ppr, "pStyle")
    if style is None:
        style = ET.Element(_w("pStyle"))
        ppr.insert(0, style)
    style.set(_attr("val"), style_id)


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


def _is_numbered_toc_heading_text(text: str) -> bool:
    return re.match(r"^\d+(?:\.\d+)?\.?\s+", text) is not None


def _is_subheading_style(style: str | None) -> bool:
    if style is None:
        return False
    return re.match(r"^Heading[2-9](?:Unlisted)?$", style) is not None


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


def _stabilize_table_width(tbl: ET.Element) -> bool:
    cols = _table_column_count(tbl)
    if cols <= 0:
        return False

    _set_full_width_table(tbl)

    total_width = 9000
    col_width = total_width // cols

    grid = _child(tbl, "tblGrid")
    if grid is None:
        grid = ET.Element(_w("tblGrid"))
        insert_at = 1 if _child(tbl, "tblPr") is not None else 0
        tbl.insert(insert_at, grid)
    for child in list(grid):
        grid.remove(child)
    for _ in range(cols):
        grid_col = ET.SubElement(grid, _w("gridCol"))
        grid_col.set(_attr("w"), str(col_width))

    for row in _table_rows(tbl):
        for cell in row.findall(_w("tc")):
            tc_pr = _child(cell, "tcPr")
            if tc_pr is None:
                tc_pr = ET.Element(_w("tcPr"))
                cell.insert(0, tc_pr)
            tc_w = ET.Element(_w("tcW"))
            tc_w.set(_attr("w"), str(col_width))
            tc_w.set(_attr("type"), "dxa")
            _set_or_replace(tc_pr, tc_w)

    return True


def _set_row_cannot_split(row: ET.Element) -> None:
    tr_pr = _child(row, "trPr")
    if tr_pr is None:
        tr_pr = ET.Element(_w("trPr"))
        row.insert(0, tr_pr)

    cant_split = ET.Element(_w("cantSplit"))
    _set_or_replace(tr_pr, cant_split)


def _remove_table_keep_next(tbl: ET.Element) -> None:
    for ppr in tbl.findall(f".//{_w('pPr')}"):
        keep_next = _child(ppr, "keepNext")
        if keep_next is not None:
            ppr.remove(keep_next)


def _remove_table_keep_next_from_all(root: ET.Element) -> None:
    for tbl in root.findall(f".//{_w('tbl')}"):
        _remove_table_keep_next(tbl)


def _keep_table_together(tbl: ET.Element) -> int:
    rows = _table_rows(tbl)
    if len(rows) <= 1:
        return 0

    kept = 0
    for row in rows:
        paragraphs = _table_cell_paragraphs(row)
        for p in paragraphs:
            _keep_lines_together(p)
    return kept


def _keep_result_table_captions_on_fresh_page(root: ET.Element) -> int:
    body = root.find(f".//{_w('body')}")
    if body is None:
        return 0

    moved = 0
    result_table_numbers = {"3.3", "3.4", "3.5"}
    offset = 0
    children = list(body)
    for index, child in enumerate(children):
        if child.tag != _w("p"):
            continue
        text = _paragraph_text(child).strip()
        match = re.match(r"^Table\s+(\d+\.\d+):", text)
        if match and match.group(1) in result_table_numbers:
            insert_at = index + offset
            current_children = list(body)
            if insert_at == 0 or not _is_page_break_paragraph(current_children[insert_at - 1]):
                body.insert(insert_at, _page_break_paragraph())
                offset += 1
            _keep_with_next(child)
            moved += 1
    return moved


def _is_block_boundary(child: ET.Element) -> bool:
    if child.tag != _w("p"):
        return child.tag == _w("tbl")

    style = _paragraph_style(child)
    text = _paragraph_text(child).strip()
    if text == "":
        return True
    if (style or "").startswith("Heading"):
        return True
    if _is_table_caption_text(text) or _is_caption_text(text):
        return True
    return False


def _apply_key_findings_layout(root: ET.Element) -> tuple[int, int]:
    body = root.find(f".//{_w('body')}")
    if body is None:
        return 0, 0

    blocks = 0
    paragraphs = 0
    children = list(body)
    for index, child in enumerate(children):
        if child.tag != _w("p") or _paragraph_text(child).strip() != "Key findings:":
            continue

        blocks += 1
        _keep_with_next(child)
        _keep_lines_together(child)
        paragraphs += 1

        follower = next((item for item in children[index + 1 :] if item.tag == _w("p")), None)
        if follower is not None and not _is_block_boundary(follower):
            _keep_with_next(follower)
            _keep_lines_together(follower)
            paragraphs += 1

    return blocks, paragraphs


def _is_content_chapter_heading(text: str) -> bool:
    return re.match(r"^[1-6]\.\s+", text) is not None


def _detoc_non_numbered_heading(style: str | None, text: str, p: ET.Element) -> bool:
    if style not in {"Heading1", "Heading2"}:
        return False
    if _is_numbered_toc_heading_text(text):
        return False

    _set_paragraph_style(p, f"{style}Unlisted")
    return True


def _apply_content_table_layout(root: ET.Element) -> int:
    body = root.find(f".//{_w('body')}")
    if body is None:
        return 0

    in_content_chapter = False
    widened_tables = 0

    for child in body:
        if child.tag == _w("p"):
            style = _paragraph_style(child)
            text = _paragraph_text(child).strip()
            if style == "Heading1":
                in_content_chapter = _is_content_chapter_heading(text)
            continue

        if child.tag != _w("tbl") or not in_content_chapter:
            continue

        _remove_table_keep_next(child)
        _set_full_width_table(child)
        widened_tables += 1

    return widened_tables


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


def _page_break_paragraph() -> ET.Element:
    p = ET.Element(_w("p"))
    r = ET.SubElement(p, _w("r"))
    br = ET.SubElement(r, _w("br"))
    br.set(_attr("type"), "page")
    return p


def _is_page_break_paragraph(p: ET.Element) -> bool:
    return p.tag == _w("p") and p.find(f".//{_w('br')}") is not None


def _is_toc_block(el: ET.Element) -> bool:
    return el.find(f".//{_w('instrText')}") is not None and any(
        "TOC" in (node.text or "") for node in el.findall(f".//{_w('instrText')}")
    )


def _move_toc_after_abstract(root: ET.Element) -> bool:
    body = root.find(f".//{_w('body')}")
    if body is None:
        return False

    children = list(body)
    toc = next((child for child in children if _is_toc_block(child)), None)
    if toc is None:
        return False

    body.remove(toc)
    children = list(body)

    list_of_figures_index = next(
        (
            index
            for index, child in enumerate(children)
            if child.tag == _w("p") and _paragraph_text(child).strip() == "List of Figures"
        ),
        None,
    )
    if list_of_figures_index is None:
        return False

    insert_index = list_of_figures_index
    if list_of_figures_index > 0 and _is_page_break_paragraph(children[list_of_figures_index - 1]):
        insert_index = list_of_figures_index - 1
        body.insert(insert_index, _page_break_paragraph())
        body.insert(insert_index + 1, toc)
    else:
        body.insert(insert_index, _page_break_paragraph())
        body.insert(insert_index + 1, toc)
        body.insert(insert_index + 2, _page_break_paragraph())
    return True


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


def _remove_style_child(style: ET.Element, child_name: str) -> None:
    child = _child(style, child_name)
    if child is not None:
        style.remove(child)


def _remove_nested_style_child(style: ET.Element, parent_name: str, child_name: str) -> None:
    parent = _child(style, parent_name)
    if parent is None:
        return
    child = _child(parent, child_name)
    if child is not None:
        parent.remove(child)


def _ensure_unlisted_heading_style(styles_root: ET.Element, source_style_id: str) -> bool:
    target_style_id = f"{source_style_id}Unlisted"
    existing = styles_root.find(f".//{_w('style')}[@{_attr('styleId')}='{target_style_id}']")
    if existing is not None:
        return False

    source = styles_root.find(f".//{_w('style')}[@{_attr('styleId')}='{source_style_id}']")
    if source is None:
        return False

    style = deepcopy(source)
    style.set(_attr("styleId"), target_style_id)

    name = _child(style, "name")
    if name is not None:
        name.set(_attr("val"), f"{source_style_id} unlisted")

    _remove_style_child(style, "link")
    _remove_style_child(style, "qFormat")
    _remove_nested_style_child(style, "pPr", "outlineLvl")

    styles_root.append(style)
    return True


def _polish_styles(xml_bytes: bytes) -> tuple[bytes, int]:
    root = ET.fromstring(xml_bytes)
    created = 0
    for style_id in ("Heading1", "Heading2"):
        if _ensure_unlisted_heading_style(root, style_id):
            created += 1
    return ET.tostring(root, encoding="utf-8", xml_declaration=True), created


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


def _polish_document(xml_bytes: bytes) -> tuple[bytes, int, int, int, int, int, int, int, int, bool, bool]:
    root = ET.fromstring(xml_bytes)
    moved_toc = _move_toc_after_abstract(root)
    split_captions = _split_inline_figure_captions(root)
    boxed_code_blocks = 0
    centered_figures = 0
    keep_next_paragraphs = 0
    stabilized_tables = 0
    detoc_headings = 0
    title_fitted = False
    _remove_table_keep_next_from_all(root)
    key_finding_blocks, key_finding_paragraphs = _apply_key_findings_layout(root)

    for tbl in root.findall(f".//{_w('tbl')}"):
        if _stabilize_table_width(tbl):
            stabilized_tables += 1

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

        if _is_subheading_style(style):
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

        if _detoc_non_numbered_heading(style, text, p):
            detoc_headings += 1

    out = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return (
        out,
        boxed_code_blocks,
        centered_figures,
        split_captions,
        keep_next_paragraphs,
        stabilized_tables,
        key_finding_blocks,
        key_finding_paragraphs,
        detoc_headings,
        title_fitted,
        moved_toc,
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

        styles_xml = None
        created_unlisted_styles = 0
        if STYLES_PATH in zin.namelist():
            styles_xml, created_unlisted_styles = _polish_styles(zin.read(STYLES_PATH))

        (
            document_xml,
            boxed_code_blocks,
            centered_figures,
            split_captions,
            keep_next_paragraphs,
            stabilized_tables,
            key_finding_blocks,
            key_finding_paragraphs,
            detoc_headings,
            title_fitted,
            moved_toc,
        ) = _polish_document(
            zin.read(DOCUMENT_PATH)
        )

        with zipfile.ZipFile(buf, "w") as zout:
            for info in zin.infolist():
                if info.filename == DOCUMENT_PATH:
                    data = document_xml
                elif info.filename == STYLES_PATH and styles_xml is not None:
                    data = styles_xml
                else:
                    data = zin.read(info.filename)
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
    _log(f"[polish_docx_layout] stabilized table widths: {stabilized_tables}", quiet=quiet)
    _log(f"[polish_docx_layout] key-finding blocks: {key_finding_blocks}", quiet=quiet)
    _log(
        f"[polish_docx_layout] key-finding keep-with-next paragraphs: {key_finding_paragraphs}",
        quiet=quiet,
    )
    _log(f"[polish_docx_layout] created unlisted heading styles: {created_unlisted_styles}", quiet=quiet)
    _log(f"[polish_docx_layout] non-TOC heading paragraphs: {detoc_headings}", quiet=quiet)
    _log(f"[polish_docx_layout] fitted title paragraph: {title_fitted}", quiet=quiet)
    _log(f"[polish_docx_layout] moved TOC after abstract: {moved_toc}", quiet=quiet)
    _log("[polish_docx_layout] done (in-place update)", quiet=quiet)


if __name__ == "__main__":
    main()
