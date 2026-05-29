#!/usr/bin/env python3
"""Prepend a prepared cover DOCX as the first page of a thesis DOCX.

DOCX files do not store "pages" as independent objects. Word and LibreOffice
calculate pages from document XML at open/export time, so raw XML concatenation
can reflow a carefully designed cover. To keep the prepared cover visually
exact, this helper renders the first page of the cover DOCX and inserts it as a
full-page cover section before the generated thesis body.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "ct": "http://schemas.openxmlformats.org/package/2006/content-types",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
}

for prefix in ("a", "pic", "r", "wp", "w"):
    ET.register_namespace(prefix, NS[prefix])


A4_WIDTH_EMU = 7_560_000
A4_HEIGHT_EMU = 10_692_000
A4_WIDTH_TWIPS = 11_906
A4_HEIGHT_TWIPS = 16_838


def qn(prefix: str, tag: str) -> str:
    return f"{{{NS[prefix]}}}{tag}"


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required tool not found in PATH: {name}")


def render_cover_page(cover_docx: Path, workdir: Path) -> Path:
    require_tool("libreoffice")
    require_tool("pdftoppm")

    pdf_dir = workdir / "cover_pdf"
    pdf_dir.mkdir()
    subprocess.run(
        [
            "libreoffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(pdf_dir),
            str(cover_docx),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    pdf_path = pdf_dir / f"{cover_docx.stem}.pdf"
    if not pdf_path.exists():
        matches = sorted(pdf_dir.glob("*.pdf"))
        if not matches:
            raise RuntimeError(f"LibreOffice did not produce a PDF for {cover_docx}")
        pdf_path = matches[0]

    png_prefix = workdir / "cover_page"
    subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-singlefile",
            "-f",
            "1",
            "-l",
            "1",
            "-r",
            "300",
            str(pdf_path),
            str(png_prefix),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    png_path = workdir / "cover_page.png"
    if not png_path.exists():
        raise RuntimeError(f"pdftoppm did not produce {png_path}")
    return png_path


def next_relationship_id(rels_root: ET.Element) -> str:
    used = {
        int(rel.attrib["Id"][3:])
        for rel in rels_root.findall(qn("pr", "Relationship"))
        if rel.attrib.get("Id", "").startswith("rId") and rel.attrib["Id"][3:].isdigit()
    }
    value = 1
    while value in used:
        value += 1
    return f"rId{value}"


def next_doc_pr_id(root: ET.Element) -> str:
    used = {
        int(node.attrib["id"])
        for node in root.findall(f".//{qn('wp', 'docPr')}")
        if node.attrib.get("id", "").isdigit()
    }
    value = 1
    while value in used:
        value += 1
    return str(value)


def ensure_png_content_type(content_types_path: Path) -> None:
    tree = ET.parse(content_types_path)
    root = tree.getroot()
    for default in root.findall(qn("ct", "Default")):
        if default.attrib.get("Extension", "").lower() == "png":
            return

    node = ET.SubElement(root, qn("ct", "Default"))
    node.set("Extension", "png")
    node.set("ContentType", "image/png")

    ET.register_namespace("", NS["ct"])
    tree.write(content_types_path, encoding="UTF-8", xml_declaration=True)
    ET.register_namespace("", NS["pr"])


def add_cover_image_relationship(workdir: Path, cover_png: Path) -> str:
    media_dir = workdir / "word" / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    target_name = "cover_page.png"
    shutil.copyfile(cover_png, media_dir / target_name)

    rels_path = workdir / "word" / "_rels" / "document.xml.rels"
    rels_tree = ET.parse(rels_path)
    rels_root = rels_tree.getroot()
    rel_id = next_relationship_id(rels_root)

    rel = ET.SubElement(rels_root, qn("pr", "Relationship"))
    rel.set("Id", rel_id)
    rel.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image")
    rel.set("Target", f"media/{target_name}")

    ET.register_namespace("", NS["pr"])
    rels_tree.write(rels_path, encoding="UTF-8", xml_declaration=True)
    return rel_id


def cover_paragraph(rel_id: str, doc_pr_id: str) -> ET.Element:
    paragraph = ET.Element(qn("w", "p"))
    ppr = ET.SubElement(paragraph, qn("w", "pPr"))
    spacing = ET.SubElement(ppr, qn("w", "spacing"))
    spacing.set(qn("w", "before"), "0")
    spacing.set(qn("w", "after"), "0")
    spacing.set(qn("w", "line"), "1")
    spacing.set(qn("w", "lineRule"), "exact")

    sect_pr = ET.SubElement(ppr, qn("w", "sectPr"))
    sect_type = ET.SubElement(sect_pr, qn("w", "type"))
    sect_type.set(qn("w", "val"), "nextPage")
    pg_sz = ET.SubElement(sect_pr, qn("w", "pgSz"))
    pg_sz.set(qn("w", "w"), str(A4_WIDTH_TWIPS))
    pg_sz.set(qn("w", "h"), str(A4_HEIGHT_TWIPS))
    pg_mar = ET.SubElement(sect_pr, qn("w", "pgMar"))
    for side in ("top", "right", "bottom", "left", "header", "footer", "gutter"):
        pg_mar.set(qn("w", side), "0")

    run = ET.SubElement(paragraph, qn("w", "r"))
    drawing = ET.SubElement(run, qn("w", "drawing"))
    anchor = ET.SubElement(drawing, qn("wp", "anchor"))
    anchor.set("distT", "0")
    anchor.set("distB", "0")
    anchor.set("distL", "0")
    anchor.set("distR", "0")
    anchor.set("simplePos", "0")
    anchor.set("relativeHeight", "251659264")
    anchor.set("behindDoc", "0")
    anchor.set("locked", "0")
    anchor.set("layoutInCell", "1")
    anchor.set("allowOverlap", "1")

    simple_pos = ET.SubElement(anchor, qn("wp", "simplePos"))
    simple_pos.set("x", "0")
    simple_pos.set("y", "0")
    position_h = ET.SubElement(anchor, qn("wp", "positionH"))
    position_h.set("relativeFrom", "page")
    ET.SubElement(position_h, qn("wp", "posOffset")).text = "0"
    position_v = ET.SubElement(anchor, qn("wp", "positionV"))
    position_v.set("relativeFrom", "page")
    ET.SubElement(position_v, qn("wp", "posOffset")).text = "0"

    extent = ET.SubElement(anchor, qn("wp", "extent"))
    extent.set("cx", str(A4_WIDTH_EMU))
    extent.set("cy", str(A4_HEIGHT_EMU))
    effect_extent = ET.SubElement(anchor, qn("wp", "effectExtent"))
    for side in ("l", "t", "r", "b"):
        effect_extent.set(side, "0")
    ET.SubElement(anchor, qn("wp", "wrapNone"))
    doc_pr = ET.SubElement(anchor, qn("wp", "docPr"))
    doc_pr.set("id", doc_pr_id)
    doc_pr.set("name", "Prepared thesis cover")
    c_nv = ET.SubElement(anchor, qn("wp", "cNvGraphicFramePr"))
    ET.SubElement(c_nv, qn("a", "graphicFrameLocks")).set("noChangeAspect", "1")

    graphic = ET.SubElement(anchor, qn("a", "graphic"))
    graphic_data = ET.SubElement(graphic, qn("a", "graphicData"))
    graphic_data.set("uri", "http://schemas.openxmlformats.org/drawingml/2006/picture")
    pic = ET.SubElement(graphic_data, qn("pic", "pic"))

    nv_pic_pr = ET.SubElement(pic, qn("pic", "nvPicPr"))
    c_nv_pr = ET.SubElement(nv_pic_pr, qn("pic", "cNvPr"))
    c_nv_pr.set("id", "0")
    c_nv_pr.set("name", "cover_page.png")
    ET.SubElement(nv_pic_pr, qn("pic", "cNvPicPr"))

    blip_fill = ET.SubElement(pic, qn("pic", "blipFill"))
    blip = ET.SubElement(blip_fill, qn("a", "blip"))
    blip.set(qn("r", "embed"), rel_id)
    stretch = ET.SubElement(blip_fill, qn("a", "stretch"))
    ET.SubElement(stretch, qn("a", "fillRect"))

    sp_pr = ET.SubElement(pic, qn("pic", "spPr"))
    xfrm = ET.SubElement(sp_pr, qn("a", "xfrm"))
    off = ET.SubElement(xfrm, qn("a", "off"))
    off.set("x", "0")
    off.set("y", "0")
    ext = ET.SubElement(xfrm, qn("a", "ext"))
    ext.set("cx", str(A4_WIDTH_EMU))
    ext.set("cy", str(A4_HEIGHT_EMU))
    prst_geom = ET.SubElement(sp_pr, qn("a", "prstGeom"))
    prst_geom.set("prst", "rect")
    ET.SubElement(prst_geom, qn("a", "avLst"))

    return paragraph


def prepend_cover(cover_docx: Path, body_docx: Path, output_docx: Path) -> None:
    if not cover_docx.exists():
        raise FileNotFoundError(f"Cover DOCX not found: {cover_docx}")
    if not body_docx.exists():
        raise FileNotFoundError(f"Body DOCX not found: {body_docx}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cover_png = render_cover_page(cover_docx, tmp_path)

        workdir = tmp_path / "docx"
        workdir.mkdir()
        with zipfile.ZipFile(body_docx) as archive:
            archive.extractall(workdir)

        document_path = workdir / "word" / "document.xml"
        document_tree = ET.parse(document_path)
        body = document_tree.getroot().find(f".//{qn('w', 'body')}")
        if body is None:
            raise ValueError("word/document.xml has no w:body")

        rel_id = add_cover_image_relationship(workdir, cover_png)
        body.insert(0, cover_paragraph(rel_id, next_doc_pr_id(document_tree.getroot())))

        document_tree.write(document_path, encoding="UTF-8", xml_declaration=True)
        ensure_png_content_type(workdir / "[Content_Types].xml")

        output_docx.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_docx, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(workdir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(workdir).as_posix())


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print("usage: prepend_cover_docx.py COVER.docx BODY.docx OUTPUT.docx", file=sys.stderr)
        return 2
    prepend_cover(Path(argv[1]), Path(argv[2]), Path(argv[3]))
    print(f"wrote {argv[3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
