"""
Put the TOC graphic on the last page of a manuscript .docx.

ACS asks for the graphic to be labelled "For Table of Contents Only" and
provided on the last page of the submitted manuscript, as well as uploaded
separately. This inserts a page break, the label and the image at the end of
the body, before the section properties.

The PNG is embedded rather than the TIF: Word renders PNG reliably and the TIF
ships alongside as the separate upload. Both are the same 975 x 525 px render,
placed at exactly 3.25 x 1.75 inches so it appears at the size ACS specifies.

Usage:  python add_toc_page.py <manuscript.docx> <toc_graphic.png> [tracked]

INPUTS   a built third-draft .docx and the PNG from make_toc_graphic.py
OUTPUTS  the same .docx, with the TOC page appended
"""

import os
import re
import shutil
import sys
import zipfile

EMU_IN = 914400
W_IN, H_IN = 3.25, 1.75
CX, CY = int(W_IN * EMU_IN), int(H_IN * EMU_IN)

LABEL = "For Table of Contents Only"


def drawing(rid):
    a = "http://schemas.openxmlformats.org/drawingml/2006/main"
    pic = "http://schemas.openxmlformats.org/drawingml/2006/picture"
    return (
        '<w:p><w:pPr><w:spacing w:before="200" w:after="60"/>'
        '<w:jc w:val="center"/></w:pPr><w:r><w:rPr><w:noProof/></w:rPr>'
        '<w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{CX}" cy="{CY}"/>'
        '<wp:effectExtent l="0" t="0" r="0" b="0"/>'
        '<wp:docPr id="900" name="TOC graphic"/>'
        f'<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="{a}" '
        'noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        f'<a:graphic xmlns:a="{a}"><a:graphicData uri="{pic}">'
        f'<pic:pic xmlns:pic="{pic}"><pic:nvPicPr>'
        '<pic:cNvPr id="0" name="ADP_TOC_graphic.png"/><pic:cNvPicPr>'
        '<a:picLocks noChangeAspect="1" noChangeArrowheads="1"/>'
        '</pic:cNvPicPr></pic:nvPicPr><pic:blipFill>'
        f'<a:blip r:embed="{rid}"/><a:srcRect/><a:stretch><a:fillRect/>'
        '</a:stretch></pic:blipFill><pic:spPr bwMode="auto"><a:xfrm>'
        f'<a:off x="0" y="0"/><a:ext cx="{CX}" cy="{CY}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
        '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing>'
        '</w:r></w:p>')


INS_ID = 9000
AUTHOR = "Oliver McQuillan"
DATE = "2026-10-20T00:00:00Z"


def as_insertion(block):
    """Mark a block of paragraphs as tracked insertions.

    The TOC page is new content relative to the second draft, so in the tracked
    file it has to carry revision marks like every other change. Each run is
    wrapped in w:ins and each paragraph mark is marked inserted, otherwise
    rejecting all changes would leave the page behind.
    """
    out, n = [], INS_ID
    for p in re.findall(r'<w:p>.*?</w:p>', block, re.S):
        n += 1
        mark = (f'<w:rPr><w:ins w:id="{n}" w:author="{AUTHOR}" '
                f'w:date="{DATE}"/></w:rPr>')
        m = re.search(r'<w:pPr>.*?</w:pPr>', p, re.S)
        ppr = m.group(0) if m else "<w:pPr></w:pPr>"
        new_ppr = ppr.replace("</w:pPr>", f"{mark}</w:pPr>")
        body = p[len(re.match(r'<w:p>', p).group(0)):-len("</w:p>")]
        if m:
            body = body.replace(ppr, "", 1)
        n += 1
        body = (f'<w:ins w:id="{n}" w:author="{AUTHOR}" w:date="{DATE}">'
                f'{body}</w:ins>')
        out.append(f"<w:p>{new_ppr}{body}</w:p>")
    return "".join(out)


def main():
    docx, png = sys.argv[1], sys.argv[2]
    tracked = len(sys.argv) > 3 and sys.argv[3] == "tracked"
    work = docx + "_tocwork"
    if os.path.isdir(work):
        shutil.rmtree(work)
    with zipfile.ZipFile(docx) as z:
        z.extractall(work)

    shutil.copy2(png, os.path.join(work, "word", "media", "toc_graphic.png"))

    rels_path = os.path.join(work, "word", "_rels", "document.xml.rels")
    rels = open(rels_path, encoding="utf-8").read()
    used = [int(n) for n in re.findall(r'Id="rId(\d+)"', rels)]
    rid = f"rId{max(used) + 1}"
    rels = rels.replace("</Relationships>",
                        f'<Relationship Id="{rid}" Type="http://schemas.'
                        'openxmlformats.org/officeDocument/2006/relationships/'
                        'image" Target="media/toc_graphic.png"/>'
                        "</Relationships>")
    open(rels_path, "w", encoding="utf-8").write(rels)

    doc_path = os.path.join(work, "word", "document.xml")
    x = open(doc_path, encoding="utf-8").read()
    if "toc_graphic" in x:
        sys.exit("TOC page already present")

    block = (
        # the page break keeps the graphic on a page of its own, as asked
        '<w:p><w:pPr><w:spacing w:line="480" w:lineRule="auto"/></w:pPr>'
        '<w:r><w:br w:type="page"/></w:r></w:p>'
        '<w:p><w:pPr><w:spacing w:before="200" w:after="100"/>'
        '<w:jc w:val="center"/></w:pPr><w:r><w:rPr><w:b/><w:bCs/></w:rPr>'
        f'<w:t xml:space="preserve">{LABEL}</w:t></w:r></w:p>'
        + drawing(rid))
    if tracked:
        block = as_insertion(block)

    # before the body's own sectPr, which has to stay last
    m = re.search(r'<w:sectPr\b', x)
    at = m.start() if m else x.rindex("</w:body>")
    x = x[:at] + block + x[at:]
    open(doc_path, "w", encoding="utf-8").write(x)

    os.remove(docx)
    with zipfile.ZipFile(docx, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(work):
            for f in files:
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, work))
    shutil.rmtree(work)
    print(f"{os.path.basename(docx)}: TOC page added as {rid}, "
          f"{W_IN} x {H_IN} in")


if __name__ == "__main__":
    main()
