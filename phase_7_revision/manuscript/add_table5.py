"""
Insert Table 5, the six-arm negative-class ablation, before the FIGURES heading.

Built to match Table 4's own properties, read out of the document rather than
guessed: 9,360 dxa wide, single borders throughout, a shaded repeating header
row and 9 pt cell text. The ablation is appended as Table 5 rather than placed
next to the other dataset tables so that every existing reference to Tables 1 to
4 keeps its meaning.

INPUTS   ms/word/document.xml, ablation_results.csv
OUTPUTS  the same document.xml, revised in place
"""

import csv
import re
import sys

DOC = "ms/word/document.xml"
RESULTS = "../ablation_results.csv"

WIDTHS = [700, 2900, 1100, 1300, 1200, 1100, 1060] # sums to the 9360 dxa table
HEADERS = ["Arm", "Training negatives", "n pos / neg", "AUC",
           "FPR hard (%)", "FPR soft (%)", "Charge sep."]

CAPTION = (
    "Table 5. Negative-Class Ablation Separating Composition from Length and "
    "Charge. Every arm keeps the 873 training positives and is scored on the "
    "published 178-row test set, so only the training negatives vary and the arms "
    "are directly comparable. AUC and the hard-negative false-positive rate are "
    "means over ten training-row permutations with their standard deviation, since "
    "row subsampling makes a single fit depend on row order by about 0.004 in AUC, "
    "which is the size of the smallest difference between arms. Arms C and D "
    "cannot reach 873 negatives, because the Basith pool holds nothing below 9 "
    "residues while 45% of the positives are shorter and DBAASP runs out of "
    "sequences at the positives' lengths, so C′ and D′ repeat them with the "
    "positives subsampled to the negative count. The hard-negative block is 45 "
    "proteome fragments and the soft-negative block 40 antimicrobial peptides, "
    "every row in each a true non-ADP, so any positive call is an error. Charge "
    "separation is the AUC of net charge alone predicting that a training row is a "
    "negative, so 0.5 is chance and an arm reaching it carries no charge shortcut. "
    "The readout is XGBoost on the fused 1,287-dimensional vector under the "
    "deployed model's hyperparameters. Arm F closes the shortcut for 0.018 in AUC; "
    "arms D and E differ by 0.03 in AUC and by 54 points in hard-negative "
    "false-positive rate."
)

TBL_PR = (
    '<w:tblPr><w:tblW w:w="9360" w:type="dxa"/><w:tblBorders>'
    '<w:top w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
    '<w:left w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
    '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
    '<w:right w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
    '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
    '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
    '</w:tblBorders><w:tblCellMar><w:left w:w="10" w:type="dxa"/>'
    '<w:right w:w="10" w:type="dxa"/></w:tblCellMar>'
    '<w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" '
    'w:lastColumn="0" w:noHBand="0" w:noVBand="1"/></w:tblPr>'
)
CELL_MAR = ('<w:tcMar><w:top w:w="60" w:type="dxa"/><w:left w:w="60" w:type="dxa"/>'
            '<w:bottom w:w="60" w:type="dxa"/><w:right w:w="60" w:type="dxa"/>'
            '</w:tcMar><w:vAlign w:val="center"/>')


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def cell(text, w, head=False):
    shd = '<w:shd w:val="clear" w:color="auto" w:fill="D9D9D9"/>' if head else ""
    rpr = ("<w:rPr>" + ("<w:b/><w:bCs/>" if head else "")
           + '<w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>')
    return (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{shd}{CELL_MAR}</w:tcPr>'
            f'<w:p><w:r>{rpr}<w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>'
            f'</w:tc>')


def row(cells, head=False):
    pr = "<w:trPr><w:tblHeader/></w:trPr>" if head else ""
    return ("<w:tr>" + pr
            + "".join(cell(c, w, head) for c, w in zip(cells, WIDTHS)) + "</w:tr>")


def main():
    with open(RESULTS, newline="") as f:
        rows = list(csv.DictReader(f))

    # the balanced repeats of the capped arms are marked with a prime rather than
    # given their own arm letter, so the six arms stay six
    rename = {"C-bal": "C′", "D-bal": "D′"}
    body = []
    for r in rows:
        body.append([
            rename.get(r["arm"], r["arm"]),
            r["negatives"],
            f'{r["n_train_pos"]} / {r["n_train_neg"]}',
            f'{float(r["auc"]):.3f} ± {float(r["auc_sd"]):.3f}',
            f'{float(r["fpr_hard_pct"]):.1f} ± {float(r["fpr_hard_sd"]):.1f}',
            f'{float(r["fpr_soft_pct"]):.1f}',
            f'{float(r["charge_separation_auc"]):.3f}',
        ])

    grid = "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{w}"/>' for w in WIDTHS) \
           + "</w:tblGrid>"
    tbl = ("<w:tbl>" + TBL_PR + grid + row(HEADERS, head=True)
           + "".join(row(b) for b in body) + "</w:tbl>")

    cap = ('<w:p><w:pPr><w:spacing w:before="200" w:after="100" '
           'w:line="276" w:lineRule="auto"/><w:jc w:val="both"/></w:pPr>'
           f'<w:r><w:rPr><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>'
           f'<w:t xml:space="preserve">{esc(CAPTION)}</w:t></w:r></w:p>')
    spacer = "<w:p/>"

    x = open(DOC, encoding="utf-8").read()

    # anchor on the FIGURES heading so the table lands after Table 4's block
    m = re.search(r'<w:p(?: [^>]*)?>(?:(?!</w:p>).)*?<w:t(?: [^>]*)?>FIGURES'
                  r'</w:t>.*?</w:p>', x, re.S)
    if not m:
        sys.exit("FIGURES heading not found")

    x = x[:m.start()] + spacer + cap + tbl + spacer + x[m.start():]
    open(DOC, "w", encoding="utf-8").write(x)
    print(f"inserted Table 5 with {len(body)} data rows before the FIGURES heading")


if __name__ == "__main__":
    main()
