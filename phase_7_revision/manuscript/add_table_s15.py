"""
Add Table S15, the five-seed selection stability of the shortlist.

Generated from results/phase7_6_seed_stability.csv so the table cannot drift
from the analysis. Placed after Table S14 and before the figure captions, which
keeps every existing S-number meaning what it did.
"""

import os
import re
import sys
import numpy as np
import pandas as pd

DOC = "si/word/document.xml"
REPO = "/Users/olivermcquillan/antidiabetic-peptide-classifier"
SEEDS = [42, 43, 44, 45, 46]
THRESH = 0.90
WIDTHS = [4700, 4660]

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
CELL_MAR = ('<w:tcMar><w:top w:w="50" w:type="dxa"/><w:left w:w="60" w:type="dxa"/>'
            '<w:bottom w:w="50" w:type="dxa"/><w:right w:w="60" w:type="dxa"/>'
            '</w:tcMar><w:vAlign w:val="center"/>')


def esc(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def cell(text, w, head=False, bold=False):
    shd = '<w:shd w:val="clear" w:color="auto" w:fill="D9D9D9"/>' if head else ""
    rpr = ("<w:rPr>" + ("<w:b/><w:bCs/>" if head or bold else "")
           + '<w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>')
    return (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{shd}{CELL_MAR}</w:tcPr>'
            f'<w:p><w:r>{rpr}<w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>'
            f'</w:tc>')


def main():
    csv = f"{REPO}/results/phase7_6_seed_stability.csv"
    if not os.path.exists(csv):
        sys.exit(f"missing {csv}: run phase7.7_seed_stability.py first")
    d = pd.read_csv(csv, keep_default_na=False)
    pub = d[d.published_discovery.astype(str).str.lower() == "true"]

    sizes = [int((d[f"consensus_seed{s}"] >= THRESH).sum()) for s in SEEDS]
    freq = pub.n_seeds_selected
    inter = int((d.n_seeds_selected == 5).sum())
    ens = int((d.consensus_mean >= THRESH).sum())
    ens_new = int(((d.consensus_mean >= THRESH)
                   & (d.published_discovery.astype(str).str.lower() != "true")).sum())

    rows = [("Candidates scored", f"{len(d):,}", True)]
    rows += [(f"  shortlist at 0.90, seed {s}", f"{n}", False)
             for s, n in zip(SEEDS, sizes)]
    rows += [
        ("  published shortlist (seed 42)", f"{len(pub)}", False),
        ("  largest over median", f"{len(pub) / np.median(sizes):.2f}×", False),
        ("Of the published 412, selected by", "", True),
    ]
    for k in range(5, 0, -1):
        n = int((freq == k).sum())
        rows.append((f"  {k} of 5 seeds", f"{n} ({100 * n / len(pub):.1f}%)", False))
    rows += [
        ("  median selection frequency", f"{int(freq.median())} of 5", False),
        ("Shortlists that do not depend on one fit", "", True),
        ("  intersection across the five seeds", f"{inter}", False),
        ("  mean ensemble at 0.90", f"{ens} ({ens_new} absent from the published "
                                    f"set)", False),
        ("Leads and known actives", "", True),
    ]
    for seq in ["FVAPFPEVF", "GPFPSIL", "LPGF", "IPAVF"]:
        r = d[d.sequence == seq]
        if len(r):
            r = r.iloc[0]
            per = ", ".join(f"{r[f'consensus_seed{s}']:.3f}" for s in SEEDS)
            rows.append((f"  {seq}", f"{int(r.n_seeds_selected)} of 5  ({per})", False))
    for t, n, lab in [(1, 216, "Tier 1"), (2, 27, "Tier 2")]:
        tier = pd.read_csv(f"{REPO}/screening/tiered_discovery.csv",
                           keep_default_na=False)
        ids = set(tier[tier.tier == t].peptide_id)
        sub = d[d.peptide_id.isin(ids)]
        rows.append((f"  {lab} (n = {len(sub)}), median frequency",
                     f"{int(sub.n_seeds_selected.median())} of 5, "
                     f"{int((sub.n_seeds_selected == 5).sum())} selected by all five",
                     False))

    cap = (
        "Table S15. Selection stability of the shortlist across the five ESM-2 "
        "fine-tuning seeds. Every one of the 3,596 candidates was rescored with "
        "each of the five DoRA adapters from the seed refits. Only the ESM-2 half "
        "of the consensus moves with the seed: XGBoost reads the frozen embedding, "
        "which the adapter does not touch, so its probability is held at the "
        "deployed value and the consensus is recomputed per seed as the mean of "
        "the two. That isolates the fine-tune's contribution. The threshold is the "
        "0.90 used throughout. The published shortlist comes from seed 42, which "
        "is both the highest of the five on test AUC (0.877 against a 0.842 mean) "
        "and the most permissive at the operating threshold, so its count and its "
        "membership are each the optimistic end of the distribution. Parenthesised "
        "figures beside each peptide are its per-seed consensus scores in seed "
        "order. Reproduced in phase7.7_seed_stability.py from the Viking job "
        "phase5.8_seed_screen."
    )

    grid = "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{w}"/>' for w in WIDTHS) \
           + "</w:tblGrid>"
    head = ("<w:tr><w:trPr><w:tblHeader/></w:trPr>"
            + cell("Quantity", WIDTHS[0], True) + cell("Value", WIDTHS[1], True)
            + "</w:tr>")
    body = "".join("<w:tr>" + cell(a, WIDTHS[0], False, bold)
                   + cell(b, WIDTHS[1], False, bold) + "</w:tr>"
                   for a, b, bold in rows)
    tbl = "<w:tbl>" + TBL_PR + grid + head + body + "</w:tbl>"
    capx = ('<w:p><w:pPr><w:spacing w:before="200" w:after="100" w:line="276" '
            'w:lineRule="auto"/><w:jc w:val="both"/></w:pPr><w:r>'
            '<w:rPr><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>'
            f'<w:t xml:space="preserve">{esc(cap)}</w:t></w:r></w:p>')

    x = open(DOC, encoding="utf-8").read()
    paras = [m for m in re.finditer(r'<w:p(?: [^>]*)?>.*?</w:p>|<w:p(?: [^>]*)?/>',
                                    x, re.S)
             if "Figure S1. Four published ADP predictors" in m.group(0)]
    if len(paras) != 1:
        sys.exit(f"Figure S1 caption matched {len(paras)} paragraphs")
    at = paras[0].start()
    x = x[:at] + "<w:p/>" + capx + tbl + "<w:p/>" + x[at:]

    # contents list
    anchor = "Table S14. The negative-class ablation under a family-aware split"
    m = re.search(r'<w:p(?: [^>]*)?>(?:(?!</w:p>).)*?' + re.escape(anchor)
                  + r'(?:(?!</w:p>).)*?</w:p>', x, re.S)
    if m:
        line = ('<w:p><w:pPr><w:spacing w:line="276" w:lineRule="auto"/></w:pPr>'
                '<w:r><w:t xml:space="preserve">Table S15. Selection stability of '
                'the shortlist across the five fine-tuning seeds</w:t></w:r></w:p>')
        x = x[:m.end()] + line + x[m.end():]
    else:
        print("note: Table S14 contents line not found, add S15 by hand")

    open(DOC, "w", encoding="utf-8").write(x)
    print(f"added Table S15 with {len(rows)} rows")


if __name__ == "__main__":
    main()
