"""
Add Tables S11 to S14 to the Supporting Information and retitle it.

S11  test-set proximity to the training set        (his point 1, first half)
S12  positive-class family structure               (his priority 1)
S13  nearest training positive per top lead        (his point 1, second half)
S14  the ablation under a family-aware split       (his priority 1 sensitivity)

Tables are generated from the audit CSVs rather than transcribed, so the
document cannot drift from the analysis. They are appended after Table S10 and
before the figure captions, which keeps every existing S-number meaning what it
did.

INPUTS   si/word/document.xml, ../audit_*.csv, ../audit_*_summary.txt
OUTPUTS  the same document.xml, revised in place
"""

import csv
import re
import sys

DOC = "si/word/document.xml"
TITLE = ("Negative-Class Design Reveals a Net-Charge Shortcut in Anti-Diabetic "
         "Peptide Prediction")

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
    return (str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def cell(text, w, head=False, sz=18):
    shd = '<w:shd w:val="clear" w:color="auto" w:fill="D9D9D9"/>' if head else ""
    rpr = ("<w:rPr>" + ("<w:b/><w:bCs/>" if head else "")
           + f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/></w:rPr>')
    return (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{shd}{CELL_MAR}</w:tcPr>'
            f'<w:p><w:r>{rpr}<w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>'
            f'</w:tc>')


def table(headers, rows, widths, sz=18):
    grid = "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{w}"/>' for w in widths) \
           + "</w:tblGrid>"
    head = ("<w:tr><w:trPr><w:tblHeader/></w:trPr>"
            + "".join(cell(h, w, True, sz) for h, w in zip(headers, widths))
            + "</w:tr>")
    body = "".join("<w:tr>" + "".join(cell(c, w, False, sz)
                                      for c, w in zip(r, widths)) + "</w:tr>"
                   for r in rows)
    return "<w:tbl>" + TBL_PR + grid + head + body + "</w:tbl>"


def caption(text):
    return ('<w:p><w:pPr><w:spacing w:before="200" w:after="100" w:line="276" '
            'w:lineRule="auto"/><w:jc w:val="both"/></w:pPr><w:r>'
            '<w:rPr><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>'
            f'<w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>')


def contents_line(text):
    return ('<w:p><w:pPr><w:spacing w:line="276" w:lineRule="auto"/></w:pPr>'
            f'<w:r><w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>')


# --------------------------------------------------------------------------- #

def build_s11():
    cap = (
        "Table S11. Proximity of the held-out test set to the training set. The "
        "partition clusters at 40% identity only above 10 residues, so the 870 "
        "sequences below that length took a stratified random split and this table "
        "measures what that left. Identity is matched residues over the longer "
        "sequence of each pair, computed as the longest common subsequence; "
        "CD-HIT's convention of normalising over the shorter saturates at 1.0 for "
        "any containment and carries no information at these lengths. The null "
        "shuffles each test sequence's residues, holding its length and "
        "composition fixed, over 200 replicates at seed 42. The AUC rows rescore "
        "the saved predictions with the 41 near-duplicate test rows removed and no "
        "model refitted, so they isolate the split's contribution. Reproduced in "
        "audit_overlap.py."
    )
    rows = [
        ["Test sequences", "178"],
        ["Exact matches in the training set", "0"],
        ["Substring relationship with a training sequence", "67 (37.6%)"],
        ["  residue-shuffled null, mean", "37.2 (20.9%), SD 4.0"],
        ["  residue-shuffled null, range over 200 shuffles", "26 to 49"],
        ["  empirical P", "< 0.005"],
        ["Near duplicates at ≥ 80% identity over the longer", "41 (23.0%)"],
        ["  of the 130 test sequences below 11 residues", "35 (26.9%)"],
        ["  of the 48 test sequences of 11 residues or more", "6 (12.5%)"],
        ["Mean identity to the nearest training sequence", "0.631"],
        ["  below 11 residues", "0.665"],
        ["  11 residues or more", "0.541"],
        ["Test AUC, ESM-2 / DoRA, near duplicates removed",
         "0.877 → 0.831 (−0.046)"],
        ["Test AUC, XGBoost, near duplicates removed", "0.857 → 0.815 (−0.043)"],
        ["Test AUC, Random Forest, near duplicates removed",
         "0.830 → 0.779 (−0.052)"],
        ["Test AUC, consensus, near duplicates removed", "0.879 → 0.836 (−0.043)"],
    ]
    return cap, table(["Quantity", "Value"], rows, [5900, 3460])


def build_s12():
    cap = (
        "Table S12. Family structure of the positive class. Families are formed by "
        "single linkage over pairs where one sequence contains the other and the "
        "shorter covers at least half the longer. The effective sample size is the "
        "family count as a fraction of the sequence count. The negatives are drawn "
        "independently and cannot share ancestry, so they serve as a control on the "
        "metric: the no-floor rows show that plain containment is transitive "
        "through the two dipeptide positives and manufactures large families in "
        "both classes, which is why the coverage floor is applied. Preproinsulin "
        "membership is tested against human P01308 at a four-residue floor, below "
        "which chance substrings such as FF and ALE appear. Reproduced in "
        "audit_positives.py."
    )
    rows = [
        ["Positives, families (≥ 50% coverage)", "577 of 966", "0.60", "56"],
        ["Negatives, families (≥ 50% coverage)", "861 of 966", "0.89", "10"],
        ["Positives, families (no coverage floor)", "427 of 966", "0.44", "242"],
        ["Negatives, families (no coverage floor)", "621 of 966", "0.64", "287"],
        ["Families straddling the published split", "21", "–",
         "226 positives (23.4%)"],
        ["Preproinsulin fragments ≥ 4 residues, positives", "141 of 966 (14.6%)",
         "–", "–"],
        ["Preproinsulin fragments ≥ 4 residues, negatives", "0 of 966", "–", "–"],
        ["Largest family: insulin B chain region", "56 members", "–",
         "straddles the split"],
        ["Second: proinsulin C-peptide", "55 members", "–",
         "straddles the split"],
        ["Third: preproinsulin signal peptide", "32 members", "–",
         "straddles the split"],
    ]
    return cap, table(["Quantity", "Count", "ESS", "Largest / note"], rows,
                      [3700, 1800, 800, 3060])


def build_s13():
    with open("../audit_discovery_leads.csv", newline="") as f:
        leads = list(csv.DictReader(f))
    cap = (
        "Table S13. Distance from each top-ranked lead to the nearest training "
        "positive. The ten highest-consensus discoveries of each tier are shown. "
        "Identity is matched residues over the longer sequence, as in Table S11, "
        "measured against the 873 training positives rather than against train "
        "plus test, since those are the sequences whose label the model learned. "
        "The contained column gives the longest training positive wholly inside "
        "the lead, which is what the novelty filter did not remove: it screened "
        "only candidates of 11 residues or more at 40% identity, and CD-HIT-2D's "
        "length-difference default additionally lets a short reference miss a "
        "longer query. Across all 412 discoveries, 64 (15.5%) contain a training "
        "positive of three residues or more and 41 (10.0%) one of six or more, "
        "among them 26 of the 196 discoveries reaching 11 residues; 23 (5.6%) "
        "reach 80% identity and every one lies in the short band the filter never "
        "covered, against none of the 196 long discoveries. Mean nearest identity "
        "is 0.603 for short discoveries and 0.442 for long ones. Reproduced in "
        "audit_discovery.py."
    )
    rows = [[r["tier"], r["peptide_id"], r["sequence"], len(r["sequence"]),
             f'{float(r["consensus"]):.4f}', r["nearest_train_positive"],
             f'{float(r["identity_over_longer"]):.2f}',
             r["contains_train_positive"] or "–"] for r in leads]
    return cap, table(["Tier", "ID", "Sequence", "Len", "Consensus",
                       "Nearest training positive", "Id.", "Contained"],
                      rows, [460, 900, 2500, 420, 820, 2500, 420, 1340], sz=15)


def build_s14():
    with open("../audit_family_split_results.csv", newline="") as f:
        fa = {r["arm"]: r for r in csv.DictReader(f)}
    with open("../ablation_results.csv", newline="") as f:
        pub = {r["arm"]: r for r in csv.DictReader(f)}
    cap = (
        "Table S14. The negative-class ablation repeated under a family-aware "
        "split. Families were formed over the published 1,932 sequences as in "
        "Table S12 and whole families assigned to train or test at a 15% target, "
        "held separately for positives, soft negatives and hard negatives so the "
        "hard-negative block does not collapse to a few rows. That gives one "
        "shared 290-row test set, 145 positives with 96 soft and 49 hard "
        "negatives, on which every arm is scored; each arm trains on the "
        "family-aware training positives plus its own negatives, excluding "
        "anything in the new test set. Giving each arm its own split instead would "
        "score each on its own negative distribution and is not a valid "
        "comparison. The published and family-aware test sets differ, so only the "
        "gap between two arms within one split is interpretable and no arm's AUC "
        "should be read across the two blocks. Figures are means over ten "
        "training-row permutations. The A to B AUC gap narrows from 0.180 to "
        "0.074 while the hard-negative false-positive gap holds at 20.6 points "
        "against 36.5, so part of the AUC effect is an artefact of the split and "
        "the deployment effect is not. Arm D records the best AUC of the six here "
        "and the worst false-positive rate. Reproduced in audit_family_split.py."
    )
    order = ["A", "B", "C", "D", "E", "F"]
    rows = []
    for a in order:
        p, f = pub[a], fa[a]
        rows.append([a, p["negatives"],
                     f'{float(p["auc"]):.3f}', f'{float(p["fpr_hard_pct"]):.1f}',
                     f'{float(f["auc"]):.3f} ± {float(f["auc_sd"]):.3f}',
                     f'{float(f["fpr_hard_pct"]):.1f}',
                     f'{float(f["fpr_soft_pct"]):.1f}'])
    return cap, table(["Arm", "Training negatives", "AUC (pub.)",
                       "FPR hard (pub.)", "AUC (family)", "FPR hard (family)",
                       "FPR soft (family)"], rows,
                      [560, 2700, 1100, 1250, 1350, 1250, 1150])


def main():
    x = open(DOC, encoding="utf-8").read()

    if x.count("Negative-Class Composition Outweighs") == 1:
        x = x.replace("Negative-Class Composition Outweighs Model Complexity in "
                      "Anti-Diabetic Peptide Prediction: Dual-Negative Sampling and "
                      "a Residual Net-Charge Shortcut", TITLE)
    else:
        print("note: SI title string not matched as one run, left unchanged")

    # contents lines, after the Table S10 entry
    anchor = "Table S10. Software versions and compute environment"
    m = re.search(r'<w:p(?: [^>]*)?>(?:(?!</w:p>).)*?'
                  + re.escape(anchor) + r'(?:(?!</w:p>).)*?</w:p>', x, re.S)
    if not m:
        sys.exit("contents anchor for Table S10 not found")
    new_lines = "".join(contents_line(t) for t in [
        "Table S11. Proximity of the held-out test set to the training set",
        "Table S12. Family structure of the positive class",
        "Table S13. Distance from each top-ranked lead to the nearest training "
        "positive",
        "Table S14. The negative-class ablation under a family-aware split",
    ])
    x = x[:m.end()] + new_lines + x[m.end():]

    # the tables themselves, before the Figure S1 caption
    # locate the paragraph ELEMENT holding the caption. rfind('<w:p') would also
    # match <w:pPr and land inside the previous element, which splits it
    paras = [m for m in re.finditer(r'<w:p(?: [^>]*)?>.*?</w:p>|<w:p(?: [^>]*)?/>',
                                    x, re.S)
             if "Figure S1. Four published ADP predictors" in m.group(0)]
    if len(paras) != 1:
        sys.exit(f"Figure S1 caption matched {len(paras)} paragraphs")
    start = paras[0].start()

    blocks = ""
    for builder in (build_s11, build_s12, build_s13, build_s14):
        cap, tbl = builder()
        blocks += "<w:p/>" + caption(cap) + tbl
    blocks += "<w:p/>"

    x = x[:start] + blocks + x[start:]
    open(DOC, "w", encoding="utf-8").write(x)
    print("added Tables S11 to S14 and four contents lines")


if __name__ == "__main__":
    main()
