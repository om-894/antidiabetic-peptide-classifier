"""
Third-draft revisions to the Supporting Information.

Terminology and claim wording, to match the manuscript, plus two corrections.
Table S1 called the positive set experimentally validated, which the released
data does not support, and Table S15 reported the mean-ensemble shortlist as
211, which came from reading a column rounded to four places where one
candidate sits at 0.899963. The deposited file now carries six places and
reports 210.

Edits are plain string replacements rather than paragraph rebuilds, because
every target here sits inside a table cell as a single run.

Usage:  python revise_si_3rd.py <second_draft_si.docx> <out_dir>

INPUTS   the second-draft Supporting Information
OUTPUTS  ADP_supporting_information_3rd_draft.docx
"""

import os
import re
import shutil
import subprocess
import sys
import zipfile

SKILL = ("/Users/olivermcquillan/Library/Application Support/Claude/"
         "local-agent-mode-sessions/skills-plugin/"
         "8c3daa7b-0fba-4809-855f-18df51b361a2/"
         "2290ba74-5977-4b52-8156-7c3da6ac754b/skills/docx")

FIX = [
 # Table S1: provenance, not an evidence claim
 ("966 non-redundant ADPs and candidates consolidated by Xie et al.",
  "966 curated positive ADP sequences consolidated by Xie et al. (899 from "
  "BioDADPep via three earlier studies, 67 later candidates)"),
 ("Largest experimentally validated positive set across the field",
  "Largest curated positive set across the field; per-sequence evidence of "
  "activity does not survive into the released data"),

 # Table S1: name the criterion rather than claiming leakage is eliminated
 ("cd-hit-2d with train as reference, iterated to zero leakage (maximum 15 "
  "passes)",
  "cd-hit-2d with train as reference, iterated until no test sequence exceeds "
  "40% identity to a training sequence (maximum 15 passes)"),
 ("CD-HIT clustering is greedy and misses some pairs. Leaked test sequences are "
  "returned to train",
  "CD-HIT clustering is greedy and misses some pairs. Test sequences above the "
  "threshold are returned to train. This removes identity above 40% and does "
  "not address substring containment or sequences below 11 residues"),

 # Table S15: the rounding artefact
 ("mean ensemble at 0.90 | 211 (7 absent from the published set)",
  "mean ensemble at 0.90 | 210 (7 absent from the published set)"),
 ("211 (7 absent from the published set)",
  "210 (7 absent from the published set)"),

 # terminology, to match the manuscript
 ("Discoveries are those reaching a consensus of 0.90",
  "Prioritised candidates are those reaching a consensus of 0.90"),
 ("Rate is discoveries as a percentage of that protein’s candidates",
  "Rate is prioritised candidates as a percentage of that protein’s candidates"),
 ("Highest-scoring discovery", "Highest-scoring candidate"),
 ("The ten highest-consensus discoveries of each tier are shown",
  "The ten highest-consensus prioritised candidates of each tier are shown"),
 ("so the discoveries are tiered rather than filtered",
  "so the prioritised candidates are tiered rather than filtered"),
 ("Composition of the 412 discoveries", "Composition of the 412 prioritised "
  "candidates"),
 ("flagged 73.6% of the 216 discoveries", "flagged 73.6% of the 216 prioritised "
  "candidates"),
 ("the 412 discoveries", "the 412 prioritised candidates"),
 ("412 discoveries", "412 prioritised candidates"),

 ("sum to 4,052 candidates and 414 discoveries rather than",
  "sum to 4,052 candidates and 414 prioritised candidates rather than"),
 ("among them 26 of the 196 discoveries reaching 11 residues",
  "among them 26 of the 196 prioritised candidates reaching 11 residues"),
 ("against none of the 196 long discoveries. Mean nearest identity is 0.603 for "
  "short discoveries and 0",
  "against none of the 196 long ones. Mean nearest identity is 0.603 for short "
  "prioritised candidates and 0"),
 ("so the discoveries were tiered rather than filtered on safety alone",
  "so the prioritised candidates were tiered rather than filtered on safety "
  "alone"),

 # Table S7 and Figure S3: reference inhibitors, and the panel scoped to itself
 ("2 positive controls (Diprotin A, IPAVF)",
  "2 reference inhibitors (Diprotin A, IPAVF)"),
 ("The negative controls establish what a genuine non-binder scores, giving the "
  "panel a floor",
  "The three rejected peptides indicate what a sequence the classifier is "
  "confident about scores under the same protocol, giving the panel a reference "
  "floor"),
 ("two positive controls and three hard negatives from the held-out test set",
  "two reference inhibitors and three hard negatives from the held-out test set"),

 ("Conventional 40% reduction would remove 782 of 1,932, since many ADPs are "
  "overlapping insulin-derived fragments",
  "Conventional 40% reduction would remove 782 of the 1,062 sequences long "
  "enough to cluster, and most of the dataset if applied to all 1,932, since "
  "many positives are overlapping insulin-derived fragments"),

 # Figure S5: calibration is a benchmark property, not a screen hit rate
 ("The consensus score is calibrated closely enough for the 0.90 threshold to "
  "carry an expected hit rate",
  "The consensus score is well calibrated on held-out benchmark peptides, which "
  "is what the 0.90 threshold was set against"),
 ("the empirical rate the discovery cut-off was adopted against",
  "the empirical rate the prioritisation cut-off was adopted against. This is a "
  "property of benchmark peptides and is not an expected hit rate for digestion "
  "fragments, which are drawn from a different distribution"),
]


def strip_empty_comments(work):
    """Drop an orphaned comments part if it holds no actual comments.

    The document carries a comments.xml with no <w:comment> in it, inherited
    from whatever produced the first draft. It is harmless, but a submission
    should not ship a part named comments at all, so it goes along with its
    content-type override and its relationship.
    """
    doc = os.path.join(work, "word", "comments.xml")
    if not os.path.exists(doc):
        return False
    if re.search(r"<w:comment[ >]", open(doc, encoding="utf-8").read()):
        return False # real comments, leave well alone

    for p in (doc, os.path.join(work, "word", "_rels", "comments.xml.rels")):
        if os.path.exists(p):
            os.remove(p)

    ct = os.path.join(work, "[Content_Types].xml")
    x = open(ct, encoding="utf-8").read()
    x = re.sub(r'<Override PartName="/word/comments\.xml"[^>]*/>', "", x)
    open(ct, "w", encoding="utf-8").write(x)

    rels = os.path.join(work, "word", "_rels", "document.xml.rels")
    x = open(rels, encoding="utf-8").read()
    x = re.sub(r'<Relationship[^>]*Target="comments\.xml"[^>]*/>', "", x)
    open(rels, "w", encoding="utf-8").write(x)
    return True


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/Downloads/ADP_supporting_information_2nd_draft.docx")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    work = os.path.join(out_dir, "_build", "si")
    if os.path.isdir(work):
        shutil.rmtree(work)
    with zipfile.ZipFile(src) as z:
        z.extractall(work)
    subprocess.run([sys.executable, f"{SKILL}/scripts/merge_runs.py", work],
                   check=True, capture_output=True)

    doc = os.path.join(work, "word", "document.xml")
    x = open(doc, encoding="utf-8").read()

    applied, missed = 0, []
    for old, new in FIX:
        o = old.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if o not in x:
            missed.append(old[:60])
            continue
        n = x.count(o)
        x = x.replace(o, new.replace("&", "&amp;").replace("<", "&lt;")
                      .replace(">", "&gt;"))
        applied += n

    open(doc, "w", encoding="utf-8").write(x)

    if strip_empty_comments(work):
        print("removed an orphaned, empty comments part")

    out = os.path.join(out_dir, "ADP_supporting_information_3rd_draft.docx")
    if os.path.exists(out):
        os.remove(out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(work):
            for f in files:
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, work))

    print(f"applied {applied} replacements across {len(FIX) - len(missed)} rules")
    for m in missed:
        print(f"  not found (already clean or reworded): {m}")


if __name__ == "__main__":
    main()
