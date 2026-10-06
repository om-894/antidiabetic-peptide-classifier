"""
Write the five-seed selection-stability result into the manuscript.

The third revision priority was outstanding when the second draft was first
assembled, so Sections 3.3, 3.6, 4.3 and the Conclusions deferred to it rather
than reporting it. The Viking job has now run and this replaces the deferrals
with the measured numbers.

Addresses paragraphs by index in the current second-draft document.xml, with
each target's opening words asserted first, as revise_ms.py does.
"""

import re
import sys

DOC = "ms/word/document.xml"
PARA = re.compile(r'<w:p(?: [^>]*)?>.*?</w:p>|<w:p(?: [^>]*)?/>', re.S)
TOKEN = re.compile(r'(\*\*.+?\*\*|\^\{[^}]+\})', re.S)

BODY = ('<w:p><w:pPr><w:spacing w:line="480" w:lineRule="auto"/>'
        '<w:jc w:val="both"/></w:pPr>{}</w:p>')


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def runs(text):
    out = []
    for part in TOKEN.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            rpr, body = "<w:rPr><w:b/><w:bCs/></w:rPr>", part[2:-2]
        elif part.startswith("^{"):
            rpr, body = '<w:rPr><w:vertAlign w:val="superscript"/></w:rPr>', part[2:-1]
        else:
            rpr, body = "", part
        out.append(f'<w:r>{rpr}<w:t xml:space="preserve">{esc(body)}</w:t></w:r>')
    return "".join(out)


def ptext(p):
    return "".join(re.findall(r'<w:t(?: [^>]*)?>(.*?)</w:t>', p, re.S))


def rebuild(p, text):
    m = re.search(r'<w:pPr>.*?</w:pPr>', p, re.S)
    ppr = m.group(0) if m else ""
    return f"{re.match(r'<w:p(?: [^>]*)?>', p).group(0)}{ppr}{runs(text)}</w:p>"


REPLACE = {

# the docking clause is shortened to pay for the new sentence on seed stability
6: ("ABSTRACT: Classifiers for anti-diabetic",
    "ABSTRACT: Classifiers for anti-diabetic peptides (ADPs) report high benchmark "
    "accuracy yet misclassify most of the proteome fragments a real screen returns. "
    "Holding the positives, hyperparameters and test set fixed, we varied the "
    "negative class alone. A dual-negative set of DBAASP antimicrobial peptides "
    "plus Gene-Ontology-filtered Swiss-Prot fragments, length-matched one-to-one "
    "against 966 positives, cut the false-positive rate on held-out proteome "
    "fragments from 93.3% to 46.7%. Architecture did not separate the learners. A "
    "DoRA fine-tuned ESM-2 650M model and a 20-feature composition Random Forest "
    "bracketed the same AUC interval at 0.877 and 0.854. Net charge survives as "
    "the residual shortcut. It is carried by the antimicrobial half of the class, "
    "reproduced by the trained model on held-out data and shared by the published "
    "state of the art, which misclassified 92.4% of 328 unseen fragments. It also "
    "selects the output, the 412 calibrated discoveries averaging −0.79 in net "
    "charge against −0.11 across the pool they came from. A six-arm ablation "
    "separates composition from length and charge. Length-matching a published "
    "negative pool leaves its false-positive rate near 80%, whereas matching "
    "charge as well drives the shortcut to chance for 0.018 in AUC. Two arms "
    "differing by 0.03 in AUC differ by 54 points in false-positive rate, so test "
    "AUC is close to blind to screen behaviour. Docking inverts the same bias, "
    "cationic non-binders drawing two thirds of their favourable score from "
    "electrostatics at an anionic site. Repeating the screen under five "
    "fine-tuning seeds returns shortlists of 153 to 412 candidates with only 111 "
    "common to all. A negative class should therefore be matched on every "
    "descriptor separating its source database from the positives and selected on "
    "deployment error rather than on test AUC, and a shortlist from a "
    "parameter-efficient fine-tune should be reported across seeds."),

# 3.6, third limitation: measured rather than deferred
90: ("The dependence of the shortlist on a single fine-tuning seed",
     "The dependence of the shortlist on a single fine-tuning seed is the third, "
     "and it is now measured rather than asserted. Only 111 of the 412 "
     "discoveries clear the threshold under all five seeds, 123 clear it under "
     "the published seed alone, and the five shortlists range from 153 to 412 "
     "(Section 3.3). The seed carried forward is both the highest of the five on "
     "test AUC and the most permissive on the screen, so the published count and "
     "its membership are each the optimistic end of their distribution. Because "
     "this bears on how any such screen should be reported, Section 4.3 treats it "
     "as a recommendation rather than only as a caveat."),

# 4.3: the deferral becomes the result
99: ("A shortlist drawn from a parameter-efficient",
     "**A shortlist drawn from a parameter-efficient fine-tune should be reported "
     "across seeds rather than from one fit.** The adapter carried forward here is "
     "the seed-42 fit, the highest of the five spanning 0.807 to 0.877, so the 412 "
     "discoveries, both tiers and the FVAPFPEVF case all rest on the most "
     "favourable ESM-2 fit rather than a representative one. Repeating the screen "
     "under all five seeds puts a number on what that costs. The shortlists range "
     "from 153 to 412 candidates and only 111 are common to all of them, so 73% of "
     "the published set does not survive a change of seed and 30% of it rests on "
     "seed 42 alone (Section 3.3, Table S15). The seed scoring highest on the test "
     "set is also the one admitting most at the operating threshold, so the two "
     "optimism effects compound rather than offset. Reporting the intersection "
     "across seeds, 111 candidates here, or the mean ensemble, 210, gives a "
     "candidate set that does not depend on a single fit; the mean ensemble is the "
     "better default, since it recovers seven candidates the published run missed "
     "while discarding most of the seed-specific tail. Sensitivity of this kind is "
     "not well characterised in the ADP literature, the nearest comparable study "
     "reporting ten-seed variability only for its classical classifier heads.^{6} "
     "The simpler remedy is to stop fine-tuning for deployment. The variance "
     "enters through the randomly initialised classification head, which moves the "
     "starting point of the whole optimisation, so it is removed by keeping the "
     "encoder frozen and fitting a classical head on the extracted representation. "
     "On the frozen 1,287-dimensional vector, logistic regression is exactly "
     "deterministic across the five seeds at AUC 0.845 and XGBoost varies by 0.003 "
     "at 0.860, against a 0.070 span for the fine-tune. Both sit inside the "
     "fine-tune's confidence interval and both retain the ESM-2 representation, so "
     "the seed sensitivity is a cost of the fine-tuning step rather than of the "
     "embedding. A screen needing a reproducible shortlist is better served by "
     "frozen features with a convex head than by the better single fit. Freezing "
     "the classification head itself is not an alternative: an untrained head "
     "carries no mapping from representation to label, so it has to be fitted. The "
     "only question is whether it is fitted alongside the encoder or on top of "
     "it."),

# conclusions: the third recommendation gains its number
103: ("Three recommendations follow for work of this kind",
      "Three recommendations follow for work of this kind. A negative class should "
      "be matched on every descriptor that separates its source database from the "
      "positives, not on length alone; where one bioactivity class cannot supply "
      "that match, the soft majority should draw across several classes or the "
      "proteome-fragment share should be raised, since those negatives are cut "
      "from UniProt on demand rather than drawn from a finite curated set. Testing "
      "every descriptor against the class labels before any model is fitted costs "
      "one ROC curve and would expose the residual skew, and a negative class "
      "should be selected on the error it makes against the sequence space a "
      "screen meets rather than on test AUC, which across six arms points the "
      "other way. A workflow pairing a sequence classifier with structure-based "
      "scoring should report the electrostatic share of every docking score rather "
      "than the total, include confidently rejected peptides as docking controls "
      "and check the charge distribution of its shortlist against its source pool, "
      "because the two stages disagree on charge in opposite directions and a "
      "peptide clearing both is neutral enough to escape two biases rather than "
      "confirmed by two methods. And a shortlist drawn from a parameter-efficient "
      "fine-tune should be reported as an intersection or mean ensemble across "
      "seeds: repeating this screen under five seeds returns shortlists of 153 to "
      "412 candidates with only 111 common to all, so 73% of a single fit's output "
      "does not survive a change of seed."),
}

# the new 3.3 paragraph goes after "The discovery set is charge selected"
INSERT_AFTER = {
74: [
 "The shortlist is also not stable across fine-tuning seeds. Rescoring all 3,596 "
 "candidates with each of the five adapters, holding the XGBoost half of the "
 "consensus fixed because the frozen embedding it reads does not move with the "
 "seed, returns shortlists of 412, 277, 238, 176 and 153 at the same 0.90 "
 "threshold (Table S15). The published 412 is the largest of the five and 1.73 "
 "times the median. Of those 412, 111 (26.9%) clear the threshold under every "
 "seed while 123 (29.9%) clear it under seed 42 alone, giving a median selection "
 "frequency of three seeds out of five. The intersection across seeds holds 111 "
 "candidates and the mean ensemble 210, seven of which are absent from the "
 "published set. Selection frequency separates the top of the ranking from the "
 "bulk: FVAPFPEVF, GPFPSIL and LPGF are selected by all five seeds at consensus "
 "scores between 0.963 and 0.998, and IPAVF by four. Tier 1 as a whole is weaker, "
 "its 216 members carrying a median selection frequency of two of five against "
 "three of five for Tier 2's 27. What survives a change of seed is the head of "
 "the ranking rather than the shortlist as a whole, and Section 4.3 states the "
 "reporting consequence.",
],
}


def main():
    x = open(DOC, encoding="utf-8").read()
    spans = [(m.start(), m.end(), m.group(0)) for m in PARA.finditer(x)]

    fails = []
    for i, (expect, _) in REPLACE.items():
        if i >= len(spans) or not ptext(spans[i][2]).lstrip().startswith(expect):
            got = ptext(spans[i][2])[:60] if i < len(spans) else "<out of range>"
            fails.append(f"para {i}: expected {expect[:40]!r}, found {got!r}")
    for i in INSERT_AFTER:
        if not ptext(spans[i][2]).lstrip().startswith("The discovery set is charge"):
            fails.append(f"para {i}: insert anchor moved")
    if fails:
        print("FAILED:")
        [print("  " + f) for f in fails]
        sys.exit(1)

    edits = [(spans[i][0], spans[i][1], rebuild(spans[i][2], new))
             for i, (_, new) in REPLACE.items()]
    for i, blocks in INSERT_AFTER.items():
        edits.append((spans[i][1], spans[i][1],
                      "".join(BODY.format(runs(t)) for t in blocks)))

    for start, end, text in sorted(edits, key=lambda e: -e[0]):
        x = x[:start] + text + x[end:]

    # Table S15 joins the contents list in the Supporting Information paragraph
    old = "the negative-class ablation under a family-aware split (Table S14);"
    new = (old[:-1] + "; selection stability of the shortlist across the five "
           "fine-tuning seeds (Table S15);")
    if x.count(esc(old)) == 1:
        x = x.replace(esc(old), esc(new))
    else:
        print("note: SI contents sentence not matched, add Table S15 by hand")

    open(DOC, "w", encoding="utf-8").write(x)
    print(f"replaced {len(REPLACE)} paragraphs, inserted "
          f"{sum(len(b) for b in INSERT_AFTER.values())}")


if __name__ == "__main__":
    main()
