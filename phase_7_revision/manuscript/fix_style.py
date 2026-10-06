"""
Remove ", and" clause joins from the paragraphs rewritten for the second draft.

House rule: two independent clauses are never joined with ", and". Each is split
into two sentences, turned into an appositive or converted to a participle. Only
text written for this revision is touched; the first draft's own prose carries
the same construction at the same density and is left for the authors to decide
on, since rewriting accepted prose unasked is not this script's job.

Applied to both the manuscript and the response document.
"""

import re
import sys

# (file, old, new) with old distinctive enough to appear once
FIXES = [
("ms/word/document.xml",
 "at 0.877 and 0.854, and no stacking variant improved on the best single model.",
 "at 0.877 and 0.854. No stacking variant improved on the best single model."),
("ms/word/document.xml",
 "differ by 54 points in false-positive rate, and under a family-aware split the "
 "best-scoring arm is the worst-deploying one, so test AUC",
 "differ by 54 points in false-positive rate. Under a family-aware split the "
 "best-scoring arm is the worst-deploying one, so test AUC"),
("ms/word/document.xml",
 "contains the baseline, and Bayesian hyperparameter optimisation moves the best "
 "tree learner by 0.011.",
 "contains the baseline. Bayesian hyperparameter optimisation moves the best tree "
 "learner by 0.011."),
("ms/word/document.xml",
 "design and reporting rules (Section 4), and the peptides reported are "
 "prioritised hypotheses rather than confirmed activities.",
 "design and reporting rules (Section 4). The peptides reported are prioritised "
 "hypotheses rather than confirmed activities."),
("ms/word/document.xml",
 "identity to a training sequence on that measure, and they concentrate in the "
 "band clustering never covered",
 "identity to a training sequence on that measure, concentrated in the band "
 "clustering never covered"),
("ms/word/document.xml",
 "inflates discrimination by about 0.046, and that is the allowance to carry when "
 "reading every AUC reported here.",
 "inflates discrimination by about 0.046, the allowance to carry when reading "
 "every AUC reported here."),
("ms/word/document.xml",
 "the C-peptide and the signal peptide, and 141 positives (14.6%) are "
 "preproinsulin fragments",
 "the C-peptide and the signal peptide. A further 141 positives (14.6%) are "
 "preproinsulin fragments"),
("ms/word/document.xml",
 "chains the set into a 242-member family, and it does the same to the negatives",
 "chains the set into a 242-member family, doing the same to the negatives"),
("ms/word/document.xml",
 "with the predictors benchmarked here, and the consequences are traced in "
 "Sections 3.1 and 3.6.",
 "with the predictors benchmarked here. The consequences are traced in Sections "
 "3.1 and 3.6."),
("ms/word/document.xml",
 "capping arm C at 303, and DBAASP runs out of sequences at the positives' "
 "lengths, capping arm D at 611.",
 "capping arm C at 303. DBAASP runs out of sequences at the positives' lengths, "
 "capping arm D at 611."),
("ms/word/document.xml",
 "so it inherits the same direction of bias, and a grouped sweep would place the "
 "operating point higher and admit fewer candidates.",
 "so it inherits the same direction of bias. A grouped sweep would place the "
 "operating point higher and admit fewer candidates."),
("ms/word/document.xml",
 "the interface backbone in its third and fourth, and then by a restrained water "
 "refinement.",
 "the interface backbone in its third and fourth, then by a restrained water "
 "refinement."),
("ms/word/document.xml",
 "rather than a representative fit, and it sits 0.035 above the five-seed mean of "
 "0.842 that the same table reports beside it.",
 "rather than a representative fit, sitting 0.035 above the five-seed mean of "
 "0.842 that the same table reports beside it."),
("ms/word/document.xml",
 "not an estimate and its uncertainty, and the adapter carried into the screen, "
 "the shortlist and the case study is that best-seed fit.",
 "not an estimate and its uncertainty. The adapter carried into the screen, the "
 "shortlist and the case study is that best-seed fit."),
("ms/word/document.xml",
 "summed over every ionisable side chain in the sequence, and what it is reading "
 "through it is the presence",
 "summed over every ionisable side chain in the sequence. What it reads through "
 "that number is the presence"),
("ms/word/document.xml",
 "at one atom, in one position, and that group is present on every peptide "
 "substrate whatever its net charge.",
 "at one atom, in one position, a group present on every peptide substrate "
 "whatever its net charge."),
("ms/word/document.xml",
 "a proxy for the interaction the enzyme makes, and a cationic side chain "
 "elsewhere in the sequence does not obstruct it.",
 "a proxy for the interaction the enzyme makes, nor does a cationic side chain "
 "elsewhere in the sequence obstruct it."),
("ms/word/document.xml",
 "reject a peptide that would satisfy the second, and why the shortcut is a "
 "property of the negative class",
 "reject a peptide that would satisfy the second, and the reason the shortcut is "
 "a property of the negative class"),
("ms/word/document.xml",
 "rather than by anything about the sequences, and the Tier-1 lead carried forward "
 "here is itself flagged as an allergen at 0.54 against a 0.3 cut-off.",
 "rather than by anything about the sequences. The Tier-1 lead carried forward "
 "here is itself flagged as an allergen at 0.54 against a 0.3 cut-off."),
("ms/word/document.xml",
 "whole cleft rather than marking the S1 pocket, and nothing in the scoring "
 "function rewards placing",
 "whole cleft rather than marking the S1 pocket, while nothing in the scoring "
 "function rewards placing"),
("ms/word/document.xml",
 "competes with a substrate would settle it, and until then the pose is best read "
 "as a plausible binding geometry of unresolved mechanism.",
 "competes with a substrate would settle it. Until then the pose is best read as "
 "a plausible binding geometry of unresolved mechanism."),
("ms/word/document.xml",
 "relative in their own fold's training rows, and grouping the folds by family "
 "lowers out-of-fold AUC by 0.020 and moves",
 "relative in their own fold's training rows, while grouping the folds by family "
 "lowers out-of-fold AUC by 0.020 and moves"),
("ms/word/document.xml",
 "calibrated without that advantage, and would admit fewer candidates than the 412 "
 "reported here.",
 "calibrated without that advantage, admitting fewer candidates than the 412 "
 "reported here."),
("ms/word/document.xml",
 "model had already seen rather than discovery, and the four shorter training "
 "positives running inside the Tier-1 lead are an instance of it.",
 "model had already seen rather than discovery. The four shorter training "
 "positives running inside the Tier-1 lead are an instance of it."),
("ms/word/document.xml",
 "highest AUCs and the highest deployment error, and under a family-aware split "
 "the best-scoring arm is the worst-deploying one.",
 "highest AUCs and the highest deployment error. Under a family-aware split the "
 "best-scoring arm is the worst-deploying one."),
("ms/word/document.xml",
 "the fine-tuning step rather than of the embedding, and a screen that needs a "
 "reproducible shortlist is better served",
 "the fine-tuning step rather than of the embedding. A screen that needs a "
 "reproducible shortlist is better served"),
("ms/word/document.xml",
 "so it has to be fitted, and the only question is whether it is fitted alongside "
 "the encoder or on top of it.",
 "so it has to be fitted. The only question is whether it is fitted alongside the "
 "encoder or on top of it."),
("ms/word/document.xml",
 "when this dataset was built, and the ablation reported here shows that we should "
 "have.",
 "when this dataset was built. The ablation reported here shows that we should "
 "have."),
("ms/word/document.xml",
 "the best of the five seeds and the 0.842 is their mean, and the adapter carried "
 "into the screen and the case study is that best-seed fit.",
 "the best of the five seeds and the 0.842 is their mean. The adapter carried "
 "into the screen and the case study is that best-seed fit."),
("ms/word/document.xml",
 "labelled “free N-terminal amine”, and the arrow marks the unmade 6.5 Å approach "
 "to the Glu206 carboxylate.",
 "labelled “free N-terminal amine”, with the arrow marking the unmade 6.5 Å "
 "approach to the Glu206 carboxylate."),
("ms/word/document.xml",
 "intersection of shortlists across seeds, or the mean ensemble across them, would "
 "give",
 "intersection of shortlists across seeds or the mean ensemble across them would "
 "give"),
]


def main():
    by_file = {}
    for path, old, new in FIXES:
        by_file.setdefault(path, []).append((old, new))

    fails, done = [], 0
    for path, pairs in by_file.items():
        x = open(path, encoding="utf-8").read()
        for old, new in pairs:
            o = old.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            n = x.count(o)
            if n != 1:
                fails.append(f"{path}: matched {n}x: {old[:60]}")
                continue
            x = x.replace(o, new.replace("&", "&amp;").replace("<", "&lt;")
                          .replace(">", "&gt;"))
            done += 1
        open(path, "w", encoding="utf-8").write(x)

    if fails:
        print("UNMATCHED:")
        for f in fails:
            print("  " + f)
    print(f"applied {done} of {len(FIXES)} fixes")
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
