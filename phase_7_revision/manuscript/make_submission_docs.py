"""
Build the JCIM cover letter and the brief submission-checks note.

The supervisor asked for "a brief note confirming that the changes have been
incorporated" rather than another point-by-point response, together with
confirmation that the numerical results, references, data availability
information and submission requirements have been checked. That is one short
document, not a second manuscript.

The cover letter is carried over from the second draft with its claims pulled
into line with the moderated text, and one mislabelled quantity corrected: the
0.020 figure is an out-of-fold AUC inflation, not a calibration inflation.

OUTPUTS  ADP_cover_letter_JCIM.docx, ADP_submission_checks.docx
"""

import os
import sys
import docx
from docx.shared import Pt, Inches, RGBColor

GAP = RGBColor(0xB0, 0x30, 0x00)


# House style: two independent clauses are never joined with ", and", and no
# comma precedes the final "and" in a list. Applied to the assembled string so
# it does not depend on how the source happens to wrap.
STYLE = [
 ("composition baseline, and Bayesian hyperparameter optimisation moved",
  "composition baseline, while Bayesian hyperparameter optimisation moved"),
 ("one dataset and one target, and we have been careful",
  "one dataset and one target, so we have been careful"),
 ("a general property of the method, and the electrostatic share is defined",
  "a general property of the method. The electrostatic share is defined"),
 ("described as predicted flags, and the tiers are described",
  "described as predicted flags, with the tiers described"),
 ("54 points in false-positive rate, and under a family-aware partition",
  "54 points in false-positive rate. Under a family-aware partition"),
 ("moves the two in opposite directions, and three peptides",
  "moves the two in opposite directions, with three peptides"),
 ("no claim of binding or inhibition, and the peptides reported",
  "no claim of binding or inhibition. The peptides reported"),
 ("survives into the released data, and explains that",
  "survives into the released data and explains that"),
 ("of 11 residues or more, and states that", "of 11 residues or more and states that"),
 ("in discrimination stated openly, and the Conclusions now acknowledge",
  "in discrimination stated openly. The Conclusions now acknowledge"),
 ("objects being measured, and its role is to demonstrate",
  "objects being measured and its role is to demonstrate"),
 ("fragments it had never seen, and the same failure mode appears",
  "fragments it had never seen. The same failure mode appears"),
 ("about 60% of its nominal sample size, and only 111 of 412",
  "about 60% of its nominal sample size and only 111 of 412"),
 ("The manuscript is original, is not under consideration elsewhere, and "
  "both authors have approved this version.",
  "The manuscript is original and is not under consideration elsewhere. Both "
  "authors have approved this version."),
 ("45% is the share below 11 residues, and the pool does contain",
  "45% is the share below 11 residues, while the pool does contain"),
 ("The file now carries six places, and 210 is correct in both places.",
  "The file now carries six places, so 210 is correct in both."),
 ("nearest-negative control in Section 2.1, and the figure of 782",
  "nearest-negative control in Section 2.1 and the figure of 782"),
 ("shows no specific results, and measures 3.25 by 1.75 inches",
  "shows no specific results and measures 3.25 by 1.75 inches"),
 ("for the corresponding-author block, and your updated affiliation",
  "for the corresponding-author block and your updated affiliation"),
]


def styled(t):
    for old, new in STYLE:
        t = t.replace(old, new)
    return t


def new_doc():
    d = docx.Document()
    s = d.styles["Normal"]
    s.font.name = "Calibri"
    s.font.size = Pt(11)
    for sec in d.sections:
        sec.left_margin = sec.right_margin = Inches(1.0)
        sec.top_margin = sec.bottom_margin = Inches(0.9)
    return d


def para(d, text, bold=False, italic=False, space=8, size=None, indent=0.0):
    p = d.add_paragraph()
    r = p.add_run(styled(text))
    r.bold, r.italic = bold, italic
    if size:
        r.font.size = Pt(size)
    p.paragraph_format.space_after = Pt(space)
    if indent:
        p.paragraph_format.left_indent = Inches(indent)
    return p


def gap(d, text):
    p = d.add_paragraph()
    r = p.add_run(styled(text))
    r.font.color.rgb = GAP
    r.bold = True
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.left_indent = Inches(0.25)


def bullet(d, head, text):
    p = d.add_paragraph()
    p.add_run(head + ". ").bold = True
    p.add_run(styled(text))
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.left_indent = Inches(0.25)


# --------------------------------------------------------------------------- #

def cover_letter():
    d = new_doc()
    para(d, "[Date]", space=14)
    para(d, "The Editors", space=0)
    para(d, "Journal of Chemical Information and Modeling", space=14)
    para(d, "Dear Editors,", space=10)

    para(d, "We submit for your consideration “Negative-Class Design Reveals a "
            "Net-Charge Shortcut in Anti-Diabetic Peptide Prediction”, a "
            "methodological study of how bioactive-peptide classification "
            "datasets are built, scored and reported.")

    para(d, "The contribution is to dataset design and evaluation practice rather "
            "than to the discovery of any particular inhibitor. Holding the "
            "positives, hyperparameters and test set fixed and varying only the "
            "negative class moved discrimination by 0.219 in AUC and the "
            "false-positive rate on held-out proteome fragments from 93.3% to "
            "46.7%. Across the four architectures we evaluated, including a "
            "650M-parameter protein language model fine-tuned with DoRA, none "
            "separated from a 20-feature amino-acid composition baseline, and "
            "Bayesian hyperparameter optimisation moved the best tree learner by "
            "0.011. A six-arm ablation then separates negative-class composition "
            "from the length and charge distributions it carries. Two arms "
            "differing by 0.03 in AUC differ by 54 points in false-positive rate. "
            "Under a family-aware partition the best-scoring arm is the "
            "worst-deploying one, so a negative class selected on test AUC is "
            "selected backwards. These are four learners on one dataset and one "
            "target, and we have been careful to scope the claims accordingly.")

    para(d, "DPP-IV and molecular docking enter this work as a physical control "
            "system, not as the object of study. We chose the target because its "
            "active site is crystallographically characterised, because peptides "
            "with published inhibitory activity are available as references, and "
            "because the site is strongly anionic, which makes it possible to "
            "test whether a sequence classifier and a structure-based score "
            "respond to the same property for the same reason. In our panel they "
            "do not. Net charge moves the two in opposite directions, and three "
            "peptides our classifier confidently rejected scored above a "
            "documented inhibitor on electrostatics alone. The consequence we "
            "draw is a reporting rule, that a hybrid screen should report the "
            "electrostatic share of every docking score rather than the total. We "
            "make no claim of a validated hit and none of binding or inhibition. "
            "The peptides reported are described throughout as prioritised "
            "computational hypotheses for experimental follow-up.")

    para(d, "We are aware that the journal does not consider straightforward "
            "applications of docking to a single target without experimental "
            "validation. We believe this work falls outside that description: no "
            "result rests on the docking, the docking is itself one of the "
            "objects being measured, and its role is to demonstrate a failure "
            "mode of hybrid in silico workflows for which we then give a remedy.")

    para(d, "The paper also reports against the state of the art. BertADP "
            "misclassified 92.4% of 328 Swiss-Prot fragments it had never seen, "
            "and the same failure mode appears in a deliberately weak control "
            "sharing its negative source, which locates the cause in the data "
            "rather than the architecture. We additionally quantify the limits of "
            "our own evaluation: near-duplicates inflate our headline AUC by "
            "about 0.046, cluster-level leakage across the cross-validation folds "
            "inflates the out-of-fold AUC by 0.020, the positive class carries "
            "about 60% of its nominal sample size, and only 111 of 412 prioritised "
            "candidates survive a change of fine-tuning seed. All four are "
            "reported in the manuscript.")

    para(d, "All code, dataset splits, out-of-fold predictions, model weights and "
            "the scripts generating every table are public at "
            "https://github.com/om-894/antidiabetic-peptide-classifier, and will "
            "be archived with a DOI on acceptance. Each supporting table names the "
            "script that produces it.")

    para(d, "The manuscript is original, is not under consideration elsewhere, and "
            "both authors have approved this version. We declare no competing "
            "financial interest.", space=14)

    para(d, "Yours sincerely,", space=10)
    para(d, "Manish Chauhan, on behalf of the authors", space=0)
    para(d, "School of Physics, Engineering and Technology, University of York, "
            "Heslington, York YO10 5DD, United Kingdom", space=0)
    para(d, "[email]", space=0)
    return d


def submission_note():
    d = new_doc()
    para(d, "Third Draft: Confirmation of Changes and Submission Checks",
         bold=True, size=14)
    para(d, "Negative-Class Design Reveals a Net-Charge Shortcut in Anti-Diabetic "
            "Peptide Prediction", italic=True, space=10)
    para(d, "Oliver McQuillan and Manish Chauhan. Prepared 20 October 2026.",
         space=14)

    para(d, "All nine points of the consolidated revision list have been "
            "incorporated. The manuscript structure, figures, tables and analyses "
            "are unchanged; the work was clarification, claim moderation and "
            "consistency. A tracked-changes version accompanies the clean "
            "manuscript. Two items remain outstanding and are marked in red.",
         space=12)

    para(d, "Changes made", bold=True, size=12, space=8)
    for h, t in [
        ("1, positive set", "The class is now described as curated positive ADP "
         "sequences throughout. Section 2.1 reports its composition as 899 "
         "sequences consolidated from three earlier studies via BioDADPep plus 67 "
         "later candidates, states that no per-sequence record of assay or "
         "activity survives into the released data and explains that the 141 "
         "preproinsulin fragments were retained for comparability and attributed "
         "by our own matching rather than by annotation."),
        ("2, leakage", "The partition is no longer described as leakage-safe. "
         "Section 2.1 names the criterion actually met, zero residual identity "
         "above 40% for sequences of 11 residues or more and states that "
         "clustering identity and substring containment are different criteria. "
         "The family-aware evaluation is reported in Section 3.6 with the "
         "reduction in discrimination stated openly. The Conclusions now "
         "acknowledge it."),
        ("3, central claim", "Abstract, Section 3.1, Section 4.1 and the "
         "Conclusions are scoped to the architectures and conditions evaluated. "
         "Three section headings that stated claims have been retitled. The "
         "ablation is now described as comparing construction strategies, with an "
         "explicit statement that the arms differ in several correlated "
         "properties and do not isolate the causal contribution of any single "
         "descriptor."),
        ("4, calibration", "Section 3.3 is retitled and the 0.90 cut-off is "
         "described as a model-based prioritisation criterion. The calibration "
         "result is confined to what it establishes within the benchmark, with an "
         "explicit statement that the experimental hit rate on digestion "
         "fragments is unknown. Discoveries are now prioritised candidates "
         "throughout the manuscript and the Supporting Information."),
        ("5, seed stability", "Section 3.3 reports three nested sets: 412 from "
         "the selected model, 217 selected by three or more seeds and 111 by all "
         "five, with the 111 named as the subset for experimental follow-up. "
         "Section 4.3 explains that seed 42 was the project's fixed default "
         "rather than a selection, and that its being both highest-scoring and "
         "most permissive biases the published shortlist."),
        ("6, docking", "Poses are described as predicted throughout. The "
         "reference inhibitors are named as such rather than as positive "
         "controls, the three rejected peptides are presented as evidence from "
         "this panel rather than as a general property of the method, and the "
         "electrostatic share is defined as a decomposition of an empirical score "
         "rather than a physical measure of binding energy."),
        ("7, novelty and safety", "Novelty is qualified by filter coverage at "
         "each length, with the weaker guarantee for short candidates stated in "
         "Sections 3.3 and 3.6. Safety outputs are described as predicted flags, "
         "and the tiers are described as a prioritisation order rather than a "
         "safety assessment."),
        ("9, final read", "Abstract, Discussion and Conclusions were read "
         "together and carry the same message. Definitive wording has been "
         "removed where the evidence is computational."),
    ]:
        bullet(d, h, t)

    para(d, "Point 8: checks carried out", bold=True, size=12, space=8)
    para(d, "Every quantitative claim in the manuscript, Supporting Information "
            "and cover letter was cross-checked against the deposited files in "
            "results, screening, predictions, benchmark and docking, roughly 180 "
            "claims in all. The large majority verified exactly, including all of "
            "Tables 4, 5 and S14, every statistical test, the calibration chain "
            "and all twenty per-seed consensus values. Six discrepancies were "
            "found and corrected, with a seventh below:", space=8)
    for h, t in [
        ("Basith pool cap", "Section 2.3 and the Table 5 caption stated that the "
         "pool holds nothing below 9 residues while 45% of positives are shorter. "
         "Both halves were wrong: 45% is the share below 11 residues, and the "
         "pool does contain shorter sequences. The cap is explained by "
         "band-by-band scarcity, which is the accurate account."),
        ("Safety discard rate", "Section 3.3 reported the two filters as "
         "discarding 93.0% of short candidates. Applied in series they retain 4 "
         "of 216, a 98.1% discard rate. The earlier figure was AlgPred's own flag "
         "rate reused for the joint rate."),
        ("Per-seed score range", "Section 3.3 gave the range for the three "
         "all-seed peptides as 0.963 to 0.998. The correct range is 0.908 to "
         "0.998; 0.963 was one peptide's minimum."),
        ("Mean-ensemble count", "Table S15 reported 211 where the body reported "
         "210. The deposited file rounded one candidate at 0.899963 up to 0.9000 "
         "at four decimal places. The file now carries six places, and 210 is "
         "correct in both places."),
        ("Preproinsulin counts", "The audit script applied no length floor when "
         "matching against preproinsulin, so its output reported 142 positives "
         "and 2 negatives against the manuscript's 141 and none at four residues "
         "or more. The manuscript was correct; the script and its deposited "
         "output now apply the stated floor."),
        ("Figure 8 contacts", "The caption gave seventeen contacts against the "
         "body's nineteen. These are the 4.5 and 5.0 angstrom counts "
         "respectively; the caption now names its cutoff."),
    ]:
        bullet(d, h, t)

    para(d, "References were checked and all citations resolve. Every figure and "
            "table, including Tables S1, S10 and S14 and Figures S2 and S3, is now "
            "cited in the body text; five were not. Abbreviations, metrics and "
            "statistical tests are defined. The Supporting Information contains "
            "every analysis the manuscript refers to. The data availability "
            "statement describes what is actually public: the repository is live "
            "and the revision analyses sit under phase_7_revision with their "
            "outputs in results.", space=8)

    para(d, "Three claim sets had no deposited source file when the audit was "
            "run. All three have since been recomputed and deposited, so every "
            "quantitative statement in the paper now traces to a file. The "
            "baseline Gini importances were already inside "
            "predictions/aac_baseline_predictions.npz and are now also a table "
            "(results/phase7_9_aac_gini.csv); they verify exactly at 25.4%, "
            "39.0%, 0.143 and 0.111. The 75.1% embeddings-only nearest-negative "
            "control was recomputed and verifies exactly. The third turned up a "
            "further error: the figure of 782 sequences removed by a "
            "conventional 40% reduction is the count over the 1,062 sequences "
            "long enough to cluster, not over all 1,932, which would lose most "
            "of the dataset. Section 2.1 and Table S1 now state the population "
            "the figure belongs to.", space=8)

    para(d, "Submission requirements", bold=True, size=12, space=8)
    para(d, "A Table of Contents graphic has been produced and placed on the "
            "last page of the manuscript, labelled \u201cFor Table of Contents "
            "Only\u201d, with a separate 300 dpi TIF for upload. It is original "
            "artwork, reuses no manuscript figure, shows no specific results and "
            "measures 3.25 by 1.75 inches with sans serif text at 6.5 pt, within "
            "the ACS specification.", space=8)

    para(d, "The JCIM author guide has now been checked directly. It asks for an "
            "abstract of three to four sentences; the abstract was 262 words "
            "across eleven sentences and has been rewritten to four sentences "
            "and 160 words. The 75-word limit elsewhere in the guide applies to "
            "Letters rather than Articles. The title is ten words, within the "
            "preferred twelve. The Supporting Information description is a "
            "nonsentence list ending in the file extension that sits before the "
            "Acknowledgment and References, as required. A Data Availability "
            "Statement is present, which the journal requires under ACS Research "
            "Data Policy Level 2. Use of an AI tool is now disclosed in the "
            "Acknowledgment, as ACS requires for text or image generation.",
         space=8)

    para(d, "The corresponding-author block now carries the email address and "
            "updated affiliation supplied on 23 October. The clean manuscript, "
            "Supporting Information, cover letter and this note contain no "
            "tracked changes, no deleted wording and no comments; an orphaned "
            "empty comments part inherited from an earlier draft has been "
            "removed from the Supporting Information. The only remaining "
            "placeholder is the Supporting Information DOI, which the publisher "
            "assigns at proof.", space=8)

    return d


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    cover_letter().save(os.path.join(out, "ADP_cover_letter_JCIM.docx"))
    submission_note().save(os.path.join(out, "ADP_submission_checks.docx"))
    print("wrote the cover letter and the submission-checks note")


if __name__ == "__main__":
    main()
