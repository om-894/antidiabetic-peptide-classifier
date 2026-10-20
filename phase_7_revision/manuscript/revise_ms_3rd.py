"""
Third-draft revisions to the manuscript, rendered clean and tracked.

One edit table, two outputs. Paragraphs are addressed by index in the second
draft because Word splits a paragraph across runs wherever formatting changes,
so a paragraph carrying a superscript citation or a bold lead-in is not a
contiguous string and cannot be matched as one. Every target's opening words
are asserted before it is touched.

Replacement text carries two bits of markup: **bold** and ^{12} for a
superscript citation. The clean renderer turns those into runs directly; the
tracked renderer word-diffs old against new and wraps only what moved, so the
citations survive as superscripts on both sides of a revision.

The edits implement the supervisor's nine consolidated points. Nothing is
recomputed: every number here already exists in results/.

INPUTS   ms/word/document.xml (second draft, runs merged)
OUTPUTS  the clean and tracked document.xml bodies
"""

import re
import sys

sys.path.insert(0, __import__("os").path.dirname(__file__))
from tracked_render import tracked_paragraph, inserted_paragraph, Ids, new_tokens, esc

PARA = re.compile(r'<w:p(?: [^>]*)?>.*?</w:p>|<w:p(?: [^>]*)?/>', re.S)
AUTHOR = "Oliver McQuillan"
DATE = "2026-10-20T00:00:00Z"
BODY_PPR = ('<w:pPr><w:spacing w:line="480" w:lineRule="auto"/>'
            '<w:jc w:val="both"/></w:pPr>')


def ptext(p):
    return "".join(re.findall(r'<w:t(?: [^>]*)?>(.*?)</w:t>', p, re.S))


def clean_runs(text):
    return "".join(f'<w:r>{rpr}<w:t xml:space="preserve">{esc(t)}</w:t></w:r>'
                   for rpr, t in _group(new_tokens(text)))


def _group(tokens):
    out, buf, cur = [], [], None
    for tok, rpr in tokens:
        if rpr != cur and buf:
            out.append((cur, "".join(buf)))
            buf = []
        cur = rpr
        buf.append(tok)
    if buf:
        out.append((cur, "".join(buf)))
    return out


def clean_paragraph(para, text):
    m = re.search(r'<w:pPr>.*?</w:pPr>', para, re.S)
    ppr = m.group(0) if m else ""
    return (f"{re.match(r'<w:p(?: [^>]*)?>', para).group(0)}{ppr}"
            f"{clean_runs(text)}</w:p>")


# --------------------------------------------------------------------------- #
# EDITS: index -> (expected opening words, replacement text)
# --------------------------------------------------------------------------- #

REPLACE = {

# --- Abstract: scoped claims, prioritisation language, curated positives ----
6: ("ABSTRACT: Classifiers for anti-diabetic",
    "ABSTRACT: Classifiers for anti-diabetic peptides (ADPs) report high benchmark "
    "accuracy yet misclassify most of the proteome fragments a real screen "
    "returns. Holding the positives, hyperparameters and test set fixed, we "
    "varied the negative class alone. A dual-negative set of antimicrobial "
    "peptides plus GO-filtered Swiss-Prot fragments, length-matched against 966 "
    "curated positive ADP sequences, cut the false-positive rate on held-out "
    "proteome fragments from 93.3% to 46.7%. Across the four architectures "
    "evaluated, none separated from the others: a DoRA "
    "fine-tuned ESM-2 650M model and a 20-feature composition baseline bracketed "
    "the same AUC interval at 0.877 and 0.854. Net charge survives as the "
    "residual shortcut, carried by the antimicrobial half of the "
    "class and shared by the published state of the art, which misclassified "
    "92.4% of 328 unseen fragments. It also shapes the output: the 412 "
    "prioritised candidates average −0.79 in net charge against −0.11 across "
    "their source pool. A six-arm ablation separates negative-class composition "
    "from the length and charge distributions it carries. Length-matching a "
    "published negative pool leaves its false-positive rate above 79%, whereas a "
    "class matched on length and charge together reduces the charge signal to "
    "chance for 0.018 in AUC. Two arms differing by 0.03 in AUC differ by 54 "
    "points in false-positive rate, so test AUC is a poor guide to screen "
    "behaviour. Docking inverts the same bias, cationic non-binders drawing two "
    "thirds of their favourable score from electrostatics. Repeating the screen "
    "under five fine-tuning seeds returns shortlists of 153 to 412 candidates "
    "with only 111 common to all. Negative-class construction therefore warrants "
    "the attention usually given to architecture; such shortlists should be "
    "reported across seeds."),

# --- 2.1 positive class: provenance, not an evidence claim (his point 1) ----
16: ("The positive class comprised the 966",
     "The positive class comprised the 966 non-redundant curated positive ADP "
     "sequences consolidated by Xie et al.,^{5} the largest such set in the "
     "field. Its composition is uneven in provenance and the distinction matters "
     "for what a positive label means here. The released dataset separates the "
     "sequences into two groups of 899 and 67. The 899 were consolidated from "
     "three earlier studies,^{3,4} all drawing on the BioDADPep database; the 67 "
     "were collected separately as peptides reported to have anti-diabetic "
     "potential, of which Xie et al. state that several have been experimentally "
     "confirmed. The two groups differ in a way that corroborates the split: no "
     "sequence in the 899 is shorter than five residues, the floor applied by the "
     "source studies, whereas 21 of the 67 are, including dipeptides. We "
     "therefore describe the class as curated rather than experimentally "
     "validated. No per-sequence record of assay, activity value or reference "
     "survives into the released data, so individual peptides cannot be ranked by "
     "strength of evidence. BioDADPep itself carries such fields, but they were "
     "discarded upstream of this work and could not be recovered. Against these "
     "we built a dual-negative class of equal size, drawn one negative per "
     "positive at the exact length of that positive, so no length signal "
     "separates the classes (Table 1, Table S1, Figure 1)."),

# --- 2.1 leakage terminology (his point 2) ---------------------------------
20: ("The partition is leakage-safe",
     "The partition removes all residual identity above the clustering threshold "
     "for every sequence long enough to cluster. CD-HIT v4.8.1^{13} clustered "
     "positives and negatives together at 40% global identity (word length 2, -l "
     "1, seed 42) and whole clusters were assigned to train or test at a 15% "
     "target, so near-duplicates at that threshold cannot straddle the split. "
     "Residual cross-set identity was removed with cd-hit-2d against train as "
     "reference, iterated until no test sequence exceeded 40% identity to a "
     "training sequence over at most 15 passes, moving test sequences to train "
     "only, so the realised test fraction landed at 9.2%. The 870 sequences below "
     "11 residues bypassed clustering and took a stratified random split per "
     "class, since 40% identity is not meaningful over so few residues. Clustering "
     "identity is one definition of sequence similarity and not the only one: it "
     "does not detect substring containment, and it does not reach the short half "
     "of this dataset at all. The partition should therefore not be described as "
     "free of all train-test relationships, and the next paragraph measures the "
     "ones it leaves. Nothing was deleted for redundancy: a conventional 40% "
     "reduction would remove a large minority of the 1,932 sequences, many "
     "positives being overlapping insulin-derived fragments, and near-duplicates "
     "inside the training set are useful hard cases."),

# --- 2.1 overlap audit: name the two definitions (his point 2) -------------
21: ("Because clustering does not reach",
     "Because clustering reaches neither the short half of the dataset nor "
     "substring relationships at any length, the residual proximity between the "
     "two partitions was measured rather than assumed small (Table S11). No test "
     "sequence occurs exactly in the training set. Sixty-seven of the 178 test "
     "sequences (37.6%) stand in a substring relationship with a training "
     "sequence, against 20.9% under a null that shuffles each test sequence's "
     "residues and so holds its length and composition fixed (200 shuffles, P < "
     "0.005). Substring containment and clustering identity are different "
     "criteria, and a partition clean under one is not thereby clean under the "
     "other. Graded identity was computed as matched residues over the longer "
     "sequence of each pair, since CD-HIT's convention of normalising over the "
     "shorter saturates at 1.0 for any containment and carries no information at "
     "these lengths. Forty-one test sequences (23.0%) reach 80% identity to a "
     "training sequence on that measure, and they concentrate in the band "
     "clustering never covered: 35 of the 130 test sequences below 11 residues "
     "against 6 of the 48 above. Removing those 41 rows and rescoring the saved "
     "predictions, with no model refitted, lowers test AUC from 0.877 to 0.831 for "
     "ESM-2/DoRA, from 0.857 to 0.815 for XGBoost and from 0.879 to 0.836 for the "
     "consensus. The split inflates discrimination by about 0.046, and that is "
     "the allowance to carry when reading every AUC reported here."),

# --- 2.1 positive redundancy and preproinsulin treatment (his point 1) -----
22: ("The positive class carries its own redundancy",
     "The positive class carries its own redundancy, which bears on what 966 "
     "sequences are evidence for rather than on leakage (Table S12). Grouping by "
     "single linkage where one sequence contains another and the shorter covers "
     "at least half the longer, the 966 positives collapse to 577 families "
     "against 861 for the independently drawn negatives, so the positive set "
     "carries about 60% of its nominal sample size. The three largest families "
     "are regions of human preproinsulin at 56, 55 and 32 members, covering the B "
     "chain, the C-peptide and the signal peptide. Counting every positive that "
     "is a substring of preproinsulin P01308 at four residues or more gives 141 "
     "(14.6%) against none of the 966 negatives; the four-residue floor excludes "
     "matches that arise by chance rather than by ancestry. That attribution is "
     "our own, derived by matching against the precursor sequence, and is not a "
     "database annotation. One hundred and forty of the 141 lie in the 899-"
     "sequence group. They were retained rather than removed, because removing "
     "them would break comparability with the source dataset and with the "
     "predictors benchmarked here, and because the model is not told which "
     "precursor a sequence came from. Twenty-one families straddle the published "
     "split, carrying 226 positives. The coverage condition is load-bearing: "
     "plain containment with no length floor is transitive through the two "
     "dipeptide positives and chains the set into a 242-member family, and it "
     "does the same to the negatives, which are drawn independently and cannot "
     "share ancestry, so a nesting-only definition measures the metric rather "
     "than the data. The consequences are traced in Sections 3.1 and 3.6."),

# --- 2.3 ablation: the caps, corrected (A6) --------------------------------
32: ("Arms C and D cannot reach 873 pairs",
     "Arms C and D cannot reach 873 pairs, because neither source database "
     "supplies enough sequences at the lengths the positives occupy. Forty-five "
     "per cent of the positives are shorter than 11 residues, and across lengths "
     "2 to 10 those positives need 435 partners while the Basith pool offers "
     "about 69, capping arm C at 303. DBAASP runs short in the same bands, "
     "capping arm D at 611. Each is therefore also fitted with the positives "
     "subsampled to its negative count, so its false-positive rate is not read "
     "off a positive-heavy training set. A published negative set cannot be "
     "repaired by length-matching when the source database does not span the "
     "lengths, which is itself part of the result. The readout is XGBoost on the "
     "fused vector under the deployed model's hyperparameters, reported as the "
     "mean over ten training-row permutations with their standard deviation: row "
     "subsampling draws rows in the order given, so a permutation alone moves "
     "this learner by about 0.004 in AUC, which is the size of the smallest "
     "difference between arms and makes single fits unusable for comparing them. "
     "Embeddings for the sequences the arms add were extracted with the same "
     "frozen encoder and the same pooling as Section 2.2."),

# --- 2.5 threshold as a prioritisation criterion (his point 4) -------------
39: ("Candidates were scored by consensus",
     "Candidates were scored by consensus, the unweighted mean of the ESM-2/DoRA "
     "and XGBoost probabilities, the two learners that multi-seed evaluation "
     "ranked strongest (Section 3.1). Cut-offs were swept from 0.50 to 0.99 in "
     "0.01 steps over the 1,754 out-of-fold training predictions for the lowest "
     "threshold retaining at least 30 sequences at a 90% empirical positive rate. "
     "That returned 0.87, so 0.90 was adopted as a conservative round figure. It "
     "is a model-based prioritisation criterion rather than a top-N cut, and it "
     "is calibrated against the benchmark only: the empirical positive rate "
     "behind it is measured on held-out benchmark peptides, not on digestion "
     "fragments, and it carries no claim about how often a prioritised candidate "
     "would show activity in assay. The association between consensus score and "
     "net charge across the candidate pool was tested by Spearman rank "
     "correlation. Those folds are stratified on class rather than grouped on "
     "sequence family, so the out-of-fold predictions the threshold was swept on "
     "do carry cluster-level leakage and the calibration in Figure S5 is "
     "correspondingly optimistic. Measured on the training rows under the family "
     "definition of Section 2.1, 540 of 1,754 (30.8%, between 29.4% and 31.9% per "
     "fold) have a family relative among their own fold's training rows. "
     "Refitting the XGBoost readout with folds grouped by family lowers "
     "out-of-fold AUC from 0.916 to 0.896 and moves the swept threshold from 0.93 "
     "to 0.97. The consensus threshold adopted here was swept on stratified folds "
     "and its ESM-2 half shares the identical fold assignment, so it inherits the "
     "same direction of bias. A grouped sweep would place the operating point "
     "higher and admit fewer candidates."),

# --- 2.5 safety tools as predictors (his point 7) --------------------------
40: ("Discoveries were screened in sequence",
     "Prioritised candidates were screened in sequence for predicted toxicity "
     "with ToxinPred2^{29} at 0.6 and predicted allergenicity with AlgPred "
     "2.0^{30} at 0.3, each an amino-acid-composition Random Forest. Both return "
     "computational flags rather than measured properties. Both also flag short "
     "peptides at higher rates, so they were applied as soft filters with "
     "continuous scores retained and the candidates tiered rather than excluded "
     "on a predicted flag alone. The dependence of each tool's flag rate on "
     "peptide length was tested by Fisher exact test."),

# --- 2.6 define the electrostatic share as a decomposition (his point 6) ---
45: ("Scores follow the water-refinement weighting",
     "Scores follow the water-refinement weighting, 1.0 E_vdw + 0.2 E_elec + 1.0 "
     "E_desolv + 0.1 E_AIR,^{36} which down-weights electrostatics to account for "
     "solvent screening. The downloadable summary archive carries no score file, "
     "so the score was recomputed per model from the REMARK energy records of "
     "each best-cluster PDB and averaged over the four models of the "
     "best-scoring cluster. Cluster Z is the best-cluster mean minus the mean of "
     "all cluster means, divided by their population standard deviation, "
     "recomputed here so that it measures separation within this run only. The "
     "HADDOCK score is left unitless by its developers and is not a binding "
     "affinity; component energies are in kcal/mol, and buried surface area is "
     "omitted because the water-refinement function does not include it. "
     "Electrostatic share is defined here as the weighted E_elec expressed as a "
     "percentage of the summed favourable weighted terms, unfavourable terms "
     "excluded. It is a decomposition of an empirical score, describing where "
     "that score came from, and it is not a physical measurement of the "
     "electrostatic contribution to binding. A high value marks a pose whose "
     "score is dominated by the charge term rather than by shape "
     "complementarity."),

# --- 2.6 reference inhibitors, not positive controls (his point 6) ---------
47: ("The panel comprised the Tier-1 lead",
     "The panel comprised the Tier-1 lead, both Tier-2 leads, two reference "
     "inhibitors (IPAVF and Diprotin A, IPI) and three hard negatives from the "
     "held-out test set that the consensus model rejected at scores below 0.01. "
     "The reference inhibitors are peptides with published DPP-IV inhibitory "
     "activity and are used as a scale against which to read the panel; the "
     "rejected peptides indicate what a sequence the classifier is confident "
     "about scores under the same protocol. Eight peptides is a small panel and "
     "it supports comparison within itself rather than a general characterisation "
     "of the method."),

# --- 3.1 heading, scoped --------------------------------------------------
53: ("3.1. Architecture Does Not Separate",
     "3.1. Negative-Class Composition Outweighs Architecture Across the Learners "
     "Tested"),

# --- 3.1 the six-arm result, causally softened (his point 3) ---------------
59: ("Part of that failure is a length shortcut",
     "Part of that failure is a length shortcut. The Basith pool is not "
     "length-matched to the positives, averaging 26.7 residues against 13.5, so "
     "the two training classes are separable on length alone. Among the 85 test "
     "negatives, the Basith-negative model wrongly accepts those averaging 9.2 "
     "residues and rejects those averaging 14.2, a length effect of AUC 0.659. "
     "The length-matched dual-negative model shows none at 0.431, and length "
     "itself carries no signal about the true label on this test set (AUC 0.448). "
     "That comparison still varies composition and length together, so six arms "
     "separate them (Table 5). "
     "Length alone does not account for the effect. Length-matching the Basith "
     "pool raises AUC from 0.682 to 0.710 while its false-positive rate on hard "
     "negatives does not improve, 91.8%, or 79.6% once the positives are "
     "subsampled to the 303 negatives that pool can supply. Composition is what "
     "moves deployment behaviour. Arms D and E are both length-matched and differ "
     "in which half of the negative class they draw from: the DBAASP half alone "
     "reaches AUC 0.798 with an 82.9% false-positive rate, the proteome half "
     "alone 0.826 with 29.1%. Two arms 0.03 apart in AUC are 54 points apart in "
     "the error that decides what a screen returns. "
     "The arm matched on length and charge together, arm F, reduces the "
     "separability of its training classes by net charge from 0.804 to 0.507, "
     "chance, at a cost of 0.018 in AUC against arm A (0.844 ± 0.007 against "
     "0.862 ± 0.004). It cuts the hard-negative false-positive rate from 45.1% "
     "to 33.1% and leaves the soft-negative rate low at 6.5%. Arm E is better "
     "still on proteome fragments at 29.1% but loses the bioactivity control, its "
     "soft-negative rate rising to 10.5%. These arms differ in more than the "
     "descriptor they were matched on: each is a different draw of sequences, and "
     "length, charge, hydrophobicity and database provenance covary across them. "
     "The comparison therefore establishes that the construction strategy changes "
     "deployment behaviour substantially, and does not isolate the causal "
     "contribution of any single physicochemical property. What it does show is "
     "that neither the published arm A nor the DBAASP-only arm D, the two with "
     "the highest charge separability, is the arm a screen would have been best "
     "built on."),

# --- 3.3 heading ----------------------------------------------------------
68: ("3.3. Calibrated in Silico Discovery",
     "3.3. Calibration and Model-Based Candidate Prioritisation"),

# --- 3.3 what calibration establishes, and what it does not (point 4) -----
70: ("A calibrated score maps onto a rate",
     "A calibrated score maps onto a rate rather than only a rank.^{43,44} Within "
     "the benchmark, the 0.90 cut-off admitted 49 test peptides of which 42 were "
     "positive, a precision of 86% against the 91% seen out-of-fold. The test "
     "figure is the one to carry forward, since the threshold was not tuned on "
     "it. That precision is a property of held-out benchmark peptides drawn from "
     "the same curated distribution as the training data. The screening pool is "
     "not that distribution: it is digestion fragments of twelve dietary "
     "proteins, and the fraction of prioritised candidates that would show "
     "activity in assay is unknown and is not estimated here. Applied to the "
     "screen, the cut-off returned 412 of 3,596 candidates, which is a ranked "
     "selection made on a fixed criterion rather than a predicted hit rate. None "
     "of the four predictors surveyed in Section 1 reports a calibration metric, "
     "so none of their positive sets can be compared on this basis."),

# --- 3.3 novelty qualified (his point 7) + terminology --------------------
71: ("In silico digestion of the twelve",
     "In silico digestion of the twelve dietary proteins yielded 3,596 candidates "
     "absent from the training and test sets, and yield varied by source (Table "
     "S9). Novelty here means absence under the filters applied, which differ by "
     "length: candidates of 11 residues or more were screened at 40% identity "
     "against the dataset, while the 3,060 shorter ones were screened by exact "
     "matching alone, so they are novel in a weaker sense (Section 3.6). "
     "Candidate count tracks precursor length, so lactoferrin (743) and serum "
     "albumin (751) dominate the pool, but prioritisation rate does not: "
     "αs1-casein, β-casein and ovalbumin returned 15.6% to 17.8% of their "
     "fragments above threshold against 6.3% for α-lactalbumin. The consensus "
     "returned 412 prioritised candidates (11.5%). ToxinPred2 passed 137 of these "
     "and AlgPred 2.0 passed 31."),

# --- 3.3 safety: predicted flags, and the corrected joint rate (A5) -------
72: ("Both safety tools flag short peptides",
     "Both safety tools flag short peptides at higher rates, so the prioritised "
     "candidates were tiered rather than filtered (Figure S6). ToxinPred2 flagged "
     "73.6% of the 216 candidates of 10 residues or fewer against 59.2% of the "
     "196 longer ones (Fisher exact test, P = 0.002), and AlgPred 2.0, applied to "
     "the 137 that ToxinPred2 passed, flagged 93.0% of 57 short peptides against "
     "66.2% of 80 longer ones (P < 0.001). Both are amino-acid-composition Random "
     "Forests, so the length dependence is a property of the tools rather than of "
     "the peptides. Neither binary cut-off should be applied as a hard filter to "
     "short hydrolysate fragments. Applied in series at the published thresholds, "
     "the two would retain only 4 of the 216 candidates of ten residues or fewer, "
     "a 98.1% discard rate driven by peptide length rather than by anything "
     "established about the sequences, and the Tier-1 lead carried forward here "
     "is itself flagged as a predicted allergen at 0.54 against a 0.3 cut-off. "
     "Work on food-derived fragments should retain the continuous scores, read "
     "them as one ranking signal among several and defer exclusion to assay. Tier "
     "1 holds all 216 candidates of 10 residues or fewer, whose safety scores are "
     "treated as a length artefact; Tier 2 holds the 27 of the 31 passing both "
     "filters that exceed 10 residues; 169 fall in neither tier. The tiers order "
     "candidates for follow-up and are not a safety assessment."),

# --- 3.3 charge selection, terminology ------------------------------------
74: ("The discovery set is charge selected",
     "The prioritised set is charge selected. The 412 average −0.79 in net charge "
     "with 10% cationic, against −0.11 and 33% across the 3,596 candidates they "
     "were drawn from, and consensus score falls with charge across the full pool "
     "(Spearman ρ = −0.33; n = 3,596; P < 0.001). The association of Section 3.2 "
     "therefore shapes what the screen returns, not only what it rejects on "
     "held-out data: a laboratory working from this shortlist would be testing a "
     "charge-selected subset of the digest."),

# --- 3.3 the three nested sets (his point 5), with A4 fixed ---------------
75: ("The shortlist is also not stable",
     "The shortlist is also not stable across fine-tuning seeds, so it is "
     "reported here as three nested sets rather than one. Rescoring all 3,596 "
     "candidates with each of the five adapters, holding the XGBoost half of the "
     "consensus fixed because the frozen embedding it reads does not move with "
     "the seed, returns shortlists of 412, 277, 238, 176 and 153 at the same 0.90 "
     "threshold for seeds 42 to 46 respectively (Table S15). The published 412 is "
     "the largest of the five and 1.73 times the median. Of those 412, 111 "
     "(26.9%) clear the threshold under every seed, 217 clear it under three or "
     "more, while 123 (29.9%) clear it under seed 42 alone, giving a median "
     "selection frequency of three seeds out of five. The three sets are "
     "therefore the 412 identified by the selected model, the 217 identified "
     "consistently across a majority of seeds and the 111 identified under all "
     "five, and the 111 is the subset we would carry into experimental follow-up. "
     "Two fit-independent alternatives are available beside them: the "
     "intersection across seeds, which is that same 111, and the mean ensemble, "
     "which holds 210 and includes 7 candidates the published run missed. "
     "Selection frequency separates the top of the ranking from the bulk: "
     "FVAPFPEVF, GPFPSIL and LPGF are selected by all five seeds, at per-seed "
     "consensus scores ranging from 0.908 to 0.998 across the three, and IPAVF by "
     "four. Tier 1 as a whole is weaker, its 216 members carrying a median "
     "selection frequency of two of five against three of five for Tier 2's 27. "
     "What survives a change of seed is the head of the ranking rather than the "
     "shortlist as a whole. Section 4.3 states the reporting consequence."),

# --- 3.4 heading ----------------------------------------------------------
76: ("3.4. Docking: The Lead Earns Its Score",
     "3.4. Docking: What Separates the Panel Is the Composition of the Score"),

# --- 3.4 the lead, as a predicted pose (his point 6) ----------------------
77: ("FVAPFPEVF returned the most favourable",
     "FVAPFPEVF returned the most favourable score of the panel at −100.1 (SD 1.6 "
     "over the four models of its best-scoring cluster) with a cluster Z of "
     "−1.83, ahead of both reference inhibitors at −67.7 for IPAVF and −39.1 for "
     "Diprotin A (Table 4). Its score came from van der Waals complementarity "
     "(−47.8) and desolvation (−46.0), while electrostatics contributed only "
     "−8.4 once the 0.2 water-stage weight is applied, 8.2% of the favourable "
     "total. The peptide carries no cationic side chain and one glutamate, giving "
     "it a net charge of −1.00, so it has no side chain to pair with the site's "
     "four carboxylates. In the predicted pose its three phenylalanines instead "
     "make 15 of the 19 heavy-atom contacts within 5 Å, seven of those against "
     "five of the eight aromatic residues lining the site, while Glu7 makes none "
     "and sits 5.3 Å from Arg125. The modelled interface therefore buries "
     "hydrophobic surface and leaves the peptide's only charged side chain out of "
     "contact. These are features of a predicted binding mode and are not "
     "evidence of binding."),

# --- 3.4 the rejected peptides, scoped to the panel (his point 6) ---------
78: ("The three negative controls",
     "The three rejected peptides, which the classifier had scored below 0.01, "
     "returned −60.6 to −66.8, approaching IPAVF at −67.7 and clearing Diprotin A "
     "by more than 20 units, but they earned it through electrostatics of −236 to "
     "−312. The site presents four acidic residues, Glu205, Glu206, Asp545 and "
     "Asp663, against two basic ones, so a cationic peptide meets a net excess of "
     "carboxylates wherever in the pocket it lands. Each carries two to four "
     "lysine or arginine side chains, and the site has no documented partner for "
     "any of them, since its S1 pocket takes proline or alanine and the side "
     "chains flanking the cleavage site face solvent.^{47} FVAPFPEVF and both "
     "reference inhibitors carry none. After weighting, charge supplies 66% to "
     "74% of the rejected peptides' favourable score, against 8% to 24% for the "
     "Tier-1 lead and the two reference inhibitors, with the Tier-2 leads between "
     "them at 31% to 41% (Figure 6a). Ranking on the raw HADDOCK score would have "
     "placed all three above Diprotin A, the strongest within 0.9 units of IPAVF. "
     "Three peptides are a small sample and this does not establish that the "
     "method produces false positives systematically; it shows that within this "
     "panel the total score did not separate peptides the classifier rejected "
     "from a documented inhibitor, while the composition of the score did. That "
     "ordering also bounds how much any single total here can carry, the lead's "
     "included: its −100.1 is consistent with a large, shape-complementary, "
     "desolvating interface rather than with tighter binding than Diprotin A. The "
     "composition of the score is reported beside every entry in Table 4 for that "
     "reason."),

# --- 3.6 limitations: family-aware promoted (his point 2) -----------------
92: ("The redundancy of the positive class",
     "The redundancy of the positive class is the fourth, and the robustness "
     "check it motivated is reported here rather than as an aside. The 966 "
     "positives collapse to 577 families and 141 of them are preproinsulin "
     "fragments against none of the negatives (Section 2.1), so the class carries "
     "about 60% of its nominal sample size and one precursor supplies a seventh "
     "of it. Rebuilding the partition family-aware, with whole families held out "
     "and every arm scored on one shared 290-row test set, is the stricter "
     "evaluation (Table S14). Under it the negative-class AUC gap narrows from "
     "0.180 to 0.074, and individual arms lose discrimination: the published arm "
     "falls from 0.862 to 0.850 and the charge-matched arm from 0.844 to 0.778. "
     "The gap in hard-negative false-positive rate holds at 20.6 points against "
     "36.5. Part of the AUC effect is therefore an artefact of the easier split "
     "and the deployment effect is not, which is the result that carries into "
     "Section 4. The same split sharpens the central point: the DBAASP-only arm "
     "records the best AUC of the six at 0.853 and the worst hard-negative "
     "false-positive rate at 88.6%, against 0.795 and 28.0% for the "
     "proteome-only arm. A negative class chosen on test AUC would have been "
     "chosen backwards. Nothing was synthesised, so the precision of the 412 is "
     "an expectation within the benchmark rather than a measurement on the "
     "screen, and every output here is a prioritised hypothesis for experimental "
     "follow-up rather than a confirmed activity. Two experiments would settle "
     "the main questions. Training the dual-negative model on the released "
     "BertADP split would separate negative-class composition from distribution "
     "shift. An inhibition assay against the recombinant enzyme would turn the "
     "Tier-1 lead into a measured candidate and would say whether its inhibition "
     "is competitive."),

# --- intro: scope the framing (his points 3 and 9) ------------------------
11: ("We separate the two effects",
     "We separate the two effects. Holding the positives, hyperparameters and "
     "test set fixed, we vary the negative class alone and measure the "
     "false-positive rate on held-out proteome fragments; separately, we vary "
     "architecture across four learners against a 20-feature composition "
     "baseline. Within these comparisons the negative class moves discrimination "
     "by 0.219 in AUC, while architecture spans a range that contains the "
     "baseline and Bayesian hyperparameter optimisation moves the best tree "
     "learner by 0.011. A six-arm ablation then separates negative-class "
     "composition from the length and charge distributions it carries, since a "
     "pool drawn from one bioactive database differs from the positives in all "
     "three at once. This is a study of how such datasets are built and scored "
     "rather than a search for new inhibitors. DPP-IV serves as a physical "
     "control system: it has a crystallographically characterised active site "
     "(Figure S2), documented peptide inhibitors available as references and a "
     "strongly anionic pocket, which together make it possible to ask whether a "
     "structure-based score and a sequence classifier agree for the right reason. "
     "The transferable output is a set of design and reporting rules (Section 4), "
     "and the peptides reported are prioritised hypotheses rather than confirmed "
     "activities."),

# --- 2.5 pre-commitment, citing Figure S3 ---------------------------------
37: ("Twelve Swiss-Prot precursors",
     "Twelve Swiss-Prot precursors with documented DPP-IV inhibitory potential at "
     "peptide or hydrolysate level were retrieved (Table S5). The protein list "
     "and the threshold were both fixed before any candidate was scored, so no "
     "protein entered on its outcome (Figure S3). Enzymatic digestion is the "
     "established route to food-derived DPP-IV inhibitors^{26,27} and returns "
     "real digestion products rather than invented sequences, so the screening "
     "pool was generated that way rather than by database sampling or de novo "
     "design."),

# --- 3.5 the case, as a prediction (his points 6 and 7) -------------------
85: ("The mechanistic case for this peptide",
     "The mechanistic case for this peptide is separable from its selection, and "
     "it is a modelling result rather than a measurement. Its nineteen "
     "heavy-atom contacts within 5 Å, the fifteen contributed by three "
     "phenylalanines, the 8.2% electrostatic share, the favourable desolvation at "
     "−46.0 and the alanine-scan localisation to Phe1 and Pro4 are all computed "
     "on a fixed receptor structure and a single predicted pose. None depends on "
     "which fine-tuning seed produced the shortlist the peptide was drawn from, "
     "and FVAPFPEVF is selected under all five (Section 3.3). What none of them "
     "establishes is that the peptide binds DPP-IV, inhibits it, or has "
     "anti-diabetic activity. A favourable docking score is not evidence of "
     "activity, and the panel of Section 3.4 shows directly that peptides the "
     "classifier rejected can attract one. We therefore report the structural "
     "findings as a hypothesis about how this peptide might engage the site, to "
     "be tested rather than relied on."),

# --- 3.5 formulation and tiers (his point 7) ------------------------------
87: ("Two formulation concerns attach to it",
     "Two predicted liabilities attach to it. Three bitterness predictors scored "
     "FVAPFPEVF as bitter and a sensory panel confirmed bitterness in the "
     "overlapping VAPFPEVF.^{49} AlgPred 2.0 flagged the peptide itself as a "
     "predicted allergen at 0.54 against a 0.3 cut-off, a call the tool makes on "
     "93.0% of the short peptides that reach it, which is why the candidates were "
     "tiered rather than filtered, though it is not a result to discount. It was "
     "carried as the Tier-1 lead because it was the highest-scoring peptide of "
     "that tier to pass ToxinPred2, at 0.588 against a 0.6 cut, the tier's "
     "top-ranked peptide GPFPSIL having been flagged at 0.652. Neither tool "
     "measures toxicity or allergenicity; both predict it from amino-acid "
     "composition, and tier membership is a prioritisation order rather than a "
     "safety clearance. The classifier predicts general anti-diabetic activity "
     "and the docking is consistent with binding at the DPP-IV active site, so "
     "the peptide is best described as an untested candidate with a documented "
     "food origin rather than a characterised inhibitor."),

# --- 3.6 novelty limitation, qualified (his point 7) ----------------------
90: ("Novelty filtering is the second",
     "Novelty filtering is the second, and it means different things at different "
     "lengths. Only 536 of the 3,596 candidates reach 11 residues, so the "
     "near-duplicate screen at 40% identity covers 15% of the pool and the "
     "remaining 3,060 rely on exact matching alone. Sequence-identity measures "
     "are in any case weak over a handful of residues, which is why the clustered "
     "partition stops at 11 (Section 2.1), so a short candidate called novel here "
     "carries a much weaker guarantee than a long one. Measured against the 873 "
     "training positives, 64 of the 412 prioritised candidates (15.5%) wholly "
     "contain one of three residues or more and 41 (10.0%) contain one of six or "
     "more, among them 26 of the 196 candidates that reach 11 residues (Table "
     "S13). Containment at any length reaches 132, but 68 of those contain only a "
     "dipeptide and the training set holds just two, so the longer floors are the "
     "informative ones. Graded identity behaves as the filter's coverage "
     "predicts: 23 candidates reach 80% identity to a training positive and every "
     "one lies in the short band the filter never reached, while none of the 196 "
     "long candidates does. The filter works where it applies. Where it does not, "
     "part of what the screen returns is extension of a motif the model had "
     "already seen rather than a new one, and the four shorter training positives "
     "running inside the Tier-1 lead are an instance of it. Mean nearest identity "
     "to a training positive is 0.603 for short candidates against 0.442 for long "
     "ones."),

# --- 4.1: the design rule, without the single-property causal claim -------
96: ("A negative class should be matched",
     "**A negative class should be matched on every descriptor that separates its "
     "source database from the positives rather than on length alone.** In the "
     "arms tested here that is achievable at modest cost. Arm F, matched on "
     "length and net charge together, reduces the charge separability of its "
     "training classes from 0.804 to 0.507, chance, for 0.018 in AUC, cutting the "
     "false-positive rate on held-out proteome fragments from 45.1% to 33.1% "
     "while keeping the soft-negative rate at 6.5%. The first draft of this work "
     "argued that forcing such a match was not worth the distortion, on the "
     "ground that no draw from a 91% cationic database can supply a non-cationic "
     "antimicrobial peptide at every positive's length. That premise is correct "
     "and the conclusion drawn from it was not. DBAASP supplies only 23% of arm "
     "F's negatives and the rest are cut from GO-filtered UniProt on demand, so "
     "the constraint binds on the curated class alone rather than on the design. "
     "The arms differ in which sequences they contain as well as in the "
     "descriptor they were matched on, so this shows that the construction "
     "strategy matters and not that net charge alone is responsible; isolating a "
     "single property would need arms that vary it independently, which these do "
     "not. Where one bioactivity class cannot supply the match, the soft majority "
     "should draw across several classes or the proteome-fragment share should be "
     "raised. "
     "Dropping the skewed class outright is the other workable route and on "
     "proteome fragments it did better here: the proteome-only arm reaches a "
     "29.1% false-positive rate against arm F's 33.1%, at the cost of the "
     "generic-bioactivity control, its soft-negative rate rising from 6.5% to "
     "10.5%. Choosing between them is a choice about which error matters and "
     "should be made explicitly rather than inherited from whichever database was "
     "at hand. What should not be done is to select on test AUC. Across these six "
     "arms the two with the highest charge separability record the highest AUCs "
     "and the highest deployment error, and under the family-aware split the "
     "best-scoring arm is the worst-deploying one. Testing every descriptor "
     "against the class labels before any model is fitted costs one ROC curve per "
     "descriptor and would expose such a skew; reporting the false-positive rate "
     "on a held-out sample of the sequence space the classifier is meant to meet "
     "is what shows whether closing it mattered. Antimicrobial peptides are "
     "cationic because their selectivity derives from electrostatic attraction to "
     "anionic bacterial membranes,^{41} so the skew is a property of that class "
     "rather than of any sample taken from it."),

# --- 4.3: why seed 42, stated plainly (his point 5) -----------------------
100: ("A shortlist drawn from a parameter-efficient",
      "**A shortlist drawn from a parameter-efficient fine-tune should be "
      "reported across seeds rather than from one fit.** Seed 42 was the "
      "project's fixed seed, used for the UMAP, the cross-validation folds, the "
      "bootstrap resampling and every base learner, so the phase 3.3 fine-tune "
      "under it produced the deployed adapter and the screen was run with that "
      "adapter before the seed-stability analysis existed. It was not chosen for "
      "scoring well. That it also turned out to be the highest of the five on "
      "test AUC and the most permissive at the operating threshold is an accident "
      "of the default, and it biases the published shortlist in both size and "
      "membership, which is why the result is reported as three nested sets in "
      "Section 3.3. Nothing here suggests the selected fit is representative of "
      "the five. The shortlists range from 153 to 412 candidates and only 111 are "
      "common to all of them, so 73% of the published set does not survive a "
      "change of seed and 30% of it rests on seed 42 alone (Table S15). Reporting "
      "the intersection across seeds, 111 candidates here, or the mean ensemble, "
      "210, gives a candidate set that does not depend on a single fit; the mean "
      "ensemble is the better default, since it recovers seven candidates the "
      "published run missed while discarding most of the seed-specific tail. "
      "Sensitivity of this kind is not well characterised in the ADP literature, "
      "the nearest comparable study reporting ten-seed variability only for its "
      "classical classifier heads.^{6} The simpler remedy is to stop fine-tuning "
      "for deployment. The variance enters through the randomly initialised "
      "classification head, which moves the starting point of the whole "
      "optimisation, so it is removed by keeping the encoder frozen and fitting a "
      "classical head on the extracted representation. On the frozen "
      "1,287-dimensional vector, logistic regression is exactly deterministic "
      "across the five seeds at AUC 0.845 and XGBoost spans 0.009 about a mean of "
      "0.860, against a 0.070 span for the fine-tune. Both sit inside the "
      "fine-tune's confidence interval and both retain the ESM-2 representation, "
      "so the seed sensitivity is a cost of the fine-tuning step rather than of "
      "the embedding. A screen needing a reproducible shortlist is better served "
      "by frozen features with a convex head than by the better single fit. "
      "Freezing the classification head itself is not an alternative: an "
      "untrained head carries no mapping from representation to label, so it has "
      "to be fitted. The only question is whether it is fitted alongside the "
      "encoder or on top of it."),

# --- conclusions (his points 3, 2 and 9) ----------------------------------
102: ("Negative-class composition, not model architecture",
      "Across the architectures and benchmark conditions evaluated here, "
      "negative-class composition exerted a substantial influence on predictive "
      "performance and on the physicochemical character of the peptides a screen "
      "prioritises. Exchanging the negatives while holding the positives, "
      "hyperparameters and test set fixed moved test AUC by 0.219 and the "
      "false-positive rate on held-out proteome fragments from 46.7% to 93.3% "
      "with recall almost unchanged, while four architectures spanning a "
      "650M-parameter fine-tune and a 20-feature composition baseline fell inside "
      "0.723 to 0.877 and stacking added nothing over the best single model. "
      "These are four learners on one dataset and one target, so the comparison "
      "does not show that architecture is unimportant in general. It does show "
      "that on this problem, effort spent on the negative class was better repaid "
      "than effort spent on the model."),

103: ("Length-matching and GO-filtered proteome fragments",
      "Length-matching and GO-filtered proteome fragments close two shortcuts. "
      "Net charge remains, and the same association appears in the errors of the "
      "model trained here, in the SHAP attributions of its tree counterpart and "
      "in the decisions of the published state of the art, so all three score a "
      "property of the negative class as part of what they call activity. It also "
      "shapes the output: the 412 prioritised candidates average −0.79 in net "
      "charge against −0.11 across the candidates they came from. Molecular "
      "docking carries the opposite bias, scoring three confidently rejected "
      "cationic peptides above Diprotin A on electrostatics of −236 to −312, so "
      "within this panel neither method checks the other on a charged sequence "
      "and a screen using both should report the electrostatic share of every "
      "docking score rather than the total alone. Under a family-aware partition "
      "the measured advantages shrink, the negative-class AUC gap narrowing from "
      "0.180 to 0.074, while the gap in deployment error holds at 20.6 points. "
      "Part of what an ordinary split reports is therefore a property of the "
      "split, and the behaviour that matters in a screen is the part that "
      "survives."),

105: ("We did not force a charge-matched negative class",
      "We did not force a charge-matched negative class when this dataset was "
      "built, and the ablation reported here indicates that we should have. "
      "Matching length and charge together reduced the charge signal to chance "
      "for 0.018 in AUC and cut the false-positive rate on proteome fragments by "
      "twelve points, without substituting a distribution the databases do not "
      "contain, because three quarters of that negative class is cut from "
      "GO-filtered UniProt on demand rather than resampled from DBAASP. The "
      "screen reported here predates that result and carries the association, so "
      "its 412 prioritised candidates are a charge-selected subset of the digest "
      "and the design we now recommend is not the design that produced them. The "
      "Tier-1 lead FVAPFPEVF scored best of the docked panel from van der Waals "
      "complementarity and desolvation rather than electrostatics, its three "
      "phenylalanines making 15 of 19 contacts within 5 Å while its single "
      "glutamate stayed out of contact, and it is selected under all five seeds. "
      "That structural case is computed on a fixed predicted pose and is a "
      "hypothesis rather than a demonstration of binding. FVAPFPEVF is an "
      "untested candidate of documented food origin rather than a characterised "
      "inhibitor, and the shortlist that contains it is provisional in membership "
      "while the mechanism that recommends it remains to be tested."),

# --- captions -------------------------------------------------------------
107: ("Table 1. Composition of the Curated Dual-Negative Dataset",
      "Table 1. Composition of the Curated Dual-Negative Dataset. The negative "
      "class is length-matched one-to-one against the positives, so no length "
      "signal separates the classes. The positives are curated ADP sequences "
      "consolidated by Xie et al. and comprise 899 drawn from BioDADPep through "
      "three earlier studies plus 67 later candidates; per-sequence evidence of "
      "activity does not survive into the released data (Section 2.1). Whole "
      "CD-HIT clusters were assigned to splits, so no test sequence of 11 "
      "residues or more exceeds 40% identity to any training sequence; the 870 "
      "sequences below that length took a stratified random split, since 40% "
      "identity is not meaningful over so few residues, and substring "
      "relationships are not covered by either procedure (Table S11). Hard "
      "negatives exclude proteins annotated for glucose homeostasis [GO:0042593], "
      "insulin receptor signalling [GO:0008286] or hormone activity "
      "[GO:0005179]."),

379: ("Figure 1. The dual-negative class closes",
      "Figure 1. The dual-negative class closes the length and "
      "bioactive-database shortcuts but not the charge association. One negative "
      "was drawn per positive at the same length, so the two length distributions "
      "are identical by construction and no length signal separates the classes. "
      "Soft negatives are antimicrobial peptides from DBAASP and control for "
      "generic bioactivity. Hard negatives are random fragments excised from "
      "GO-filtered Swiss-Prot proteins and represent the sequence space a real "
      "screen returns. Net charge is not closed by this design; its consequences "
      "are traced in Section 3.2. Only the 1,062 sequences of 11 residues or more "
      "entered CD-HIT, so the 40% identity criterion covers those while the "
      "remaining 870 took a stratified random split, and substring relationships "
      "are outside what either procedure removes. Full construction parameters "
      "are given in Table S1."),

389: ("Figure 6. Charge carries the negative controls",
      "Figure 6. Charge carries the rejected peptides' docking scores, while the "
      "lead and the reference inhibitors earn theirs from shape and desolvation. "
      "(a) Electrostatic share is the weighted electrostatic term as a percentage "
      "of the summed terms that favour binding, so penalties are excluded and the "
      "denominator is not the HADDOCK score. It decomposes an empirical score and "
      "is not a physical measure of binding energy. The three peptides the "
      "consensus model scored below 0.01 draw 66% to 74% of their score from "
      "charge, against 8% to 24% for the Tier-1 lead and both reference "
      "inhibitors, with the Tier-2 leads between them at 31% to 41%. (b) "
      "Desolvation against the unweighted electrostatic energy in kcal/mol, the "
      "dashed line marking zero desolvation. More negative is more favourable "
      "throughout. The rejected peptides pair the largest electrostatic energies "
      "in the panel with unfavourable desolvation. CAND3554 falls on the same "
      "side of that line at +7.6, so the split is between those three and the "
      "Tier-1 lead plus reference inhibitors rather than between them and all "
      "leads. Both Tier-2 leads are labelled by candidate ID, their sequences "
      "running to 21 and 32 residues. No statistical test was applied, since n = "
      "8, and the panel supports comparison within itself rather than a general "
      "claim about the method."),

393: ("Figure 8. The lead binds across the catalytic cleft",
      "Figure 8. The predicted pose spans the catalytic cleft without reproducing "
      "the Glu-Glu recognition of a substrate amine. FVAPFPEVF docked into human "
      "DPP-IV (PDB 4A5S), on the top-ranked model of its best-scoring cluster. "
      "(a) The pose across the catalytic cleft. The peptide is magenta and the "
      "receptor grey, with contacted active-site residues in cyan. Every contact "
      "within 4.5 Å to the eight site residues shown is drawn as a dashed line "
      "labelled by distance, six of the seventeen the pose makes at that cutoff "
      "across all 22 restrained residues; the nineteen quoted in the text are "
      "contacts within 5.0 Å, the counting cutoff of Section 2.6. Ser630 and "
      "His740 are two of the three catalytic triad residues, the third being "
      "Asp708. (b) Close-up on Phe1 with the receptor cartoon made "
      "semi-transparent. The aromatic side chain reaches Glu205 and Glu206 at 3.3 "
      "Å, while the free N-terminal amine lies 6.5 Å from Glu206, so the pose "
      "does not reproduce the Glu-Glu recognition of a substrate amine. The amine "
      "is drawn as an orange sphere, labelled, with the arrow marking the unmade "
      "6.5 Å approach to the Glu206 carboxylate. The pose is a prediction and is "
      "not evidence of binding."),

# --- 3.1 the shared-negative comparison, terminology ----------------------
61: ("The comparison isolates the negative class",
     "The comparison isolates the negative class rather than the architecture. "
     "BertADP is a fine-tuned ProtBert and the Basith-negative arm is a "
     "fine-tuned ESM-2; what they share is the negative class, which Xie et al. "
     "adopted from Basith et al. Two different fine-tuned PLMs on the same "
     "negatives differ by 6.6 percentage points, whereas one PLM on two different "
     "negative classes differs by 46.6. The deliberately weak arm could be "
     "dismissed as built to fail; BertADP cannot. Only the dual-negative model "
     "saw hard negatives of this kind, so it alone is tested in-distribution. The "
     "result indicates that a shared negative class produces a shared failure "
     "mode, not that ESM-2 outperforms ProtBert."),

# --- 3.3 known actives, terminology ---------------------------------------
73: ("Two peptides with measured DPP-IV",
     "Two peptides with measured DPP-IV inhibitory activity survived the cut-off: "
     "LPGF at rank 4 of 3,596, with an IC50 of 451.49 μM,^{45} and IPAVF at rank "
     "26, at 44.7 μM.^{46} Each differs from a training positive by a single "
     "residue, KLPGF and IPAVFK, so this shows that the threshold retains known "
     "actives rather than that the pipeline generalises to unfamiliar chemistry. "
     "Of the 412 prioritised candidates, 324 also reach 0.90 on XGBoost alone, "
     "FVAPFPEVF among them at 0.995; the other 88 fall between 0.823 and 0.90, so "
     "XGBoost would not have selected them by itself."),

# --- 3.4 desolvation, terminology -----------------------------------------
79: ("Desolvation is what separates them",
     "Desolvation is what separates them. It is favourable for the Tier-1 lead "
     "and both reference inhibitors at −46.0, −21.3 and −7.3, and unfavourable "
     "for all three rejected peptides at +5.5 to +12.2 (Figure 6b): in these "
     "models a cationic peptide holding its charges at a solvated interface pays "
     "rather than gains on displacing water. The Tier-2 lead CAND3554 falls on "
     "the unfavourable side of that line at +7.6, alongside the rejected peptides "
     "rather than the leads, which is why its 40.9% electrostatic share matters "
     "more than its −62.0 total."),

# --- 3.4 counter-directional charge, terminology --------------------------
80: ("Charge moves the two methods in opposite directions",
     "Charge moves the two methods in opposite directions. DPP-IV does recognise "
     "substrates in part through the Glu205/Glu206 motif, whose carboxylates hold "
     "the protonated N-terminal amine in position for cleavage.^{47} But the "
     "enzyme reads that charge at one atom in one position, whereas HADDOCK's "
     "E_elec is a single intermolecular term summing charge anywhere across the "
     "interface.^{31} The three rejected peptides average +2.01 in net charge "
     "where the 45 held-out hard negatives average +0.01, so a selection criterion "
     "meant to pick confidently rejected sequences drew a cationic set out of a "
     "charge-neutral pool. Within this panel a cationic peptide is therefore "
     "penalised by the classifier while attracting a more favourable docking "
     "score. A screen that filters on the first and ranks on the second would "
     "inherit both tendencies at once: a cationic true binder is rejected before "
     "it reaches the docking, while a peptide clearing both is neutral enough to "
     "escape two biases rather than confirmed by two methods. The total score "
     "does not show this. The electrostatic share does, which is why Table 4 "
     "reports it beside every score. Section 4.2 states the consequence for "
     "hybrid workflows generally."),

# --- 3.4 panel independence, terminology ----------------------------------
81: ("The panel comparison does not depend",
     "The panel comparison does not depend on which classifier fit selected the "
     "lead. Both reference inhibitors are documented inhibitors chosen from the "
     "literature rather than from any model output, and the three rejected "
     "peptides sit at consensus scores below 0.01. Since the consensus is the "
     "mean of two probabilities, moving a peptide from below 0.01 to the 0.90 "
     "threshold would require its ESM-2 probability to rise by more than 1.78, "
     "which is impossible for a bounded probability. Those three therefore give "
     "the same reference floor under any refit, and the energetic decomposition "
     "in Table 4 is a property of the eight poses rather than of the screen that "
     "produced them."),

# --- 3.6 seed limitation, terminology -------------------------------------
91: ("The dependence of the shortlist on a single",
     "The dependence of the shortlist on a single fine-tuning seed is the third, "
     "now measured rather than asserted. Only 111 of the 412 prioritised "
     "candidates clear the threshold under all five seeds while 123 clear it "
     "under the published seed alone, and the five shortlists range from 153 to "
     "412 (Section 3.3). The seed carried forward is both the highest of the five "
     "on test AUC and the most permissive on the screen, so the published count "
     "and its membership are each the optimistic end of their distribution. "
     "Because this bears on how any such screen should be reported, Section 4.3 "
     "treats it as a recommendation rather than only as a caveat."),

# --- SI contents list, terminology ----------------------------------------
397: ("Construction of the dual-negative dataset",
      "Construction of the dual-negative dataset (Table S1); feature "
      "representation (Table S2); architectures, training settings and tuning "
      "search spaces (Table S3); ESM-2/DoRA fine-tuning configuration (Table S4); "
      "dietary protein precursors (Table S5); simulated digestion and candidate "
      "filtering (Table S6); HADDOCK 2.4 docking protocol (Table S7); training "
      "and evaluation of the three models compared on hard negatives (Table S8); "
      "candidates recovered from each dietary protein (Table S9); software "
      "versions and compute environment (Table S10); test-set proximity to the "
      "training set (Table S11); positive-class family structure (Table S12); "
      "distance from each prioritised candidate to the nearest training positive "
      "(Table S13); the negative-class ablation under a family-aware split (Table "
      "S14); selection stability of the shortlist across the five fine-tuning "
      "seeds (Table S15); methodological gaps across four published ADP "
      "predictors (Figure S1); DPP-IV inhibition mechanism (Figure S2); screening "
      "design and pre-commitment (Figure S3); amino-acid composition importances "
      "(Figure S4); consensus calibration (Figure S5); safety-filter length "
      "dependence (Figure S6) (PDF)."),

# --- 4.2 hybrid workflows, scoped to this panel (his point 6) -------------
98: ("Charge moves the classifier and the docking",
     "**Charge moves the classifier and the docking in opposite directions here, "
     "so on a charged sequence neither checks the other.** A screen that filters "
     "on the classifier and ranks on the docking would inherit both tendencies at "
     "once. A cationic peptide is penalised at the first step while attracting a "
     "more favourable score at the second, so clearing both marks it neutral "
     "enough to escape two biases rather than confirmed by two methods, and a "
     "cationic true binder could be rejected before it ever reaches the docking. "
     "The site presents Glu205, Glu206, Asp545 and Asp663 against two basic "
     "residues, and where the enzyme reads that charge at one atom in one "
     "position, HADDOCK's E_elec sums it anywhere across the "
     "interface.^{31,47} Ranking on the raw HADDOCK score would have placed all "
     "three rejected peptides above Diprotin A (Section 3.4), so the total score "
     "does not show this. The composition of the score does, which is why Table 4 "
     "reports it beside every entry. This rests on one panel of eight peptides "
     "against one target, so it is a caution for workflows of this shape rather "
     "than a measured property of docking in general. The practical step is "
     "cheap: report the electrostatic share beside every score, include "
     "confidently rejected peptides in the panel and check the charge "
     "distribution of a shortlist against its source pool."),

# --- Table 5 caption: the corrected cap explanation (A6) ------------------
312: ("Table 5. Negative-Class Ablation Separating Composition",
      "Table 5. Negative-Class Ablation Separating Composition from Length and "
      "Charge. Every arm keeps the 873 training positives and is scored on the "
      "published 178-row test set, so only the training negatives vary and the "
      "arms are directly comparable. AUC and the hard-negative false-positive "
      "rate are means over ten training-row permutations with their standard "
      "deviation, since row subsampling makes a single fit depend on row order by "
      "about 0.004 in AUC, which is the size of the smallest difference between "
      "arms. Arms C and D cannot reach 873 negatives, because neither source "
      "database holds enough sequences at the lengths the positives occupy: 45% "
      "of the positives are shorter than 11 residues and across lengths 2 to 10 "
      "they need 435 partners, against about 69 available from the Basith pool. "
      "C′ and D′ repeat those two arms with the positives subsampled to the "
      "negative count. The hard-negative block is 45 proteome fragments and the "
      "soft-negative block 40 antimicrobial peptides, every row in each a true "
      "non-ADP, so any positive call is an error. Charge separation is the AUC of "
      "net charge alone predicting that a training row is a negative, so 0.5 is "
      "chance. The readout is XGBoost on the fused 1,287-dimensional vector under "
      "the deployed model's hyperparameters. The arms differ in which sequences "
      "they contain as well as in the descriptor they were matched on, so they "
      "compare construction strategies rather than isolating single properties. "
      "Arm F reduces the charge signal to chance for 0.018 in AUC; arms D and E "
      "differ by 0.03 in AUC and by 54 points in hard-negative false-positive "
      "rate."),

# --- data availability, citing Table S10 ----------------------------------
412: ("All code, dataset splits",
      "All code, dataset splits, out-of-fold predictions, tuned hyperparameters "
      "and result tables are available at "
      "https://github.com/om-894/antidiabetic-peptide-classifier, archived at "
      "[Zenodo DOI to be assigned on release]. Scripts are numbered by pipeline "
      "phase; the analyses added in revision are under phase_7_revision with "
      "their outputs in results, and each supporting table names the script that "
      "generates it. Software versions and the compute environment are given in "
      "Table S10. The positive set derives from Xie et al.^{5} and the negative "
      "class from DBAASP^{9} and UniProt.^{10} Receptor coordinates are PDB "
      "4A5S."),
}


# --------------------------------------------------------------------------- #
# HOUSE STYLE
# --------------------------------------------------------------------------- #
# Two independent clauses are never joined with ", and". Each pair below splits
# one into two sentences, turns it into an appositive, or makes the second a
# participle. Applied to the replacement text before either renderer sees it, so
# the clean and tracked files carry identical prose.

STYLE = [
 ("attention usually given to architecture, and a shortlist",
  "attention usually given to architecture. A shortlist"),
 ("reporting rules (Section 4), and the peptides reported are",
  "reporting rules (Section 4), with the peptides reported being"),
 ("it does not detect substring containment, and it does not reach",
  "it does not detect substring containment, nor does it reach"),
 ("free of all train-test relationships, and the next paragraph measures",
  "free of all train-test relationships. The next paragraph measures"),
 ("overlapping insulin-derived fragments, and near-duplicates inside",
  "overlapping insulin-derived fragments, while near-duplicates inside"),
 ("clustering identity are different criteria, and a partition clean under one",
  "clustering identity are different criteria, so a partition clean under one"),
 ("to a training sequence on that measure, and they concentrate",
  "to a training sequence on that measure, concentrated"),
 ("inflates discrimination by about 0.046, and that is the allowance",
  "inflates discrimination by about 0.046. That is the allowance"),
 ("matching against the precursor sequence, and is not a database annotation",
  "matching against the precursor sequence, not a database annotation"),
 ("with the predictors benchmarked here, and because the model is not told",
  "with the predictors benchmarked here, and because nothing tells the model"),
 ("chains the set into a 242-member family, and it does the same to the negatives",
  "chains the set into a 242-member family, doing the same to the negatives"),
 ("shorter than 11 residues, and across lengths 2 to 10 those positives need",
  "shorter than 11 residues, so across lengths 2 to 10 those positives need"),
 ("rather than a top-N cut, and it is calibrated against the benchmark only",
  "rather than a top-N cut, calibrated against the benchmark only"),
 ("not on digestion fragments, and it carries no claim",
  "not on digestion fragments, so it carries no claim"),
 ("component energies are in kcal/mol, and buried surface area is omitted",
  "component energies are in kcal/mol, while buried surface area is omitted"),
 ("describing where that score came from, and it is not a physical measurement",
  "describing where that score came from rather than measuring the physical"),
 ("model shows none at 0.431, and length itself carries no signal",
  "model shows none at 0.431, while length itself carries no signal"),
 ("each is a different draw of sequences, and length, charge",
  "each is a different draw of sequences, so length, charge"),
 ("deployment behaviour substantially, and does not isolate",
  "deployment behaviour substantially. It does not isolate"),
 ("fragments of twelve dietary proteins, and the fraction of prioritised",
  "fragments of twelve dietary proteins. The fraction of prioritised"),
 ("absent from the training and test sets, and yield varied by source",
  "absent from the training and test sets, with yield varying by source"),
 ("anything established about the sequences, and the Tier-1 lead carried",
  "anything established about the sequences. The Tier-1 lead carried"),
 ("candidates they were drawn from, and consensus score falls with charge",
  "candidates they were drawn from, with consensus score falling with charge"),
 ("the 111 identified under all five, and the 111 is the subset",
  "the 111 identified under all five. The 111 is the subset"),
 ("from 0.908 to 0.998 across the three, and IPAVF by four",
  "from 0.908 to 0.998 across the three, with IPAVF selected by four"),
 ("lysine or arginine side chains, and the site has no documented partner",
  "lysine or arginine side chains, yet the site has no documented partner"),
 ("separable from its selection, and it is a modelling result",
  "separable from its selection, and is a modelling result"),
 ("the shortlist the peptide was drawn from, and FVAPFPEVF is selected",
  "the shortlist the peptide was drawn from, FVAPFPEVF being selected"),
 ("score is not evidence of activity, and the panel of Section 3.4 shows",
  "score is not evidence of activity, as the panel of Section 3.4 shows"),
 ("predict it from amino-acid composition, and tier membership is",
  "predict it from amino-acid composition, while tier membership is"),
 ("Novelty filtering is the second, and it means different things",
  "Novelty filtering is the second, meaning different things"),
 ("already seen rather than a new one, and the four shorter training positives",
  "already seen rather than a new one. The four shorter training positives"),
 ("positive class is the fourth, and the robustness check it motivated",
  "positive class is the fourth, with the robustness check it motivated"),
 ("narrows from 0.180 to 0.074, and individual arms lose discrimination",
  "narrows from 0.180 to 0.074, while individual arms lose discrimination"),
 ("rather than a measurement on the screen, and every output here is",
  "rather than a measurement on the screen. Every output here is"),
 ("AUCs and the highest deployment error, and under the family-aware split",
  "AUCs and the highest deployment error, while under the family-aware split"),
 ("threshold is an accident of the default, and it biases the published",
  "threshold is an accident of the default, biasing the published"),
 ("close two shortcuts. Net charge remains, and the same association appears",
  "close two shortcuts. Net charge remains, the same association appearing"),
 ("therefore a property of the split, and the behaviour that matters",
  "therefore a property of the split. The behaviour that matters"),
 ("when this dataset was built, and the ablation reported here indicates",
  "when this dataset was built. The ablation reported here indicates"),
 ("single glutamate stayed out of contact, and it is selected under all five",
  "single glutamate stayed out of contact, and is selected under all five"),
 ("than a characterised inhibitor, and the shortlist that contains it",
  "than a characterised inhibitor. The shortlist that contains it"),
 ("not meaningful over so few residues, and substring relationships are not",
  "not meaningful over so few residues. Substring relationships are not"),
 ("took a stratified random split, and substring relationships are outside",
  "took a stratified random split. Substring relationships lie outside"),
 ("since n = 8, and the panel supports comparison",
  "since n = 8, so the panel supports comparison"),
 ("with their outputs in results, and each supporting table names",
  "with their outputs in results, each supporting table naming"),
 ("Two different fine-tuned PLMs on the same negatives differ by 6.6 percentage "
  "points, whereas", "Two different fine-tuned PLMs on the same negatives differ "
  "by 6.6 percentage points, where"),
 ("reference inhibitors at −46.0, −21.3 and −7.3, and unfavourable",
  "reference inhibitors at −46.0, −21.3 and −7.3, while it is unfavourable"),
 ("rather than from any model output, and the three rejected peptides sit",
  "rather than from any model output, while the three rejected peptides sit"),
 ("under all five seeds while 123 clear it under the published seed alone, and "
  "the five shortlists", "under all five seeds while 123 clear it under the "
  "published seed alone. The five shortlists"),
 ("(Fisher exact test, P = 0.002), and AlgPred 2.0, applied to",
  "(Fisher exact test, P = 0.002). AlgPred 2.0, applied to"),
 ("the same reference floor under any refit, and the energetic decomposition",
  "the same reference floor under any refit. The energetic decomposition"),
 ("while its single glutamate stayed out of contact, and is selected under all "
  "five seeds.", "while its single glutamate stayed out of contact. That peptide "
  "is selected under all five seeds."),
 ("escape two biases rather than confirmed by two methods, and a cationic true "
  "binder could be", "escape two biases rather than confirmed by two methods, "
  "while a cationic true binder could be"),
 ("against two basic residues, and where the enzyme reads that charge",
  "against two basic residues, so where the enzyme reads that charge"),

 # abstract length: ACS Articles are commonly held to about 250 words
 ("A six-arm ablation separates negative-class composition from the length and "
  "charge distributions it carries.",
  "A six-arm ablation separates composition from the length and charge "
  "distributions it carries."),
 ("two thirds of their favourable score from electrostatics. Repeating the "
  "screen under five fine-tuning seeds returns shortlists of 153 to 412 "
  "candidates with only 111 common to all.",
  "two thirds of their score from electrostatics. Repeating the screen under "
  "five seeds returns shortlists of 153 to 412 with only 111 common to all."),
]


def styled(text):
    """Apply the house-style clause fixes to one replacement string."""
    for old, new in STYLE:
        text = text.replace(old, new)
    return text


for _i, (_e, _t) in list(REPLACE.items()):
    REPLACE[_i] = (_e, styled(_t))


def apply(x, tracked):
    """Return a revised document body, clean or tracked."""
    spans = [(m.start(), m.end(), m.group(0)) for m in PARA.finditer(x)]
    fails = []
    for i, (expect, _) in REPLACE.items():
        if i >= len(spans) or not ptext(spans[i][2]).lstrip().startswith(expect):
            got = ptext(spans[i][2])[:60] if i < len(spans) else "<out of range>"
            fails.append(f"para {i}: expected {expect[:40]!r}, found {got!r}")
    if fails:
        print("FAILED:")
        for f in fails:
            print("  " + f)
        sys.exit(1)

    ids = Ids()
    edits = []
    for i, (_, new) in REPLACE.items():
        para = spans[i][2]
        out = (tracked_paragraph(para, new, AUTHOR, DATE, ids) if tracked
               else clean_paragraph(para, new))
        edits.append((spans[i][0], spans[i][1], out))

    for start, end, text in sorted(edits, key=lambda e: -e[0]):
        x = x[:start] + text + x[end:]
    return x
