"""
Apply the second-draft revisions to the manuscript's word/document.xml.

Paragraphs are addressed by their index in document order, not by matching
their text. Word splits a paragraph across runs wherever formatting changes, so
a paragraph carrying a superscript citation, a bold lead-in or a rendered page
break is not a contiguous string in the XML and cannot be matched as one. The
index is stable because this script is the only thing editing the file, and
every target's current opening words are asserted before it is touched.

Each replacement keeps the original paragraph properties and rebuilds its runs
from a small markup: **bold** and ^{12} for a superscript citation. That is what
lets ACS-style citation numbers survive a rewritten paragraph.

Figures stay untouched: only document.xml changes, so the eight embedded images
and their relationships carry through unchanged.

INPUTS   ms/word/document.xml (runs already merged)
OUTPUTS  the same file, revised in place
"""

import re
import sys

DOC = "ms/word/document.xml"
PARA = re.compile(r'<w:p(?: [^>]*)?>.*?</w:p>|<w:p(?: [^>]*)?/>', re.S)
TOKEN = re.compile(r'(\*\*.+?\*\*|\^\{[^}]+\})', re.S)


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def runs(text):
    """Build w:r runs from text carrying **bold** and ^{n} superscript markup."""
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
    """Same paragraph properties, new runs."""
    m = re.search(r'<w:pPr>.*?</w:pPr>', p, re.S)
    ppr = m.group(0) if m else ""
    open_tag = re.match(r'<w:p(?: [^>]*)?>', p).group(0)
    return f"{open_tag}{ppr}{runs(text)}</w:p>"


BODY = ('<w:p><w:pPr><w:spacing w:line="480" w:lineRule="auto"/>'
        '<w:jc w:val="both"/></w:pPr>{}</w:p>')


# --------------------------------------------------------------------------- #
# EDITS: index -> (expected opening words, new text) ; inserts go after index
# --------------------------------------------------------------------------- #

REPLACE = {

0: ("Negative-Class Composition Outweighs",
    "Negative-Class Design Reveals a Net-Charge Shortcut in Anti-Diabetic Peptide "
    "Prediction"),

# his point 6: the dense clause becomes separate sentences
6: ("ABSTRACT: Classifiers for anti-diabetic",
    "ABSTRACT: Classifiers for anti-diabetic peptides (ADPs) report high benchmark "
    "accuracy yet misclassify most of the proteome fragments a real screen returns. "
    "Holding the positives, hyperparameters and test set fixed, we varied the "
    "negative class alone. A dual-negative set of DBAASP antimicrobial peptides "
    "plus Gene-Ontology-filtered Swiss-Prot fragments, length-matched one-to-one "
    "against 966 positives, cut the false-positive rate on held-out proteome "
    "fragments from 93.3% to 46.7%. Architecture did not separate the learners. A "
    "DoRA fine-tuned ESM-2 650M model and a 20-feature composition Random Forest "
    "bracketed the same AUC interval at 0.877 and 0.854, and no stacking variant "
    "improved on the best single model. Net charge survives as the residual "
    "shortcut. It is carried by the antimicrobial half of the class, reproduced by "
    "the trained model on held-out data and shared by the published state of the "
    "art, which misclassified 92.4% of 328 unseen fragments. It also selects the "
    "output, the 412 calibrated discoveries averaging −0.79 in net charge against "
    "−0.11 across the pool they came from. A six-arm ablation separates "
    "composition from length and charge. Length-matching a published negative pool "
    "leaves its false-positive rate near 80%, whereas matching charge as well "
    "drives the shortcut to chance for 0.018 in AUC. Two arms differing by 0.03 in "
    "AUC differ by 54 points in false-positive rate, and under a family-aware "
    "split the best-scoring arm is the worst-deploying one, so test AUC is close "
    "to blind to screen behaviour. Docking inverts the same bias, cationic "
    "non-binders drawing 66% to 74% of their favourable score from electrostatics "
    "at a site presenting four carboxylates. A negative class should therefore be "
    "matched on every descriptor separating its source database from the positives "
    "and selected on deployment error rather than on test AUC."),

# JCIM framing: methodological contribution, DPP-IV as a control system
11: ("We separate the two effects",
     "We separate the two effects. Holding the positives, hyperparameters and test "
     "set fixed, we vary the negative class alone and measure the false-positive "
     "rate on held-out proteome fragments; separately, we vary architecture across "
     "four learners against a 20-feature composition baseline. The negative class "
     "moves discrimination by 0.219 in AUC. Architecture spans a range that "
     "contains the baseline, and Bayesian hyperparameter optimisation moves the "
     "best tree learner by 0.011. A six-arm ablation then separates negative-class "
     "composition from the length and charge distributions it carries, since a "
     "pool drawn from one bioactive database differs from the positives in all "
     "three at once. This is a study of how such datasets are built and scored "
     "rather than a search for new inhibitors. DPP-IV serves as a physical control "
     "system: it has a crystallographically characterised active site, documented "
     "peptide inhibitors available as positive controls and a strongly anionic "
     "pocket, which together make it possible to ask whether a structure-based "
     "score and a sequence classifier agree for the right reason. The transferable "
     "output is a set of design and reporting rules (Section 4), and the peptides "
     "reported are prioritised hypotheses rather than confirmed activities."),

# his point 2: fold-level leakage behind the operating threshold
34: ("Candidates were scored by consensus",
     "Candidates were scored by consensus, the unweighted mean of the ESM-2/DoRA "
     "and XGBoost probabilities, the two learners that multi-seed evaluation ranked "
     "strongest (Section 3.1). Cut-offs were swept from 0.50 to 0.99 in 0.01 steps "
     "over the 1,754 out-of-fold training predictions for the lowest threshold "
     "retaining at least 30 sequences at a 90% empirical positive rate. That "
     "returned 0.87, so 0.90 was adopted as a conservative round figure. This is a "
     "calibrated cut-off, not a top-N shortlist. The association between consensus "
     "score and net charge across the candidate pool was tested by Spearman rank "
     "correlation. Those folds are stratified on class rather than grouped on "
     "sequence family, so the out-of-fold predictions the threshold was swept on do "
     "carry cluster-level leakage and the calibration in Figure S5 is "
     "correspondingly optimistic. Measured on the training rows under the family "
     "definition of Section 2.1, 540 of 1,754 (30.8%, between 29.4% and 31.9% per "
     "fold) have a family relative among their own fold's training rows. Refitting "
     "the XGBoost readout with folds grouped by family lowers out-of-fold AUC from "
     "0.916 to 0.896 and moves the swept threshold from 0.93 to 0.97. The consensus "
     "threshold adopted here was swept on stratified folds and its ESM-2 half "
     "shares the identical fold assignment, so it inherits the same direction of "
     "bias, and a grouped sweep would place the operating point higher and admit "
     "fewer candidates. The test-set precision reported in Section 3.3 is measured "
     "on the cluster-wise held-out split and is not affected by this."),

# corrects active vs passive, and reports HADDOCK's own accessibility trim
38: ("The receptor was human DPP-IV",
     "The receptor was human DPP-IV (PDB 4A5S, 1.62 Å), chain A only, with "
     "heteroatoms removed and alternate conformations collapsed. The 22 chain-A "
     "residues with any heavy atom within 5.0 Å of the co-crystallised ligand N7F "
     "were supplied as receptor restraints (residues 125, 205, 206, 357, "
     "545–547, 554, 627–632, 656, 659, 662, 663, 666, 710, 711 and 740). "
     "HADDOCK's own solvent-accessibility filter retained nine of them as active "
     "(125, 205, 357, 545, 547, 554, 629, 630 and 740) and discarded the rest as "
     "buried below 15% relative accessibility, with neighbouring residues reached "
     "through automatic passive definition at 6.5 Å. Every residue of each peptide "
     "was defined passive rather than active, so a peptide was free to select its "
     "own contacts instead of being restrained to form them. Four of the active "
     "residues are acidic and one basic, so the site carries a net negative charge, "
     "which governs the control comparison in Section 3.4."),

# his point 3: backbone flexibility during semi-flexible refinement
39: ("Short peptides have no single native conformation",
     "Short peptides have no single native conformation, so each was supplied as a "
     "three-model ensemble at fixed φ/ψ built with PeptideBuilder^{34} on "
     "Biopython^{35} at three fixed conformations: extended (−139, 135), "
     "α-helical (−57, −47) and polyproline-II (−75, 145). That geometry sets the "
     "starting point only and is not carried through to scoring. Rigid-body "
     "docking is followed by HADDOCK's semi-flexible simulated annealing, which "
     "frees interface side chains in its second stage and the interface backbone in "
     "its third and fourth, and then by a restrained water refinement. Because "
     "every peptide residue lies at the interface, each whole chain fell inside the "
     "semi-flexible segment and its backbone was free to move, so the backbone of a "
     "reported pose is not the backbone that was built and the three conformers act "
     "as seeds for a search that can leave them rather than as shapes the peptide "
     "is held in while it is scored. Sampling used the server defaults of 1,000 "
     "rigid-body, 200 semi-flexible and 200 water-refinement models, confirmed "
     "against the run parameters recovered from the server together with HADDOCK "
     "2.4-2022.01 and the EASY interface."),

# his point 9: 0.877 is the best of five, not the seed mean
51: ("Stability, not accuracy, separates the learners",
     "Stability, not accuracy, separates the learners. Refitting under five random "
     "seeds, XGBoost returned a mean AUC of 0.862 (SD 0.002) against 0.842 (SD "
     "0.026) for ESM-2, so it is higher on average and an order of magnitude more "
     "stable (Figure 3c). Both tree ensembles combine hundreds of trees, so a "
     "change of seed shifts only which rows and features each split sees, whereas "
     "fine-tuning restarts from a randomly initialised classification head and "
     "moves the starting point of the whole optimisation. The five ESM-2 runs span "
     "0.807 to 0.877, so the 0.877 carried in Table 2 is the best of the five "
     "rather than a representative fit, and it sits 0.035 above the five-seed mean "
     "of 0.842 that the same table reports beside it. The two numbers in that row "
     "are a single favourable fit and the distribution it was drawn from, not an "
     "estimate and its uncertainty, and the adapter carried into the screen, the "
     "shortlist and the case study is that best-seed fit. Section 4.3 returns to "
     "what follows from it. The 1D-CNN spans 0.166 across the same seeds, more than "
     "twice the range of any other learner."),

# his priority 2: the six-arm result
54: ("Part of that failure is a length shortcut",
     "Part of that failure is a length shortcut. The Basith pool is not "
     "length-matched to the positives, averaging 26.7 residues against 13.5, so the "
     "two training classes are separable on length alone. Among the 85 test "
     "negatives, the Basith-negative model wrongly accepts those averaging 9.2 "
     "residues and rejects those averaging 14.2, a length effect of AUC 0.659. The "
     "length-matched dual-negative model shows none at 0.431, and length itself "
     "carries no signal about the true label on this test set (AUC 0.448). That "
     "comparison still varies composition and length together, so six arms separate "
     "them (Table 5)."),

# his point 4: global net charge against localised N-terminal charge
58: ("Length was matched by construction",
     "Length was matched by construction and bioactive-database provenance by the "
     "hard negatives. Net charge was not, and it is the most consequential of the "
     "three. Charge alone separates the 873 training positives from the 598 "
     "antimicrobial negatives at AUC 0.912 and from the 283 proteome fragments at "
     "only 0.577, reaching 0.804 against the complete negative class (Figure 5a,b). "
     "Two thirds of the negative class therefore supply a rule that the remaining "
     "third does not. If low charge marked anti-diabetic activity, it would "
     "separate the positives from any non-ADP; what it identifies is antimicrobial "
     "peptides. The quantity doing the work here should be kept apart from the one "
     "the enzyme uses. What the classifier learns is net charge, a single number "
     "summed over every ionisable side chain in the sequence, and what it is "
     "reading through it is the presence of several cationic side chains anywhere "
     "along the chain. DPP-IV's own recognition of charge is neither of those: the "
     "Glu205 and Glu206 carboxylates hold one protonated α-amino group, at one "
     "atom, in one position, and that group is present on every peptide substrate "
     "whatever its net charge.^{47} A low net charge is therefore not a proxy for "
     "the interaction the enzyme makes, and a cationic side chain elsewhere in the "
     "sequence does not obstruct it. The two are independent, which is why a rule "
     "learned on the first can reject a peptide that would satisfy the second, and "
     "why the shortcut is a property of the negative class rather than a coarse "
     "version of the biology."),

# his point 10: do not use the safety cut-offs as hard filters
67: ("Both safety tools flag short peptides",
     "Both safety tools flag short peptides at higher rates, so the discoveries "
     "were tiered rather than filtered (Figure S6). ToxinPred2 flagged 73.6% of the "
     "216 discoveries of 10 residues or fewer against 59.2% of the 196 longer ones "
     "(Fisher exact test, P = 0.002), and AlgPred 2.0, applied to the 137 that "
     "ToxinPred2 passed, flagged 93.0% of 57 short peptides against 66.2% of 80 "
     "longer ones (P < 0.001). Both are amino-acid-composition Random Forests, so "
     "the length dependence is a property of the tools rather than of the peptides. "
     "Neither binary cut-off should be applied as a hard filter to short "
     "hydrolysate fragments. At the published thresholds the two together would "
     "discard 93.0% of the discoveries of ten residues or fewer, a rate set by "
     "peptide length rather than by anything about the sequences, and the Tier-1 "
     "lead carried forward here is itself flagged as an allergen at 0.54 against a "
     "0.3 cut-off. Work on food-derived fragments should retain the continuous "
     "scores, read them as one ranking signal among several and defer exclusion to "
     "assay, rather than removing candidates on a threshold calibrated against "
     "proteins two orders of magnitude longer. Tier 1 holds all 216 leads of 10 "
     "residues or fewer, whose safety scores are treated as a length artefact; Tier "
     "2 holds the 27 of the 31 passing both filters that exceed 10 residues; 169 "
     "discoveries fall in neither tier."),

# his point 4: how much the pose can carry given the control ordering
72: ("The three negative controls",
     "The three negative controls, peptides the classifier had confidently rejected "
     "at consensus scores below 0.01, scored −60.6 to −66.8, approaching IPAVF at "
     "−67.7 and clearing Diprotin A by more than 20 units, but they earned it "
     "through electrostatics of −236 to −312. The site presents four acidic "
     "residues, Glu205, Glu206, Asp545 and Asp663, against two basic ones, so a "
     "cationic peptide meets a net excess of carboxylates wherever in the pocket it "
     "lands. Each control carries two to four lysine or arginine side chains, and "
     "the site has no documented partner for any of them, since its S1 pocket takes "
     "proline or alanine and the side chains flanking the cleavage site face "
     "solvent.^{47} FVAPFPEVF and both positive controls carry none. After "
     "weighting, charge supplies 66% to 74% of the controls' favourable score, "
     "against 8% to 24% for the Tier-1 lead and the two positive controls, with the "
     "Tier-2 leads between them at 31% to 41% (Figure 6a). Ranking on the raw "
     "HADDOCK score would have promoted all three negative controls above Diprotin "
     "A, the strongest within 0.9 units of IPAVF. That ordering bounds how much any "
     "single total in this panel can carry, the lead's included. A function that "
     "places three peptides the classifier rejected below 0.01 above a documented "
     "inhibitor is not ranking these eight peptides by propensity to bind, so the "
     "lead's −100.1 is evidence that it forms a large, shape-complementary, "
     "desolvating interface rather than that it binds more tightly than Diprotin A. "
     "The composition of the score separates the panel where the total does not, "
     "which is why it is reported beside every entry in Table 4."),

# his point 4: non-competitive mechanism or protocol limitation
78: ("The structural results do not mirror",
     "The structural results do not mirror that ranking, which is consistent with "
     "proline acting through backbone conformation rather than through direct "
     "contact with the enzyme. In the best pose, Phe1 sits 3.3 Å from Glu205 and "
     "Glu206, against their backbone atoms rather than either carboxylate, while "
     "the free N-terminus stays at least 6.5 Å from the pair, so the pose does not "
     "reproduce the Glu-Glu recognition of a substrate amine (Figure 8). Phe5 "
     "reaches the Ser630 hydroxyl at 3.5 Å and Val8 the His740 backbone carbonyl at "
     "4.0 Å, two of the three catalytic triad residues. Phe9 made more contacts "
     "than any other position, within 5 Å of eight of the 22 site residues and 2.9 "
     "Å from Asp545. Model-derived residue importance and static docking contacts "
     "therefore capture different aspects of the interaction. The missing Glu-Glu "
     "contact admits two readings and the data here do not separate them. It may be "
     "real, in which case a nonamer occupying the cleft without delivering an "
     "α-amino group to the Glu pair is not positioned for cleavage and would "
     "obstruct the site rather than turn over in it, which is a non-competitive or "
     "mixed mode rather than the substrate-like binding of Diprotin A. It may "
     "equally be an artefact of the protocol: ambiguous interaction restraints "
     "reward buried interface area, the nine active receptor residues span the whole "
     "cleft rather than marking the S1 pocket, and nothing in the scoring function "
     "rewards placing one particular amine on one particular carboxylate pair, so a "
     "pose that satisfies the restraints by packing three phenylalanines is "
     "favoured over one that gives up contact area to make a single salt bridge. "
     "Separating the two needs experiment rather than more docking: an inhibition "
     "assay reporting whether the peptide competes with a substrate would settle "
     "it, and until then the pose is best read as a plausible binding geometry of "
     "unresolved mechanism."),

# his point 2, restated as a limitation with a number on it
83: ("The operating threshold is the first exposure",
     "The operating threshold is the first exposure. It was set on inner folds "
     "split at random, so near-identical sequences could be scored by a model that "
     "had effectively seen them, unlike the clean cluster-wise train–test split, "
     "and its hit rate falls from 91% out-of-fold to 86% on test. Section 2.5 puts "
     "a number on it: 30.8% of the training rows have a family relative in their "
     "own fold's training rows, and grouping the folds by family lowers out-of-fold "
     "AUC by 0.020 and moves the swept threshold from 0.93 to 0.97. Building folds "
     "cluster-wise, as applied by Asl et al.,^{6} would give an operating point "
     "calibrated without that advantage, and would admit fewer candidates than the "
     "412 reported here."),

# his point 1, second half: how close the shortlist sits to the training positives
84: ("Novelty filtering is the second",
     "Novelty filtering is the second. Only 536 of the 3,596 candidates reach 11 "
     "residues, so the near-duplicate screen at 40% identity covers 15% of the pool "
     "and the remaining 3,060 rely on exact matching alone. Measured against the "
     "873 training positives, 64 of the 412 discoveries (15.5%) wholly contain one "
     "of three residues or more and 41 (10.0%) contain one of six or more, among "
     "them 26 of the 196 discoveries that reach 11 residues (Table S13). "
     "Containment at any length reaches 132, but 68 of those contain only a "
     "dipeptide and the training set holds just two, so the longer floors are the "
     "informative ones. Graded identity behaves as the filter's coverage predicts: "
     "23 discoveries reach 80% identity to a training positive and every one lies "
     "in the short band the filter never reached, while none of the 196 long "
     "discoveries does. The filter works where it applies. Where it does not, part "
     "of what the screen returns is extension of a motif the model had already seen "
     "rather than discovery, and the four shorter training positives running inside "
     "the Tier-1 lead are an instance of it. Mean nearest identity to a training "
     "positive is 0.603 for short discoveries against 0.442 for long ones."),

# his priority 1 as a limitation, plus the family-aware split result
86: ("Nothing was synthesised",
     "The redundancy of the positive class is the fourth. The 966 positives "
     "collapse to 577 families and 141 of them are preproinsulin fragments against "
     "none of the negatives (Section 2.1), so the class carries about 60% of its "
     "nominal sample size and one precursor supplies a seventh of it. Rebuilding "
     "the partition family-aware, with whole families held out and every arm scored "
     "on one shared 290-row test set, narrows the negative-class AUC gap from 0.180 "
     "to 0.074 while the gap in hard-negative false-positive rate holds at 20.6 "
     "points against 36.5. Part of the AUC effect is an artefact of the split and "
     "the deployment effect is not, which is the result that carries into Section "
     "4. The same split sharpens the central point: the DBAASP-only arm records the "
     "best AUC of the six at 0.853 and the worst hard-negative false-positive rate "
     "at 88.6%, against 0.795 and 28.0% for the proteome-only arm. A negative class "
     "selected on test AUC would have been selected backwards. Nothing was "
     "synthesised, so the precision of the 412 is an expectation rather than a "
     "measurement, and every output here is a prioritised hypothesis for "
     "experimental follow-up rather than a confirmed activity. Two experiments "
     "would settle the main questions. Training the dual-negative model on the "
     "released BertADP split would separate negative-class composition from "
     "distribution shift. An inhibition assay against the recombinant enzyme would "
     "turn the Tier-1 lead into a measured candidate and would say whether its "
     "inhibition is competitive."),

# his priority 2, consequence: 4.1 reverses
90: ("A negative class should be matched on every descriptor",
     "**A negative class should be matched on every descriptor that separates its "
     "source database from the positives rather than on length alone.** The "
     "ablation shows this is both achievable and cheap. Arm F matches length and "
     "net charge together and drives the charge separability of the training "
     "classes from 0.804 to 0.507, chance, for 0.018 in AUC, cutting the "
     "false-positive rate on held-out proteome fragments from 45.1% to 33.1% while "
     "keeping the soft-negative rate at 6.5%. The first draft of this work argued "
     "that forcing such a match was not worth the distortion, on the ground that no "
     "draw from a 91% cationic database can supply a non-cationic antimicrobial "
     "peptide at every positive's length. That premise is correct and the "
     "conclusion drawn from it was not. DBAASP supplies only 23% of arm F's "
     "negatives and the rest are cut from GO-filtered UniProt on demand, so the "
     "constraint binds on the curated class alone rather than on the design. Where "
     "one bioactivity class cannot supply the match, the soft majority should draw "
     "across several classes or the proteome-fragment share should be raised. "
     "Dropping the skewed class outright is the other workable route and on "
     "proteome fragments it is the better one: the proteome-only arm reaches a "
     "29.1% false-positive rate against arm F's 33.1%, at the cost of the "
     "generic-bioactivity control, its soft-negative rate rising from 6.5% to "
     "10.5%. Choosing between them is a choice about which error matters and should "
     "be made explicitly rather than inherited from whichever database was at hand. "
     "What should not be done is to select on test AUC. Across the six arms the two "
     "with the highest charge separability record the highest AUCs and the highest "
     "deployment error, and under a family-aware split the best-scoring arm is the "
     "worst-deploying one. Testing every descriptor against the class labels before "
     "any model is fitted costs one ROC curve per descriptor and would expose the "
     "skew; reporting the false-positive rate on a held-out sample of the sequence "
     "space the classifier is meant to meet is what shows whether closing it "
     "mattered. Antimicrobial peptides are cationic because their selectivity "
     "derives from electrostatic attraction to anionic bacterial membranes,^{41} so "
     "the skew is a property of that class rather than of any sample taken from "
     "it."),

# his point 5: frozen features remove the seed sensitivity
94: ("A shortlist drawn from a parameter-efficient",
     "**A shortlist drawn from a parameter-efficient fine-tune should be reported "
     "across seeds rather than from one fit.** The adapter carried forward here is "
     "the seed-42 fit, the highest of the five spanning 0.807 to 0.877, and the "
     "consensus is the mean of that model's probability with XGBoost's, so the 412 "
     "discoveries, both tiers and the FVAPFPEVF case all rest on the most "
     "favourable ESM-2 fit rather than a representative one. Reporting the "
     "intersection of shortlists across seeds, or the mean ensemble across them, "
     "would give a candidate set that does not depend on a single fit. Sensitivity "
     "of this kind is not well characterised in the ADP literature, the nearest "
     "comparable study reporting ten-seed variability only for its classical "
     "classifier heads.^{6} The simpler remedy is to stop fine-tuning for "
     "deployment. The variance enters through the randomly initialised "
     "classification head, which moves the starting point of the whole "
     "optimisation, so it is removed by keeping the encoder frozen and fitting a "
     "classical head on the extracted representation. On the frozen "
     "1,287-dimensional vector, logistic regression is exactly deterministic across "
     "the five seeds at AUC 0.845 and XGBoost varies by 0.003 at 0.860, against a "
     "0.070 span for the fine-tune. Both sit inside the fine-tune's confidence "
     "interval and both retain the ESM-2 representation, so the seed sensitivity is "
     "a cost of the fine-tuning step rather than of the embedding, and a screen "
     "that needs a reproducible shortlist is better served by frozen features with "
     "a convex head than by the better single fit. Freezing the classification head "
     "itself is not an alternative: an untrained head carries no mapping from "
     "representation to label, so it has to be fitted, and the only question is "
     "whether it is fitted alongside the encoder or on top of it."),

# conclusions: the reversal
99: ("We did not force a charge-matched negative class",
     "We did not force a charge-matched negative class when this dataset was built, "
     "and the ablation reported here shows that we should have. Matching length and "
     "charge together drives the shortcut to chance for 0.018 in AUC and cuts the "
     "false-positive rate on proteome fragments by twelve points, without "
     "substituting a distribution the databases do not contain, because three "
     "quarters of that negative class is cut from GO-filtered UniProt on demand "
     "rather than resampled from DBAASP. The screen reported here predates that "
     "result and carries the shortcut, so its 412 discoveries are a charge-selected "
     "subset of the digest and the design we now recommend is not the design that "
     "produced them. The Tier-1 lead FVAPFPEVF scored best of the docked panel from "
     "van der Waals complementarity and desolvation rather than electrostatics, its "
     "three phenylalanines making 15 of 19 contacts within 5 Å while its single "
     "glutamate stayed out of contact. That structural case is measured on a fixed "
     "pose and does not depend on the fit that selected the peptide; its position "
     "in the shortlist does. FVAPFPEVF is therefore an untested candidate of "
     "documented food origin rather than a characterised inhibitor, and the "
     "shortlist that contains it is provisional in membership while the mechanism "
     "that recommends it is not."),

# his point 9: say it in the caption too
127: ("Table 2. Discrimination on the Held-Out Test Set",
      "Table 2. Discrimination on the Held-Out Test Set (n = 178). Point estimates "
      "carry percentile 95% confidence intervals from 1,000 bootstrap resamples, "
      "with MCC taken at a threshold of 0.5. The seed column covers only the four "
      "base learners, since reseeding the stack means refitting all four. It "
      "reports five independent refits against a single fit in the AUC column, so "
      "the two are not an estimate and its uncertainty: for ESM-2/DoRA the 0.877 is "
      "the best of the five seeds and the 0.842 is their mean, and the adapter "
      "carried into the screen and the case study is that best-seed fit. "
      "ESM-2/DoRA (Basith negatives) is the negative-class ablation of Section 3.1 "
      "and differs from the dual-negative model in its training negatives alone."),

# his point 8: label the orange sphere
321: ("Figure 8. The lead binds across the catalytic cleft",
      "Figure 8. The lead binds across the catalytic cleft without reproducing the "
      "Glu-Glu recognition of a substrate amine. FVAPFPEVF docked into human DPP-IV "
      "(PDB 4A5S), on the top-ranked model of its best-scoring cluster. (a) The "
      "pose across the catalytic cleft. The peptide is magenta and the receptor "
      "grey, with contacted active-site residues in cyan. Every contact within 4.5 "
      "Å to the eight site residues shown is drawn as a dashed line labelled by "
      "distance, six of the seventeen the pose makes across all 22 restrained "
      "residues. Ser630 and His740 are two of the three catalytic triad residues, "
      "the third being Asp708. (b) Close-up on Phe1 with the receptor cartoon made "
      "semi-transparent. The aromatic side chain reaches Glu205 and Glu206 at 3.3 "
      "Å, while the free N-terminal amine lies 6.5 Å from Glu206, so the pose does "
      "not reproduce the Glu-Glu recognition of a substrate amine. The amine is "
      "drawn as an orange sphere, labelled “free N-terminal amine”, and the arrow "
      "marks the unmade 6.5 Å approach to the Glu206 carboxylate."),
}


INSERT_AFTER = {

# his point 1 and priority 1, in Methods 2.1
20: [
 "Because clustering does not reach the short half of the dataset, the residual "
 "proximity between the two partitions was measured rather than assumed small "
 "(Table S11). No test sequence occurs exactly in the training set. Sixty-seven of "
 "the 178 test sequences (37.6%) stand in a substring relationship with a training "
 "sequence, against 20.9% under a null that shuffles each test sequence's residues "
 "and so holds its length and composition fixed (200 shuffles, P < 0.005). Graded "
 "identity was computed as matched residues over the longer sequence of each pair, "
 "since CD-HIT's convention of normalising over the shorter saturates at 1.0 for "
 "any containment and carries no information at these lengths. Forty-one test "
 "sequences (23.0%) reach 80% identity to a training sequence on that measure, and "
 "they concentrate in the band clustering never covered: 35 of the 130 test "
 "sequences below 11 residues against 6 of the 48 above. Removing those 41 rows "
 "and rescoring the saved predictions, with no model refitted, lowers test AUC "
 "from 0.877 to 0.831 for ESM-2/DoRA, from 0.857 to 0.815 for XGBoost and from "
 "0.879 to 0.836 for the consensus. The split inflates discrimination by about "
 "0.046, and that is the allowance to carry when reading every AUC reported here.",

 "The positive class carries its own redundancy, which bears on what 966 sequences "
 "are evidence for rather than on leakage (Table S12). Grouping by single linkage "
 "where one sequence contains another and the shorter covers at least half the "
 "longer, the 966 positives collapse to 577 families against 861 for the "
 "independently drawn negatives, so the positive set carries about 60% of its "
 "nominal sample size. The three largest families are regions of human "
 "preproinsulin at 56, 55 and 32 members, covering the B chain, the C-peptide and "
 "the signal peptide, and 141 positives (14.6%) are preproinsulin fragments of "
 "four residues or more against none of the 966 negatives. Twenty-one families "
 "straddle the published split, carrying 226 positives. The coverage condition is "
 "load-bearing: plain containment with no length floor is transitive through the "
 "two dipeptide positives and chains the set into a 242-member family, and it does "
 "the same to the negatives, which are drawn independently and cannot share "
 "ancestry, so a nesting-only definition measures the metric rather than the data. "
 "The positive set was left intact, since reducing it would break comparability "
 "with the source dataset and with the predictors benchmarked here, and the "
 "consequences are traced in Sections 3.1 and 3.6.",
],

# his priority 2, in Methods 2.3
28: [
 "That comparison varies composition and length together, because a pool drawn "
 "from a curated bioactive database differs from the positives in both. Six arms "
 "separate them (Table 5). Arm A is the dual-negative class as published. Arm B is "
 "the Basith pool at its own lengths and arm C the same pool length-matched. Arm D "
 "is the DBAASP half alone and arm E the proteome half alone, both length-matched "
 "one-to-one. Arm F is matched on length and on net charge together, accepting a "
 "negative within 0.5 charge units of its paired positive, drawn from DBAASP first "
 "and from GO-filtered UniProt where that database cannot supply a match at the "
 "required length and charge. Every arm keeps the 873 training positives and is "
 "scored on the published 178-row test set, so only the training negatives vary "
 "and the arms are directly comparable.",

 "Arms C and D cannot reach 873 pairs. The Basith pool holds nothing below 9 "
 "residues while 45% of the positives are shorter, capping arm C at 303, and "
 "DBAASP runs out of sequences at the positives' lengths, capping arm D at 611. "
 "Each is therefore also fitted with the positives subsampled to its negative "
 "count, so its false-positive rate is not read off a positive-heavy training set. "
 "A published negative set cannot be repaired by length-matching when the source "
 "database lacks the lengths, which is itself part of the result. The readout is "
 "XGBoost on the fused vector under the deployed model's hyperparameters, reported "
 "as the mean over ten training-row permutations with their standard deviation: "
 "row subsampling draws rows in the order given, so a permutation alone moves this "
 "learner by about 0.004 in AUC, which is the size of the smallest difference "
 "between arms and makes single fits unusable for comparing them. Embeddings for "
 "the sequences the arms add were extracted with the same frozen encoder and the "
 "same pooling as Section 2.2.",

 "The dependence of the negative-class effect on the partition was tested by "
 "rebuilding the split family-aware. Families were formed over the published 1,932 "
 "sequences under the definition of Section 2.1 and whole families were assigned "
 "to train or test at a 15% target, held separately for positives, soft negatives "
 "and hard negatives so the hard-negative block does not shrink to a handful of "
 "rows. That gives one shared 290-row test set, 145 positives with 96 soft and 49 "
 "hard negatives, on which every arm is scored. Each arm trains on the "
 "family-aware training positives together with its own negatives, excluding "
 "anything appearing in the new test set. The published and family-aware test sets "
 "differ, so only the gap between two arms within one split is interpretable and "
 "no arm's AUC is compared across splits.",
],
}


def main():
    x = open(DOC, encoding="utf-8").read()
    spans = [(m.start(), m.end(), m.group(0)) for m in PARA.finditer(x)]
    fails = []

    # check every target before editing, so a stale index fails loudly
    for i, (expect, _) in REPLACE.items():
        if i >= len(spans) or not ptext(spans[i][2]).lstrip().startswith(expect):
            got = ptext(spans[i][2])[:60] if i < len(spans) else "<out of range>"
            fails.append(f"para {i}: expected {expect[:40]!r}, found {got!r}")
    if fails:
        print("FAILED:")
        [print("  " + f) for f in fails]
        sys.exit(1)

    # apply back to front so earlier offsets stay valid
    edits = []
    for i, (_, new) in REPLACE.items():
        edits.append((spans[i][0], spans[i][1], rebuild(spans[i][2], new)))
    for i, blocks in INSERT_AFTER.items():
        add = "".join(BODY.format(runs(t)) for t in blocks)
        edits.append((spans[i][1], spans[i][1], add))

    for start, end, text in sorted(edits, key=lambda e: -e[0]):
        x = x[:start] + text + x[end:]

    open(DOC, "w", encoding="utf-8").write(x)
    print(f"replaced {len(REPLACE)} paragraphs, inserted "
          f"{sum(len(b) for b in INSERT_AFTER.values())}")


if __name__ == "__main__":
    main()
