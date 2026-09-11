# OJCS-2026-04-0386 revision tracker

Deadline 15 Sep 2026. Last audited 11 Sep 2026, after two audits. The first checked every promise in
the response letter against the manuscript. The second, an external critical read, found statements in
the manuscript that contradicted the code, and those are listed below as settled.
Main paper 12 pages and self contained. Supplement 8 pages, submitted alongside for extra detail.
Both compile with no undefined reference and no overfull line.
Response letter has no unresolved bracket and every section, table and figure reference verified.

| ID | Concern | Where it is answered | Status |
|---|---|---|---|
| R1-1 | cooperation map has no loss enforcing an uncertainty meaning | IX H measures it, rank correlation minus 0.258, renamed, calibrated removed | CLOSED |
| R1-2 | no predicate level ablation | IX D, all six removals on the reported checkpoint | CLOSED |
| R1-3 | risk of artificial sharpening and hallucinated boundaries | IX J, four measures, two reported as costs, named in Limitations | CLOSED |
| R1-4 | fuzzy rule weights look heuristic | IX E, fourteen variations, none turns a measure negative | CLOSED |
| R1-5 | missing expert network and neuro symbolic literature | II C, 34 references all resolved | CLOSED |
| R1-6 | typos and grammar | style script clean, mechanical checks clean | CLOSED |
| R2-1 | LOO protocol contradiction | Table 3 carries both protocols, S7 documents the adapted one | CLOSED |
| R2-2 | explain the PSNR drop, SwinIR minus 1.69 dB | X A, cache defect fixed, SwinIR now minus 0.17 dB | CLOSED |
| R3-1 | highlight the new ideas | I B | CLOSED |
| R3-2 | motivation not clear | I A | CLOSED |
| R3-3 | cite three specific modeling papers | II C, all three cited and discussed | CLOSED |
| R3-4 | define stability | VII A, Definition 1 before the theorem | CLOSED |
| R3-5 | state the optimization problem and how it is solved | III B | CLOSED |
| R3-6 | embedded notation is hard to read | Table 1 defines every symbol before first use | CLOSED |
| R3-7 | define the operator in equation 6 | III A, Hadamard product defined in words and symbols | CLOSED |
| R3-8 | drop the multiplication dot | removed throughout, none in source | CLOSED |
| R3-9 | justify the six predicates and give sensitivity | Table 2 and IX D and IX E | CLOSED |
| R3-10 | training algorithm unclear | VIII, five parts, Algorithm 1 of the supplement | CLOSED |
| R3-11 | safety validation on lesions, fluid, thin layers | IX J, expert reading stated as a limitation | CLOSED |
| R3-12 | link theory to the results section | IX I, per constraint failure counts added | CLOSED |
| R3-13 | public code and data link without password | answered with a reason, double blind, archive on acceptance | ANSWERED BY DECISION |
| R3-14 | follows from 13 | release documents the five defects and the checks | CLOSED |
| R3-15 | do not call them experiments | zero occurrences in the source | CLOSED |
| R3-16 | fair comparison at similar complexity | Table 3 lower blocks, text narrowed to match the code | CLOSED |
| R3-17 | how were the fuzzy parameters computed | V B and IX E and the constants table of the supplement | CLOSED |
| R3-18 | transferability needs cost and variability numbers | IX K, wall clock and seed spread per backbone | CLOSED |
| R3-19 | verify the assumed constants and bounds hold | IX I, honest majority statement, failures reported | CLOSED |

## Self containment

The main paper carries every closed form, both proofs, every constant with its origin, the training
schedule and the selection grid. Four pointers into the supplement remain and none is load bearing,
namely the per subject spreads of the adapted protocol, the per cell seed spreads of Table 3, the
full resolution visual comparison, and a restatement of the learning rates.

## Not a reviewer concern, still open

- The received date field is left empty, at the author's instruction. The journal fills it at
  production, and guessing it from the manuscript number would have been a fabrication.
- The seeded zeroshot result files carry no checkpoint field. Provenance was established from the
  sweep log ordering and from the fact that they differ from the pre sweep run, but it is not
  recorded in the files themselves.
- The matched complexity baseline was fitted before the gate search, so it carries the percentile
  gate rather than the tissue bounded gate NAFNet was finally selected with. Its checkpoint is gone,
  so the rest of its configuration cannot be read back. The paper says this rather than claiming a
  match it cannot verify.

## Settled in the second audit, each verified against the code or the checkpoints

- Equation 16 printed a sum of squared differences between each property score and its threshold.
  The code compares the corrected score against the backbone score with weights 3, 2, 2 and 1 and a
  penalty below one half, and the thresholds appear nowhere in it. That is why the thresholds take no
  gradient and hold their literature values. The equation now matches the code.
- Equation 14 printed squared error plus one minus the structural similarity index and mentioned a
  dead zone that did not appear in it. The dead zone is one of the two selected quantities, so the
  equation now shows the three part function the code implements.
- The paper said the three safety constraints must hold on every image. The rule accepts on two of
  three, and 38 images pass only two while still receiving the full candidate. It is now described as
  a majority acceptance test.
- The deployment advice said tightening the third constraint moves the output toward the backbone.
  It does not when the other two hold. It now says to lower the gain bound.
- The parameter overhead was given as about 2.4 percent, which is the shared part alone. The whole
  wrapper is 2.3 to 5.8 percent, since the DnCNN head is larger than the shared wrapper.
- The Lipschitz gap was called a conservative proof, contradicting the supplement, which says the
  constant is attained. It now says the perturbations never reach the worst case.
- The cooperation map diagnostic did not say which map was correlated with the backbone error. It is
  the demand map, and a well ordered demand map would give a positive correlation.
- The adapted rows are now labelled in the main paper as selected on the subject they report.
- The 173 test pairs come from five clean image identities, which the measures paragraph now says.
- The loss weights are now given numerically, and the inactive learned weight path is disclosed.

## Settled since the last audit

- The selection criterion is blind to the two safety measures. Measured on the validation split, the
  runner up gate for NAFNet raises weak structure retention to 43.9 percent where the selected gate
  lowers it to 37.6. Reported in Limitations. No reselection, because the criterion was fixed before
  the test split was scored.
- All three safety constraints are now reported. Energy and bounded change hold on all 173 images,
  the no sacrifice constraint fails on 38, and the layer returns the candidate in full everywhere.
