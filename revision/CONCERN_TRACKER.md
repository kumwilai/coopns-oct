# OJCS-2026-04-0386 revision tracker

Deadline 15 Sep 2026. Last audited 11 Sep 2026, after two audits. The first checked every promise in
the response letter against the manuscript. The second, an external critical read, found statements in
the manuscript that contradicted the code, and those are listed below as settled.
Main paper 12 pages and self contained. Supplement 8 pages, submitted alongside for extra detail.
Both compile with no undefined reference and no overfull line.
Response letter has no unresolved bracket and every section, table and figure reference verified.

| ID | Concern | Where it is answered | Status |
|---|---|---|---|
| R1-1 | cooperation map has no loss enforcing an uncertainty meaning | V H measures it, rank correlation minus 0.258, renamed, calibrated removed | CLOSED |
| R1-2 | no predicate level ablation | V F, all six removals on the reported checkpoint, maps in Fig. 2 | CLOSED |
| R1-3 | risk of artificial sharpening and hallucinated boundaries | V J, four measures, two reported as costs, named in Limitations | CLOSED |
| R1-4 | fuzzy rule weights look heuristic | V F, fourteen variations, none turns a measure negative | CLOSED |
| R1-5 | missing expert network and neuro symbolic literature | I C, 26 references all resolved | CLOSED |
| R1-6 | typos and grammar | style script clean, mechanical checks clean | CLOSED |
| R2-1 | LOO protocol contradiction | Table 3 carries the no adaptation protocol only, Table S7 reports the adapted one as a test selected historical diagnostic | CLOSED |
| R2-2 | explain the PSNR drop, SwinIR minus 1.69 dB | VI A, cache defect fixed, SwinIR now minus 0.17 dB | CLOSED |
| R3-1 | highlight the new ideas | I B | CLOSED |
| R3-2 | motivation not clear | I A | CLOSED |
| R3-3 | cite three specific modeling papers | I C, all three cited and discussed | CLOSED |
| R3-4 | define stability | III C, Definition 1 before the theorem | CLOSED |
| R3-5 | state the optimization problem and how it is solved | II B | CLOSED |
| R3-6 | embedded notation is hard to read | Table 1 defines every symbol before first use | CLOSED |
| R3-7 | define the operator in equation 6 | II A, Hadamard product defined in words and symbols | CLOSED |
| R3-8 | drop the multiplication dot | removed throughout, the remaining \cdot are argument placeholders | CLOSED |
| R3-9 | justify the six predicates and give sensitivity | Table 2 and V F | CLOSED |
| R3-10 | training algorithm unclear | V, six loss terms with every weight, the schedule, the selection rule, the constants, Algorithm 1 of the supplement | CLOSED |
| R3-11 | safety validation on lesions, fluid, thin layers | V J, expert reading stated as a limitation | CLOSED |
| R3-12 | link theory to the results section | V I for the theorems, V J for the three constraints | CLOSED |
| R3-13 | public code and data link without password | the release is built and goes public after acceptance, under a permanent identifier | ANSWERED BY DECISION |
| R3-14 | follows from 13 | release documents the six defects and the checks | CLOSED |
| R3-15 | do not call them experiments | zero occurrences in the source | CLOSED |
| R3-16 | fair comparison at similar complexity | Table 5, lower block, text narrowed to match the code | CLOSED |
| R3-17 | how were the fuzzy parameters computed | III C and V F and the constants table of the supplement | CLOSED |
| R3-18 | transferability needs cost and variability numbers | V K, wall clock and seed spread per backbone | CLOSED |
| R3-19 | verify the assumed constants and bounds hold | V I and V J, honest majority statement, failures reported | CLOSED |

## Self containment

The main paper carries every closed form, both proofs, every constant with its origin, the training
schedule, the selection grid, the six failure maps and the full resolution visual comparison. Three
pointers into the supplement remain and none is load bearing, namely the adapted protocol of the
original submission, which is reported there because its numbers are selected on the data they
describe, the per cell seed spreads of Tables 3 and 4, and a restatement of the learning rates.

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
- The adapted rows have left the main paper. Each of their folds keeps the epoch that scores best on
  the subject it then reports, so they are selected on the data they describe. Table S7 of the
  supplementary file carries them, labeled a historical diagnostic, and no claim of the paper rests
  on them.
- The 173 test pairs come from five clean image identities, which the measures paragraph now says.
- The loss weights are now given numerically, and the inactive learned weight path is disclosed.

## Round four, the letter synchronised and the remaining gaps closed

- Response 2.2 asserted that tightening the third constraint moves the output toward the backbone and
  then, four sentences later, withdrew the same claim. The correction had been appended and the
  original left standing. The old paragraph is gone and one explanation remains.
- Response 3.5 still said the constraints enter training as penalties, which the main paper now
  contradicts. Replaced.
- Response 3.1 said four contributions and listed five. It now lists the four the paper has.
- Table 3 had lost the seed spread on fidelity, which appears nowhere else, since Table S6 carries
  only the six clinical measures. Restored, and Table 4 is now the same shape and also one column, so
  a row here compares against a row there without re reading a header.
- The DnCNN overhead is 5.7 percent, not 5.8. Arithmetic from our own counts.
- The claim that DnCNN and KBNet each have a measure that falls on test was wrong for KBNet, whose
  edge preservation is plus 0.0005 percent with one seed negative. Both are now described separately
  from the selection outcome.
- The supplement said the repair restores the gradient to every rule parameter. It restores it to
  four weighted rules and the base. The fifth is inactive because its truth value is the constant
  zero when a single corrector proposes, which is now the stated reason rather than an observation.
- The two Lipschitz constants now carry their domains. 0.6014 is exact on the whole cube and 0.5489
  is exact on the face where the fifth coordinate is fixed, which bounds any smaller attainable set.
  "Real perturbations never reach the worst case" became "the perturbations tested did not attain it".
- The P1 disclosure claimed the allocation is unaffected. Only the failure map is. The score reaches
  the soft failure and the acceptance energy, so the narrower claim is that the subterm is non
  discriminative, not that the output is independent of it.
- Theorem S1 of the supplement gained the positive output background variance the main theorem has.
- The p value column of the adapted table is removed. Every checkpoint there was chosen on the
  subject it is scored on, so the test does not carry its usual meaning.
- Enforcement language softened where only a finite penalty acts, the mean absolute change renamed
  away from the gate symbol, the selection score no longer called a confidence bound, and the
  explanation claim narrowed to the multiplicative gain, since the edge branch and the background
  operation do not pass through the allocation.
- make_submission.sh discarded compiler output, so a hard TeX error left the previous PDF in place
  and every later check read a stale file. It now fails loudly and prints the error.

## Round three, second pass, the predicates and the objective read against the code

- The $P_1$ continuity subterm carries no information. Erosion after dilation by the same element is
  a closing, which contains the set it closes, so the inner product equals the edge count and the
  ratio is one. Measured on 256 by 256 fields it reads 0.952 and every pixel of the shortfall is on
  the zero padded border. In the interior it is exactly 1. The supplement now says so, and says the
  score varies only through the other two subterms while the failure map does not read it at all.
- $P_3$ and $P_4$ do not reach the unit interval by construction. The variance ratio of $P_3$ may
  exceed one and the correlation in $P_4$ may be negative. Both are clipped in the code and the
  clips are now printed.
- The $P_5$ target is measured on the absolute log residual while $1/\sqrt{k}$ is the coefficient of
  variation of a Gamma variable. A coefficient of variation does not survive that transform, so the
  target is now called speckle motivated with the right depth ordering rather than derived.
- The objective is optimized at the candidate and not at the output. The acceptance rule is skipped
  whenever the model is in training mode.
- The constraints do not enter training at all. The paper said they enter as penalties. No term of
  the loss stands in for them, and the verifier that evaluates them is called from one place, inside
  the branch that training skips.
- Symbols that collided are separated. The scalar fidelity drop is $D$ rather than $\Delta$, which
  is the gain map, the mean absolute change in Equation 19 is $g$ rather than $a$, which is the
  allocation, and the fixed image bands are $\sigma_R$ and $\mu_U$ rather than the anatomical sets of
  Theorem 2. The smoothness penalty acts on the admitted gain, not the allocation.
- Every squared norm in the loss is a mean over pixels and the batch, now stated, and the clipping of
  the first two penalties is printed in the equation rather than described after it.
- The Pearson term is not differentiable everywhere. It is undefined for a constant map, where the
  code returns zero, and its invariance is to positive rescaling only.
- The nine quantities of the clinical term and the exact selection rule are now written out in the
  supplement, since the main paper points at both.

## Third party review, round three, settled against the code

- The manuscript explained the fixed thresholds by saying the rule layer runs without gradient
  tracking. That is wrong. A training step on the selected model shows the base level and the weights
  of rules one to four carry gradients, and their values differ from what they were restored with.
  The thresholds are fixed because they have no gradient path, which is a different statement.
  Section V-I now reports the audit parameter by parameter.
- Rule five is inert. Its weight has no gradient path and is identical to its initial value in all
  four checkpoints, because the conflict it tests for needs more than one corrector proposing at a
  pixel. The reported Lipschitz constant 0.5489 was already the four rule sum, while printed
  Equation 6 sums five, so the theorem now says a rule that never fires drops out and the paper gives
  both constants with their domains.
- Two of the four backbones cleared no admissibility level at all. NAFNet and SwinIR were admitted at
  the strictest, and for DnCNN and KBNet the script reported the least bad setting of the grid. Those
  are the two backbones with a falling measure. This was visible in the recorded selection and was
  not stated. It is now, in the main paper and in the supplement, along with the exact bound.
- The abstract said only a small head is refitted while Section V-K says the whole wrapper is. The
  abstract now separates a common architecture from separately fitted weights.
- The discussion said fidelity falls whenever the wrapper acts, which the paper's own Figure 3
  disproves at plus 0.02 dB on NAFNet. It now reports a mean and names the exception. The comparison
  with the variation between two acquisitions of the same eye is withdrawn, since we never measured it.
- Section IV was titled a guaranteed safety decision while its rule is a majority vote that never
  intervened on the test set. It is now Bounded Correction and Majority Based Acceptance.
- Theorem 2 needed a positive output background variance to be defined, the design was said to make
  its hypotheses hold rather than encourage them, and the 60 image count covers the two conditions we
  measure and not the whole theorem. All three corrected.
- Theorem 1 tightness was attributed to the Lukasiewicz conjunction. It follows from the allocation
  being affine in the truth values. The budget identities are sufficient conditions, not a derivation.
- The paper listed the initialization as default while every run restores the submitted checkpoint.
  Both stages are now distinguished.
- The supplement concluded that adaptation reliably improves contrast, from folds that select on the
  subject they report. That conclusion is withdrawn.
- Equation numbers printed in the supplement are now generated from the compiled main paper by
  gen_eqnums.py, so a section merge can never leave them stale again.
- Twelve response letter defects corrected, including the parameter totals, which are 173289 for
  KBNet and 402903 for DnCNN, below NAFNet and above SwinIR rather than between them.

## Restructured for readability, 12 pages held

- Table 3 of the previous version merged four studies into one float, so a reader comparing two
  backbones had to work out which block a row belonged to. It is now Table 3 for the test set,
  Table 4 for transfer and Table 5 for the component removals and the alternative operators.
- The six failure maps and the full resolution visual comparison moved from the supplementary file
  into the main paper as Figures 2 and 4. The supplement is 7 pages and carries nothing a claim
  rests on.
- The full resolution comparison used a square crop, which made a four row figure too tall for the
  main paper. Retinal layers run across the scan, so the crop is now 120 by 300 and the figure is
  less than half its old height. Its backbone names moved out of the B-scans, where dark type on
  dark speckle was barely readable.
- Section V cut by about a third and every other section tightened. The references are 26, down
  from 34. Nothing a reviewer asked for was removed.
- Eleven numbered sections down to seven. Related work became a subsection of the introduction. The
  method split into two sections named for what they contribute rather than for what they contain,
  and each absorbed the theorem that belongs to it, so the stability result now sits with the rule
  layer it bounds and the contrast result with the safety decision it protects. Training joined the
  tests and results. Every section pointer in the response letter and in this file was remapped from
  the compiled numbering rather than by hand, then checked against the sections the built PDF has.

## Settled in the third audit, the objective read line by line against the code

- Equation 15 printed the clinical term as a sum of negative log ratios over the contrast to noise
  ratio and the tissue contrast index. The code uses a weighted one sided penalty against a target
  taken from the backbone, over nine quantities with weights 1, 5, 1, 15, 40, 5, 8, 40 and 10 and
  factors 1.12, 1.12, 1.00, 1.05, 1.20, 1.12, 1.10, 1.15 and 0.87, with a symmetric pull on three of
  them. The tissue contrast index is not among the nine. The equation now matches the code.
- Equation 18 printed unweighted gradient matching. The code weights it by the clean edge map and
  adds two parts, a normalized edge magnitude error and a hinge that refuses an edge preservation
  index below five percent above the backbone. The equation now shows all three.
- The objective of Equation 13 listed five terms. Eight more were active, a floor and a ceiling on
  the size of the change, a smoothness penalty on the allocation map, a ceiling on the edge branch
  and four terms protecting the background and the tissue brightness. They are now a sixth term,
  Equation 19, with every constant printed.
- The flag for the edge weight never reaches the loss. It is now named with the other inert flags.
- The section pointers of this tracker still named the eleven subsection layout of Section V. The
  merge to nine had moved eight of them. All are checked against the compiled document.
- The one line gloss of the property term in Section II B still said it calibrates the scores
  against their thresholds, which the corrected Equation 16 contradicts. It now says what the code
  does.

## Settled since the last audit

- The selection criterion is blind to the two safety measures. Measured on the validation split, the
  runner up gate for NAFNet raises weak structure retention to 43.9 percent where the selected gate
  lowers it to 37.6. Reported in Limitations. No reselection, because the criterion was fixed before
  the test split was scored.
- All three safety constraints are now reported. Energy and bounded change hold on all 173 images,
  the no sacrifice constraint fails on 38, and the layer returns the candidate in full everywhere.
