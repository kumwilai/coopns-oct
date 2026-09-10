# OJCS-2026-04-0386 revision tracker

Deadline 15 Sep 2026. Today 09 Sep 2026.
Deliverables: revised manuscript (LaTeX) and response to reviewers (docx).

Author style rules for the whole manuscript.
No em dash. No colon. No semicolon. Plain words. Top down explanation.
Every variable and equation defined before first use. Figures redrawn.

## Status legend
TODO, RUNNING, DONE, BLOCKED

| ID | Reviewer concern | Closure plan | Needs new run | Status |
|----|------------------|--------------|---------------|--------|
| R1-1 | Confidence head has no loss enforcing an uncertainty meaning | Measure whether the map predicts backbone error. Report rank correlation, sparsification error, reliability curve. Rename honestly if it fails. | yes E4 | TODO |
| R1-2 | No predicate level ablation | Leave one predicate out for P1 to P6 at inference on PKU37 | yes E2 | TODO |
| R1-3 | Risk of artificial sharpening and hallucinated boundaries | New safety section. False edge rate against clean reference, weak structure retention, worst case per image table | yes E6 | TODO |
| R1-4 | Fuzzy rule weights look heuristic | Sensitivity sweep over rule weights and alternative t norms. State how each constant was chosen | yes E3 | TODO |
| R1-5 | Missing deep expert network and neuro symbolic literature | Expand related work with recent expert network and neuro symbolic image processing papers | no | TODO |
| R1-6 | Typos and grammar | Full rewrite pass | no | TODO |
| R2-1 | LOO protocol contradiction, what is trained per fold | Code shows per fold few shot adaptation from the PKU37 corrector with EWC. Report protocol exactly and add a true zero shot transfer table | yes E1 | RUNNING |
| R2-2 | Explain the PSNR drop, SwinIR minus 1.69 dB | New subsection on the fidelity and clinical trade off with a per backbone explanation | no | TODO |
| R3-1 | Highlight the new ideas | Rewrite introduction with an explicit novelty list | no | TODO |
| R3-2 | Motivation not clear | Rewrite motivation top down from the clinical problem | no | TODO |
| R3-3 | Cite three specific modeling papers | Add the three references and discuss them where they are relevant | no | TODO |
| R3-4 | Define stability | Give a formal definition of the stability notion used before the theorem | no | TODO |
| R3-5 | State the optimization problem and how it is solved | New problem statement section with the objective, the constraints, the variables and the solver | no | TODO |
| R3-6 | Embedded notation is hard to read | Notation table first, one symbol per concept, no nested subscripts | no | TODO |
| R3-7 | Define the operator in equation 6 | Define the Hadamard product in the notation table before first use | no | TODO |
| R3-8 | Drop the multiplication dot | Remove explicit dots for scalar products | no | TODO |
| R3-9 | Justify the six predicates and give sensitivity | Clinical and mathematical justification per predicate plus the E2 sensitivity study | yes E2 | TODO |
| R3-10 | Training algorithm unclear | Full training algorithm box, targets, losses, initialization, parameter selection, and a clear statement about paired data use | no | TODO |
| R3-11 | Safety validation on lesions, fluid, thin layers, expert reading | Quantitative safety study on dark fluid like regions and thin layers. State plainly that no expert reading was performed | yes E6 | TODO |
| R3-12 | Link theory to the results section | Add a subsection that tests each theoretical claim numerically | yes E5 | TODO |
| R3-13 | Public code and data link without password | Prepare a public release bundle and put the link in the paper | no | TODO |
| R3-14 | Follows from 13 | Same as R3-13 | no | TODO |
| R3-15 | Do not call them experiments | Global rename to tests and results | no | TODO |
| R3-16 | Fair comparison at similar complexity | Train a plain corrector with the same parameter budget and the same loss, plus classical post processing baselines | yes E7 | TODO |
| R3-17 | How were the fuzzy parameters computed | Document every constant, its source, and its sensitivity | yes E3 | TODO |
| R3-18 | Transferability needs cost and variability numbers | Report per backbone head size, adaptation data, epochs, wall clock, and fold variability | yes E8 | TODO |
| R3-19 | Verify the assumed constants and bounds hold | Empirical check of the Lipschitz constant, the margins and the background leakage | yes E5 | TODO |

## Internal problems found that must be fixed regardless of the reviewers

| ID | Problem | Evidence | Fix |
|----|---------|----------|-----|
| X1 | Paper says the corrector is trained only on PKU37 and transferred. The code fine tunes it on the n minus 1 Duke images of every fold with EWC | run_duke17_loo.py lines 74 to 212 | Report the real protocol and add a zero shot table |
| X2 | Paper Section IV describes a negotiator that is not the one used | SymbolicNegotiator in the cooperative module implements five different rules | Rewrite the section around the implemented rules and reprove stability for them |
| X3 | The SwinIR PKU37 row has no result file on disk | Only four PKU37 result files match the table, SwinIR is not one of them | Recompute the SwinIR row |
