# OJCS-2026-04-0386 revision tracker

Deadline 15 Sep 2026. Last audited 10 Sep 2026 evening.
Main paper 12 pages. Supplement 9 pages. Both compile with no undefined reference.
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

## Not a reviewer concern, still open

- Received date 14 April 2026 is inferred from the manuscript number and needs the journal acknowledgement email to confirm.
- Whether the selection criterion is blind to the safety measures is being measured on the validation split for the selected gate and the runner up gate. One Limitations sentence will report the outcome either way.
- The seeded zeroshot result files carry no checkpoint field. Provenance was established from the sweep log ordering and from the fact that they differ from the pre sweep run, but it is not recorded in the files.
