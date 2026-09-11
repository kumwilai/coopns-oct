# Which script produces which item in the paper

Section, table and figure numbers refer to the revised manuscript. Every generated number passes
through a JSON file in outputs/revision and then through revision/make_tables.py, which writes the
LaTeX table bodies to revision/sections/generated. Nothing in a table is typed by hand, and a
missing result prints as a dash rather than a number.

The manuscript has three tables and two figures. The supplementary file has seven tables, three
figures and one algorithm, numbered S1 onward.

| Paper item | Script | Result file | Generated file |
|---|---|---|---|
| Section VIII D, how the operating point was chosen | scripts/train_all.sh then scripts/select_config.sh | sw_BACKBONE_SETTING_sSEED_val.json, winners.json | none, the level that admitted each backbone is in winners.json |
| Table 1, notation | none, definitions | | |
| Table 2, the six clinical properties | none, definitions, closed forms in Table S1 | | |
| Table 3, PKU37 and both Duke sets, all four backbones, three seeds | scripts/eval_in_distribution.sh and scripts/eval_transfer.sh | test_BACKBONE_sSEED.json, zeroshot_BACKBONE_sSEED.json | tab_in_distribution.tex |
| Table 3, component blocks | scripts/eval_studies.sh | comp_no_negotiator.json, comp_no_edge.json, comp_no_uncertainty.json, comp_no_bg_smooth.json, lopo_none.json for the full method row | tab_in_distribution.tex |
| Table 3, matched complexity blocks | scripts/eval_studies.sh | matched_plain_eval.json, classical_unsharp.json, classical_clahe.json, classical_tuning.json | tab_in_distribution.tex |
| Table 3, adapted protocol block | scripts/eval_transfer_adapted.sh | loo_duke17_nafnet/loo_results.json, loo_duke2013_nafnet/loo_results.json | tab_in_distribution.tex, guarded so the block is refused if the recorded resume checkpoint is not the selected NAFNet model |
| Section IX D, one clinical property removed at a time | scripts/eval_studies.sh | lopo_none.json, lopo_drop_P1.json to lopo_drop_P6.json | tab_lopo.tex and Figure S3 left |
| Section IX E, sensitivity to the rule constants and the conjunction | scripts/eval_studies.sh | fuzz_ref.json, fuzz_base_*.json, fuzz_usecorr_*.json, fuzz_boost_*.json, fuzz_tnorm_*.json | Figure S3 middle |
| Section IX H, does the cooperation map predict error | scripts/run_diagnostics.sh | diagnostics_nafnet.json, key calibration | theory_numbers.tex, macros CalibRho and CalibAuse |
| Section IX I, numerical check of both theorems and of which constraint binds | scripts/run_diagnostics.sh | diagnostics_nafnet.json | theory_numbers.tex, macros LipTheory, LipMeasured, CondMargin, CondSigma, EtaT, EtaB, EtaTPos, EtaBNeg, SigmaRatio, Alloc*, ConEnergy, ConPareto, ConLip, ConAllThree, BlendOne |
| Section IX J, safety study | scripts/run_diagnostics.sh | diagnostics_nafnet.json | theory_numbers.tex, macros EdgeInv*, EdgeInvUp, WeakRet*, WeakRetDown, Dark*, IlmShift*, RpeShift*, GlobalChange, GlobalMax, and Figure S3 right |
| Section IX K, cost of moving to a new backbone | no script, the wall clock comes from the timestamps in the sweep logs and the seed spread from the same test files as Table 3 | test_BACKBONE_sSEED.json for the spread | none, the numbers are in the prose |
| Section X B, what the selection criterion does not see | scripts/run_diagnostics.sh on the validation split, both gates | val_diag_selected.json, val_diag_runnerup.json | none, the numbers are in the prose |
| Figure 1, the six stages | scripts/make_figures.sh, revision/fig_architecture.py | none | revision/figures/fig_architecture.pdf |
| Figure 2, visual comparison across four backbones | scripts/make_figures.sh, revision/fig_subjective.py | test_BACKBONE_s*.json and the four checkpoints under checkpointpaper/selected | revision/figures/fig_subjective_pku37.pdf |
| Table S1, closed form of the six properties | none, definitions | | |
| Table S2 and Table S3, every constant and its origin | none, the values are in the source | neuro_symbolic_corrector_v8_cooperative.py, class SymbolicNegotiator, and train_v8_cooperative.py, the argument parser | |
| Table S4, the gate rule and dead zone selected for each backbone | scripts/select_config.sh | winners.json, winners_intensity.json | none, four rows typed from winners.json |
| Table S6, every clinical cell with its seed spread | scripts/eval_in_distribution.sh and scripts/eval_transfer.sh | test_BACKBONE_sSEED.json, zeroshot_BACKBONE_sSEED.json | tab_full_spreads.tex |
| Table S7, the adapted protocol against the frozen one | scripts/eval_transfer_adapted.sh | loo_duke17_nafnet/loo_results.json, loo_duke2013_nafnet/loo_results.json | none, the table is typed from the two files and the frozen column comes from Table 3 |
| Figure S1, the six property maps | scripts/make_figures.sh, revision/fig_predicates.py | reads checkpointpaper/nafnet_pku37_cooperative.pth, the submitted checkpoint, not the selected one | revision/figures/fig_predicate_maps.pdf |
| Figure S2, full resolution comparison for all four backbones | scripts/make_figures.sh, revision/fig_subjective.py --supp | the four checkpoints under checkpointpaper/selected | revision/figures/fig_subjective_all.pdf |
| Figure S3, the three studies | scripts/make_figures.sh, revision/fig_studies.py | lopo_*, fuzz_*, diagnostics_nafnet.json | revision/figures/fig_studies.pdf |
| Algorithm 1 of the supplementary file | none, it describes train_v8_cooperative.py | | |
| The Lipschitz constant quoted in the text | neuro_symbolic_corrector_v8_cooperative.py, method SymbolicNegotiator.lipschitz_constant | reported inside diagnostics_nafnet.json | macro LipTheory |
| The submitted version of the tables | any scoring script with LEGACY_SATURATING_ALLOCATION=1 and the warm start checkpoints | | |

The writing rules of the manuscript, no em dash, no colon, no semicolon, and no use of the word
experiment, are checked by revision/style_check.py. Run it with no argument to check every section
file of both documents.

Two things a reader should know about provenance. Every result file records the checkpoint it
scored, and revision/fig_subjective.py asserts the md5 of each checkpoint it draws and cross checks
it against the checkpoint the matching result file names, so the pixels and the printed numbers
cannot come from two different models. revision/make_tables.py refuses to emit the adapted block if
its recorded resume checkpoint is not the selected NAFNet model.
