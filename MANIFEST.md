# Which script produces which result

| Paper item | Script | Output |
|---|---|---|
| Table 3, in distribution results | scripts/eval_in_distribution.sh | outputs/eval_BACKBONE.json |
| Table 4, transfer with no adaptation | scripts/eval_transfer.sh | outputs/zeroshot_BACKBONE.json |
| Table 5, constants | none, the values are in the source | code/neuro_symbolic_corrector_v8_cooperative.py |
| Table 6, component ablation and matched complexity | scripts/eval_ablation.sh | outputs/comp_*.json, outputs/matched_plain_eval.json, outputs/classical_*.json |
| Figure 1, architecture | scripts/make_figures.sh | figures/fig_architecture.pdf |
| Figure 2, property failure maps | scripts/make_figures.sh | figures/fig_predicate_maps.pdf |
| Figure 3, studies | scripts/make_figures.sh | figures/fig_studies.pdf |
| Figure 4, visual comparison | scripts/make_figures.sh | figures/fig_subjective_pku37.png |
| Section on the numerical check of the theory | scripts/run_diagnostics.sh | outputs/diagnostics_nafnet.json |
| The Lipschitz constant quoted in the text | code, method lipschitz_constant | printed by run_diagnostics.sh |
