# Selected correctors of the revised submission

One file per backbone, the seed 0 model of the setting the selector chose on the
validation split. These are the checkpoints every number in the revised paper is
computed from, including Table 3, both subjective figures, the component study,
the sensitivity study, the safety study and the theory check.

| Backbone | Setting | Background gate | Fidelity dead zone |
|---|---|---|---|
| NAFNet | halo41_dz06 | Otsu tissue mask dilated by 41 px | 0.6 dB |
| DnCNN | int_dz09 | 15th intensity percentile | 0.9 dB |
| SwinIR | int_dz09 | 15th intensity percentile | 0.9 dB |
| KBNet | int_dz09 | 15th intensity percentile | 0.9 dB |

The gate must be set to match. Scoring a checkpoint under a gate it was not
selected under changes its output. `revision/fig_subjective.py` pins both the
file and the gate per backbone and refuses to run if either does not match.

The frozen backbones are in the parent directory and are not modified by any
run. Two seeds beyond seed 0 exist for every setting and are used for the seed
spreads reported in the paper. They are not shipped here because they are large
and reproducible from the recorded commands.
