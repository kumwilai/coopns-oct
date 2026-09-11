# Weights

The four correctors reported in the revised paper are committed here, under selected/, because they
are what the tables describe and a reader should not have to fetch anything to inspect them. The
frozen backbones and the warm start correctors are not committed. They total about 330 megabytes,
they never change, and git is a poor store for that. Download the archive attached to the release
and unpack it at the root of this repository so that the paths below exist.

## Frozen backbones and warm start correctors, in this directory

    checkpointpaper/nafnet_backbone.pth           frozen denoiser, about 7 million parameters
    checkpointpaper/dncnn_backbone.pth
    checkpointpaper/kbnet_backbone.pth
    checkpointpaper/swinir_backbone.pth
    checkpointpaper/nafnet_pku37_cooperative.pth  corrector of the SUBMITTED version, warm start only
    checkpointpaper/dncnn_pku37_cooperative.pth
    checkpointpaper/kbnet_pku37_cooperative.pth
    checkpointpaper/swinir_pku37_cooperative.pth

The four correctors here were trained with the saturating allocation rule described in
ALLOCATION_FIX.md. They are not the models reported in the revised paper. They are the starting
point from which every reported model was trained, see scripts/train_all.sh, and they are the models
whose numbers appear in the submitted version. Scoring them with LEGACY_SATURATING_ALLOCATION=1
reproduces the submitted tables.

## Selected correctors of the revised paper, in outputs/revision

The reported models are the checkpoints chosen by scripts/select_config.sh. Their names follow

    outputs/revision/sw_BACKBONE_SETTING_sSEED/best_model_cooperative.pth

for the setting recorded in outputs/revision/winners.json and seeds 0, 1 and 2. The release
archive contains exactly those twelve checkpoints, so every table can be regenerated without any
training. The other settings of the search can be reproduced with scripts/train_all.sh.

## The correctors reported in the revised paper, committed under selected/

    checkpointpaper/selected/nafnet_halo41_dz06_s0_cooperative.pth
    checkpointpaper/selected/dncnn_int_dz09_s0_cooperative.pth
    checkpointpaper/selected/swinir_int_dz09_s0_cooperative.pth
    checkpointpaper/selected/kbnet_int_dz09_s0_cooperative.pth

One per backbone, the seed 0 model of the setting the selector chose on the validation split. Every
number in the revised paper comes from these four, including the merged results table, the
comparison figure, the component study, the sensitivity study, the safety study and the theory
check. The background gate must be set to match, since scoring a checkpoint under a gate it was not
selected under changes its output. selected/README.md gives the gate and the dead zone for each, and
revision/fig_subjective.py pins both and refuses to run if either disagrees.

Seeds 1 and 2 exist for every setting and produce the spreads the paper reports. They are not
committed, because they are reproducible from the recorded commands and would triple the size here.
