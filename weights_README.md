# Weights

The trained weights are attached to the GitHub release rather than committed, because they total
about 286 megabytes and git is a poor store for binary files that never change.

Download the archive attached to the latest release and unpack it here, so that this directory
contains

    nafnet_backbone.pth              the frozen denoiser
    nafnet_pku37_cooperative.pth     the trained corrector
    dncnn_backbone.pth
    dncnn_pku37_cooperative.pth
    kbnet_backbone.pth
    kbnet_pku37_cooperative.pth
    swinir_backbone.pth
    swinir_pku37_cooperative.pth
