# CoopNS-OCT, anonymous release for peer review

This archive contains everything needed to reproduce every number and every figure in the
manuscript. It requires no password and no account.

The archive is anonymous. Author names, institutional paths and version control history have been
removed for double blind review. A permanent citable archive will replace it on acceptance.

## What is here

    code/        the model, the training script and the evaluation scripts
    scripts/     one driver per table and per figure in the paper
    weights/     the frozen backbones and the trained correctors
    MANIFEST.md  which script produces which table or figure

## Requirements

Python 3.12 with torch 2.6, numpy, scipy, scikit image, opencv and matplotlib. A GPU is not
required. Every result in the paper was produced on two CPU cores, which takes longer but needs no
special hardware.

    pip install torch numpy scipy scikit-image opencv-python-headless matplotlib

## Reproducing the paper

    bash scripts/reproduce_all.sh

That runs, in order, the training of every corrector, the in distribution evaluation, the transfer
evaluation with no adaptation, the property leave one out study, the rule sensitivity study, the
component ablation, the matched complexity comparison and the diagnostics pass. It writes one JSON
file per result and then rebuilds every figure.

To reproduce a single table instead, see MANIFEST.md.

## The training recipe

Every corrector in the paper comes from one command, recorded in scripts/train_all.sh. No result in
the paper uses a hand tuned variation of it.

## A note on a defect found during this revision

An earlier version of this code combined the fuzzy rule activations in a way that always exceeded
the upper limit of the clamp that follows them. The allocation map was therefore the constant one at
every pixel, the rule layer had no effect on the output, and its parameters received no gradient.
The defect is described in full in ALLOCATION_FIX.md, together with the corrected formula and the
proof of its bound.

The original behaviour can be restored exactly by setting the environment variable
LEGACY_SATURATING_ALLOCATION to 1, so the numbers of the earlier submission remain reproducible from
this same code.
