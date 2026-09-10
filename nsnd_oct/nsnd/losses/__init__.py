"""NSND Loss Functions for TMI Publication"""

from .clinical_weighted import (
    ClinicalWeightedL1Loss,
    ClinicalWeightedMSELoss,
    LayerSSIMLoss,
    BoundarySharpnessLoss,
    PerceptualTextureLoss,
    TMIJointLoss,
    compute_psnr,
    compute_ssim,
    CLINICAL_WEIGHTS_4CLASS,
    CLINICAL_WEIGHTS_3CLASS,
    LAYER_NAMES_4CLASS,
    LAYER_NAMES_3CLASS,
)

from .neuro_symbolic import (
    compute_layer_ordering_loss,
    compute_thickness_loss,
    compute_intensity_order_loss,
    compute_boundary_continuity_loss,
    compute_head_diversity_loss,
    AnatomyTemplateLoss,
    NeuroSymbolicLoss,
    CLASS_NAMES_3,
    THICKNESS_PRIOR_3CLASS,
    INTENSITY_ORDER_3CLASS,
)

__all__ = [
    # Clinical weighted losses
    "ClinicalWeightedL1Loss",
    "ClinicalWeightedMSELoss",
    "LayerSSIMLoss",
    "BoundarySharpnessLoss",
    "PerceptualTextureLoss",
    "TMIJointLoss",
    "compute_psnr",
    "compute_ssim",
    "CLINICAL_WEIGHTS_4CLASS",
    "CLINICAL_WEIGHTS_3CLASS",
    "LAYER_NAMES_4CLASS",
    "LAYER_NAMES_3CLASS",
    # Neuro-symbolic losses (KEY TMI CONTRIBUTION)
    "compute_layer_ordering_loss",
    "compute_thickness_loss",
    "compute_intensity_order_loss",
    "compute_boundary_continuity_loss",
    "compute_head_diversity_loss",
    "AnatomyTemplateLoss",
    "NeuroSymbolicLoss",
    "CLASS_NAMES_3",
    "THICKNESS_PRIOR_3CLASS",
    "INTENSITY_ORDER_3CLASS",
]
