"""Utility functions for NSND"""

from typing import Optional

try:
    from .visualization import (
        visualize_noise_decomposition,
        visualize_uncertainty,
        generate_noise_report,
        plot_component_denoisers,
    )
    _viz_available = True
except ImportError:
    _viz_available = False

from .metrics import (
    compute_psnr,
    compute_ssim,
    compute_enl,
    evaluate_denoising,
)

__all__ = [
    "compute_psnr",
    "compute_ssim",
    "compute_enl",
    "evaluate_denoising",
]

if _viz_available:
    __all__.extend([
        "visualize_noise_decomposition",
        "visualize_uncertainty",
        "generate_noise_report",
        "plot_component_denoisers",
    ])
