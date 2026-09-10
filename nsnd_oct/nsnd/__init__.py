"""
NSND-OCT: Neuro-Symbolic Noise Decomposition for Adaptive OCT Denoising

A vendor-agnostic OCT denoising framework that combines:
- Neural perception of noise characteristics
- Symbolic reasoning for interpretable noise decomposition
- Physics-based component denoisers
- Learned fusion for optimal results
"""

__version__ = "0.1.0"
__author__ = "OCT Research Team"

from .models.nsnd_model import NSNDModel
from .models.symbolic_analyzer import SymbolicNoiseAnalyzer, NeuroSymbolicNoiseAnalyzer
from .models.fusion_network import NeuralFusionNetwork
from .models.adaptive_ensemble import AdaptiveEnsembleNSND, train_ensemble_weights

__all__ = [
    "NSNDModel",
    "SymbolicNoiseAnalyzer",
    "NeuroSymbolicNoiseAnalyzer",
    "NeuralFusionNetwork",
    "AdaptiveEnsembleNSND",
    "train_ensemble_weights",
]
