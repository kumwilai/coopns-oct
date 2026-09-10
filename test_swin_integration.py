
import sys
import os
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.getcwd())

# Mock dependencies
mock_hybrid = MagicMock()
mock_hybrid.HybridCNNSymbolicAnalyzer.return_value.load_state_dict.return_value = ([], [])
sys.modules["nsnd.models.hybrid_analyzer"] = mock_hybrid

mock_sym = MagicMock()
mock_sym.NeuroSymbolicNoiseAnalyzer.return_value.load_state_dict.return_value = ([], [])
mock_sym.SymbolicNoiseAnalyzer.return_value.load_state_dict.return_value = ([], [])
sys.modules["nsnd.models.symbolic_analyzer"] = mock_sym

sys.modules["nsnd.models.nafnet"] = MagicMock()
sys.modules["nsnd.models.adaptive_multihead_refinement"] = MagicMock()
sys.modules["nsnd.models.component_denoisers"] = MagicMock()
sys.modules["nsnd.datasets.synthetic_oct"] = MagicMock()
sys.modules["scripts.fix_adaptive_denoising"] = MagicMock()
sys.modules["nsnd.training.synthetic_noise"] = MagicMock()
sys.modules["nsnd.utils.metrics"] = MagicMock()

# Mock Swin Head dependencies (timm is often missing in minimal envs)
sys.modules["sota.models.swinir_fair"] = MagicMock()

# Import the class to test
from importlib.machinery import SourceFileLoader
target_file = "nsnd_oct/scripts/train_hybrid_nsnd_multitask.py"
module_name = "train_hybrid_nsnd_multitask"

loader = SourceFileLoader(module_name, target_file)
mod = loader.load_module()
MultiTaskHybridNSND = mod.MultiTaskHybridNSND

# Mock SwinResidualHead since we are testing integration, not the head itself
class MockSwinHead(nn.Module):
    def __init__(self, img_size=64, in_chans=1, embed_dim=32):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
    def forward(self, x):
        return x # Identity for smoke test

mod.SwinResidualHead = MockSwinHead
mod.NAFNetResidualHead = MagicMock(return_value=MockSwinHead())
mod.NAFNetSmall = MagicMock(return_value=MockSwinHead())
mod.SpatialWeightRefiner = MagicMock(return_value=MockSwinHead())

def test_swin_integration():
    print("Testing Swin Integration...")
    
    with patch("torch.load") as mock_load:
        mock_load.return_value = {"state_dict": {}}
        
        # Instantiate with Swin enabled
        model = MultiTaskHybridNSND(
            hybrid_analyzer_ckpt="dummy.pth",
            use_log_domain_speckle=False,
            use_swin_speckle=True,
            residual_head_width=32,
            use_base_nafnet=True,
            device="cpu"
        )
        
        # Check if Swin head is used for speckle
        if "speckle" in model.residual_heads:
            head = model.residual_heads["speckle"]
            if isinstance(head, MockSwinHead):
                print("✅ SUCCESS: 'speckle' head is SwinResidualHead")
            else:
                print(f"❌ FAILURE: 'speckle' head is {type(head)}")
        else:
            print("❌ FAILURE: 'speckle' head missing")

        # Check Forward Pass
        x = torch.randn(2, 1, 64, 64)
        model.cnn_analyzer = MagicMock(return_value=({
            "speckle": torch.rand(2), "banding": torch.rand(2),
            "gaussian": torch.rand(2), "shot": torch.rand(2)
        }, {}))
        model.symbolic_analyzer = None
        model.spatial_refiner = None
        
        try:
            output, weights, extras = model(x)
            print("✅ SUCCESS: Forward pass complete")
        except Exception as e:
            print(f"❌ FAILURE: Forward pass crashed: {e}")

if __name__ == "__main__":
    test_swin_integration()
