
import sys
import os
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.getcwd())

# Mock dependencies before importing the model
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

# Import the class to test
# We need to use source_file_loader because the file is a script, not a module in a package structure that is easily importable if dependencies are missing
from importlib.machinery import SourceFileLoader
target_file = "nsnd_oct/scripts/train_hybrid_nsnd_multitask.py"
module_name = "train_hybrid_nsnd_multitask"

# We need to mock the imports INSIDE the file as well.
# Since we already mocked sys.modules, normal imports should work.
# But we need to make sure we can import the class `MultiTaskHybridNSND`
# We will use patch to mock torch.load since __init__ loads checkpoints.

# To actually import the file, we might need to handle the fact that it runs code at top level?
# No, the top level code is imports and class definitions. `if __name__ == "__main__":` protects execution.

loader = SourceFileLoader(module_name, target_file)
mod = loader.load_module()
MultiTaskHybridNSND = mod.MultiTaskHybridNSND
NAFNetResidualHead = mod.NAFNetResidualHead  # The script imports it, so it should be available on the module
LogDomainSpeckleResidualHead = mod.LogDomainSpeckleResidualHead

# Mock NAFNetResidualHead and LogDomainSpeckleResidualHead to be simple nn.Modules we can inspect
class MockHead(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.width = width
        self.conv = nn.Conv2d(1, 1, 1) # dummy param
    def forward(self, x):
        return x

class MockLogHead(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.width = width
        self.conv = nn.Conv2d(1, 1, 1) # dummy param
    def forward(self, r, x):
        return r

# Patch the classes in the module
mod.NAFNetResidualHead = MockHead
mod.LogDomainSpeckleResidualHead = MockLogHead
mod.NAFNetSmall = MagicMock(return_value=MockHead()) # Base denoiser mock
mod.NAFNetSmallFiLM = MagicMock(return_value=MockHead())
mod.NAFNet = MagicMock(return_value=MockHead())
mod.SignalDependentExpert = MagicMock(return_value=MockHead())
mod.SpatialWeightRefiner = MagicMock(return_value=MockHead())

def test_independent_heads_log_domain():
    print("Testing Independent Heads (Log Domain Speckle)...")
    
    with patch("torch.load") as mock_load:
        # Mock checkpoint return
        mock_load.return_value = {"state_dict": {}}
        
        # Instantiate model with use_log_domain_speckle=True
        model = MultiTaskHybridNSND(
            hybrid_analyzer_ckpt="dummy.pth",
            use_log_domain_speckle=True,
            shared_residual=False, # Independent heads
            residual_head_width=16,
            use_base_nafnet=True,
            device="cpu"
        )
        
        # Check 1: "speckle" should NOT be in residual_heads
        if "speckle" in model.residual_heads:
            print("❌ FAILURE: 'speckle' found in residual_heads when use_log_domain_speckle=True")
        else:
            print("✅ SUCCESS: 'speckle' correctly omitted from residual_heads")
            
        # Check 2: log_speckle_head should exist
        if model.log_speckle_head is None:
            print("❌ FAILURE: log_speckle_head is None")
        else:
            print("✅ SUCCESS: log_speckle_head initialized")
            
        # Check 3: Forward pass
        x = torch.randn(2, 1, 64, 64)
        
        # Mock analyzer output
        model.cnn_analyzer = MagicMock(return_value=({
            "speckle": torch.rand(2), "banding": torch.rand(2),
            "gaussian": torch.rand(2), "shot": torch.rand(2)
        }, {}))
        model.symbolic_analyzer = None # Disable symbolic for simplicity
        model.spatial_refiner = None
        
        try:
            output, weights, extras = model(x)
            expert_outputs = extras["expert_outputs"]
            if "speckle" in expert_outputs:
                print("✅ SUCCESS: Forward pass produced 'speckle' output")
            else:
                print("❌ FAILURE: Forward pass missing 'speckle' output")
        except Exception as e:
            print(f"❌ FAILURE: Forward pass crashed: {e}")
            import traceback
            traceback.print_exc()

def test_independent_heads_standard():
    print("\nTesting Independent Heads (Standard)...")
    
    with patch("torch.load") as mock_load:
        mock_load.return_value = {"state_dict": {}}
        
        # Instantiate model with use_log_domain_speckle=False
        model = MultiTaskHybridNSND(
            hybrid_analyzer_ckpt="dummy.pth",
            use_log_domain_speckle=False,
            shared_residual=False,
            residual_head_width=16,
            use_base_nafnet=True,
            device="cpu"
        )
        
        # Check 1: "speckle" SHOULD be in residual_heads
        if "speckle" in model.residual_heads:
            print("✅ SUCCESS: 'speckle' found in residual_heads")
        else:
            print("❌ FAILURE: 'speckle' missing from residual_heads when use_log_domain_speckle=False")
            
        # Check 2: log_speckle_head should be None
        if model.log_speckle_head is None:
            print("✅ SUCCESS: log_speckle_head is None")
        else:
            print("❌ FAILURE: log_speckle_head initialized when disabled")

        # Check 3: Forward pass
        x = torch.randn(2, 1, 64, 64)
        model.cnn_analyzer = MagicMock(return_value=({
            "speckle": torch.rand(2), "banding": torch.rand(2),
            "gaussian": torch.rand(2), "shot": torch.rand(2)
        }, {}))
        model.symbolic_analyzer = None
        model.spatial_refiner = None
        
        try:
            output, weights, extras = model(x)
            expert_outputs = extras["expert_outputs"]
            if "speckle" in expert_outputs:
                print("✅ SUCCESS: Forward pass produced 'speckle' output")
            else:
                print("❌ FAILURE: Forward pass missing 'speckle' output")
        except Exception as e:
            print(f"❌ FAILURE: Forward pass crashed: {e}")

if __name__ == "__main__":
    test_independent_heads_log_domain()
    test_independent_heads_standard()
