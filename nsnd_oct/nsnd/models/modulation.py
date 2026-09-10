
import torch
import torch.nn as nn

class NoiseModulationBlock(nn.Module):
    """
    Lightweight Feature Modulation Block.
    Maps a 4-dim noise vector to channel-wise scaling factors.
    
    Architecture:
    Linear(4 -> hidden) -> ReLU -> Linear(hidden -> channels) -> Sigmoid
    
    Init:
    Initializes to output 1.0 (Identity) to prevent training instability.
    """
    def __init__(self, input_dim=4, hidden_dim=16, output_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_channels),
            nn.Sigmoid()
        )
        
        # CRITICAL: Identity Initialization
        # The conditioner MUST start outputting ~1.0 (identity scaling)
        # to prevent "init shock" that destroys pre-trained head performance
        #
        # Strategy:
        # - First layer: Normal initialization (can learn from noise vector)
        # - Second layer: Near-zero weights + large positive bias
        #   → sigmoid(near_zero × relu(x) + 4.0) ≈ sigmoid(4.0) ≈ 0.98

        # First layer: Standard Kaiming init (allows learning)
        nn.init.kaiming_uniform_(self.net[0].weight, nonlinearity='relu')
        nn.init.zeros_(self.net[0].bias)

        # Second layer: CRITICAL identity init
        nn.init.normal_(self.net[2].weight, mean=0.0, std=1e-4)  # Near zero
        nn.init.constant_(self.net[2].bias, 4.0)  # sigmoid(4.0) ≈ 0.982

    def forward(self, noise_vector):
        """
        Args:
            noise_vector: (B, 4) - noise probability distribution
        Returns:
            scale: (B, C, 1, 1) - channel-wise scaling factors in range [0, 1]
        """
        # Safety check: ensure input is valid
        if torch.isnan(noise_vector).any() or torch.isinf(noise_vector).any():
            # Fallback to identity scaling
            B = noise_vector.size(0)
            C = self.net[2].out_features
            return torch.ones(B, C, 1, 1, device=noise_vector.device)

        scale = self.net(noise_vector)  # [B, C] in range [0, 1] (sigmoid output)

        # Additional safety: clamp to valid range (should already be [0,1] from sigmoid)
        scale = scale.clamp(0.0, 1.0)

        return scale.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
