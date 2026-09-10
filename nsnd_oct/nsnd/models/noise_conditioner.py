"""
Noise-Adaptive Feature Modulation (Active Conditioning)

Lightweight module that modulates internal features based on noise composition.
Allows heads to adapt their processing strength dynamically based on noise severity.

Key Innovation: Heads can now process "High Speckle" differently from "Low Speckle"
rather than just scaling their final output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseConditioner(nn.Module):
    """
    Lightweight conditioning module that converts noise probability vector
    into channel-wise feature scaling factors.

    Architecture:
        noise_vector (4,) -> Linear -> ReLU -> Linear -> Sigmoid -> gamma (C,)

    Key Properties:
    - Identity initialization: outputs 1.0 at initialization (no "init shock")
    - Lightweight: only 2 linear layers
    - Channel-wise modulation: scales each feature channel independently
    - Range (0, 1]: Sigmoid ensures valid scaling factors

    Args:
        noise_dim: Size of input noise vector (default: 4 for speckle/banding/gaussian/shot)
        feature_channels: Number of feature channels to modulate
        hidden_dim: Hidden layer size (default: 16 for lightweight design)

    Example:
        >>> conditioner = NoiseConditioner(noise_dim=4, feature_channels=64)
        >>> noise_vec = torch.tensor([0.8, 0.1, 0.05, 0.05])  # High speckle
        >>> gamma = conditioner(noise_vec)  # Shape: (64,)
        >>> modulated_features = features * gamma.view(1, -1, 1, 1)
    """

    def __init__(
        self,
        noise_dim: int = 4,
        feature_channels: int = 64,
        hidden_dim: int = 16,
    ):
        super().__init__()

        self.noise_dim = noise_dim
        self.feature_channels = feature_channels
        self.hidden_dim = hidden_dim

        # Two-layer MLP
        self.fc1 = nn.Linear(noise_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, feature_channels)
        self.sigmoid = nn.Sigmoid()

        # CRITICAL: Identity initialization
        # This ensures the conditioner outputs ~1.0 at initialization
        # Prevents "init shock" that would destroy pre-trained stability
        self._init_identity()

    def _init_identity(self):
        """
        Initialize to output 1.0 (identity scaling).

        Strategy:
        - fc2.weight ≈ 0 (small random noise for symmetry breaking)
        - fc2.bias = 0 (so sigmoid(0) = 0.5, but we'll adjust)

        Actually, we want sigmoid(x) ≈ 1.0, so x should be large positive.
        But for stability, we use:
        - fc2.weight ≈ 0 (very small)
        - fc2.bias ≈ +4.0 (sigmoid(4) ≈ 0.98, close to 1.0)

        This gives initial output ~0.98, which is close enough to identity.
        """
        # First layer: standard initialization is fine (it's gated by second layer)
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)

        # Second layer: CRITICAL identity initialization
        # Very small weights (near zero)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1e-4)

        # Bias set to ~4.0 so sigmoid(4) ≈ 0.98 (close to 1.0)
        # This ensures initial modulation is approximately identity
        nn.init.constant_(self.fc2.bias, 4.0)

    def forward(self, noise_vector: torch.Tensor) -> torch.Tensor:
        """
        Convert noise probability vector to channel-wise scaling factors.

        Args:
            noise_vector: Noise composition [B, 4] or [4] (batch or single sample)

        Returns:
            gamma: Channel-wise scaling factors [B, C] or [C]
                   Range: (0, 1] via Sigmoid

        Example:
            High speckle (0.9, 0.05, 0.03, 0.02) -> gamma might emphasize texture channels
            Low speckle (0.1, 0.1, 0.7, 0.1) -> gamma might emphasize smoothing channels
        """
        # Handle both batched [B, 4] and unbatched [4] inputs
        if noise_vector.dim() == 1:
            noise_vector = noise_vector.unsqueeze(0)  # [4] -> [1, 4]
            squeeze_output = True
        else:
            squeeze_output = False

        # Two-layer MLP
        h = self.relu(self.fc1(noise_vector))  # [B, hidden]
        logits = self.fc2(h)  # [B, C]
        gamma = self.sigmoid(logits)  # [B, C] in range (0, 1]

        if squeeze_output:
            gamma = gamma.squeeze(0)  # [1, C] -> [C]

        return gamma

    def get_modulation_stats(self, noise_vector: torch.Tensor) -> dict:
        """
        Get statistics about the modulation for debugging/visualization.

        Returns:
            stats: Dict with mean, std, min, max of gamma values
        """
        with torch.no_grad():
            gamma = self.forward(noise_vector)
            return {
                'mean': gamma.mean().item(),
                'std': gamma.std().item(),
                'min': gamma.min().item(),
                'max': gamma.max().item(),
            }


class GatedNoiseConditioner(NoiseConditioner):
    """
    Robust Confidence-Gated Conditioner.
    
    Safety Mechanism:
    - Scales the modulation strength by the Analyzer's confidence score.
    - If confidence is low, gamma -> 0.5 (sigmoid output) -> ~Identity modulation?
    - Wait, NoiseConditioner outputs sigmoid (0-1).
    - If we want Identity, we want the modulation to be "neutral".
    - In FiLM (gamma * x + beta), neutral gamma is 1.0, neutral beta is 0.0.
    
    Refining the logic:
    NoiseConditioner outputs `gamma` directly via Sigmoid.
    Wait, `ConditionalNAFBlock` uses `_apply_film`:
        scale, shift = film_params.chunk(2, dim=1)
        scale = torch.tanh(scale)
        return x * (1.0 + scale) + shift
    
    So `film_params` (output of the projection layer inside the block) determines modulation.
    The conditioner (this class) produces the `noise_vector` (or embedding) that goes INTO the block's projection layer.
    
    So if we want Identity, we want the INPUT to the block's projection to be Zero?
    If input is 0, projection (Linear) output is 0 (bias is 0).
    Tanh(0) = 0. Scale = 0.
    x * (1 + 0) + 0 = x. Identity!
    
    So `GatedNoiseConditioner` should gate the **embedding** it produces.
    
    Args:
        confidence: (B, 1) scalar score [0, 1]
    """
    
    def forward(self, noise_vector: torch.Tensor, confidence: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            noise_vector: (B, 4)
            confidence: (B, 1) or None. If None, assumes 1.0 (full confidence).
        
        Returns:
            gated_embedding: (B, feature_channels) - modulated embedding
        """
        # Get raw embedding (gamma in base class, but here we treat it as a latent code)
        # We want a latent code that the ConditionalNAFBlock projects to scale/shift.
        # The base class outputs sigmoid(logits). 
        # But ConditionalNAFBlock expects a raw vector `cond`.
        # So we should probably just use the raw noise vector or an embedding of it.
        
        # Actually, ConditionalNAFBlock takes `cond_dim`.
        # If we use `GatedNoiseConditioner` as a wrapper around the noise vector itself?
        
        # Let's keep it simple: This class processes the noise vector into a "Conditioning Embedding".
        # This embedding is what gets passed to the NAFBlock.
        
        h = self.relu(self.fc1(noise_vector))
        embedding = self.fc2(h) # Raw logits (no sigmoid)
        
        # Gating
        if confidence is not None:
            embedding = embedding * confidence.view(-1, 1)
            
        return embedding

    def _init_identity(self):
        # We want the embedding to be near zero initially so the Block sees 0 -> Identity
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)


class SpatialBasisModulator(nn.Module):
    """Spatial DBM: predict noise maps and provide per-stage basis vectors."""

    def __init__(
        self,
        feature_channels: int,
        stage_channels: dict,
        num_noise_types: int = 4,
        hidden_channels: int = 64,
        alpha: float = 0.1,
        gate_floor: float = 0.0,
        basis_init_std: float = 5e-3,
    ):
        super().__init__()
        self.num_noise_types = int(num_noise_types)
        self.alpha = float(alpha)
        self.gate_floor = float(gate_floor)
        self.basis_init_std = float(basis_init_std)

        self.noisy_proj = nn.Sequential(
            nn.Conv2d(1, feature_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.spatial_head = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, self.num_noise_types, 3, padding=1),
        )

        self.basis_gamma = nn.ParameterDict()
        self.basis_beta = nn.ParameterDict()
        for name, channels in stage_channels.items():
            gamma = nn.Parameter(torch.zeros(self.num_noise_types, channels))
            beta = nn.Parameter(torch.zeros(self.num_noise_types, channels))
            nn.init.normal_(gamma, mean=0.0, std=self.basis_init_std)
            nn.init.normal_(beta, mean=0.0, std=self.basis_init_std)
            self.basis_gamma[name] = gamma
            self.basis_beta[name] = beta

    def forward(
        self,
        feature_map: torch.Tensor,
        global_weights: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
        noisy: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
        if feature_map is None and noisy is None:
            raise ValueError("SpatialBasisModulator requires a feature_map or noisy tensor.")
        if noisy is not None:
            noisy_feat = self.noisy_proj(noisy)
            if feature_map is not None and noisy_feat.shape[-2:] != feature_map.shape[-2:]:
                noisy_feat = F.interpolate(
                    noisy_feat,
                    size=feature_map.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            feature_map = noisy_feat if feature_map is None else feature_map + noisy_feat
        logits = self.spatial_head(feature_map)
        if global_weights is not None:
            logits = logits + global_weights.view(-1, self.num_noise_types, 1, 1)
        spatial_map = torch.softmax(logits, dim=1)
        gate = None
        if confidence is not None:
            gate = confidence.view(-1, 1, 1, 1)
            if self.gate_floor > 0.0:
                gate = gate.clamp(min=self.gate_floor)
        basis = {
            name: {"gamma": self.basis_gamma[name], "beta": self.basis_beta[name]}
            for name in self.basis_gamma.keys()
        }
        return spatial_map, gate, basis

    @staticmethod
    def _orthogonality_loss(basis: torch.Tensor) -> torch.Tensor:
        v = F.normalize(basis, dim=1)
        gram = v @ v.t()
        eye = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
        return F.mse_loss(gram, eye)

    def regularization_loss(
        self,
        ortho_weight: float = 0.0,
        sparsity_weight: float = 0.0,
    ) -> dict:
        device = next(iter(self.basis_gamma.values())).device
        ortho = torch.tensor(0.0, device=device)
        sparse = torch.tensor(0.0, device=device)
        for name in self.basis_gamma.keys():
            gamma = self.basis_gamma[name]
            beta = self.basis_beta[name]
            if ortho_weight > 0:
                ortho = ortho + self._orthogonality_loss(gamma)
            if sparsity_weight > 0:
                sparse = sparse + gamma.abs().mean() + beta.abs().mean()
        total = ortho_weight * ortho + sparsity_weight * sparse
        return {"total": total, "ortho": ortho, "sparse": sparse}
