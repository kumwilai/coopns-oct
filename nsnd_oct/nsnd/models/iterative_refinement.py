"""
Phase 2: Iterative Multi-Stage Refinement

Architecture:
    Stage 1: Multi-Head Refinement → Partial denoising
    ↓ Analyze remaining noise
    Stage 2: Targeted refinement based on dominant residual noise
    ↓ Analyze again
    Stage 3: Final polish

Expected gain: +0.5-1.0 dB over single-stage
Target: 28.2-28.5 dB
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
from .adaptive_multihead_refinement import AdaptiveMultiHeadRefinement


class IterativeMultiStageRefinement(nn.Module):
    """
    Iterative refinement with symbolic guidance at each stage

    Key Innovation:
    - Don't stop after one pass
    - Analyze remaining noise after each stage
    - Use symbolic routing to select specialists for residual noise
    - Each stage targets specific remaining noise types
    """

    def __init__(
        self,
        nsnd_symbolic_analyzer,
        num_stages: int = 3,
        channels: int = 16,
        device: str = 'cpu',
        dropout: float = 0.1
    ):
        """
        Args:
            nsnd_symbolic_analyzer: Trained NSND symbolic analyzer
            num_stages: Number of refinement stages (2-4 recommended)
            channels: Channels in each head
            device: Device to run on
            dropout: Dropout rate (default 0.1 for Phase 2, less than Phase 1's 0.2)
        """
        super().__init__()

        self.device = device
        self.num_stages = num_stages
        self.nsnd_analyzer = nsnd_symbolic_analyzer

        # Create multiple refinement stages
        # Each stage is a full multi-head refinement model with reduced dropout
        self.stages = nn.ModuleList([
            AdaptiveMultiHeadRefinement(
                nsnd_symbolic_analyzer=nsnd_symbolic_analyzer,
                channels=channels,
                device=device,
                dropout=dropout
            )
            for _ in range(num_stages)
        ])

        # Stage-specific blending weights (learnable)
        # Later stages should be more conservative
        self.stage_alphas = nn.Parameter(
            torch.tensor([0.3, 0.2, 0.1][:num_stages], dtype=torch.float32)
        )

    def forward(
        self,
        noisy: torch.Tensor,
        clean: Optional[torch.Tensor] = None,
        return_intermediates: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Iterative refinement with symbolic guidance

        Args:
            noisy: Noisy input [B, 1, H, W]
            clean: Clean ground truth (for training analysis) [B, 1, H, W]
            return_intermediates: Return all stage outputs

        Returns:
            output: Final denoised image [B, 1, H, W]
            intermediates: Dict with stage-by-stage info
        """
        B, C, H, W = noisy.shape

        current = noisy
        stage_outputs = []
        stage_noise_profiles = []
        stage_improvements = []

        for stage_idx in range(self.num_stages):
            # Run refinement stage
            refined, stage_info = self.stages[stage_idx](current, return_intermediates=True)

            # Blend with current (conservative update)
            alpha = torch.sigmoid(self.stage_alphas[stage_idx])  # Ensure [0, 1]
            current = current * (1 - alpha) + refined * alpha

            # Store stage output
            stage_outputs.append(current.clone())
            stage_noise_profiles.append(stage_info['noise_profile'])

            # Analyze improvement if ground truth available
            if clean is not None:
                psnr = 10 * torch.log10(1.0 / (F.mse_loss(current, clean) + 1e-8))
                stage_improvements.append(psnr.item())

            # Early stopping: if noise profile shows minimal remaining noise
            if stage_idx < self.num_stages - 1:
                # Check if remaining noise is < 10% total
                total_noise = sum(stage_info['noise_profile'][k].mean()
                                for k in ['speckle', 'banding', 'gaussian', 'shot'])
                if total_noise < 0.1:
                    print(f"Early stop at stage {stage_idx+1}: minimal residual noise")
                    break

        # Collect intermediates
        intermediates = {}
        if return_intermediates:
            intermediates = {
                'stage_outputs': stage_outputs,
                'stage_noise_profiles': stage_noise_profiles,
                'stage_improvements': stage_improvements,
                'stage_alphas': self.stage_alphas.data,
                'num_stages_used': len(stage_outputs)
            }

        return current, intermediates

    def get_stage_summary(self, intermediates: Dict) -> str:
        """
        Generate human-readable summary of iterative refinement

        Args:
            intermediates: Dict from forward pass

        Returns:
            summary: Text summary
        """
        summary = "Iterative Refinement Summary:\n"
        summary += "=" * 60 + "\n"

        stage_improvements = intermediates.get('stage_improvements', [])
        stage_alphas = intermediates.get('stage_alphas', [])

        for i, (improvement, alpha) in enumerate(zip(stage_improvements, stage_alphas)):
            summary += f"\nStage {i+1}:\n"
            summary += f"  PSNR: {improvement:.2f} dB\n"
            summary += f"  Blend weight α: {torch.sigmoid(alpha).item():.3f}\n"

            if i < len(intermediates['stage_noise_profiles']):
                noise_profile = intermediates['stage_noise_profiles'][i]
                summary += "  Noise composition:\n"
                for noise_type in ['speckle', 'banding', 'gaussian', 'shot']:
                    weight = noise_profile[noise_type].mean().item() * 100
                    summary += f"    {noise_type}: {weight:.1f}%\n"

        summary += "\n" + "=" * 60
        return summary


class ProgressiveStageRefinement(nn.Module):
    """
    Alternative: Progressive refinement with specialized stages

    Stage 1: Broad denoising (all heads active)
    Stage 2: Target dominant remaining noise
    Stage 3: Fine-grained polish
    """

    def __init__(
        self,
        nsnd_symbolic_analyzer,
        channels: int = 16,
        device: str = 'cpu'
    ):
        super().__init__()

        self.device = device
        self.nsnd_analyzer = nsnd_symbolic_analyzer

        # Stage 1: Full multi-head (aggressive)
        self.stage1 = AdaptiveMultiHeadRefinement(
            nsnd_symbolic_analyzer=nsnd_symbolic_analyzer,
            channels=channels,
            device=device
        )
        self.stage1.alpha = 0.4  # More aggressive

        # Stage 2: Full multi-head (moderate)
        self.stage2 = AdaptiveMultiHeadRefinement(
            nsnd_symbolic_analyzer=nsnd_symbolic_analyzer,
            channels=channels//2,  # Smaller capacity
            device=device
        )
        self.stage2.alpha = 0.2  # More conservative

        # Stage 3: Final polish (gentle)
        self.stage3 = AdaptiveMultiHeadRefinement(
            nsnd_symbolic_analyzer=nsnd_symbolic_analyzer,
            channels=channels//4,  # Even smaller
            device=device
        )
        self.stage3.alpha = 0.1  # Very conservative

    def forward(
        self,
        noisy: torch.Tensor,
        return_intermediates: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Progressive refinement

        Args:
            noisy: Noisy input [B, 1, H, W]
            return_intermediates: Return stage outputs

        Returns:
            output: Final denoised [B, 1, H, W]
            intermediates: Stage info
        """
        # Stage 1: Aggressive denoising
        out1, info1 = self.stage1(noisy, return_intermediates=True)

        # Stage 2: Refine stage 1 output
        out2, info2 = self.stage2(out1, return_intermediates=True)

        # Stage 3: Final polish
        out3, info3 = self.stage3(out2, return_intermediates=True)

        intermediates = {}
        if return_intermediates:
            intermediates = {
                'stage1_output': out1,
                'stage2_output': out2,
                'stage3_output': out3,
                'stage1_routing': info1['head_weights'],
                'stage2_routing': info2['head_weights'],
                'stage3_routing': info3['head_weights'],
            }

        return out3, intermediates


def train_iterative_refinement(
    model: IterativeMultiStageRefinement,
    train_loader,
    device: str = 'cpu',
    epochs: int = 50,
    lr: float = 5e-4
):
    """
    Train iterative refinement model

    Key: Each stage is trained jointly
    Loss is applied to final output + intermediate supervision

    Args:
        model: Iterative refinement model
        train_loader: Training data
        device: Device
        epochs: Number of epochs
        lr: Learning rate (lower than single-stage)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("\n" + "="*70)
    print("TRAINING ITERATIVE MULTI-STAGE REFINEMENT")
    print("="*70)
    print(f"Number of stages: {model.num_stages}")
    print("Innovation: Each stage targets residual noise")
    print("="*70)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_psnr = 0.0
        num_batches = 0

        for noisy, clean in train_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()

            # Forward pass (with intermediate supervision)
            output, intermediates = model(noisy, clean=clean, return_intermediates=True)

            # Multi-stage loss
            # Final output loss (most important)
            loss_final = F.mse_loss(output.float(), clean.float())

            # Intermediate stage losses (auxiliary supervision)
            loss_intermediate = 0.0
            stage_outputs = intermediates['stage_outputs']
            for i, stage_out in enumerate(stage_outputs[:-1]):  # Exclude final
                weight = 0.3 * (0.5 ** i)  # Exponentially decreasing weight
                loss_intermediate += weight * F.mse_loss(stage_out.float(), clean.float())

            # Total loss
            loss = loss_final + 0.2 * loss_intermediate

            # PSNR (on final output)
            psnr = 10 * torch.log10(torch.tensor(1.0, device=device) / (loss_final + 1e-8))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_psnr += psnr.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        avg_psnr = epoch_psnr / num_batches

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.2f} dB")

            # Show stage improvements
            if num_batches > 0:
                stage_improvements = intermediates.get('stage_improvements', [])
                if stage_improvements:
                    improvements_str = " → ".join([f"{p:.2f}" for p in stage_improvements])
                    print(f"  Stage PSNR progression: {improvements_str} dB")

    print("="*70)
    print("Iterative training complete!")
    print("="*70)

    return model
