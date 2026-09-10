#!/usr/bin/env python3
"""
Adapter-based Continual Learning for Differentiable Shortest Path (DSP)

This module implements domain-specific adapters for DSP boundary detection,
enabling continual learning across different:
- OCT devices (Zeiss, Heidelberg, Topcon, etc.)
- Clinical sites (different acquisition protocols)
- Patient populations (different pathologies)

Key Design Principles:
1. Shared DSP backbone (frozen after initial training)
2. Small domain-specific adapters (~5-10% of backbone parameters)
3. No replay buffer needed (important for medical data privacy)
4. Easy to extend with new domains without retraining

Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ADAPTER-BASED DSP                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input Features ──► Shared DSP Backbone (frozen) ──► Base Costs             │
│                              │                           │                  │
│                              ▼                           ▼                  │
│                     ┌─────────────────┐         ┌──────────────┐            │
│                     │  Adapter Bank   │         │ Cost Adapter │            │
│                     │  ┌───────────┐  │         │  (per domain)│            │
│                     │  │ Adapter_A │  │         └──────────────┘            │
│                     │  │ Adapter_B │  │                │                    │
│                     │  │ Adapter_C │  │                ▼                    │
│                     │  └───────────┘  │    Adapted Costs = Base + Δ         │
│                     └─────────────────┘                │                    │
│                                                        ▼                    │
│                                          DSP Shortest Path ──► Boundaries   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

References:
- Houlsby et al., "Parameter-Efficient Transfer Learning for NLP", ICML 2019
- Rebuffi et al., "Learning multiple visual domains with residual adapters", NeurIPS 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math


class DSPAdapter(nn.Module):
    """
    Lightweight adapter module for domain-specific DSP adaptation.

    Architecture (bottleneck design):
    ┌─────────────────────────────────────────────┐
    │  Input (C channels)                         │
    │         │                                   │
    │         ▼                                   │
    │  Down-projection (C → C/r)                  │
    │         │                                   │
    │         ▼                                   │
    │  Non-linearity (GELU)                       │
    │         │                                   │
    │         ▼                                   │
    │  Up-projection (C/r → C)                    │
    │         │                                   │
    │         ▼                                   │
    │  Scale (learnable α)                        │
    │         │                                   │
    │         ▼                                   │
    │  Output = Input + α * Adapter(Input)        │
    └─────────────────────────────────────────────┘

    The bottleneck ratio r controls the parameter efficiency.
    Typical r=4 means adapter has ~1/4 parameters of a full layer.
    """

    def __init__(
        self,
        in_channels: int,
        bottleneck_ratio: int = 4,
        dropout: float = 0.1,
        init_scale: float = 0.01,
    ):
        """
        Args:
            in_channels: Number of input channels
            bottleneck_ratio: Reduction ratio for bottleneck (higher = fewer params)
            dropout: Dropout rate
            init_scale: Initial scale for adapter output (start small for stability)
        """
        super().__init__()

        self.in_channels = in_channels
        bottleneck_dim = max(in_channels // bottleneck_ratio, 8)

        # Bottleneck layers
        self.down_proj = nn.Conv2d(in_channels, bottleneck_dim, 1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Conv2d(bottleneck_dim, in_channels, 1)

        # Learnable scale (starts small for stable training)
        self.scale = nn.Parameter(torch.tensor(init_scale))

        # Initialize to near-identity (adapter starts with minimal effect)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for near-identity behavior at start."""
        nn.init.kaiming_normal_(self.down_proj.weight, mode='fan_out')
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)  # Zero init for residual
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply adapter transformation.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Output tensor [B, C, H, W] = x + scale * adapter(x)
        """
        # Bottleneck transformation
        residual = self.down_proj(x)
        residual = self.activation(residual)
        residual = self.dropout(residual)
        residual = self.up_proj(residual)

        # Scaled residual connection
        return x + self.scale * residual

    def get_num_params(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BoundaryAdapter(nn.Module):
    """
    Adapter specifically for boundary cost adjustment.

    This adapter learns domain-specific adjustments to boundary costs,
    accounting for differences in:
    - Layer appearance across devices
    - Noise characteristics
    - Scan quality variations
    """

    def __init__(
        self,
        num_boundaries: int = 4,
        hidden_dim: int = 32,
        use_boundary_specific: bool = True,
    ):
        """
        Args:
            num_boundaries: Number of boundaries to adapt
            hidden_dim: Hidden dimension for adapter
            use_boundary_specific: If True, use separate adapters per boundary
        """
        super().__init__()

        self.num_boundaries = num_boundaries
        self.use_boundary_specific = use_boundary_specific

        if use_boundary_specific:
            # Separate small adapter per boundary
            self.boundary_adapters = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(1, hidden_dim, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(hidden_dim, 1, 3, padding=1),
                )
                for _ in range(num_boundaries)
            ])
        else:
            # Shared adapter for all boundaries
            self.shared_adapter = nn.Sequential(
                nn.Conv2d(num_boundaries, hidden_dim, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, num_boundaries, 3, padding=1),
            )

        # Learnable scales per boundary
        self.scales = nn.Parameter(torch.ones(num_boundaries) * 0.1)

        self._init_weights()

    def _init_weights(self):
        """Initialize for near-identity at start."""
        if self.use_boundary_specific:
            for adapter in self.boundary_adapters:
                nn.init.zeros_(adapter[-1].weight)
                nn.init.zeros_(adapter[-1].bias)
        else:
            nn.init.zeros_(self.shared_adapter[-1].weight)
            nn.init.zeros_(self.shared_adapter[-1].bias)

    def forward(self, costs: torch.Tensor) -> torch.Tensor:
        """
        Adapt boundary costs.

        Args:
            costs: [B, num_boundaries, H, W] boundary cost volumes

        Returns:
            Adapted costs [B, num_boundaries, H, W]
        """
        B, N, H, W = costs.shape

        if self.use_boundary_specific:
            adapted = []
            for i in range(N):
                cost_i = costs[:, i:i+1, :, :]  # [B, 1, H, W]
                delta_i = self.boundary_adapters[i](cost_i)  # [B, 1, H, W]
                adapted_i = cost_i + self.scales[i] * delta_i
                adapted.append(adapted_i)
            return torch.cat(adapted, dim=1)
        else:
            delta = self.shared_adapter(costs)  # [B, N, H, W]
            scales = self.scales.view(1, N, 1, 1)
            return costs + scales * delta


class AdapterBank(nn.Module):
    """
    Manages multiple domain-specific adapters.

    Provides:
    - Registration of new adapters
    - Selection of adapter by domain name
    - Optional automatic domain detection
    - Adapter freezing/unfreezing for continual learning
    """

    def __init__(
        self,
        feature_channels: int = 64,
        num_boundaries: int = 4,
        bottleneck_ratio: int = 4,
    ):
        """
        Args:
            feature_channels: Number of feature channels for feature adapters
            num_boundaries: Number of boundaries for cost adapters
            bottleneck_ratio: Bottleneck ratio for feature adapters
        """
        super().__init__()

        self.feature_channels = feature_channels
        self.num_boundaries = num_boundaries
        self.bottleneck_ratio = bottleneck_ratio

        # Dictionary of registered adapters
        self.feature_adapters = nn.ModuleDict()
        self.cost_adapters = nn.ModuleDict()

        # Current active adapter
        self.active_adapter: Optional[str] = None

        # Domain classifier for automatic selection (optional)
        self.domain_classifier: Optional[nn.Module] = None

    def register_adapter(
        self,
        domain_name: str,
        include_feature_adapter: bool = True,
        include_cost_adapter: bool = True,
    ):
        """
        Register a new adapter for a domain.

        Args:
            domain_name: Unique name for the domain (e.g., 'zeiss', 'heidelberg')
            include_feature_adapter: Whether to add feature adapter
            include_cost_adapter: Whether to add cost adapter
        """
        if include_feature_adapter:
            self.feature_adapters[domain_name] = DSPAdapter(
                in_channels=self.feature_channels,
                bottleneck_ratio=self.bottleneck_ratio,
            )

        if include_cost_adapter:
            self.cost_adapters[domain_name] = BoundaryAdapter(
                num_boundaries=self.num_boundaries,
            )

        print(f"Registered adapter for domain: {domain_name}")
        self._print_adapter_stats(domain_name)

    def _print_adapter_stats(self, domain_name: str):
        """Print parameter statistics for an adapter."""
        total_params = 0

        if domain_name in self.feature_adapters:
            params = self.feature_adapters[domain_name].get_num_params()
            total_params += params
            print(f"  Feature adapter: {params:,} params")

        if domain_name in self.cost_adapters:
            params = sum(p.numel() for p in self.cost_adapters[domain_name].parameters())
            total_params += params
            print(f"  Cost adapter: {params:,} params")

        print(f"  Total: {total_params:,} params")

    def set_active_adapter(self, domain_name: Optional[str]):
        """
        Set the active adapter for forward pass.

        Args:
            domain_name: Domain name or None to disable adapters
        """
        if domain_name is not None:
            if domain_name not in self.feature_adapters and domain_name not in self.cost_adapters:
                raise ValueError(f"Unknown domain: {domain_name}. "
                               f"Available: {list(self.feature_adapters.keys())}")
        self.active_adapter = domain_name

    def freeze_adapter(self, domain_name: str):
        """Freeze an adapter (no gradient updates)."""
        if domain_name in self.feature_adapters:
            for param in self.feature_adapters[domain_name].parameters():
                param.requires_grad = False
        if domain_name in self.cost_adapters:
            for param in self.cost_adapters[domain_name].parameters():
                param.requires_grad = False
        print(f"Frozen adapter: {domain_name}")

    def unfreeze_adapter(self, domain_name: str):
        """Unfreeze an adapter (enable gradient updates)."""
        if domain_name in self.feature_adapters:
            for param in self.feature_adapters[domain_name].parameters():
                param.requires_grad = True
        if domain_name in self.cost_adapters:
            for param in self.cost_adapters[domain_name].parameters():
                param.requires_grad = True
        print(f"Unfrozen adapter: {domain_name}")

    def adapt_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply active feature adapter."""
        if self.active_adapter is None:
            return features
        if self.active_adapter not in self.feature_adapters:
            return features
        return self.feature_adapters[self.active_adapter](features)

    def adapt_costs(self, costs: torch.Tensor) -> torch.Tensor:
        """Apply active cost adapter."""
        if self.active_adapter is None:
            return costs
        if self.active_adapter not in self.cost_adapters:
            return costs
        return self.cost_adapters[self.active_adapter](costs)

    def get_adapter_names(self) -> List[str]:
        """Get list of registered adapter names."""
        return list(set(list(self.feature_adapters.keys()) +
                       list(self.cost_adapters.keys())))

    def add_domain_classifier(self, num_domains: int, feature_dim: int = 64):
        """
        Add automatic domain classifier for adapter selection.

        Args:
            num_domains: Number of domains to classify
            feature_dim: Feature dimension for classification
        """
        self.domain_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_domains),
        )

    def classify_domain(self, features: torch.Tensor) -> torch.Tensor:
        """
        Classify domain from features (for automatic adapter selection).

        Args:
            features: [B, C, H, W] input features

        Returns:
            domain_logits: [B, num_domains] classification logits
        """
        if self.domain_classifier is None:
            raise ValueError("Domain classifier not initialized. Call add_domain_classifier first.")
        return self.domain_classifier(features)


class AdaptiveDSPBoundaryDetector(nn.Module):
    """
    DSP Boundary Detector with Adapter-based Continual Learning.

    This extends the base DSPBoundaryDetector with:
    1. Adapter bank for domain-specific adaptation
    2. Backbone freezing for continual learning
    3. Domain-specific boundary refinement

    Usage for Continual Learning:
    1. Train on initial domain (e.g., 'zeiss') with full model
    2. Freeze backbone
    3. Register new adapter for new domain (e.g., 'heidelberg')
    4. Train only the new adapter on new domain data
    5. Repeat for additional domains
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 64,
        num_boundaries: int = 4,
        smoothness_weight: float = 1.0,
        temperature: float = 0.1,
        min_gap: int = 5,
        use_gradient_hint: bool = True,
        adapter_bottleneck_ratio: int = 4,  # Alias for compatibility
        adapter_dropout: float = 0.1,  # Adapter dropout (stored for reference)
        initial_domain: str = 'spectralis',  # Initial domain name
        bottleneck_ratio: int = None,  # Deprecated, use adapter_bottleneck_ratio
    ):
        super().__init__()

        # Handle deprecated bottleneck_ratio parameter
        if bottleneck_ratio is not None:
            adapter_bottleneck_ratio = bottleneck_ratio

        self.in_channels = in_channels
        self.num_boundaries = num_boundaries

        # =====================================================================
        # Shared Backbone (frozen after initial training)
        # =====================================================================

        # Feature trunk (shared across domains)
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Per-boundary cost heads (shared)
        self.boundary_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels // 2, 1, 1),
            )
            for _ in range(num_boundaries)
        ])

        # Gradient hint fusion (optional)
        self.use_gradient_hint = use_gradient_hint
        if use_gradient_hint:
            self.gradient_fusion = nn.Conv2d(hidden_channels + 1, hidden_channels, 1)

        # DSP module (shared)
        from .differentiable_shortest_path import DifferentiableShortestPath
        self.dsp = DifferentiableShortestPath(
            num_boundaries=num_boundaries,
            smoothness_weight=smoothness_weight,
            temperature=temperature,
            min_gap=min_gap,
            use_soft_dtw=False,
        )

        # Boundary smoother (shared)
        self.smoother = nn.Sequential(
            nn.Conv1d(num_boundaries, num_boundaries * 2, 11, padding=5, groups=num_boundaries),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_boundaries * 2, num_boundaries, 11, padding=5, groups=num_boundaries),
        )
        # BUG FIX: Initialize last conv to zero for proper residual learning
        nn.init.zeros_(self.smoother[-1].weight)
        nn.init.zeros_(self.smoother[-1].bias)

        # =====================================================================
        # Adapter Bank (domain-specific, trainable)
        # =====================================================================
        self.adapter_bank = AdapterBank(
            feature_channels=hidden_channels,
            num_boundaries=num_boundaries,
            bottleneck_ratio=adapter_bottleneck_ratio,
        )

        # Register initial domain
        self.initial_domain = initial_domain
        self.adapter_bank.register_adapter(initial_domain)
        self.adapter_bank.set_active_adapter(initial_domain)

        # Track backbone frozen state
        self.backbone_frozen = False

    def freeze_backbone(self):
        """Freeze shared backbone for continual learning."""
        for param in self.trunk.parameters():
            param.requires_grad = False
        for head in self.boundary_heads:
            for param in head.parameters():
                param.requires_grad = False
        if self.use_gradient_hint:
            for param in self.gradient_fusion.parameters():
                param.requires_grad = False
        for param in self.smoother.parameters():
            param.requires_grad = False

        self.backbone_frozen = True
        print("DSP backbone frozen. Only adapters will be trained.")

    def unfreeze_backbone(self):
        """Unfreeze backbone (for fine-tuning or initial training)."""
        for param in self.trunk.parameters():
            param.requires_grad = True
        for head in self.boundary_heads:
            for param in head.parameters():
                param.requires_grad = True
        if self.use_gradient_hint:
            for param in self.gradient_fusion.parameters():
                param.requires_grad = True
        for param in self.smoother.parameters():
            param.requires_grad = True

        self.backbone_frozen = False
        print("DSP backbone unfrozen. Full model will be trained.")

    def register_domain(self, domain_name: str):
        """Register a new domain adapter."""
        self.adapter_bank.register_adapter(domain_name)

    def set_domain(self, domain_name: Optional[str]):
        """Set active domain for forward pass."""
        self.adapter_bank.set_active_adapter(domain_name)

    def set_active_domain(self, domain_name: Optional[str]):
        """Alias for set_domain (for compatibility)."""
        self.set_domain(domain_name)

    @property
    def active_domain(self) -> Optional[str]:
        """Get the currently active domain name."""
        return self.adapter_bank.active_adapter

    def get_domain_names(self) -> List[str]:
        """Get list of all registered domain names."""
        return list(self.adapter_bank.feature_adapters.keys())

    def get_adapter_params(self) -> List[nn.Parameter]:
        """Alias for get_trainable_params (for compatibility)."""
        return self.get_trainable_params()

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get list of trainable parameters (for optimizer)."""
        params = []

        # Add backbone params if not frozen
        if not self.backbone_frozen:
            params.extend(self.trunk.parameters())
            for head in self.boundary_heads:
                params.extend(head.parameters())
            if self.use_gradient_hint:
                params.extend(self.gradient_fusion.parameters())
            params.extend(self.smoother.parameters())

        # Add adapter params (always trainable for active domain)
        for adapter in self.adapter_bank.feature_adapters.values():
            params.extend(adapter.parameters())
        for adapter in self.adapter_bank.cost_adapters.values():
            params.extend(adapter.parameters())

        return [p for p in params if p.requires_grad]

    def forward(
        self,
        features: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        return_costs: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Detect boundaries with domain adaptation.

        Args:
            features: [B, C, H, W] image features
            image: [B, 1, H, W] original image (optional)
            return_costs: Whether to return cost volumes

        Returns:
            Dict with boundaries, costs, etc.
        """
        B, C, H, W = features.shape

        # Apply feature adapter (domain-specific)
        features = self.adapter_bank.adapt_features(features)

        # Shared trunk
        x = self.trunk(features)

        # Add gradient hint if available
        if self.use_gradient_hint and image is not None:
            grad_y = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
            grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')
            grad_hint = 1.0 - grad_y
            x = self.gradient_fusion(torch.cat([x, grad_hint], dim=1))

        # Predict costs for each boundary
        costs = []
        for head in self.boundary_heads:
            cost = head(x)
            costs.append(cost)
        costs = torch.cat(costs, dim=1)  # [B, num_boundaries, H, W]

        # Apply cost adapter (domain-specific boundary adjustment)
        costs = self.adapter_bank.adapt_costs(costs)

        # Find optimal boundaries via DSP
        boundaries, path_costs = self.dsp(costs, return_path_costs=True)

        # Smooth boundaries
        boundaries_smooth = self.smoother(boundaries) + boundaries

        # Enforce ordering
        boundaries_ordered = self._enforce_ordering(boundaries_smooth)
        boundaries_ordered = boundaries_ordered.clamp(0.01, 0.99)

        outputs = {
            'boundaries': boundaries_ordered,
            'boundaries_pixels': boundaries_ordered * (H - 1),
            'path_costs': path_costs,
        }

        if return_costs:
            outputs['costs'] = costs

        return outputs

    def _enforce_ordering(self, boundaries: torch.Tensor) -> torch.Tensor:
        """Enforce ascending order of boundaries."""
        B, N, W = boundaries.shape

        deltas = torch.zeros_like(boundaries)
        deltas[:, 0, :] = boundaries[:, 0, :]
        deltas[:, 1:, :] = boundaries[:, 1:, :] - boundaries[:, :-1, :]

        min_gap_normalized = 0.02
        deltas = F.relu(deltas) + min_gap_normalized

        ordered = torch.cumsum(deltas, dim=1)
        max_val = ordered[:, -1:, :]
        ordered = ordered / (max_val + 1e-8)

        return ordered


class ContinualDSPTrainer:
    """
    Trainer for continual learning with adapter-based DSP.

    Manages the training process for:
    1. Initial domain training (full model)
    2. New domain adaptation (adapters only)
    3. Optional regularization to prevent forgetting
    """

    def __init__(
        self,
        model: AdaptiveDSPBoundaryDetector,
        device: torch.device,
        learning_rate: float = 1e-4,
        adapter_lr_multiplier: float = 10.0,  # Adapters train faster
    ):
        """
        Args:
            model: AdaptiveDSPBoundaryDetector model
            device: Training device
            learning_rate: Base learning rate
            adapter_lr_multiplier: Multiplier for adapter learning rate
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.adapter_lr_multiplier = adapter_lr_multiplier

        # Track trained domains
        self.trained_domains: List[str] = []

    def train_initial_domain(
        self,
        domain_name: str,
        train_loader,
        val_loader,
        loss_fn,
        epochs: int = 50,
    ) -> Dict[str, float]:
        """
        Train on initial domain with full model.

        Args:
            domain_name: Name for this domain
            train_loader: Training data loader
            val_loader: Validation data loader
            loss_fn: Loss function
            epochs: Number of epochs

        Returns:
            Training metrics
        """
        print(f"\n{'='*60}")
        print(f"INITIAL DOMAIN TRAINING: {domain_name}")
        print(f"{'='*60}")

        # Ensure backbone is unfrozen
        self.model.unfreeze_backbone()

        # Register domain adapter
        self.model.register_domain(domain_name)
        self.model.set_domain(domain_name)

        # Create optimizer for full model
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )

        # Training loop
        best_loss = float('inf')
        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader, optimizer, loss_fn)
            val_loss = self._validate(val_loader, loss_fn)

            if val_loss < best_loss:
                best_loss = val_loss

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Freeze backbone after initial training
        self.model.freeze_backbone()
        self.trained_domains.append(domain_name)

        print(f"\nInitial training complete. Backbone frozen.")
        print(f"Trained domains: {self.trained_domains}")

        return {'train_loss': train_loss, 'val_loss': best_loss}

    def adapt_to_new_domain(
        self,
        domain_name: str,
        train_loader,
        val_loader,
        loss_fn,
        epochs: int = 20,
        freeze_previous: bool = True,
    ) -> Dict[str, float]:
        """
        Adapt to new domain using only adapters.

        Args:
            domain_name: Name for new domain
            train_loader: Training data loader
            val_loader: Validation data loader
            loss_fn: Loss function
            epochs: Number of epochs
            freeze_previous: Whether to freeze previous domain adapters

        Returns:
            Training metrics
        """
        print(f"\n{'='*60}")
        print(f"NEW DOMAIN ADAPTATION: {domain_name}")
        print(f"{'='*60}")

        # Freeze previous adapters if requested
        if freeze_previous:
            for prev_domain in self.trained_domains:
                self.model.adapter_bank.freeze_adapter(prev_domain)

        # Register new adapter
        self.model.register_domain(domain_name)
        self.model.set_domain(domain_name)

        # Create optimizer for adapter only (higher LR)
        adapter_params = []
        if domain_name in self.model.adapter_bank.feature_adapters:
            adapter_params.extend(
                self.model.adapter_bank.feature_adapters[domain_name].parameters()
            )
        if domain_name in self.model.adapter_bank.cost_adapters:
            adapter_params.extend(
                self.model.adapter_bank.cost_adapters[domain_name].parameters()
            )

        optimizer = torch.optim.AdamW(
            adapter_params,
            lr=self.learning_rate * self.adapter_lr_multiplier,
            weight_decay=1e-4,
        )

        # Training loop
        best_loss = float('inf')
        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader, optimizer, loss_fn)
            val_loss = self._validate(val_loader, loss_fn)

            if val_loss < best_loss:
                best_loss = val_loss

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        self.trained_domains.append(domain_name)

        print(f"\nAdaptation complete.")
        print(f"Trained domains: {self.trained_domains}")

        return {'train_loss': train_loss, 'val_loss': best_loss}

    def _train_epoch(self, loader, optimizer, loss_fn) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0

        for batch in loader:
            features = batch['features'].to(self.device)
            gt_boundaries = batch['boundaries'].to(self.device)

            optimizer.zero_grad()

            outputs = self.model(features, return_costs=True)
            loss = loss_fn(outputs['boundaries'], gt_boundaries)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def _validate(self, loader, loss_fn) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in loader:
                features = batch['features'].to(self.device)
                gt_boundaries = batch['boundaries'].to(self.device)

                outputs = self.model(features, return_costs=True)
                loss = loss_fn(outputs['boundaries'], gt_boundaries)

                total_loss += loss.item()

        return total_loss / len(loader)

    def evaluate_all_domains(self, domain_loaders: Dict[str, any], loss_fn) -> Dict[str, float]:
        """
        Evaluate on all trained domains.

        Args:
            domain_loaders: Dict mapping domain names to data loaders
            loss_fn: Loss function

        Returns:
            Dict of domain -> loss
        """
        results = {}

        for domain_name, loader in domain_loaders.items():
            if domain_name in self.trained_domains:
                self.model.set_domain(domain_name)
                loss = self._validate(loader, loss_fn)
                results[domain_name] = loss
                print(f"Domain {domain_name}: Loss = {loss:.4f}")

        return results


# =============================================================================
# Test
# =============================================================================
if __name__ == '__main__':
    print("Testing Adapter-based DSP...")

    device = 'cpu'

    # Create model
    model = AdaptiveDSPBoundaryDetector(
        in_channels=64,
        hidden_channels=64,
        num_boundaries=4,
    ).to(device)

    # Count backbone parameters
    backbone_params = sum(p.numel() for p in model.trunk.parameters())
    backbone_params += sum(p.numel() for head in model.boundary_heads for p in head.parameters())
    print(f"Backbone parameters: {backbone_params:,}")

    # Register domains
    model.register_domain('zeiss')
    model.register_domain('heidelberg')
    model.register_domain('topcon')

    # Test forward pass
    features = torch.randn(2, 64, 256, 256)
    image = torch.randn(2, 1, 256, 256)

    # Test each domain
    for domain in ['zeiss', 'heidelberg', 'topcon']:
        model.set_domain(domain)
        outputs = model(features, image, return_costs=True)
        print(f"\nDomain: {domain}")
        print(f"  Boundaries: {outputs['boundaries'].shape}")
        print(f"  Costs: {outputs['costs'].shape}")

    # Test without adapter
    model.set_domain(None)
    outputs = model(features, image)
    print(f"\nNo adapter:")
    print(f"  Boundaries: {outputs['boundaries'].shape}")

    # Test freezing
    model.freeze_backbone()
    trainable = model.get_trainable_params()
    print(f"\nTrainable params after freezing backbone: {sum(p.numel() for p in trainable):,}")

    print("\nAll tests passed!")
