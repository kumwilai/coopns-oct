"""
MambaIR: A Simple Baseline for Image Restoration with State-Space Model (ECCV 2024)

Faithful reproduction of the official architecture (github.com/csguoh/MambaIR).
Pure PyTorch implementation: replaces CUDA selective_scan_fn with JIT-compiled
sequential scan. All other architectural details match the official exactly.

Configured with embed_dim=101 for ~7M parameters (vs official embed_dim=180).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@torch.jit.script
def ssm_scan(dt: torch.Tensor, B_val: torch.Tensor, C_val: torch.Tensor,
             x_input: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """JIT-compiled sequential SSM scan with full gradient flow.

    Implements the S6 selective scan: h_t = A_bar * h_{t-1} + B_bar * x_t, y_t = C * h_t
    where A_bar = exp(dt * A) and B_bar = dt * B (first-order Euler discretization).

    Args:
        dt: (batch, L, C) discretization timesteps (after softplus)
        B_val: (batch, L, d_state) input projection
        C_val: (batch, L, d_state) output projection
        x_input: (batch, L, C) input features
        A: (C, d_state) state transition matrix (negative)
    Returns:
        y_out: (batch, C, L)
    """
    batch_size = dt.shape[0]
    L = dt.shape[1]
    C_dim = dt.shape[2]
    d_state = A.shape[1]
    A_unsq = A.unsqueeze(0)  # (1, C, d_state)
    h = torch.zeros(batch_size, C_dim, d_state, device=dt.device, dtype=dt.dtype)
    y_out = torch.empty(batch_size, C_dim, L, device=dt.device, dtype=dt.dtype)

    for t in range(L):
        dt_t = dt[:, t, :].unsqueeze(-1)  # (batch, C, 1)
        h = torch.exp(dt_t * A_unsq) * h + dt_t * B_val[:, t].unsqueeze(1) * x_input[:, t].unsqueeze(-1)
        y_out[:, :, t] = (C_val[:, t].unsqueeze(1) * h).sum(dim=-1)

    return y_out


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization (no timm dependency)."""
    with torch.no_grad():
        def norm_cdf(x):
            return (1. + math.erf(x / math.sqrt(2.))) / 2.

        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


class DropPath(nn.Module):
    """Stochastic depth (drop path) regularization."""

    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.

    Args:
        num_feat: Channel number of intermediate features.
        squeeze_factor: Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    """Channel Attention Block from official MambaIR.

    Args:
        num_feat: Number of feature channels.
        compress_ratio: Compression ratio for intermediate convolutions. Default: 3.
        squeeze_factor: Squeeze factor for channel attention. Default: 30.
    """

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super().__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor),
        )

    def forward(self, x):
        return self.cab(x)


class SS2D(nn.Module):
    """Selective Scan 2D — core SSM module matching the official MambaIR.

    Scans the 2D feature map in 4 directions (H-W, W-H, and their reverses),
    sums the results, then gates with a parallel branch.

    Args:
        d_model: Input/output feature dimension.
        d_state: SSM state dimension. Default: 16.
        d_conv: Depthwise conv kernel size. Default: 3.
        expand: Hidden dimension expansion factor. Default: 2.
        dt_min: Minimum dt for initialization. Default: 0.001.
        dt_max: Maximum dt for initialization. Default: 0.1.
        dt_init_floor: Floor for dt initialization. Default: 1e-4.
    """

    def __init__(self, d_model, d_state=16, d_conv=3, expand=2.,
                 dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16)
        self.n_dirs = 4

        # Input projection (bias=False, matching official)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        # Depthwise conv on 2D feature map (before unfolding to 1D)
        self.conv2d = nn.Conv2d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            padding=(d_conv - 1) // 2, groups=self.d_inner, bias=True,
        )
        self.act = nn.SiLU()

        # Per-direction x_proj: d_inner -> (dt_rank + d_state * 2)
        # Stored as batched weight: (4, dt_rank + 2*d_state, d_inner)
        x_proj_list = [
            nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
            for _ in range(self.n_dirs)
        ]
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in x_proj_list], dim=0)
        )

        # Per-direction dt_proj: dt_rank -> d_inner
        # Stored as batched weight: (4, d_inner, dt_rank) and bias: (4, d_inner)
        dt_proj_list = [
            self._init_dt_proj(self.dt_rank, self.d_inner, dt_min, dt_max, dt_init_floor)
            for _ in range(self.n_dirs)
        ]
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in dt_proj_list], dim=0)
        )
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in dt_proj_list], dim=0)
        )

        # A: per-direction state transition (log-space, S4D init)
        # Shape: (4 * d_inner, d_state)
        A = torch.arange(1, d_state + 1, dtype=torch.float32
                         ).unsqueeze(0).expand(self.d_inner, -1).contiguous()
        A_log = torch.log(A)
        A_log = A_log.unsqueeze(0).expand(self.n_dirs, -1, -1
                                          ).contiguous().flatten(0, 1)
        self.A_logs = nn.Parameter(A_log)

        # D: per-direction skip parameter. Shape: (4 * d_inner,)
        D = torch.ones(self.d_inner).unsqueeze(0).expand(self.n_dirs, -1
                                                         ).contiguous().flatten(0, 1)
        self.Ds = nn.Parameter(D)

        # Output normalization and projection (matching official)
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    @staticmethod
    def _init_dt_proj(dt_rank, d_inner, dt_min, dt_max, dt_init_floor):
        """Initialize dt_proj following Mamba convention."""
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank ** -0.5
        nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    def _forward_core(self, x):
        """4-directional selective scan matching official forward_core.

        Args:
            x: (B, d_inner, H, W) after depthwise conv + SiLU
        Returns:
            y: (B, d_inner, H*W) summed output from all 4 directions
        """
        B, C, H, W = x.shape
        L = H * W

        # Create 4 directional sequences (matching official exactly):
        # dir 0: H-W (row-major flatten)
        # dir 1: W-H (column-major flatten)
        # dir 2: reversed H-W
        # dir 3: reversed W-H
        x_hw = x.reshape(B, C, L)
        x_wh = x.transpose(2, 3).contiguous().reshape(B, C, L)
        seqs = [x_hw, x_wh, x_hw.flip(-1), x_wh.flip(-1)]

        As = -torch.exp(self.A_logs.float())  # (4*C, d_state)

        ys = []
        for k in range(self.n_dirs):
            seq = seqs[k].float()  # (B, C, L)
            seq_t = seq.transpose(1, 2).contiguous()  # (B, L, C)

            # Per-direction projection: (B, L, C) @ (dt_rank+2*d_state, C).T
            x_dbc = seq_t @ self.x_proj_weight[k].T  # (B, L, dt_rank+2*d_state)
            dt_raw, B_val, C_val = x_dbc.split(
                [self.dt_rank, self.d_state, self.d_state], dim=-1
            )

            # dt projection with bias, then softplus
            dt = F.softplus(
                dt_raw @ self.dt_projs_weight[k].T + self.dt_projs_bias[k]
            )  # (B, L, C)

            # SSM scan
            A_k = As[k * C: (k + 1) * C]  # (C, d_state)
            y_k = ssm_scan(dt, B_val, C_val, seq_t, A_k)  # (B, C, L)

            # D skip connection
            D_k = self.Ds[k * C: (k + 1) * C].float()  # (C,)
            y_k = y_k + D_k.unsqueeze(0).unsqueeze(-1) * seq  # (B, C, L)

            ys.append(y_k)

        # Reconstruct: reverse backward dirs and transpose W-H dirs back to H-W
        out0 = ys[0]
        out1 = ys[1].reshape(B, C, W, H).transpose(2, 3).contiguous().reshape(B, C, L)
        out2 = ys[2].flip(-1)
        out3 = ys[3].flip(-1).reshape(B, C, W, H).transpose(2, 3).contiguous().reshape(B, C, L)

        return out0 + out1 + out2 + out3  # sum, not concat (matching official)

    def forward(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            out: (B, H, W, C)
        """
        B, H, W, C = x.shape

        # Input projection and split
        xz = self.in_proj(x)  # (B, H, W, 2*d_inner)
        x_path, z = xz.chunk(2, dim=-1)  # each (B, H, W, d_inner)

        # Depthwise conv in spatial format
        x_path = x_path.permute(0, 3, 1, 2).contiguous()  # (B, d_inner, H, W)
        x_path = self.act(self.conv2d(x_path))  # (B, d_inner, H, W)

        # 4-directional scan
        y = self._forward_core(x_path)  # (B, d_inner, H*W)

        # Reshape and normalize
        y = y.transpose(1, 2).contiguous().reshape(B, H, W, -1)  # (B, H, W, d_inner)
        y = self.out_norm(y)

        # Gating
        y = y * F.silu(z)

        # Output projection
        return self.out_proj(y)  # (B, H, W, C)


class VSSBlock(nn.Module):
    """Visual State Space Block matching official MambaIR.

    Contains SS2D with learnable skip scales, DropPath, and CAB.

    Args:
        hidden_dim: Feature dimension.
        drop_path: Drop path rate.
        d_state: SSM state dimension.
        expand: SS2D expansion factor.
        d_conv: Depthwise conv kernel size.
    """

    def __init__(self, hidden_dim, drop_path=0., d_state=16, expand=2., d_conv=3):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.self_attention = SS2D(
            d_model=hidden_dim, d_state=d_state, expand=expand, d_conv=d_conv,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x, x_size):
        """
        Args:
            x: (B, L, C) token sequence
            x_size: (H, W) spatial dimensions
        Returns:
            x: (B, L, C)
        """
        B, L, C = x.shape
        x = x.view(B, *x_size, C).contiguous()  # (B, H, W, C)

        # SS2D branch with learnable skip scale
        x = x * self.skip_scale + self.drop_path(self.self_attention(self.ln_1(x)))

        # CAB branch with learnable skip scale (CAB operates in BCHW)
        x = x * self.skip_scale2 + self.conv_blk(
            self.ln_2(x).permute(0, 3, 1, 2).contiguous()
        ).permute(0, 2, 3, 1).contiguous()

        return x.view(B, -1, C).contiguous()  # (B, L, C)


class BasicLayer(nn.Module):
    """Stack of VSSBlocks.

    Args:
        dim: Feature dimension.
        depth: Number of VSSBlocks.
        d_state: SSM state dimension.
        expand: SS2D expansion factor.
        drop_path: Drop path rates (list or scalar).
        d_conv: Depthwise conv kernel size.
        use_checkpoint: Whether to use gradient checkpointing.
    """

    def __init__(self, dim, depth, d_state=16, expand=2., drop_path=0.,
                 d_conv=3, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                d_state=d_state,
                expand=expand,
                d_conv=d_conv,
            )
            for i in range(depth)
        ])

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(blk, x, x_size, use_reentrant=False)
            else:
                x = blk(x, x_size)
        return x


class ResidualGroup(nn.Module):
    """Residual State Space Group (RSSG) matching official MambaIR.

    Args:
        dim: Feature dimension.
        depth: Number of VSSBlocks.
        d_state: SSM state dimension.
        expand: SS2D expansion factor.
        drop_path: Drop path rates.
        d_conv: Depthwise conv kernel size.
        use_checkpoint: Use gradient checkpointing.
    """

    def __init__(self, dim, depth, d_state=16, expand=2., drop_path=0.,
                 d_conv=3, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.residual_group = BasicLayer(
            dim=dim, depth=depth, d_state=d_state, expand=expand,
            drop_path=drop_path, d_conv=d_conv, use_checkpoint=use_checkpoint,
        )
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, x_size):
        # x: (B, L, C)
        res = self.residual_group(x, x_size)  # (B, L, C)
        # Convert to spatial for conv, then back to tokens
        B = res.shape[0]
        res = res.transpose(1, 2).reshape(B, self.dim, *x_size)  # (B, C, H, W)
        res = self.conv(res)
        res = res.flatten(2).transpose(1, 2)  # (B, L, C)
        return res + x


class MambaIR(nn.Module):
    """MambaIR: Image Restoration with State-Space Model (ECCV 2024).

    Faithful reproduction of the official architecture with pure PyTorch SSM scan.

    Args:
        in_channels: Number of input channels. Default: 1.
        out_channels: Number of output channels. Default: 1.
        embed_dim: Feature embedding dimension. Default: 101.
        depths: Number of VSSBlocks per RSSG. Default: (6,6,6,6,6,6).
        d_state: SSM state dimension. Default: 16.
        d_conv: Depthwise conv kernel size in SS2D. Default: 3.
        expand: SS2D expansion factor. Default: 2.
        drop_path_rate: Maximum stochastic depth rate. Default: 0.1.
        use_checkpoint: Use gradient checkpointing. Default: True.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        embed_dim=101,
        depths=(6, 6, 6, 6, 6, 6),
        d_state=16,
        d_conv=3,
        expand=2.,
        drop_path_rate=0.1,
        use_checkpoint=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = len(depths)

        # --- 1. Shallow feature extraction ---
        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)

        # --- 2. Deep feature extraction ---
        # Stochastic depth decay rule (linearly increasing)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(ResidualGroup(
                dim=embed_dim,
                depth=depths[i],
                d_state=d_state,
                expand=expand,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                d_conv=d_conv,
                use_checkpoint=use_checkpoint,
            ))

        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # --- 3. Reconstruction (denoising mode) ---
        self.conv_last = nn.Conv2d(embed_dim, out_channels, 3, 1, 1)

        # Initialize weights (matching official: Linear + LayerNorm only)
        self.apply(self._init_weights)

        # Print parameter count
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"MambaIR | Total params: {n_params:,} ({n_params/1e6:.2f}M) | "
              f"Trainable: {n_trainable:,} ({n_trainable/1e6:.2f}M)")

    def _init_weights(self, m):
        """Weight initialization matching official MambaIR.

        Only initializes Linear and LayerNorm modules. Conv2d modules keep
        PyTorch defaults (kaiming_uniform_). SS2D's x_proj_weight, dt_projs_weight,
        dt_projs_bias, A_logs, and Ds are bare Parameters (not modules) and are
        initialized in SS2D.__init__, so they are not affected here.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """Deep feature extraction in token space.

        Args:
            x: (B, C, H, W) spatial features
        Returns:
            x: (B, C, H, W) spatial features after deep extraction
        """
        x_size = (x.shape[2], x.shape[3])
        # 2D -> 1D tokens
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # (B, L, C)

        # 1D -> 2D
        x = x.transpose(1, 2).reshape(x.shape[0], self.embed_dim, *x_size)
        return x

    def forward(self, x):
        """
        Args:
            x: (B, in_channels, H, W)
        Returns:
            output: (B, out_channels, H, W)
        """
        # Denoising forward path (matching official)
        x_first = self.conv_first(x)
        res = self.conv_after_body(self.forward_features(x_first)) + x_first
        output = x + self.conv_last(res)
        return output.clamp(0.0, 1.0)


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    print("=" * 80)
    print("MambaIR: Testing model creation and forward pass")
    print("=" * 80)

    model = MambaIR(
        in_channels=1,
        out_channels=1,
        embed_dim=101,
        depths=[6, 6, 6, 6, 6, 6],
        d_state=16,
        d_conv=3,
        expand=2.,
    )

    total, trainable = count_parameters(model)
    print(f"\nParameter count: {total:,} ({total/1e6:.2f}M)")

    # Test with different spatial sizes (MambaIR handles any size)
    test_sizes = [(1, 1, 64, 64), (1, 1, 32, 32)]
    for size in test_sizes:
        x = torch.randn(*size)
        with torch.no_grad():
            y = model(x)
        assert y.shape == size, f"Shape mismatch: input {size} -> output {y.shape}"
        print(f"Input {size} -> Output {y.shape}")

    print("\nAll tests passed.")
