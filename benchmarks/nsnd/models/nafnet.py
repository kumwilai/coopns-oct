"""
NAFNet: Simple Baselines for Image Restoration (Chen et al., ECCV 2022)

SOTA simple architecture that removes unnecessary components (LayerNorm, SimpleGate).
Serves as a strong modern baseline for comparison with NSND.

Paper: https://arxiv.org/abs/2204.04676
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormFunction(torch.autograd.Function):
    """Custom LayerNorm for better numerical stability"""

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    """2D Layer Normalization"""

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    """Simple gating mechanism (split and multiply)"""

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """
    NAF Block: The core building block

    Key components:
    - SimpleGate (gating mechanism)
    - LayerNorm2d (normalization)
    - No activation functions!
    """

    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class ConditionalNAFBlock(nn.Module):
    """NAF block with FiLM-style conditioning."""

    def __init__(self, c, cond_dim, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.film1 = nn.Linear(cond_dim, c * 2)
        self.film2 = nn.Linear(cond_dim, c * 2)
        nn.init.zeros_(self.film1.weight)
        nn.init.zeros_(self.film1.bias)
        nn.init.zeros_(self.film2.weight)
        nn.init.zeros_(self.film2.bias)

    @staticmethod
    def _apply_film(x, film_params):
        scale, shift = film_params.chunk(2, dim=1)
        scale = torch.tanh(scale).unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + scale) + shift

    def forward(self, inp, cond):
        if cond is None:
            raise ValueError("ConditionalNAFBlock requires a conditioning vector.")

        x = self.norm1(inp)
        x = self._apply_film(x, self.film1(cond))

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = self._apply_film(x, self.film2(cond))
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class DBMNAFBlock(NAFBlock):
    """NAF block with spatial Disentangled Basis Modulation (DBM)."""

    @staticmethod
    def _scale_alpha(alpha: torch.Tensor | float | None, x: torch.Tensor) -> torch.Tensor:
        if alpha is None:
            return torch.zeros(1, 1, 1, 1, device=x.device, dtype=x.dtype)
        if torch.is_tensor(alpha):
            if alpha.ndim == 0:
                return alpha.view(1, 1, 1, 1)
            if alpha.ndim == 1:
                return alpha.view(-1, 1, 1, 1)
            return alpha
        return torch.tensor(alpha, device=x.device, dtype=x.dtype).view(1, 1, 1, 1)

    @staticmethod
    def _apply_dbm(
        x: torch.Tensor,
        spatial_map: torch.Tensor,
        basis_gamma: torch.Tensor,
        basis_beta: torch.Tensor | None,
        alpha: torch.Tensor | float | None,
    ) -> torch.Tensor:
        alpha_t = DBMNAFBlock._scale_alpha(alpha, x)
        proj_gamma = torch.einsum("bkhw,kc->bchw", spatial_map, basis_gamma)
        proj_gamma = torch.tanh(proj_gamma)
        if basis_beta is not None:
            proj_beta = torch.einsum("bkhw,kc->bchw", spatial_map, basis_beta)
            proj_beta = torch.tanh(proj_beta)
        else:
            proj_beta = torch.zeros_like(proj_gamma)
        return x * (1.0 + alpha_t * proj_gamma) + alpha_t * proj_beta

    def forward(
        self,
        inp: torch.Tensor,
        spatial_map: torch.Tensor | None = None,
        basis_gamma: torch.Tensor | None = None,
        basis_beta: torch.Tensor | None = None,
        alpha: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        if spatial_map is None or basis_gamma is None:
            return super().forward(inp)

        x = self.norm1(inp)
        x = self._apply_dbm(x, spatial_map, basis_gamma, basis_beta, alpha)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = self._apply_dbm(x, spatial_map, basis_gamma, basis_beta, alpha)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):
    """
    NAFNet Architecture

    Args:
        img_channel: Number of input channels (1 for grayscale OCT)
        width: Base channel width (default: 32)
        middle_blk_num: Number of blocks in the middle (bottleneck)
        enc_blk_nums: Number of blocks in each encoder stage
        dec_blk_nums: Number of blocks in each decoder stage
    """

    def __init__(self, img_channel=1, width=32, middle_blk_num=1,
                 enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class NAFNetSmall(nn.Module):
    """
    Smaller NAFNet for faster training and fair comparison

    Reduced width and depth compared to full NAFNet
    """

    def __init__(self, img_channel=1, width=16):
        super().__init__()

        # Simpler configuration
        enc_blk_nums = [1, 1, 1, 1]  # 1 block per stage
        dec_blk_nums = [1, 1, 1, 1]
        middle_blk_num = 1

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class NAFNetSmallFiLM(nn.Module):
    """
    Conditional NAFNetSmall with FiLM modulation from a noise vector.
    """

    def __init__(self, img_channel=1, width=16, cond_dim=4):
        super().__init__()

        enc_blk_nums = [1, 1, 1, 1]
        dec_blk_nums = [1, 1, 1, 1]
        middle_blk_num = 1

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            blocks = nn.ModuleList([ConditionalNAFBlock(chan, cond_dim) for _ in range(num)])
            self.encoders.append(blocks)
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.ModuleList([ConditionalNAFBlock(chan, cond_dim) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            blocks = nn.ModuleList([ConditionalNAFBlock(chan, cond_dim) for _ in range(num)])
            self.decoders.append(blocks)

        self.padder_size = 2 ** len(self.encoders)

    @staticmethod
    def _run_blocks(blocks, x, cond):
        for block in blocks:
            x = block(x, cond)
        return x

    def forward(self, inp, cond):
        if cond is None:
            raise ValueError("NAFNetSmallFiLM requires a conditioning vector.")

        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for blocks, down in zip(self.encoders, self.downs):
            x = self._run_blocks(blocks, x, cond)
            encs.append(x)
            x = down(x)

        for block in self.middle_blks:
            x = block(x, cond)

        for blocks, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = self._run_blocks(blocks, x, cond)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class NAFNetFullFiLM(nn.Module):
    """
    Robust Adaptive NAFNet (Full Capacity) with Selective FiLM and Spatial Cues.
    
    Design:
    - Inherits structure from NAFNet (width=64, deep stages)
    - Supports loading pre-trained NAFNet-64 weights (standard blocks match)
    - Replaces specific blocks (middle, decoders) with ConditionalNAFBlock based on flags
    - Adds parallel projection for shallow spatial cues (noise maps)
    """

    def __init__(self, img_channel=1, width=32, middle_blk_num=1,
                 enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1],
                 cond_dim=4, condition_middle=True, condition_decoders=True,
                 use_spatial_cue=True, spatial_cue_channels=4):
        super().__init__()
        
        self.condition_middle = condition_middle
        self.condition_decoders = condition_decoders
        self.use_spatial_cue = use_spatial_cue

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1, bias=True)
        
        # Parallel projection for spatial cue (noise maps)
        # This is added to the intro features, preserving the intro weights
        if self.use_spatial_cue:
            self.cue_projection = nn.Sequential(
                nn.Conv2d(spatial_cue_channels, width, 3, 1, 1),
                nn.Tanh() # Tanh to keep the cue modulation controlled (-1 to 1)
            )
            # Init to near-zero so it doesn't disrupt pre-trained features initially
            nn.init.xavier_uniform_(self.cue_projection[0].weight, gain=0.01)
            nn.init.zeros_(self.cue_projection[0].bias)
        else:
            self.cue_projection = None

        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.dbm_stage_channels = {}
        self.dbm_stage_names = []

        chan = width
        # Encoders (Standard NAFBlocks to match pre-trained weights perfectly)
        # We don't condition encoders usually, as they extract features.
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        # Middle Blocks (Selective Conditioning)
        if self.condition_middle:
            self.middle_blks = nn.ModuleList(
                [DBMNAFBlock(chan) for _ in range(middle_blk_num)]
            )
            self.dbm_stage_channels["middle"] = chan
        else:
            self.middle_blks = nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        # Decoders (Selective Conditioning)
        for idx, num in enumerate(dec_blk_nums):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            stage_name = f"dec{len(dec_blk_nums) - 1 - idx}"
            self.dbm_stage_names.append(stage_name)
            
            if self.condition_decoders:
                self.decoders.append(
                    nn.ModuleList([DBMNAFBlock(chan) for _ in range(num)])
                )
                self.dbm_stage_channels[stage_name] = chan
            else:
                self.decoders.append(
                    nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
                )

        self.padder_size = 2 ** len(self.encoders)

    @staticmethod
    def _get_basis(basis: dict | None, name: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not basis or name not in basis:
            return None, None
        entry = basis[name]
        return entry.get("gamma"), entry.get("beta")

    @staticmethod
    def _resize_map(spatial_map: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if spatial_map.shape[-2:] == x.shape[-2:]:
            return spatial_map
        return F.interpolate(spatial_map, size=x.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, inp, spatial_map=None, basis=None, alpha=0.1, gate=None):
        """
        Args:
            inp: Input image (B, C, H, W)
            spatial_map: Spatial noise map (B, 4, H, W) - Optional
            basis: Dict of stage -> {"gamma": (4, C), "beta": (4, C)}
            alpha: Modulation scale (float or tensor)
            gate: Optional confidence gate (B, 1, 1, 1)
        """
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        alpha_value = alpha
        if gate is not None:
            alpha_value = alpha_value * gate
        
        # Feature Extraction
        x = self.intro(inp)
        
        # Inject Spatial Cue (Shallow "TypeTag")
        if self.use_spatial_cue and spatial_map is not None:
            # Resize cue to match features if needed (due to padding)
            if spatial_map.shape[-2:] != x.shape[-2:]:
                spatial_map = F.interpolate(spatial_map, size=x.shape[-2:], mode='bilinear', align_corners=False)
            
            cue_feat = self.cue_projection(spatial_map)
            x = x + cue_feat  # Additive injection

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        # Middle Blocks (Conditioned)
        if self.condition_middle:
            map_mid = None
            if spatial_map is not None:
                map_mid = self._resize_map(spatial_map, x)
            basis_gamma, basis_beta = self._get_basis(basis, "middle")
            for block in self.middle_blks:
                x = block(x, map_mid, basis_gamma, basis_beta, alpha_value)
        else:
            x = self.middle_blks(x)

        for stage_name, decoder, up, enc_skip in zip(
            self.dbm_stage_names, self.decoders, self.ups, encs[::-1]
        ):
            x = up(x)
            x = x + enc_skip
            
            if self.condition_decoders:
                map_dec = None
                if spatial_map is not None:
                    map_dec = self._resize_map(spatial_map, x)
                basis_gamma, basis_beta = self._get_basis(basis, stage_name)
                for block in decoder:
                    x = block(x, map_dec, basis_gamma, basis_beta, alpha_value)
            else:
                x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

    def forward_patch_based(self, inp, patch_size=64, stride=32, spatial_map=None, basis=None, alpha=0.1, gate=None):
        """
        Patch-based inference for large images to avoid memory issues.

        Args:
            inp: Input image (B, C, H, W)
            patch_size: Size of patches to process (default: 64)
            stride: Stride between patches (default: 32, 50% overlap)
            spatial_map: Spatial noise map (B, 4, H, W) - Optional
            basis: Dict of stage -> {"gamma": (4, C), "beta": (4, C)}
            alpha: Modulation scale (float or tensor)
            gate: Optional confidence gate (B, 1, 1, 1)

        Returns:
            Denoised image (B, C, H, W)
        """
        B, C, H, W = inp.shape
        device = inp.device
        dtype = inp.dtype

        # Initialize output and weight map for blending
        output = torch.zeros_like(inp)
        weight_map = torch.zeros_like(inp)

        # Calculate number of patches
        h_patches = (H - patch_size) // stride + 1 if H > patch_size else 1
        w_patches = (W - patch_size) // stride + 1 if W > patch_size else 1

        # Process patches
        for i in range(h_patches):
            for j in range(w_patches):
                # Calculate patch coordinates
                h_start = i * stride
                w_start = j * stride
                h_end = min(h_start + patch_size, H)
                w_end = min(w_start + patch_size, W)

                # Adjust start if we're at the edge
                if h_end - h_start < patch_size:
                    h_start = max(0, h_end - patch_size)
                if w_end - w_start < patch_size:
                    w_start = max(0, w_end - patch_size)

                # Extract patch
                patch = inp[:, :, h_start:h_end, w_start:w_end]

                # Extract corresponding spatial map patch if provided
                map_patch = None
                if spatial_map is not None:
                    map_patch = spatial_map[:, :, h_start:h_end, w_start:w_end]

                # Process patch
                with torch.no_grad():
                    denoised_patch = self.forward(patch, spatial_map=map_patch, basis=basis, alpha=alpha, gate=gate)

                # Blend patch into output with Gaussian weights for smooth transitions
                h_patch, w_patch = denoised_patch.shape[2:]
                weight_patch = self._create_blend_weights(h_patch, w_patch, device, dtype)

                # Add weighted patch to output
                output[:, :, h_start:h_start+h_patch, w_start:w_start+w_patch] += denoised_patch * weight_patch
                weight_map[:, :, h_start:h_start+h_patch, w_start:w_start+w_patch] += weight_patch

                # Clear patch memory
                del patch, denoised_patch, weight_patch
                if map_patch is not None:
                    del map_patch

        # Normalize by weight map to average overlapping regions
        output = output / (weight_map + 1e-8)

        # Clean up
        del weight_map
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output

    @staticmethod
    def _create_blend_weights(h, w, device, dtype):
        """
        Create Gaussian-like blend weights for smooth patch transitions.

        Args:
            h: Patch height
            w: Patch width
            device: Device for the tensor
            dtype: Data type for the tensor

        Returns:
            Weight map (1, 1, h, w)
        """
        # Create 1D weight arrays with higher weights in the center
        y = torch.linspace(-1, 1, h, device=device, dtype=dtype)
        x = torch.linspace(-1, 1, w, device=device, dtype=dtype)

        # Apply Gaussian-like falloff
        y_weights = torch.exp(-y**2)
        x_weights = torch.exp(-x**2)

        # Create 2D weight map
        weights = y_weights.view(-1, 1) * x_weights.view(1, -1)

        return weights.view(1, 1, h, w)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test NAFNet
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Standard NAFNet (width=32)
    nafnet = NAFNet(width=32, enc_blk_nums=[1, 1, 1, 1], dec_blk_nums=[1, 1, 1, 1]).to(device)
    print(f"NAFNet (width=32): {count_parameters(nafnet):,} parameters")

    # Small NAFNet (width=16)
    nafnet_small = NAFNetSmall(width=16).to(device)
    print(f"NAFNet Small (width=16): {count_parameters(nafnet_small):,} parameters")

    # Test forward pass
    x = torch.randn(1, 1, 64, 64).to(device)
    with torch.no_grad():
        y = nafnet(x)
        y_small = nafnet_small(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape (NAFNet): {y.shape}")
    print(f"Output shape (NAFNet Small): {y_small.shape}")
