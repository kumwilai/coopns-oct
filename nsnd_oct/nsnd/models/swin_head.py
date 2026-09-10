import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sota.models.swinir_fair import (
    PatchEmbed,
    PatchUnEmbed,
    DropPath,
    Mlp,
    to_2tuple,
    trunc_normal_,
    window_partition,
    window_reverse,
)
from .modulation import NoiseModulationBlock

class StableWindowAttention(nn.Module):
    """
    Window-based attention with optional cosine similarity stabilization.
    """
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
        attn_mode="cosine",
        cosine_logit_scale=10.0,
    ):
        super().__init__()
        if attn_mode not in ("scaled_dot", "cosine"):
            raise ValueError(f"attn_mode must be 'scaled_dot' or 'cosine', got {attn_mode}")
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.attn_mode = attn_mode

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.logit_scale = None
        if self.attn_mode == "cosine":
            init = math.log(max(cosine_logit_scale, 1e-4))
            self.logit_scale = nn.Parameter(torch.tensor(init))

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.attn_mode == "cosine":
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            logit_scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
            attn = (q @ k.transpose(-2, -1)) * logit_scale
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class StableSwinTransformerBlock(nn.Module):
    """Swin Transformer Block with optional post-norm for stability."""
    def __init__(
        self,
        dim,
        num_heads,
        window_size=8,
        shift_size=0,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_mode="cosine",
        cosine_logit_scale=10.0,
        post_norm=True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.post_norm = post_norm
        self._attn_mask_cache = {}

        self.norm1 = norm_layer(dim)
        self.attn = StableWindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_mode=attn_mode,
            cosine_logit_scale=cosine_logit_scale,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def _get_attn_mask(self, x_size, device):
        cache_key = (x_size[0], x_size[1], str(device))
        cached = self._attn_mask_cache.get(cache_key)
        if cached is not None:
            return cached
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        self._attn_mask_cache[cache_key] = attn_mask
        return attn_mask

    def _window_attention(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        if pad_b > 0 or pad_r > 0:
            x = x.permute(0, 3, 1, 2)
            x = F.pad(x, (0, pad_r, 0, pad_b))
            x = x.permute(0, 2, 3, 1)
        Hp, Wp = x.shape[1], x.shape[2]

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        if self.shift_size > 0:
            attn_mask = self._get_attn_mask((Hp, Wp), x.device)
        else:
            attn_mask = None
        attn_windows = self.attn(x_windows, mask=attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_b > 0 or pad_r > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        return x

    def forward(self, x, x_size):
        if self.post_norm:
            shortcut = x
            x = self._window_attention(x, x_size)
            x = shortcut + self.drop_path(x)
            x = self.norm1(x)
            x = x + self.drop_path(self.mlp(x))
            x = self.norm2(x)
            return x

        shortcut = x
        x = self.norm1(x)
        x = self._window_attention(x, x_size)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class StableBasicLayer(nn.Module):
    """A basic Swin layer using StableSwinTransformerBlock."""
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        attn_mode="cosine",
        cosine_logit_scale=10.0,
        post_norm=True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            StableSwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_mode=attn_mode,
                cosine_logit_scale=cosine_logit_scale,
                post_norm=post_norm,
            )
            for i in range(depth)
        ])

    def forward(self, x, x_size):
        for blk in self.blocks:
            x = blk(x, x_size)
        return x

class SwinResidualHead(nn.Module):
    """
    Swin Transformer Head for Residual Refinement.
    
    Architecture:
    1. Shallow Feature Extraction (Conv 3x3)
    2. Deep Feature Extraction (Swin Transformer Blocks)
    3. Reconstruction (Conv 3x3)
    4. Residual Output (Tanh)
    
    Used for Speckle Head to capture global texture coherence.
    """
    def __init__(
        self,
        img_size=64,
        in_chans=2,
        embed_dim=32,
        depths=[2, 2],
        num_heads=[4, 4],
        window_size=8,
        mlp_ratio=2.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_conditioning=False,
        noise_dim=4,
        conditioner_hidden=16,
        dual_stream=True,
        fusion="concat",
        attn_mode="cosine",
        cosine_logit_scale=10.0,
        post_norm=True,
    ):
        super().__init__()

        self.use_conditioning = use_conditioning
        self.dual_stream = bool(dual_stream)
        self.fusion = fusion

        if self.dual_stream and in_chans != 2:
            raise ValueError(f"dual_stream requires in_chans=2, got {in_chans}")

        # FIX 1: Pre-normalize amplified input to prevent explosion
        # Using GroupNorm(in_chans) = InstanceNorm to keep Residual and Base stats separate
        self.input_norm = nn.GroupNorm(num_groups=in_chans, num_channels=in_chans)

        # 1. Shallow Feature Extraction
        if self.dual_stream:
            self.conv_first_res = nn.Conv2d(1, embed_dim, 3, 1, 1)
            self.conv_first_ctx = nn.Conv2d(1, embed_dim, 3, 1, 1)
        else:
            self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # FIX 2: Conservative initialization for first conv to handle amplified inputs
        with torch.no_grad():
            if self.dual_stream:
                self.conv_first_res.weight.mul_(0.1)
                if self.conv_first_res.bias is not None:
                    self.conv_first_res.bias.zero_()
                self.conv_first_ctx.weight.mul_(0.1)
                if self.conv_first_ctx.bias is not None:
                    self.conv_first_ctx.bias.zero_()
            else:
                self.conv_first.weight.mul_(0.1)  # 10x smaller than default
                if self.conv_first.bias is not None:
                    self.conv_first.bias.zero_()

        # 2. Deep Feature Extraction (Swin Blocks)
        if self.dual_stream:
            if self.fusion not in ("sum", "concat"):
                raise ValueError(f"fusion must be 'sum' or 'concat', got {self.fusion}")
            self.patch_embed_res = PatchEmbed(
                img_size=img_size, patch_size=1, in_chans=embed_dim,
                embed_dim=embed_dim, norm_layer=norm_layer if patch_norm else None
            )
            self.patch_embed_ctx = PatchEmbed(
                img_size=img_size, patch_size=1, in_chans=embed_dim,
                embed_dim=embed_dim, norm_layer=norm_layer if patch_norm else None
            )
            if self.fusion == "concat":
                self.token_fuse = nn.Linear(embed_dim * 2, embed_dim)
                self.feature_fuse = nn.Conv2d(embed_dim * 2, embed_dim, 1)
                nn.init.xavier_uniform_(self.token_fuse.weight, gain=0.1)
                if self.token_fuse.bias is not None:
                    nn.init.zeros_(self.token_fuse.bias)
                with torch.no_grad():
                    self.feature_fuse.weight.mul_(0.1)
                    if self.feature_fuse.bias is not None:
                        self.feature_fuse.bias.zero_()
            else:
                self.token_fuse = None
                self.feature_fuse = None
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=1, in_chans=embed_dim,
                embed_dim=embed_dim, norm_layer=norm_layer if patch_norm else None
            )
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=1, in_chans=embed_dim, embed_dim=embed_dim
        )
        
        self.layers = nn.ModuleList()
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i_layer in range(len(depths)):
            layer = StableBasicLayer(
                dim=embed_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur:cur + depths[i_layer]],
                norm_layer=norm_layer,
                attn_mode=attn_mode,
                cosine_logit_scale=cosine_logit_scale,
                post_norm=post_norm,
            )
            self.layers.append(layer)
            cur += depths[i_layer]
            
        self.norm = norm_layer(embed_dim)

        # Noise-Adaptive Feature Modulation (Active Conditioning)
        if self.use_conditioning:
            self.conditioner = NoiseModulationBlock(
                input_dim=noise_dim,
                output_channels=embed_dim,
                hidden_dim=conditioner_hidden,
            )
        else:
            self.conditioner = None

        # 3. Reconstruction
        self.conv_last = nn.Conv2d(embed_dim, 1, 3, 1, 1)
        
        # Zero-initialize the last layer for stable residual learning
        # The head starts by predicting "no residual" and learns from there
        nn.init.zeros_(self.conv_last.weight)
        if self.conv_last.bias is not None:
            nn.init.zeros_(self.conv_last.bias)

    def forward(self, x, condition_vector=None):
        """
        Forward pass with optional noise-adaptive conditioning.

        Args:
            x: Input tensor [B, 2, H, W] (amplified residual + base context)
            condition_vector: Noise probability vector [B, 4] or None
                              If provided and use_conditioning=True, modulates internal features

        Returns:
            out: Refined residual [B, 1, H, W]
        """
        # x is the amplified 2-channel input: [Residual * scale, Base]
        # FIX 1: Normalize input BEFORE processing to prevent activation explosion
        x = self.input_norm(x)

        # Shallow feature extraction
        if self.dual_stream:
            residual = x[:, :1, :, :]
            context = x[:, 1:2, :, :]
            x_res = self.conv_first_res(residual)
            x_ctx = self.conv_first_ctx(context)
            if self.fusion == "sum":
                x_first = x_res + x_ctx
            else:
                x_first = self.feature_fuse(torch.cat([x_res, x_ctx], dim=1))

            res_tokens = self.patch_embed_res(x_res)
            ctx_tokens = self.patch_embed_ctx(x_ctx)
            if self.fusion == "sum":
                x_tokens = res_tokens + ctx_tokens
            else:
                x_tokens = self.token_fuse(torch.cat([res_tokens, ctx_tokens], dim=-1))
        else:
            x_first = self.conv_first(x)
            x_tokens = self.patch_embed(x_first)

        x_size = (x_first.shape[-2], x_first.shape[-1])

        # Swin Transformer Blocks
        for layer in self.layers:
            x_tokens = layer(x_tokens, x_size)

        x_tokens = self.norm(x_tokens)

        # Patch UnEmbed
        x_feat = self.patch_unembed(x_tokens, x_size)  # [B, C, H, W]

        # ACTIVE FEATURE MODULATION (Late Fusion)
        # Apply noise-adaptive scaling to features BEFORE reconstruction
        # This allows the head to adjust its processing strength dynamically
        if self.use_conditioning and self.conditioner is not None and condition_vector is not None:
            # Get channel-wise scaling factors from noise vector
            # NoiseModulationBlock outputs [B, C, 1, 1] directly
            gamma = self.conditioner(condition_vector)  # [B, C, 1, 1]
            x_feat = x_feat * gamma  # Broadcasting works directly

        # Global Residual Connection (like SwinIR)
        x_feat = x_feat + x_first

        # Reconstruction
        out = self.conv_last(x_feat)

        # Output is a residual correction, bounded by Tanh for stability
        return torch.tanh(out)
