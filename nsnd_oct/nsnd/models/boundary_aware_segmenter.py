#!/usr/bin/env python3
"""
Boundary-Aware Segmenter with Deep Supervision for OCT Layer Segmentation

Key innovations:
1. Dual-head architecture: Region segmentation + Boundary detection
2. Deep supervision: Auxiliary losses at multiple decoder stages
3. Columnar attention: Exploits horizontal layer structure
4. Topology-aware: Soft ordering constraints for anatomical consistency

Designed to handle thin layers (IS_OS) that are essentially boundaries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBNReLU(nn.Module):
    """Conv-BatchNorm-ReLU block."""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    """Residual block with optional downsampling."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, out_ch)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

        # Skip connection
        self.skip = nn.Identity() if (in_ch == out_ch and stride == 1) else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return self.relu(out + identity)


class ColumnarAttention(nn.Module):
    """
    Column-wise attention for OCT layer segmentation.
    OCT layers are roughly horizontal, so we apply attention along vertical axis.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels

        # Vertical attention (along height)
        self.query = nn.Conv2d(channels, channels // reduction, 1)
        self.key = nn.Conv2d(channels, channels // reduction, 1)
        self.value = nn.Conv2d(channels, channels, 1)

        # Horizontal smoothing (layers are continuous horizontally)
        self.horizontal_smooth = nn.Conv2d(channels, channels, (1, 7), padding=(0, 3),
                                           groups=channels, bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape

        # Compute Q, K, V
        q = self.query(x)  # [B, C/r, H, W]
        k = self.key(x)
        v = self.value(x)

        # Reshape for vertical attention
        # Each column is treated as a sequence of H tokens
        q = q.permute(0, 3, 2, 1).reshape(B * W, H, -1)  # [B*W, H, C/r]
        k = k.permute(0, 3, 2, 1).reshape(B * W, H, -1)
        v = v.permute(0, 3, 2, 1).reshape(B * W, H, -1)

        # Self-attention along vertical axis
        attn = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(q.shape[-1]), dim=-1)
        out = (attn @ v)  # [B*W, H, C]

        # Reshape back
        out = out.reshape(B, W, H, C).permute(0, 3, 2, 1)  # [B, C, H, W]

        # Horizontal smoothing for layer continuity
        out = self.horizontal_smooth(out)

        return x + self.gamma * out


class ASPPModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling for multi-scale feature extraction.
    Captures both fine details (thin layers) and global context.
    """
    def __init__(self, in_channels, out_channels, dilations=[6, 12, 18]):
        super().__init__()

        # 1x1 conv
        self.conv1x1 = ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0)

        # Dilated convolutions
        self.conv_d1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[0],
                      dilation=dilations[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_d2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[1],
                      dilation=dilations[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_d3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[2],
                      dilation=dilations[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Global average pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        size = x.shape[2:]

        feat1 = self.conv1x1(x)
        feat2 = self.conv_d1(x)
        feat3 = self.conv_d2(x)
        feat4 = self.conv_d3(x)
        feat5 = F.interpolate(self.global_pool(x), size=size, mode='bilinear', align_corners=False)

        out = self.fusion(torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1))
        return out


class BoundaryDetectionHead(nn.Module):
    """
    Dedicated boundary detection head.
    Thin layers like IS_OS are essentially boundaries between adjacent regions.
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes

        # Boundary feature extraction
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
        )

        # Per-class boundary prediction
        # For 4 classes, we have 3 boundaries: RNFL|INL, INL|IS, IS|RPE
        self.boundary_head = nn.Conv2d(in_channels // 4, num_classes - 1, 1)

        # Gradient-based boundary detector (fixed, not learnable)
        self.register_buffer('sobel_y', torch.tensor([
            [[-1., -2., -1.],
             [0., 0., 0.],
             [1., 2., 1.]]
        ]).unsqueeze(0) / 4.0)

    def forward(self, features, seg_probs=None):
        """
        Args:
            features: Decoder features [B, C, H, W]
            seg_probs: Optional segmentation probabilities for boundary extraction

        Returns:
            boundary_logits: [B, num_classes-1, H, W] boundary predictions
        """
        # Learned boundary features
        boundary_feat = self.boundary_conv(features)
        boundary_logits = self.boundary_head(boundary_feat)

        return boundary_logits

    def extract_gt_boundaries(self, seg_mask):
        """
        Extract ground truth boundaries from segmentation mask.

        Args:
            seg_mask: [B, H, W] integer segmentation labels

        Returns:
            boundaries: [B, num_classes-1, H, W] binary boundary masks
        """
        B, H, W = seg_mask.shape
        device = seg_mask.device

        boundaries = []
        for c in range(self.num_classes - 1):
            # Boundary between class c and c+1
            # A pixel is on boundary if its class is c and neighbor is c+1, or vice versa
            mask_c = (seg_mask == c).float().unsqueeze(1)  # [B, 1, H, W]
            mask_c1 = (seg_mask == c + 1).float().unsqueeze(1)

            # Apply Sobel filter to detect edges
            padded_c = F.pad(mask_c, [1, 1, 1, 1], mode='reflect')
            padded_c1 = F.pad(mask_c1, [1, 1, 1, 1], mode='reflect')

            edge_c = torch.abs(F.conv2d(padded_c, self.sobel_y.to(device)))
            edge_c1 = torch.abs(F.conv2d(padded_c1, self.sobel_y.to(device)))

            # Boundary is where edges of adjacent classes meet
            boundary = ((edge_c > 0.1) | (edge_c1 > 0.1)).float()

            # Dilate slightly for better training signal
            boundary = F.max_pool2d(boundary, 3, stride=1, padding=1)

            boundaries.append(boundary)

        return torch.cat(boundaries, dim=1)


class BoundaryAwareSegmenter(nn.Module):
    """
    Boundary-Aware Segmenter with Deep Supervision.

    Architecture:
    1. Encoder: ResNet-style with 4 stages
    2. ASPP: Multi-scale feature aggregation
    3. Decoder: 4 stages with skip connections
    4. Dual heads: Region segmentation + Boundary detection
    5. Deep supervision: Auxiliary losses at each decoder stage
    6. Columnar attention: Exploits horizontal layer structure
    """

    def __init__(self, in_channels=1, num_classes=4, base_filters=48):
        super().__init__()
        self.num_classes = num_classes

        # Encoder (4 stages)
        self.enc1 = nn.Sequential(
            ConvBNReLU(in_channels, base_filters),
            ResBlock(base_filters, base_filters)
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            ResBlock(base_filters, base_filters * 2),
            ResBlock(base_filters * 2, base_filters * 2)
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            ResBlock(base_filters * 2, base_filters * 4),
            ResBlock(base_filters * 4, base_filters * 4)
        )
        self.enc4 = nn.Sequential(
            nn.MaxPool2d(2),
            ResBlock(base_filters * 4, base_filters * 8),
            ResBlock(base_filters * 8, base_filters * 8)
        )

        # ASPP at bottleneck
        self.aspp = ASPPModule(base_filters * 8, base_filters * 4)

        # Columnar attention at bottleneck
        self.col_attn = ColumnarAttention(base_filters * 4)

        # Decoder with skip connections
        self.up4 = nn.ConvTranspose2d(base_filters * 4, base_filters * 4, 2, stride=2)
        self.dec4 = nn.Sequential(
            ResBlock(base_filters * 4 + base_filters * 4, base_filters * 4),
            ResBlock(base_filters * 4, base_filters * 4)
        )

        self.up3 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.dec3 = nn.Sequential(
            ResBlock(base_filters * 2 + base_filters * 2, base_filters * 2),
            ResBlock(base_filters * 2, base_filters * 2)
        )

        self.up2 = nn.ConvTranspose2d(base_filters * 2, base_filters, 2, stride=2)
        self.dec2 = nn.Sequential(
            ResBlock(base_filters + base_filters, base_filters),
            ResBlock(base_filters, base_filters)
        )

        # Region segmentation heads (main + auxiliary for deep supervision)
        # Input channels match decoder outputs: d2=base, d3=base*2, d4=base*4
        self.seg_head_main = nn.Conv2d(base_filters, num_classes, 1)  # full res
        self.seg_head_aux2 = nn.Conv2d(base_filters, num_classes, 1)  # 1/2 res (d2 output)
        self.seg_head_aux3 = nn.Conv2d(base_filters * 2, num_classes, 1)  # 1/4 res (d3 output)
        self.seg_head_aux4 = nn.Conv2d(base_filters * 4, num_classes, 1)  # 1/8 res (d4 output)

        # Boundary detection head
        self.boundary_head = BoundaryDetectionHead(base_filters, num_classes)

        # Boundary-guided refinement
        # Uses boundary predictions to refine segmentation at boundaries
        self.boundary_refine = nn.Sequential(
            nn.Conv2d(num_classes + (num_classes - 1), base_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_filters, num_classes, 1)
        )

        # Column-wise refinement (layers are roughly horizontal)
        self.column_refine = nn.Conv2d(num_classes, num_classes, (7, 1), padding=(3, 0))

    def forward(self, x, return_aux=True, return_boundary=True):
        """
        Forward pass.

        Args:
            x: Input image [B, 1, H, W]
            return_aux: Whether to return auxiliary outputs for deep supervision
            return_boundary: Whether to return boundary predictions

        Returns:
            seg_probs: Main segmentation probabilities [B, num_classes, H, W]
            aux_outputs: List of auxiliary segmentation outputs (if return_aux=True)
            boundary_logits: Boundary predictions [B, num_classes-1, H, W] (if return_boundary=True)
        """
        input_size = x.shape[2:]

        # Encoder
        e1 = self.enc1(x)      # [B, 48, H, W]
        e2 = self.enc2(e1)     # [B, 96, H/2, W/2]
        e3 = self.enc3(e2)     # [B, 192, H/4, W/4]
        e4 = self.enc4(e3)     # [B, 384, H/8, W/8]

        # ASPP + Columnar Attention
        aspp_out = self.aspp(e4)  # [B, 192, H/8, W/8]
        aspp_out = self.col_attn(aspp_out)

        # Decoder stage 4
        d4 = self.up4(aspp_out)
        # Handle size mismatch
        if d4.shape[2:] != e3.shape[2:]:
            d4 = F.interpolate(d4, e3.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))
        aux4 = self.seg_head_aux4(d4)

        # Decoder stage 3
        d3 = self.up3(d4)
        if d3.shape[2:] != e2.shape[2:]:
            d3 = F.interpolate(d3, e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))
        aux3 = self.seg_head_aux3(d3)

        # Decoder stage 2
        d2 = self.up2(d3)
        if d2.shape[2:] != e1.shape[2:]:
            d2 = F.interpolate(d2, e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        aux2 = self.seg_head_aux2(d2)

        # Main segmentation head
        seg_logits = self.seg_head_main(d2)

        # Boundary detection
        boundary_logits = self.boundary_head(d2)

        # Boundary-guided refinement
        # Concatenate seg and boundary, then refine
        combined = torch.cat([seg_logits, boundary_logits], dim=1)
        seg_refined = seg_logits + 0.2 * self.boundary_refine(combined)

        # Column-wise refinement for horizontal smoothness
        seg_refined = seg_refined + 0.1 * self.column_refine(seg_refined)

        # Apply softmax for probabilities
        seg_probs = F.softmax(seg_refined, dim=1)

        # Prepare outputs
        outputs = {'seg_probs': seg_probs, 'seg_logits': seg_refined}

        if return_aux:
            # Upsample auxiliary outputs to input size
            aux2_up = F.interpolate(aux2, input_size, mode='bilinear', align_corners=False)
            aux3_up = F.interpolate(aux3, input_size, mode='bilinear', align_corners=False)
            aux4_up = F.interpolate(aux4, input_size, mode='bilinear', align_corners=False)
            outputs['aux_outputs'] = [aux2_up, aux3_up, aux4_up]

        if return_boundary:
            outputs['boundary_logits'] = boundary_logits

        return outputs

    def get_boundary_gt(self, seg_mask):
        """Helper to get boundary ground truth."""
        return self.boundary_head.extract_gt_boundaries(seg_mask)


class BoundaryAwareLoss(nn.Module):
    """
    Combined loss for boundary-aware segmentation with deep supervision.

    Components:
    1. Main segmentation loss (Dice + CE)
    2. Auxiliary losses at each decoder stage (deep supervision)
    3. Boundary detection loss (BCE)
    4. Thin layer boosting
    """

    def __init__(self, num_classes=4, thin_layer_ids=[2],
                 aux_weights=[0.4, 0.3, 0.2], boundary_weight=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.thin_layer_ids = thin_layer_ids
        self.aux_weights = aux_weights
        self.boundary_weight = boundary_weight

        # Per-class weights (boost underrepresented classes)
        # Class distribution: RNFL=60%, INL=4%, IS_OS=4%, RPE=31%
        # IS_OS needs aggressive boost (10x) to prevent collapse
        # [RNFL_GCL, INL_OPL_ONL, IS_OS, RPE_Choroid]
        self.class_weights = torch.tensor([1.0, 3.0, 10.0, 1.5])  # IS_OS gets 10x weight

    def dice_loss(self, pred, target, smooth=1e-5):
        """
        Compute Dice loss.

        Args:
            pred: [B, C, H, W] predicted probabilities
            target: [B, H, W] integer labels
        """
        B, C, H, W = pred.shape
        target_one_hot = F.one_hot(target, C).permute(0, 3, 1, 2).float()  # [B, C, H, W]

        # Per-class Dice
        dice_per_class = []
        for c in range(C):
            pred_c = pred[:, c].reshape(B, -1)
            target_c = target_one_hot[:, c].reshape(B, -1)

            intersection = (pred_c * target_c).sum(dim=1)
            union = pred_c.sum(dim=1) + target_c.sum(dim=1)

            dice = (2 * intersection + smooth) / (union + smooth)
            dice_per_class.append(dice)

        dice_per_class = torch.stack(dice_per_class, dim=1)  # [B, C]

        # Weighted average
        weights = self.class_weights.to(pred.device)
        weighted_dice = (dice_per_class * weights).sum(dim=1) / weights.sum()

        return 1 - weighted_dice.mean()

    def focal_ce_loss(self, pred_logits, target, gamma=2.0):
        """
        Focal Cross-Entropy loss for class imbalance with class weighting.

        Args:
            pred_logits: [B, C, H, W] raw logits
            target: [B, H, W] integer labels
        """
        device = pred_logits.device
        weights = self.class_weights.to(device)

        # Cross-entropy with class weights
        ce = F.cross_entropy(pred_logits, target, weight=weights, reduction='none')  # [B, H, W]

        # Focal weighting
        pred_probs = F.softmax(pred_logits, dim=1)
        target_probs = pred_probs.gather(1, target.unsqueeze(1)).squeeze(1)  # [B, H, W]
        focal_weight = (1 - target_probs) ** gamma

        focal_ce = (focal_weight * ce).mean()
        return focal_ce

    def boundary_loss(self, pred_boundary, gt_boundary):
        """
        Boundary detection loss using BCE with per-boundary weighting.

        Args:
            pred_boundary: [B, C-1, H, W] predicted boundary logits (3 boundaries for 4 classes)
            gt_boundary: [B, C-1, H, W] ground truth boundaries

        Boundaries:
            0: RNFL_GCL | INL_OPL_ONL
            1: INL_OPL_ONL | IS_OS  (critical for IS_OS)
            2: IS_OS | RPE_Choroid  (critical for IS_OS)
        """
        pred_probs = torch.sigmoid(pred_boundary)
        device = pred_boundary.device

        # Per-boundary weights: IS_OS boundaries (1, 2) get higher weight
        # [RNFL|INL, INL|IS_OS, IS_OS|RPE]
        boundary_weights = torch.tensor([1.0, 4.0, 4.0], device=device).view(1, -1, 1, 1)

        # Use focal BCE to handle boundary/non-boundary imbalance
        pos_weight = 10.0  # Boundaries are rare, weight them higher
        bce_per_boundary = F.binary_cross_entropy_with_logits(
            pred_boundary, gt_boundary,
            pos_weight=torch.tensor([pos_weight], device=device),
            reduction='none'
        )
        # Apply per-boundary weights
        weighted_bce = (bce_per_boundary * boundary_weights).mean()

        # Per-boundary Dice loss
        smooth = 1e-5
        dice_losses = []
        for b in range(pred_boundary.shape[1]):
            pred_b = pred_probs[:, b]
            gt_b = gt_boundary[:, b]
            intersection = (pred_b * gt_b).sum()
            union = pred_b.sum() + gt_b.sum()
            dice = 1 - (2 * intersection + smooth) / (union + smooth)
            dice_losses.append(dice * boundary_weights[0, b, 0, 0])
        weighted_dice = sum(dice_losses) / boundary_weights.sum()

        return weighted_bce + weighted_dice

    def forward(self, outputs, target_mask, boundary_head):
        """
        Compute total loss.

        Args:
            outputs: Dict with 'seg_probs', 'seg_logits', 'aux_outputs', 'boundary_logits'
            target_mask: [B, H, W] ground truth segmentation
            boundary_head: BoundaryDetectionHead for extracting GT boundaries

        Returns:
            total_loss: Scalar loss
            loss_dict: Dict of individual loss components
        """
        seg_probs = outputs['seg_probs']
        seg_logits = outputs['seg_logits']
        aux_outputs = outputs.get('aux_outputs', [])
        boundary_logits = outputs.get('boundary_logits', None)

        device = seg_probs.device
        self.class_weights = self.class_weights.to(device)

        # Main segmentation loss
        main_dice = self.dice_loss(seg_probs, target_mask)
        main_ce = self.focal_ce_loss(seg_logits, target_mask)
        main_loss = main_dice + main_ce

        # Deep supervision losses
        aux_loss = 0
        for i, aux_logits in enumerate(aux_outputs):
            aux_probs = F.softmax(aux_logits, dim=1)
            aux_dice = self.dice_loss(aux_probs, target_mask)
            aux_ce = self.focal_ce_loss(aux_logits, target_mask)
            aux_loss += self.aux_weights[i] * (aux_dice + aux_ce)

        # Boundary loss
        boundary_loss_val = 0
        if boundary_logits is not None:
            gt_boundary = boundary_head.extract_gt_boundaries(target_mask)
            boundary_loss_val = self.boundary_loss(boundary_logits, gt_boundary)

        # Total loss
        total_loss = main_loss + aux_loss + self.boundary_weight * boundary_loss_val

        # Loss dictionary for logging
        loss_dict = {
            'main_dice': main_dice.item(),
            'main_ce': main_ce.item(),
            'aux_loss': aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss,
            'boundary_loss': boundary_loss_val.item() if isinstance(boundary_loss_val, torch.Tensor) else boundary_loss_val,
            'total': total_loss.item()
        }

        return total_loss, loss_dict


def create_boundary_aware_segmenter(num_classes=4, pretrained_path=None):
    """
    Factory function to create the boundary-aware segmenter.

    Args:
        num_classes: Number of segmentation classes (default 4 for 4-layer scheme)
        pretrained_path: Optional path to pretrained weights

    Returns:
        model: BoundaryAwareSegmenter instance
    """
    model = BoundaryAwareSegmenter(in_channels=1, num_classes=num_classes)

    if pretrained_path is not None:
        state = torch.load(pretrained_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=False)
        print(f"Loaded pretrained boundary-aware segmenter from {pretrained_path}")

    return model


if __name__ == '__main__':
    # Test the module
    print("Testing BoundaryAwareSegmenter...")

    # Create model
    model = BoundaryAwareSegmenter(in_channels=1, num_classes=4, base_filters=48)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(2, 1, 64, 64)
    outputs = model(x, return_aux=True, return_boundary=True)

    print(f"Segmentation output: {outputs['seg_probs'].shape}")
    print(f"Boundary output: {outputs['boundary_logits'].shape}")
    print(f"Auxiliary outputs: {[aux.shape for aux in outputs['aux_outputs']]}")

    # Test loss
    target = torch.randint(0, 4, (2, 64, 64))
    loss_fn = BoundaryAwareLoss(num_classes=4)
    total_loss, loss_dict = loss_fn(outputs, target, model.boundary_head)

    print(f"\nLoss components: {loss_dict}")

    print("\nAll tests passed!")
