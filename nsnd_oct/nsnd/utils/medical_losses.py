"""
Medical Image Quality Losses

For medical imaging, we need to optimize BOTH:
- PSNR: Pixel-wise accuracy
- SSIM: Structural similarity (perceptual quality, edge preservation)

SSIM is especially important because:
- Preserves diagnostic features (edges, textures)
- Better correlates with human perception
- Critical for clinical interpretation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Loss

    SSIM ranges from -1 to 1 (higher is better)
    SSIM Loss = 1 - SSIM (for minimization)
    """

    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1

        # Create Gaussian window
        self.window = self._create_window(window_size, self.channel)

    def _gaussian(self, window_size, sigma=1.5):
        """Create Gaussian kernel"""
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size//2)**2 / (2*sigma**2)))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        """Create 2D Gaussian window"""
        _1D_window = self._gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        """Compute SSIM"""
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        """
        Args:
            img1, img2: [B, 1, H, W]

        Returns:
            loss: 1 - SSIM (lower is better)
        """
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        ssim_value = self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

        return 1 - ssim_value  # Convert to loss (minimize)


class MedicalImageLoss(nn.Module):
    """
    Combined loss for medical image denoising

    Optimizes BOTH:
    - PSNR (via MSE): Pixel accuracy
    - SSIM: Structural/perceptual quality

    For medical imaging, SSIM weight should be HIGH
    """

    def __init__(self, ssim_weight=0.5, mse_weight=0.5):
        """
        Args:
            ssim_weight: Weight for SSIM loss (0-1)
            mse_weight: Weight for MSE loss (0-1)

        Recommended for medical imaging:
            ssim_weight = 0.5-0.7 (emphasize structure)
            mse_weight = 0.3-0.5
        """
        super().__init__()

        assert 0 <= ssim_weight <= 1
        assert 0 <= mse_weight <= 1

        self.ssim_weight = ssim_weight
        self.mse_weight = mse_weight

        self.ssim_loss = SSIMLoss()

    def forward(self, output, target):
        """
        Args:
            output: Denoised image [B, 1, H, W]
            target: Clean ground truth [B, 1, H, W]

        Returns:
            loss: Combined loss
            metrics: Dict with individual losses
        """
        # MSE loss (for PSNR)
        mse = F.mse_loss(output, target)

        # SSIM loss (for structure)
        ssim = self.ssim_loss(output, target)

        # Combined loss
        total_loss = self.mse_weight * mse + self.ssim_weight * ssim

        # Compute metrics for monitoring
        psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
        ssim_value = 1 - ssim  # Convert back to SSIM (0-1)

        metrics = {
            'total_loss': total_loss.item(),
            'mse_loss': mse.item(),
            'ssim_loss': ssim.item(),
            'psnr': psnr.item(),
            'ssim': ssim_value.item()
        }

        return total_loss, metrics


class EdgePreservingLoss(nn.Module):
    """
    Edge-preserving loss for medical imaging

    Penalizes edge degradation more heavily
    Critical for preserving diagnostic boundaries
    """

    def __init__(self, edge_weight=0.3):
        super().__init__()
        self.edge_weight = edge_weight

        # Sobel filters for edge detection
        self.sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        self.sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

    def forward(self, output, target):
        """
        Args:
            output: Denoised [B, 1, H, W]
            target: Clean [B, 1, H, W]

        Returns:
            loss: MSE + edge-preserving term
        """
        # Move Sobel filters to same device
        if output.is_cuda:
            self.sobel_x = self.sobel_x.cuda(output.get_device())
            self.sobel_y = self.sobel_y.cuda(output.get_device())

        self.sobel_x = self.sobel_x.type_as(output)
        self.sobel_y = self.sobel_y.type_as(output)

        # MSE loss
        mse = F.mse_loss(output, target)

        # Edge loss
        # Compute edges for output
        output_pad = F.pad(output, (1, 1, 1, 1), mode='reflect')
        output_edge_x = F.conv2d(output_pad, self.sobel_x)
        output_edge_y = F.conv2d(output_pad, self.sobel_y)
        output_edge = torch.sqrt(output_edge_x**2 + output_edge_y**2)

        # Compute edges for target
        target_pad = F.pad(target, (1, 1, 1, 1), mode='reflect')
        target_edge_x = F.conv2d(target_pad, self.sobel_x)
        target_edge_y = F.conv2d(target_pad, self.sobel_y)
        target_edge = torch.sqrt(target_edge_x**2 + target_edge_y**2)

        # Edge MSE
        edge_mse = F.mse_loss(output_edge, target_edge)

        # Combined
        total_loss = mse + self.edge_weight * edge_mse

        return total_loss


class MedicalMultiMetricLoss(nn.Module):
    """
    Complete loss for medical imaging

    Combines:
    - MSE (PSNR optimization)
    - SSIM (structural similarity)
    - Edge preservation
    """

    def __init__(self, mse_weight=0.4, ssim_weight=0.5, edge_weight=0.1):
        super().__init__()

        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight

        self.ssim_loss = SSIMLoss()
        self.edge_loss = EdgePreservingLoss(edge_weight=1.0)  # Will be scaled below

    def forward(self, output, target):
        """
        Args:
            output: Denoised [B, 1, H, W]
            target: Clean [B, 1, H, W]

        Returns:
            loss: Combined loss
            metrics: Dict with all metrics
        """
        # MSE
        mse = F.mse_loss(output, target)

        # SSIM
        ssim_loss_value = self.ssim_loss(output, target)

        # Edge preservation
        edge_loss_value = self.edge_loss(output, target)

        # Combined
        total_loss = (
            self.mse_weight * mse +
            self.ssim_weight * ssim_loss_value +
            self.edge_weight * edge_loss_value
        )

        # Metrics
        psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
        ssim_value = 1 - ssim_loss_value

        metrics = {
            'total_loss': total_loss.item(),
            'mse': mse.item(),
            'ssim_loss': ssim_loss_value.item(),
            'edge_loss': edge_loss_value.item(),
            'psnr': psnr.item(),
            'ssim': ssim_value.item()
        }

        return total_loss, metrics
