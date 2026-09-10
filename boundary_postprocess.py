"""
Boundary Post-Processing for OCT Layer Segmentation

Improves Dice scores for thin layers through:
1. Monotonic ordering enforcement (b0 < b1 < b2 < b3)
2. Minimum layer thickness constraints
3. Boundary smoothing (reduces jitter)
4. Optional gradient-based refinement
"""

import torch
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.signal import savgol_filter


def postprocess_boundaries(
    boundaries: np.ndarray,
    image: np.ndarray = None,
    min_thickness: dict = None,
    smooth_method: str = 'savgol',
    smooth_window: int = 11,
    enforce_order: bool = True,
    use_gradient_refinement: bool = False,
    gradient_search_range: int = 5,
) -> np.ndarray:
    """
    Post-process boundary predictions to improve Dice scores.

    Args:
        boundaries: Array of shape (4, W) with normalized [0,1] boundary positions
        image: Optional image array (H, W) for gradient-based refinement
        min_thickness: Dict of minimum layer thicknesses in normalized coords
                      e.g., {'RNFL': 0.02, 'INL': 0.015, 'IS_OS': 0.02}
        smooth_method: 'median', 'gaussian', 'savgol', or None
        smooth_window: Window size for smoothing
        enforce_order: Whether to enforce monotonic ordering
        use_gradient_refinement: Whether to refine using image gradients
        gradient_search_range: Pixel range to search for gradient peaks

    Returns:
        Processed boundaries of shape (4, W)
    """
    bounds = boundaries.copy()
    W = bounds.shape[1]

    # Default minimum thicknesses (in normalized [0,1] coordinates)
    if min_thickness is None:
        min_thickness = {
            'RNFL': 0.025,    # ~6px at 256px, ~5px at 192px
            'INL': 0.020,     # ~5px at 256px, ~4px at 192px
            'IS_OS': 0.025,   # ~6px at 256px, ~5px at 192px
        }

    # Step 1: Smooth boundaries to reduce jitter
    if smooth_method is not None:
        bounds = smooth_boundaries(bounds, method=smooth_method, window=smooth_window)

    # Step 2: Enforce monotonic ordering with minimum thickness
    if enforce_order:
        bounds = enforce_monotonic_order(bounds, min_thickness)

    # Step 3: Optional gradient-based refinement
    if use_gradient_refinement and image is not None:
        bounds = refine_with_gradients(bounds, image, search_range=gradient_search_range)
        # Re-enforce ordering after refinement
        if enforce_order:
            bounds = enforce_monotonic_order(bounds, min_thickness)

    return bounds


def smooth_boundaries(boundaries: np.ndarray, method: str = 'savgol', window: int = 11) -> np.ndarray:
    """
    Smooth boundary predictions to reduce column-to-column jitter.
    """
    bounds = boundaries.copy()

    for i in range(4):
        if method == 'median':
            bounds[i] = median_filter(bounds[i], size=window, mode='nearest')
        elif method == 'gaussian':
            bounds[i] = gaussian_filter1d(bounds[i], sigma=window/4, mode='nearest')
        elif method == 'savgol':
            # Savitzky-Golay filter preserves peaks better than Gaussian
            if window >= 5 and len(bounds[i]) >= window:
                bounds[i] = savgol_filter(bounds[i], window_length=window, polyorder=3, mode='nearest')
        # else: no smoothing

    return bounds


def enforce_monotonic_order(boundaries: np.ndarray, min_thickness: dict) -> np.ndarray:
    """
    Enforce b0 < b1 < b2 < b3 with minimum layer thickness.

    Layer mapping:
    - RNFL (RNFL_GCL): between b0 (ILM) and b1 (RNFL_INL)
    - INL (INL_OPL_ONL): between b1 and b2 (INL_ISOS)
    - IS_OS: between b2 and b3 (ISOS_RPE)
    """
    bounds = boundaries.copy()
    W = bounds.shape[1]

    min_gaps = [
        min_thickness.get('RNFL', 0.025),  # b0 to b1
        min_thickness.get('INL', 0.020),   # b1 to b2
        min_thickness.get('IS_OS', 0.025), # b2 to b3
    ]

    for col in range(W):
        # Start from b0 and push down if needed
        for i in range(3):
            min_gap = min_gaps[i]
            if bounds[i+1, col] < bounds[i, col] + min_gap:
                bounds[i+1, col] = bounds[i, col] + min_gap

        # Clamp to valid range [0, 1]
        bounds[:, col] = np.clip(bounds[:, col], 0, 1)

        # If b3 exceeds 1, push everything up proportionally
        if bounds[3, col] > 1.0:
            excess = bounds[3, col] - 1.0
            # Distribute excess upward
            for i in range(4):
                bounds[i, col] = max(0, bounds[i, col] - excess * (4-i) / 4)

    return bounds


def refine_with_gradients(
    boundaries: np.ndarray,
    image: np.ndarray,
    search_range: int = 5
) -> np.ndarray:
    """
    Refine boundary positions using image gradients.
    Search for gradient peaks near predicted boundaries.
    """
    bounds = boundaries.copy()
    H, W = image.shape

    # Compute vertical gradient
    gradient = np.abs(np.gradient(image.astype(float), axis=0))

    for i in range(4):
        for col in range(W):
            # Current boundary position in pixels
            pos = int(bounds[i, col] * (H - 1))

            # Search range
            start = max(0, pos - search_range)
            end = min(H, pos + search_range + 1)

            # Find gradient peak in search range
            if end > start:
                search_region = gradient[start:end, col]
                if len(search_region) > 0:
                    peak_offset = np.argmax(search_region)
                    new_pos = start + peak_offset
                    bounds[i, col] = new_pos / (H - 1)

    return bounds


def postprocess_batch(
    boundaries: torch.Tensor,
    images: torch.Tensor = None,
    **kwargs
) -> torch.Tensor:
    """
    Post-process a batch of boundary predictions.

    Args:
        boundaries: Tensor of shape (B, 4, W)
        images: Optional tensor of shape (B, 1, H, W)
        **kwargs: Arguments passed to postprocess_boundaries

    Returns:
        Processed boundaries tensor of shape (B, 4, W)
    """
    B = boundaries.shape[0]
    device = boundaries.device

    bounds_np = boundaries.cpu().numpy()

    if images is not None:
        images_np = images.cpu().numpy()
    else:
        images_np = [None] * B

    processed = []
    for i in range(B):
        img = images_np[i, 0] if images is not None else None
        proc = postprocess_boundaries(bounds_np[i], image=img, **kwargs)
        processed.append(proc)

    return torch.from_numpy(np.stack(processed)).to(device)


# Convenience functions for different post-processing levels

def postprocess_light(boundaries: np.ndarray) -> np.ndarray:
    """Light post-processing: just enforce ordering with minimum thickness."""
    return postprocess_boundaries(
        boundaries,
        smooth_method=None,
        enforce_order=True,
        use_gradient_refinement=False,
    )


def postprocess_medium(boundaries: np.ndarray) -> np.ndarray:
    """Medium post-processing: smoothing + ordering."""
    return postprocess_boundaries(
        boundaries,
        smooth_method='savgol',
        smooth_window=11,
        enforce_order=True,
        use_gradient_refinement=False,
    )


def postprocess_heavy(boundaries: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Heavy post-processing: smoothing + ordering + gradient refinement."""
    return postprocess_boundaries(
        boundaries,
        image=image,
        smooth_method='savgol',
        smooth_window=11,
        enforce_order=True,
        use_gradient_refinement=True,
        gradient_search_range=5,
    )


if __name__ == '__main__':
    # Test the post-processing
    import json
    from PIL import Image
    from physics_enhanced_v3 import PhysicsEnsembleV3, boundaries_to_segmentation

    print("=== Testing Boundary Post-Processing ===\n")

    # Load model
    model = PhysicsEnsembleV3(in_channels=1, hidden_channels=48, num_boundaries=4)
    ckpt = torch.load('outputs/physics_v3_dice_v2/stage1_128/best_model.pt',
                      map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded model from epoch {ckpt['epoch']}")

    # Load test sample
    with open('combined_val.jsonl', 'r') as f:
        sample = json.loads(f.readline())

    H = 192  # Test at stage 2 resolution

    # Load and preprocess
    img = Image.open(sample['image_path']).convert('L')
    img_resized = img.resize((H, H), Image.BILINEAR)
    img_np = np.array(img_resized)
    img_tensor = torch.from_numpy(img_np).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)

    # Load GT mask
    mask = Image.open(sample['mask_path']).convert('L')
    mask_resized = np.array(mask.resize((H, H), Image.NEAREST))
    gt_mask = torch.from_numpy(mask_resized)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor, return_aux=True)
        pred_bounds = outputs['boundaries'][0].numpy()  # (4, W)

    # Compute Dice before post-processing
    def compute_dice(bounds, gt_mask, H):
        bounds_tensor = torch.from_numpy(bounds).unsqueeze(0)
        pred_seg = boundaries_to_segmentation(bounds_tensor, H, num_classes=4)[0]

        dice_scores = {}
        for c, name in enumerate(['RNFL_GCL', 'INL_OPL_ONL', 'IS_OS', 'RPE_Choroid']):
            pred_c = (pred_seg == c).float()
            gt_c = (gt_mask == (c + 1)).float()
            intersection = (pred_c * gt_c).sum()
            union = pred_c.sum() + gt_c.sum()
            dice = (2 * intersection + 1e-8) / (union + 1e-8)
            dice_scores[name] = dice.item()
        dice_scores['avg'] = np.mean(list(dice_scores.values()))
        return dice_scores

    print(f"\nTesting at {H}x{H} resolution")
    print("-" * 60)

    # Before post-processing
    dice_before = compute_dice(pred_bounds, gt_mask, H)
    print(f"Before post-processing:")
    print(f"  Avg Dice: {dice_before['avg']:.3f}")
    print(f"  RNFL_GCL: {dice_before['RNFL_GCL']:.3f}, INL: {dice_before['INL_OPL_ONL']:.3f}, "
          f"IS_OS: {dice_before['IS_OS']:.3f}, RPE: {dice_before['RPE_Choroid']:.3f}")

    # Light post-processing
    bounds_light = postprocess_light(pred_bounds)
    dice_light = compute_dice(bounds_light, gt_mask, H)
    print(f"\nLight (ordering only):")
    print(f"  Avg Dice: {dice_light['avg']:.3f} ({dice_light['avg'] - dice_before['avg']:+.3f})")
    print(f"  RNFL_GCL: {dice_light['RNFL_GCL']:.3f}, INL: {dice_light['INL_OPL_ONL']:.3f}, "
          f"IS_OS: {dice_light['IS_OS']:.3f}, RPE: {dice_light['RPE_Choroid']:.3f}")

    # Medium post-processing
    bounds_medium = postprocess_medium(pred_bounds)
    dice_medium = compute_dice(bounds_medium, gt_mask, H)
    print(f"\nMedium (smoothing + ordering):")
    print(f"  Avg Dice: {dice_medium['avg']:.3f} ({dice_medium['avg'] - dice_before['avg']:+.3f})")
    print(f"  RNFL_GCL: {dice_medium['RNFL_GCL']:.3f}, INL: {dice_medium['INL_OPL_ONL']:.3f}, "
          f"IS_OS: {dice_medium['IS_OS']:.3f}, RPE: {dice_medium['RPE_Choroid']:.3f}")

    # Heavy post-processing
    bounds_heavy = postprocess_heavy(pred_bounds, img_np / 255.0)
    dice_heavy = compute_dice(bounds_heavy, gt_mask, H)
    print(f"\nHeavy (smoothing + ordering + gradient):")
    print(f"  Avg Dice: {dice_heavy['avg']:.3f} ({dice_heavy['avg'] - dice_before['avg']:+.3f})")
    print(f"  RNFL_GCL: {dice_heavy['RNFL_GCL']:.3f}, INL: {dice_heavy['INL_OPL_ONL']:.3f}, "
          f"IS_OS: {dice_heavy['IS_OS']:.3f}, RPE: {dice_heavy['RPE_Choroid']:.3f}")

    print("\n=== Test Complete ===")
