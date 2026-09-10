"""
Evaluate baseline models (DRUNet, SwinIR, Noise2Void) from sota module.
"""
import torch
import argparse
import numpy as np

# Import from sota module
from sota.train_eval import build_model, evaluate_pairs

def evaluate_baseline(checkpoint_path, val_pairs, model_name, image_size=64):
    """Evaluate a baseline model checkpoint."""
    print(f"Loading {model_name} checkpoint: {checkpoint_path}")

    # Build model using sota build_model
    model = build_model(model_name)

    # Load weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

    print(f"Evaluating on validation set: {val_pairs}")
    print(f"Image size: {image_size}x{image_size}")

    # Use sota evaluate_pairs function (already has progress printing)
    psnr, ssim, time_ms = evaluate_pairs(model, val_pairs, size=image_size)

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")
    print(f"Avg time: {time_ms:.2f} ms/image")
    print("="*60)

    return {
        'psnr': psnr,
        'ssim': ssim,
        'time_ms': time_ms,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--val_pairs", type=str, required=True, help="Validation pairs file")
    parser.add_argument("--model_name", type=str, required=True,
                       choices=['drunet', 'swinir', 'noise2void', 'nafnet', 'speckle2speckle'],
                       help="Model name")
    parser.add_argument("--image_size", type=int, default=64, help="Image size")
    args = parser.parse_args()

    evaluate_baseline(args.checkpoint, args.val_pairs, args.model_name, args.image_size)
