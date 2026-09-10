"""Check if analyzer predictions match ground truth noise proportions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "nsnd_oct"))

import torch
import numpy as np
from nsnd.models.hybrid_analyzer import HybridCNNSymbolicAnalyzer
from PIL import Image
import json

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HybridCNNSymbolicAnalyzer(use_log_domain=True).to(device)
ckpt = torch.load("checkpoints/hybrid_analyzer_improved_seed0.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model.eval()
print(f"Loaded checkpoint from epoch {ckpt.get('epoch')}, val_top1={ckpt.get('val_top1'):.1f}%")
print()

# Load validation data
pairs_file = Path("val_pairs_duke_analysis_sub50.txt")
weights_jsonl = Path("weights_duke_analysis_val_sub50.jsonl")

# Parse weights
weights_map = {}
with weights_jsonl.open("r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        noisy_path = entry.get("noisy_path") or entry.get("noisy")
        weights = entry.get("weights")
        if noisy_path and weights:
            weights_map[noisy_path] = weights

# Parse pairs
pairs = []
with pairs_file.open("r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            clean_path, noisy_path = parts[0], parts[1]
            if noisy_path in weights_map:
                pairs.append((noisy_path, weights_map[noisy_path]))

print(f"Loaded {len(pairs)} validation samples")
print()

# Evaluate on first 20 samples
predictions = []
ground_truths = []

for i, (noisy_path, gt_weights) in enumerate(pairs[:20]):
    img = np.array(Image.open(noisy_path).convert('L'), dtype=np.float32) / 255.0
    # Center crop to 64x64
    h, w = img.shape
    top = (h - 64) // 2
    left = (w - 64) // 2
    img = img[top:top+64, left:left+64]

    x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred_dict, features = model(x)

    pred = [
        pred_dict['speckle'][0].item(),
        pred_dict['banding'][0].item(),
        pred_dict['gaussian'][0].item(),
        pred_dict['shot'][0].item(),
    ]
    gt = [
        gt_weights['speckle'],
        gt_weights['banding'],
        gt_weights['gaussian'],
        gt_weights['shot'],
    ]

    predictions.append(pred)
    ground_truths.append(gt)

    # Print individual sample
    print(f"Sample {i+1}:")
    print(f"  GT:   speckle={gt[0]:.3f}, banding={gt[1]:.3f}, gaussian={gt[2]:.3f}, shot={gt[3]:.3f}")
    print(f"  Pred: speckle={pred[0]:.3f}, banding={pred[1]:.3f}, gaussian={pred[2]:.3f}, shot={pred[3]:.3f}")

    # Compute errors
    abs_err = [abs(p - g) for p, g in zip(pred, gt)]
    mae = np.mean(abs_err)
    print(f"  MAE: {mae:.4f}, Max err: {max(abs_err):.4f}")

    # Check dominant class
    gt_dom = np.argmax(gt)
    pred_dom = np.argmax(pred)
    dom_names = ['speckle', 'banding', 'gaussian', 'shot']
    match = "✓" if gt_dom == pred_dom else "✗"
    print(f"  Dominant: GT={dom_names[gt_dom]} vs Pred={dom_names[pred_dom]} {match}")
    print()

# Overall statistics
predictions = np.array(predictions)
ground_truths = np.array(ground_truths)

overall_mae = np.mean(np.abs(predictions - ground_truths))
per_class_mae = np.mean(np.abs(predictions - ground_truths), axis=0)

print("=" * 60)
print("OVERALL STATISTICS (20 samples):")
print(f"  Overall MAE: {overall_mae:.4f}")
print(f"  Per-class MAE: speckle={per_class_mae[0]:.4f}, banding={per_class_mae[1]:.4f}, "
      f"gaussian={per_class_mae[2]:.4f}, shot={per_class_mae[3]:.4f}")

# Check if predictions are just uniform
pred_mean = predictions.mean(axis=0)
pred_std = predictions.std(axis=0)
print(f"  Pred mean: speckle={pred_mean[0]:.3f}, banding={pred_mean[1]:.3f}, "
      f"gaussian={pred_mean[2]:.3f}, shot={pred_mean[3]:.3f}")
print(f"  Pred std:  speckle={pred_std[0]:.3f}, banding={pred_std[1]:.3f}, "
      f"gaussian={pred_std[2]:.3f}, shot={pred_std[3]:.3f}")

# Top-1 accuracy
gt_dominant = np.argmax(ground_truths, axis=1)
pred_dominant = np.argmax(predictions, axis=1)
top1_acc = np.mean(gt_dominant == pred_dominant) * 100
print(f"  Top-1 accuracy: {top1_acc:.1f}%")

# Check correlation
correlations = []
for i in range(4):
    corr = np.corrcoef(ground_truths[:, i], predictions[:, i])[0, 1]
    correlations.append(corr)
print(f"  Correlations: speckle={correlations[0]:.3f}, banding={correlations[1]:.3f}, "
      f"gaussian={correlations[2]:.3f}, shot={correlations[3]:.3f}")
