"""Check speckle noise detection accuracy specifically."""
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
ckpt = torch.load("checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model.eval()
print(f"Loaded checkpoint from epoch {ckpt.get('epoch')}, val_top1={ckpt.get('val_top1'):.1f}%")
print()

# Load validation data
pairs_file = Path("val_pairs_duke_analysis.txt")
weights_jsonl = Path("weights_duke_analysis_val.jsonl")

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

print(f"Total validation samples: {len(pairs)}")
print()

# Filter for speckle-dominant samples (GT speckle > 0.5)
speckle_dominant = []
speckle_present = []  # GT speckle > 0.3
speckle_minor = []   # GT speckle > 0.1
speckle_absent = []  # GT speckle < 0.1

for noisy_path, weights in pairs:
    speckle_weight = weights['speckle']
    if speckle_weight > 0.5:
        speckle_dominant.append((noisy_path, weights))
    elif speckle_weight > 0.3:
        speckle_present.append((noisy_path, weights))
    elif speckle_weight > 0.1:
        speckle_minor.append((noisy_path, weights))
    else:
        speckle_absent.append((noisy_path, weights))

print(f"Speckle-dominant samples (GT > 0.5): {len(speckle_dominant)}")
print(f"Speckle-present samples (0.3 < GT < 0.5): {len(speckle_present)}")
print(f"Speckle-minor samples (0.1 < GT < 0.3): {len(speckle_minor)}")
print(f"Speckle-absent samples (GT < 0.1): {len(speckle_absent)}")
print()

# Evaluate on speckle-dominant samples
print("=" * 70)
print("SPECKLE-DOMINANT SAMPLES (GT speckle > 0.5):")
print("=" * 70)

speckle_gt = []
speckle_pred = []
correct_top1 = 0

for i, (noisy_path, gt_weights) in enumerate(speckle_dominant[:20]):  # Show first 20
    img = np.array(Image.open(noisy_path).convert('L'), dtype=np.float32) / 255.0
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

    speckle_gt.append(gt[0])
    speckle_pred.append(pred[0])

    gt_dom = np.argmax(gt)
    pred_dom = np.argmax(pred)
    match = "✓" if gt_dom == pred_dom else "✗"
    if gt_dom == pred_dom:
        correct_top1 += 1

    dom_names = ['speckle', 'banding', 'gaussian', 'shot']

    print(f"Sample {i+1}:")
    print(f"  GT:   speckle={gt[0]:.3f}, banding={gt[1]:.3f}, gaussian={gt[2]:.3f}, shot={gt[3]:.3f}")
    print(f"  Pred: speckle={pred[0]:.3f}, banding={pred[1]:.3f}, gaussian={pred[2]:.3f}, shot={pred[3]:.3f}")
    print(f"  Speckle error: {abs(pred[0] - gt[0]):.4f}")
    print(f"  Dominant: GT={dom_names[gt_dom]} vs Pred={dom_names[pred_dom]} {match}")
    print()

# All speckle-dominant samples
all_speckle_gt = []
all_speckle_pred = []
all_correct = 0

for noisy_path, gt_weights in speckle_dominant:
    img = np.array(Image.open(noisy_path).convert('L'), dtype=np.float32) / 255.0
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

    all_speckle_gt.append(gt[0])
    all_speckle_pred.append(pred[0])

    if np.argmax(pred) == np.argmax(gt):
        all_correct += 1

print("=" * 70)
print(f"SPECKLE-DOMINANT STATISTICS (all {len(speckle_dominant)} samples):")
print("=" * 70)
all_speckle_gt = np.array(all_speckle_gt)
all_speckle_pred = np.array(all_speckle_pred)
mae = np.mean(np.abs(all_speckle_pred - all_speckle_gt))
corr = np.corrcoef(all_speckle_gt, all_speckle_pred)[0, 1]
top1 = 100.0 * all_correct / len(speckle_dominant)

print(f"Speckle weight MAE: {mae:.4f}")
print(f"Speckle weight correlation: {corr:.3f}")
print(f"Top-1 accuracy (speckle as dominant): {top1:.1f}%")
print(f"Mean GT speckle weight: {all_speckle_gt.mean():.3f}")
print(f"Mean predicted speckle weight: {all_speckle_pred.mean():.3f}")
print()

# Check confusion: when GT is speckle-dominant, what does model predict?
confusion = {'speckle': 0, 'banding': 0, 'gaussian': 0, 'shot': 0}
for noisy_path, gt_weights in speckle_dominant:
    img = np.array(Image.open(noisy_path).convert('L'), dtype=np.float32) / 255.0
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

    dom_names = ['speckle', 'banding', 'gaussian', 'shot']
    pred_dom_name = dom_names[np.argmax(pred)]
    confusion[pred_dom_name] += 1

print("Confusion matrix (GT=speckle, Pred=?):")
for noise_type, count in confusion.items():
    pct = 100.0 * count / len(speckle_dominant)
    print(f"  {noise_type}: {count}/{len(speckle_dominant)} ({pct:.1f}%)")
print()

# Check speckle-absent samples - does model avoid false positives?
print("=" * 70)
print("SPECKLE-ABSENT SAMPLES (GT speckle < 0.1) - checking false positives:")
print("=" * 70)

false_positives = 0
absent_speckle_pred = []
for noisy_path, gt_weights in speckle_absent[:50]:
    img = np.array(Image.open(noisy_path).convert('L'), dtype=np.float32) / 255.0
    h, w = img.shape
    top = (h - 64) // 2
    left = (w - 64) // 2
    img = img[top:top+64, left:left+64]

    x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred_dict, features = model(x)

    pred_speckle = pred_dict['speckle'][0].item()
    absent_speckle_pred.append(pred_speckle)

    if pred_speckle > 0.3:  # False positive
        false_positives += 1

print(f"False positive rate (pred > 0.3 when GT < 0.1): {false_positives}/{len(speckle_absent[:50])} ({100.0*false_positives/len(speckle_absent[:50]):.1f}%)")
print(f"Mean predicted speckle weight on speckle-absent: {np.mean(absent_speckle_pred):.3f}")
