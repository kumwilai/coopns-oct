"""Check shot noise detection accuracy specifically."""
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
ckpt = torch.load("checkpoints/hybrid_analyzer_shot_fixed_seed1.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model.eval()

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

# Filter for shot-dominant samples (GT shot > 0.5)
shot_dominant = []
shot_present = []  # GT shot > 0.3
shot_minor = []   # GT shot > 0.1
shot_absent = []  # GT shot < 0.1

for noisy_path, weights in pairs:
    shot_weight = weights['shot']
    if shot_weight > 0.5:
        shot_dominant.append((noisy_path, weights))
    elif shot_weight > 0.3:
        shot_present.append((noisy_path, weights))
    elif shot_weight > 0.1:
        shot_minor.append((noisy_path, weights))
    else:
        shot_absent.append((noisy_path, weights))

print(f"Shot-dominant samples (GT > 0.5): {len(shot_dominant)}")
print(f"Shot-present samples (0.3 < GT < 0.5): {len(shot_present)}")
print(f"Shot-minor samples (0.1 < GT < 0.3): {len(shot_minor)}")
print(f"Shot-absent samples (GT < 0.1): {len(shot_absent)}")
print()

# Evaluate on shot-dominant samples
print("=" * 70)
print("SHOT-DOMINANT SAMPLES (GT shot > 0.5):")
print("=" * 70)

shot_gt = []
shot_pred = []
correct_top1 = 0

for i, (noisy_path, gt_weights) in enumerate(shot_dominant[:20]):  # Show first 20
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

    shot_gt.append(gt[3])
    shot_pred.append(pred[3])

    gt_dom = np.argmax(gt)
    pred_dom = np.argmax(pred)
    match = "✓" if gt_dom == pred_dom else "✗"
    if gt_dom == pred_dom:
        correct_top1 += 1

    dom_names = ['speckle', 'banding', 'gaussian', 'shot']

    print(f"Sample {i+1}:")
    print(f"  GT:   speckle={gt[0]:.3f}, banding={gt[1]:.3f}, gaussian={gt[2]:.3f}, shot={gt[3]:.3f}")
    print(f"  Pred: speckle={pred[0]:.3f}, banding={pred[1]:.3f}, gaussian={pred[2]:.3f}, shot={pred[3]:.3f}")
    print(f"  Shot error: {abs(pred[3] - gt[3]):.4f}")
    print(f"  Dominant: GT={dom_names[gt_dom]} vs Pred={dom_names[pred_dom]} {match}")
    print()

# All shot-dominant samples
all_shot_gt = []
all_shot_pred = []
all_correct = 0

for noisy_path, gt_weights in shot_dominant:
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

    all_shot_gt.append(gt[3])
    all_shot_pred.append(pred[3])

    if np.argmax(pred) == np.argmax(gt):
        all_correct += 1

print("=" * 70)
print(f"SHOT-DOMINANT STATISTICS (all {len(shot_dominant)} samples):")
print("=" * 70)
all_shot_gt = np.array(all_shot_gt)
all_shot_pred = np.array(all_shot_pred)
mae = np.mean(np.abs(all_shot_pred - all_shot_gt))
corr = np.corrcoef(all_shot_gt, all_shot_pred)[0, 1]
top1 = 100.0 * all_correct / len(shot_dominant)

print(f"Shot weight MAE: {mae:.4f}")
print(f"Shot weight correlation: {corr:.3f}")
print(f"Top-1 accuracy (shot as dominant): {top1:.1f}%")
print(f"Mean GT shot weight: {all_shot_gt.mean():.3f}")
print(f"Mean predicted shot weight: {all_shot_pred.mean():.3f}")
print()

# Check confusion: when GT is shot-dominant, what does model predict?
confusion = {'speckle': 0, 'banding': 0, 'gaussian': 0, 'shot': 0}
for noisy_path, gt_weights in shot_dominant:
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

print("Confusion matrix (GT=shot, Pred=?):")
for noise_type, count in confusion.items():
    pct = 100.0 * count / len(shot_dominant)
    print(f"  {noise_type}: {count}/{len(shot_dominant)} ({pct:.1f}%)")
print()

# Check shot-absent samples - does model avoid false positives?
print("=" * 70)
print("SHOT-ABSENT SAMPLES (GT shot < 0.1) - checking false positives:")
print("=" * 70)

false_positives = 0
absent_shot_pred = []
for noisy_path, gt_weights in shot_absent[:50]:
    img = np.array(Image.open(noisy_path).convert('L'), dtype=np.float32) / 255.0
    h, w = img.shape
    top = (h - 64) // 2
    left = (w - 64) // 2
    img = img[top:top+64, left:left+64]

    x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred_dict, features = model(x)

    pred_shot = pred_dict['shot'][0].item()
    absent_shot_pred.append(pred_shot)

    if pred_shot > 0.3:  # False positive
        false_positives += 1

print(f"False positive rate (pred > 0.3 when GT < 0.1): {false_positives}/{len(shot_absent[:50])} ({100.0*false_positives/len(shot_absent[:50]):.1f}%)")
print(f"Mean predicted shot weight on shot-absent: {np.mean(absent_shot_pred):.3f}")
