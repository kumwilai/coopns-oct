# Improving Top-1 Accuracy from 64% to 80%+

## Problem Analysis
Your previous run achieved only **64% Top-1 accuracy** because:

1. **Classification loss weight too low**: `0.1` → analyzer gets weak gradient signal
2. **Limited training data**: Only 200 samples → insufficient learning
3. **Short training**: Only 5 epochs → poor convergence
4. **Training from scratch**: No pre-trained weights → slower learning
5. **Imbalanced losses**: Usage loss (0.5) >> Classification loss (0.1)

## Solutions Provided

### Option 1: Balanced Training (Recommended First Try)
**Script**: `run_end_to_end_accurate.sh`

**Key Changes**:
- Classification loss: `0.1 → 1.0` (10x increase)
- Training samples: `200 → 500`
- Epochs: `5 → 15`
- **Uses pre-trained analyzer** for warm start
- Target: **Top-1 > 80%**

**Run**:
```bash
bash run_end_to_end_accurate.sh
```

### Option 2: Maximum Accuracy Mode (Aggressive)
**Script**: `run_end_to_end_max_accuracy.sh`

**Key Changes**:
- Classification loss: `0.1 → 2.0` (20x increase!)
- Usage loss: `0.5 → 0.3` (reduced to prioritize classification)
- Training samples: **ALL** (no limit)
- Epochs: `5 → 20`
- Lower learning rate: `1e-4 → 5e-5` (fine-tuning)
- **Uses pre-trained analyzer**
- Target: **Top-1 > 85%**

**Run**:
```bash
bash run_end_to_end_max_accuracy.sh
```

## New Monitoring Features

### 1. Per-Class Accuracy Breakdown
Now you'll see which noise types are being misclassified:

```
Top-1 Accuracy:    75.0%
Per-class Accuracy:
  speckle : 85.0%  (n=20)
  banding : 60.0%  (n=15)  ← problematic!
  gaussian: 80.0%  (n=10)
  shot    : 75.0%  (n=5)
```

This helps you identify:
- Which noise types are hard to classify
- If the dataset is imbalanced
- Where to focus improvements

### 2. Gain Monitoring
Every epoch shows your method vs. baseline:

```
📊 Gain over noisy: +5.32 dB
🎯 Gain over base:  +1.24 dB  ← KEY METRIC
```

## Understanding the Loss Weights

The total loss is:
```
Total = Reconstruction + (Usage × 0.5) + (Classification × weight)
```

**Original (64% accuracy)**:
- Reconstruction: ~0.01
- Usage: ~0.001 × 0.5 = 0.0005
- Classification: ~1.0 × 0.1 = 0.1
- **Classification contributes only ~10% of total loss**

**Improved (80%+ accuracy)**:
- Reconstruction: ~0.01
- Usage: ~0.001 × 0.5 = 0.0005
- Classification: ~1.0 × 1.0 = 1.0
- **Classification contributes ~50% of total loss**

## Expected Results

### Original Configuration (64% Top-1)
```
CLASSIFY_LOSS=0.1
EPOCHS=5
MAX_TRAIN_SAMPLES=200
ANALYZER_INIT=""  # From scratch
```

### Accurate Configuration (80%+ Top-1)
```
CLASSIFY_LOSS=1.0
EPOCHS=15
MAX_TRAIN_SAMPLES=500
ANALYZER_INIT="checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth"
```

### Max Accuracy Configuration (85%+ Top-1)
```
CLASSIFY_LOSS=2.0
USAGE_LOSS=0.3  # Reduced
EPOCHS=20
MAX_TRAIN_SAMPLES=ALL
ANALYZER_INIT="checkpoints/hybrid_analyzer_speckle_fixed_seed2.pth"
LR=5e-5  # Lower for fine-tuning
```

## Troubleshooting

### If accuracy is still below 75%:
1. **Check per-class accuracy** - is one class dragging it down?
2. **Increase `CLASSIFY_LOSS` to 3.0** or higher
3. **Reduce `USAGE_LOSS` to 0.1** - give classification more weight
4. **Train longer** - try 30+ epochs
5. **Check analyzer initialization** - make sure pre-trained weights loaded

### If denoising quality drops:
1. **Don't reduce reconstruction loss** (keep at 1.0 implicit)
2. **Increase `USAGE_LOSS` back to 0.5**
3. **Reduce `CLASSIFY_LOSS` to 1.5**
4. **Balance is key**: You need both good classification AND good denoising

## Comparison Table

| Configuration | Classify Loss | Epochs | Samples | Pre-trained | Expected Top-1 | Training Time |
|---------------|---------------|--------|---------|-------------|----------------|---------------|
| **Original**  | 0.1           | 5      | 200     | No          | ~64%           | Fast          |
| **Accurate**  | 1.0           | 15     | 500     | Yes         | ~80%           | Medium        |
| **Max Acc**   | 2.0           | 20     | ALL     | Yes         | ~85%+          | Slower        |

## What Changed in Code

1. **`train_end_to_end.py`**:
   - Added base NAFNet comparison in validation
   - Added per-class accuracy tracking
   - Enhanced metrics display with per-class breakdown
   - Checkpoint saves gain metrics

2. **New Scripts**:
   - `run_end_to_end_accurate.sh` - balanced accuracy improvement
   - `run_end_to_end_max_accuracy.sh` - maximum accuracy mode

## Next Steps

1. **Start with**: `bash run_end_to_end_accurate.sh`
2. **Monitor**: Watch per-class accuracy each epoch
3. **Identify**: Which noise types have low accuracy?
4. **Adjust**: If needed, try max accuracy mode
5. **Iterate**: Tune loss weights based on your specific needs

Good luck achieving 80%+ Top-1 accuracy! 🎯
