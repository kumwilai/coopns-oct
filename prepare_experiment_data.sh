#!/bin/bash
# Prepare all data splits for experiments

echo "Creating experiment data splits..."
mkdir -p experiments/data_splits

# ============================================================================
# 1. FEW-SHOT ADAPTATION SPLITS
# ============================================================================
echo "Creating few-shot splits..."
head -50 train_pairs_universal.txt > experiments/data_splits/train_fewshot_50.txt
head -100 train_pairs_universal.txt > experiments/data_splits/train_fewshot_100.txt
head -200 train_pairs_universal.txt > experiments/data_splits/train_fewshot_200.txt
head -500 train_pairs_universal.txt > experiments/data_splits/train_fewshot_500.txt

# ============================================================================
# 2. CROSS-NOISE GENERALIZATION SPLITS
# ============================================================================
echo "Creating cross-noise splits..."

# Training sets (single noise type)
grep "noisy_gaussian" train_pairs_universal.txt > experiments/data_splits/train_gaussian_only.txt
grep "noisy_rayleigh" train_pairs_universal.txt > experiments/data_splits/train_rayleigh_only.txt
grep "noisy_poisson" train_pairs_universal.txt > experiments/data_splits/train_poisson_only.txt

# Validation sets (different noise types)
grep "noisy_gaussian" val_pairs_universal.txt > experiments/data_splits/val_gaussian_only.txt
grep "noisy_rayleigh" val_pairs_universal.txt > experiments/data_splits/val_rayleigh_only.txt
grep "noisy_poisson" val_pairs_universal.txt > experiments/data_splits/val_poisson_only.txt
grep "noisy_moderate_gamma" val_pairs_universal.txt > experiments/data_splits/val_moderate_gamma_only.txt
grep "noisy_heavy_gamma" val_pairs_universal.txt > experiments/data_splits/val_heavy_gamma_only.txt

# ============================================================================
# 3. CROSS-PATHOLOGY GENERALIZATION SPLITS
# ============================================================================
echo "Creating cross-pathology splits..."

# Training sets (single pathology)
grep "/normal/" train_pairs_universal.txt > experiments/data_splits/train_normal_only.txt
grep "/dme/" train_pairs_universal.txt > experiments/data_splits/train_dme_only.txt
grep "/cnv/" train_pairs_universal.txt > experiments/data_splits/train_cnv_only.txt
grep "/drusen/" train_pairs_universal.txt > experiments/data_splits/train_drusen_only.txt

# Validation sets (different pathologies)
grep "/normal/" val_pairs_universal.txt > experiments/data_splits/val_normal_only.txt
grep "/dme/" val_pairs_universal.txt > experiments/data_splits/val_dme_only.txt
grep "/cnv/" val_pairs_universal.txt > experiments/data_splits/val_cnv_only.txt
grep "/drusen/" val_pairs_universal.txt > experiments/data_splits/val_drusen_only.txt

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "========================================================================"
echo "DATA SPLITS CREATED"
echo "========================================================================"
echo ""
echo "Few-shot splits:"
wc -l experiments/data_splits/train_fewshot_*.txt
echo ""
echo "Cross-noise training splits:"
wc -l experiments/data_splits/train_*_only.txt | grep -E "(gaussian|rayleigh|poisson)"
echo ""
echo "Cross-noise validation splits:"
wc -l experiments/data_splits/val_*_only.txt | grep -E "(gaussian|rayleigh|poisson|gamma)"
echo ""
echo "Cross-pathology training splits:"
wc -l experiments/data_splits/train_*_only.txt | grep -E "(normal|dme|cnv|drusen)"
echo ""
echo "Cross-pathology validation splits:"
wc -l experiments/data_splits/val_*_only.txt | grep -E "(normal|dme|cnv|drusen)"
echo ""
echo "========================================================================"
