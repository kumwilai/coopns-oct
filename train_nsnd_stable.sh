#!/bin/bash
# Train NSND with stable losses (no speckle cycle explosion)

python nsnd_oct/scripts/train_hybrid_nsnd_multitask.py \
    --pairs_train /home/kumwilai/OCT/train_pairs_duke_analysis.txt \
    --pairs_val /home/kumwilai/OCT/val_pairs_duke_analysis.txt \
    --weights_jsonl_train /home/kumwilai/OCT/weights_duke_analysis_train.jsonl \
    --weights_jsonl_val /home/kumwilai/OCT/weights_duke_analysis_val.jsonl \
    --max_samples 2000 \
    --val_samples 400 \
    --batch_size 4 \
    --epochs 50 \
    --use_log_domain_analyzer \
    --use_log_domain_speckle \
    --speckle_cycle_weight 0.0 \
    --composition_loss_weight 0.2 \
    --composition_consistency_weight 0.05 \
    --composition_group_size 2 \
    --use_joint_signal_expert \
    --joint_expert_channels 16 \
    --ns_use_neural_predicates \
    --ns_use_neural_weights \
    --shared_residual \
    --shared_trunk_width 24 \
    --shared_adapter_channels 8 \
    --shared_adapter_hidden 8 \
    --base_nafnet_width 32 \
    --nafnet_width 24 \
    --noise_cycle_weight 0.01 \
    --noise_cycle_use_true \
    --param_reg_weight 0.1 \
    --freeze_analyzer_epochs 50 \
    --hybrid_analyzer_ckpt checkpoints/hybrid_cnn_symbolic_duke_analysis_twostage2_seed0.pth \
    --seed 0

# Output will be saved to:
# checkpoints/multitask_hybrid_nsnd_lambda0p5to0p1_linear_best.pth
