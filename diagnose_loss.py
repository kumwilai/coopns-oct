#!/usr/bin/env python3
"""Diagnostic: Analyze loss calculation from training log."""

# Simulate the loss calculation based on the user's output
# From the logs:
# Batch 0050/1000 | Loss: 1584592463.3187 (Denoise: 0.1082, Interp: 1.1022, Param: 0.1036)

# Component losses (these are AVERAGES over 50 batches)
avg_denoise = 0.1082
avg_interp = 1.1022
avg_param = 0.1036

# Reported total loss (should be average too)
reported_avg_total = 1584592463.3187

# If these were correctly averaged, let's check what the sum would be
batches = 50  # At batch 50

# If avg_total = total_loss_sum / batches, then:
implied_total_loss_sum = reported_avg_total * batches
print(f"Batch 0050:")
print(f"  Reported avg_total: {reported_avg_total:,.2f}")
print(f"  Batches: {batches}")
print(f"  Implied total_loss_sum: {implied_total_loss_sum:,.2f}")
print()

# Expected sum of component losses over 50 batches
expected_denoise_sum = avg_denoise * batches  # 5.41
expected_interp_sum = avg_interp * batches    # 55.11
expected_param_sum = avg_param * batches      # 5.18

print(f"Expected sums (if components are averaged correctly):")
print(f"  Denoise sum: {expected_denoise_sum:.2f}")
print(f"  Interp sum: {expected_interp_sum:.2f}")
print(f"  Param sum: {expected_param_sum:.2f}")
print(f"  Expected total (without extras): {expected_denoise_sum + expected_interp_sum + expected_param_sum:.2f}")
print()

# Difference - this is what the "extra" losses contribute
extra_loss_sum = implied_total_loss_sum - (expected_denoise_sum + expected_interp_sum + expected_param_sum)
print(f"Extra loss contribution (total_sum - component_sums):")
print(f"  Extra loss sum: {extra_loss_sum:,.2f}")
print(f"  Extra loss average per batch: {extra_loss_sum / batches:,.2f}")
print()

# Now let's analyze Batch 100
print("="*60)
print("Batch 0100:")
reported_avg_total_100 = 624873351534.0498
batches_100 = 100
implied_total_loss_sum_100 = reported_avg_total_100 * batches_100

avg_denoise_100 = 0.1081
avg_interp_100 = 1.1093
avg_param_100 = 0.1031

print(f"  Reported avg_total: {reported_avg_total_100:,.2f}")
print(f"  Implied total_loss_sum: {implied_total_loss_sum_100:,.2f}")
print()

# Check if total_loss_sum is growing exponentially
growth_factor = implied_total_loss_sum_100 / implied_total_loss_sum
print(f"Growth factor from batch 50 to 100: {growth_factor:,.2f}x")
print()

# Hypothesis: What if 'batches' is NOT being used for averaging?
# What if avg_total is actually just total_loss_sum?
print("="*60)
print("HYPOTHESIS CHECK:")
print()
print("If 'avg_total' is actually 'total_loss_sum' (BUG: missing division):")
print(f"  At batch 50: total_loss_sum would be {reported_avg_total:.2f}")
print(f"  At batch 100: total_loss_sum would be {reported_avg_total_100:.2f}")
print(f"  Component sums look correct (small values)")
print()
print("This suggests a bug where:")
print("  Line: avg_total = total_loss_sum / max(1, batches)")
print("  Is NOT being executed, or batches=0, or total_loss_sum is exploding")
