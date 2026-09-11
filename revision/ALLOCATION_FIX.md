# Allocation rule fix for the symbolic negotiator

File changed. `neuro_symbolic_corrector_v8_cooperative.py`, class `SymbolicNegotiator`.

## 1. What was wrong

The negotiator computes a per pixel allocation `a` that multiplies the corrector output
(`weighted_alpha = correction * allocation * lambda_val`). The released formula was

    a = clamp( sigmoid(1.5) + 0.9 w2 r2 + 0.8 w3 r3 - 0.05 w1 r1 (+ rule 4 and rule 5 terms), 0, 1 )

followed by a second multiplication by 1.5 and a second clamp. The positive budget is
0.818 + 0.884 + 0.762 = 2.46 before the boost, so the clamp at 1.0 is active on every
pixel. Measured on the released checkpoint the allocation map is the constant 1.0 on
100 percent of pixels and the gradient of the loss with respect to every negotiator
parameter is exactly zero. The rules had no effect on any output and were never trained.

Setting `LEGACY_SATURATING_ALLOCATION=1` restores that behavior bit for bit. This was
verified against the pre submission backup on twenty random inputs (single corrector and
three corrector cases) with `torch.equal` on the maps and equality on the rule traces.

## 2. New formula

Let r1 to r5 in [0,1] be the five Lukasiewicz rule activations (unchanged), let
w_k = sigmoid(rule_weight_k) in (0,1) be the five learnable rule weights (unchanged
parameters), and let beta be the learnable base logit (`base_allocation`, unchanged
parameter). The allocation is

    b = B_MIN + (B_MAX - B_MIN) * sigmoid(beta)

    a = b - C1 w1 r1 + C2 w2 r2 + C3 w3 r3 + C4 w4 r4 - C5 w5 r5

    a = clamp(a, 0, 1)

with the fixed budgets

    B_MIN = 0.25   B_MAX = 0.50
    C1 = 0.15  (rule 1, trust backbone, negative vote)
    C2 = 0.25  (rule 2, use corrector, positive vote)
    C3 = 0.15  (rule 3, failing predicate, positive vote)
    C4 = 0.10  (rule 4, balance, positive vote)
    C5 = 0.10  (rule 5, conflict, negative vote)

The rule 5 term is present only when two or more correctors are negotiated, exactly as in
the released code. The priority boost step in `forward` is a no op on this path (see
section 6).

Reading. Each named rule casts a signed vote whose size is its fixed budget C_k times its
learned importance w_k times its fuzzy truth value r_k. The base b is the allocation
when no rule fires. This is a weighted bounded sum of fuzzy truth values, which is the
Lukasiewicz reading of a weighted disjunction of the positive rules and a weighted
negation of the negative rules.

## 3. The clamp is inert

Because 0 <= r_k <= 1 and 0 < w_k < 1 for every k and every parameter value,

    a >= B_MIN - C1 - C5 = 0.25 - 0.15 - 0.10 = 0
    a <= B_MAX + C2 + C3 + C4 = 0.50 + 0.25 + 0.15 + 0.10 = 1

so the value before the clamp already lies in [0,1] and the clamp never changes it. The
bounds are approached only when every rule of one sign is fully true and every weight is
at the limit of the sigmoid, which cannot happen with finite logits. A randomised test of
2000 negotiators with logits drawn from N(0,16) and 64 random rule vectors each produced
zero values outside [0,1]. Saturation is therefore impossible, not merely rare.

## 4. Lipschitz bound and proof

Claim. For any two rule vectors r and r' at the same pixel and the same parameters

    | a(r) - a(r') |  <=  L * max_k | r_k - r'_k |

with

    L = C1 w1 + C2 w2 + C3 w3 + C4 w4 + C5 w5
      = 0.15 sigmoid(theta1) + 0.25 sigmoid(theta2) + 0.15 sigmoid(theta3)
      + 0.10 sigmoid(theta4) + 0.10 sigmoid(theta5)

where theta_k are the stored rule weight logits. When only one corrector is negotiated the
rule 5 term is absent and L drops its last summand.

Proof. Write s_k for the sign of rule k (minus for k in {1,5}, plus otherwise). Before the
clamp, a(r) - a(r') = sum_k s_k C_k w_k (r_k - r'_k). Taking absolute values and the
triangle inequality gives |a(r) - a(r')| <= sum_k C_k w_k |r_k - r'_k| <= (sum_k C_k w_k)
max_k |r_k - r'_k| = L ||r - r'||_inf. The clamp is 1 Lipschitz so it cannot enlarge the
difference, and by section 3 it is in fact the identity on this path. The bound is
attained by choosing r_k - r'_k = s_k t for any t, so L is the exact Lipschitz constant and
not merely an upper bound. The randomised test above confirmed no violation and a
measured ratio of 0.9945 of the bound, with the attained direction verified to machine
precision in float64.

Because w_k < 1, L < C1 + C2 + C3 + C4 + C5 = 0.75 for every parameter value. At the
released checkpoint weights (theta = 0.1, 4.0, 3.0, 1.5, 0.1) the value is L = 0.6014 with
all five rules and L = 0.5489 for the single corrector deployment used in the paper.
The method `SymbolicNegotiator.lipschitz_constant()` returns this number from the live
parameters so the paper can quote it from the code.

## 5. Why the negotiator parameters now receive gradient

The loss depends on the corrected image, which depends on `correction * a * lambda`. On
the legacy path a was the constant 1.0 produced by a clamp, and the derivative of a clamp
at its bound is zero, so the chain rule stopped at the clamp. On the new path the clamp is
the identity (section 3), so the derivative of the loss with respect to a is passed
through unchanged and then

    da/dbeta      = (B_MAX - B_MIN) * sigmoid'(beta)          nonzero for finite beta
    da/dtheta_k   = s_k C_k r_k sigmoid'(theta_k)              nonzero wherever r_k > 0
    da/dthreshold = s_k C_k w_k * dr_k/dthreshold              nonzero wherever the rule is
                                                              not saturated

Every rule activation is a product of sigmoids with slope 10, so dr_k/dthreshold is
nonzero on the soft band around each threshold, and the released maps show large soft
bands (rule 2 mean 0.60, rule 3 mean 0.88, rule 1 mean 0.01, rule 4 mean 0.03 on the
test image). Measured on a 96 by 96 crop of one PKU37 image with the released checkpoint
the absolute gradient sums are

    base_allocation 0.0502, low_confidence 0.584, predicate_failing 0.114,
    high_potential 0.0315, use_corrector 0.0057, boost_failing 0.0085,
    balance 0.0012, trust_nafnet 5.5e-8, high_confidence 1.1e-4, low_potential 1.1e-4

versus exactly 0.0 for every one of them on the legacy path. The small values for rule 1
reflect that rule 1 is nearly false on this image (mean 0.01), not a structural block.

Two honest caveats. First, rule 5 (conflict) and its two parameters `conservative` and
`conflict_threshold` only exist when two or more correctors are negotiated. The paper
deployment negotiates a single `gain` corrector so those two parameters cannot receive
gradient in that configuration, on either path. This is by construction of rule 5 and is
not part of the saturation defect. In the three corrector configuration both now receive
gradient (measured 30.6 and 257.1 on a random input) because the hard count
`(potential > threshold).float()` was replaced on the new path by the soft count
`sigmoid(10 (potential - threshold))`, which is the same soft comparison the other rules
already use. Second, `logic.sharpness` is used only by the parameterized t norm and is
unused under the Lukasiewicz t norm, so it never receives gradient. Neither caveat
should be described in the paper as a trained parameter.

## 6. Other changes on the new path

The priority boost in `forward` multiplied the allocation by up to 1.5 times
(1 + predicate deficit) with the deficit computed through `.item()`, so it carried no
gradient, it re saturated the clamp, and it duplicated rule 3, which already carries the
failing predicate signal per pixel and differentiably. On the new path this step is
skipped and the trace reports `priority_boost = 1.0`. On the legacy path it is untouched.

Rule 4 in the released code was `a * (1 - 0.3 e4) + 0.6 e4` with e4 = w4 r4. For a in [0,1]
this always increases a (its derivative in e4 is 0.6 - 0.3 a > 0), so it was a positive
vote in practice despite the comment about pulling toward a moderate value. The new path
makes that explicit as a bounded positive vote with budget C4 = 0.10.

## 7. The stored base value

The released checkpoint stores `base_allocation = 1.5`. The parameter is kept with the same
name and shape so `load_state_dict(strict=False)` loads all 456 tensors with nothing
missing and nothing unexpected. Its meaning is rescaled rather than its value. The new base
is b = 0.25 + 0.25 sigmoid(beta), so the stored 1.5 gives b = 0.454, which is inside the
allowed range and is a sensible warm start. No load time rewriting is done. A fresh model
initializes beta = 0.0 (b = 0.375). No parameter was added or removed, the negotiator has
13 scalar parameters before and after.

## 8. Chosen constants

| Constant | Value | One line justification |
|---|---|---|
| B_MIN | 0.25 | Equals C1 + C5 so the strongest negative votes bring the allocation exactly to 0 and never below, keeping the clamp inert. |
| B_MAX | 0.50 | Equals 1 - (C2 + C3 + C4) so the strongest positive votes bring the allocation exactly to 1 and never above. |
| C1 | 0.15 | Rule 1 (trust backbone) must be able to remove a visible share of the base. The legacy value 0.05 could move the allocation by at most 0.026 and was inert. |
| C2 | 0.25 | Rule 2 (use corrector) keeps the largest budget, matching the legacy ordering where it dominated (0.9). |
| C3 | 0.15 | Rule 3 (failing predicate) is the second largest positive vote, matching the legacy ordering (0.8) while leaving room under the bound. |
| C4 | 0.10 | Rule 4 (balance) is a small positive vote, matching its small legacy effect (0.3 and 0.6 scaled by w4 r4 with r4 rarely above 0.05). |
| C5 | 0.10 | Rule 5 (conflict) is a small negative vote, matching the legacy 0.2 multiplicative reduction, and is only active with two or more correctors. |
| Steepness 10 | 10 | Unchanged from the released soft comparisons. Reused for the rule 5 soft count so all five rules share one comparison operator. |
| Init beta | 0.0 | Gives b = 0.375, the midpoint of the base range, for training from scratch. |

Sum check. B_MAX + C2 + C3 + C4 = 1.00 and B_MIN - C1 - C5 = 0.00.

## 9. Measured results on the released checkpoint

Checkpoint `checkpointpaper/nafnet_pku37_cooperative.pth`, backbone
`checkpointpaper/nafnet_backbone.pth`, image `noisy/003105.tif` (first line of
`revision/pku37_subset40.jsonl`, 640 by 640), quantity `info["allocation_maps"]["gain"]`.

| Path | min | mean | max | std | fraction at 1.0 | fraction at 0.0 |
|---|---|---|---|---|---|---|
| Legacy (`LEGACY_SATURATING_ALLOCATION=1`) | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 |
| New (default) | 0.5513 | 0.7297 | 0.7974 | 0.0618 | 0.0000 | 0.0000 |

Cross image check on the first four subset images (new path). Means 0.7297, 0.7244,
0.7216, 0.7292 and per pixel standard deviations 0.0618, 0.0609, 0.0570, 0.0611, with no
pixel at either clamp in any image. The rule traces are identical between the two paths
because the rule activations themselves were never the problem, only their combination.

## 10. What this means for retraining

The correctors in the released checkpoint were trained with a constant allocation of 1.0.
On the new path the mean allocation on PKU37 is about 0.73, so the effective correction
strength is scaled down by that factor until the warm start fine tune adapts the corrector
and lambda predictor. All 456 tensors load, the negotiator now has nonzero gradient, and a
96 by 96 crop forward and backward pass runs in a few seconds on two CPU cores, so a warm
start from the released checkpoint is feasible. The numbers in the paper tables that depend
on the negotiator must be regenerated after that fine tune. The legacy switch remains
available so the original numbers can be reproduced exactly.
