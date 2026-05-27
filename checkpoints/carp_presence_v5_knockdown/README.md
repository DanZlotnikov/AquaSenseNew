# carp_presence_v5_knockdown

**Single-ring knockdown augmentation — best model as of v5.**

## Motivation
v3/v4 could learn to fire on a single strong F or B electrode, because nothing
in the training data showed that single-ring activation is *not* a fish.
Knockdown augmentation directly creates those counterexamples.

## Architecture
Same 12-channel temporal CNN as v3. No architectural changes.

## Key change from v3
**Knockdown negatives** added to the training split only:
- For every positive (GT event) window, two synthetic negatives are created:
  - **F-knockdown**: zero out the 6 front electrode columns → single B-ring signal → label 0
  - **B-knockdown**: zero out the 6 back electrode columns → single F-ring signal → label 0
- Val and test sets are identical to v3, so metrics remain directly comparable.

Training set: 544 pos + 2176 original neg + 1088 knockdown neg = **3808 total (6:1 ratio)**.

## Performance vs v3 and v4 (ACQ_5_3_2026, threshold=0.5, merge_gap=79)
|                          | v3     | v4     | v5_knockdown | v5_coactivation |
|--------------------------|--------|--------|--------------|-----------------|
| Total GT detections      | 427    | 427    | 427          | 427             |
| Total ML detections      | 736    | 809    | **698**      | 710             |
| Over-count %             | +72.4% | +89.5% | **+63.5%**   | +66.3%          |
| GT=0 files with FPs      | 15     | 24     | **4**        | 15              |
| GT>0 files missed        | 0      | 0      | 0            | 0               |
| Best val_loss (ensemble) | 0.173  | 0.206  | 0.223        | 0.216           |

**v5_knockdown is the new production model.** FP files dropped from 15 → 4 while
over-detection improved from +72.4% → +63.5% and no events were missed.

## Why it works
The knockdown negatives show the model what it should NOT respond to: a
window where only one ring is active, even at high amplitude. v3 never saw such
examples labelled as negative, so it could learn "strong F electrode = fish" as a
sufficient condition. With knockdowns, single-ring activations are unambiguously
negative during training.

## Val_loss note
Val_loss is slightly higher than v3 (0.223 vs 0.173) because the val set is
unchanged (clean, no knockdowns) while training now includes harder examples.
This is expected and does not reflect lower quality — ACQ inference confirms
the improvement.
