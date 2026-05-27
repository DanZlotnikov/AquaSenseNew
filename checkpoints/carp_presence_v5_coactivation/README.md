# carp_presence_v5_coactivation

**Additive co-activation gate on min(F_peak, B_peak).**

## Motivation
Same as v5_knockdown: reduce single-electrode FPs by incorporating the physical
constraint that a real carp crossing activates BOTH rings simultaneously.

## Architecture
12-channel temporal CNN + learned co-activation gate added to the presence logit.

After the temporal branch produces a base logit, the gate computes:
```
f_peak = max(|z|) over time, across front 6 electrodes  → clamped to [0, 1] (peak/5)
b_peak = max(|z|) over time, across back  6 electrodes  → clamped to [0, 1] (peak/5)
joint_min = min(f_peak, b_peak)                          → scalar in [0, 1]
gate = Linear(1→1)(joint_min)                            → additive logit correction
logit_presence = base_logit + gate
```

**Init**: `weight=2.0, bias=-1.0` — gate=-1 when inactive (suppresses detection),
gate=0 at joint_min=0.5, gate=+1 when both rings fully active.

The gate is learned end-to-end alongside the temporal CNN.

## Dataset
Same as v3 (`data/carp_presence_v3/`). The co-activation constraint is purely
architectural — no data augmentation.

## Performance vs v3 (ACQ_5_3_2026, threshold=0.5, merge_gap=79)
|                          | v3     | v5_coactivation |
|--------------------------|--------|-----------------|
| Total ML detections      | 736    | 710             |
| Over-count %             | +72.4% | +66.3%          |
| GT=0 files with FPs      | 15     | 15              |
| GT>0 files missed        | 0      | 0               |

Slightly fewer total detections than v3, but the number of FP files is unchanged at 15.
The gate reduced over-counting but did not filter out the specific noise events that
caused whole-file FPs. **v5_knockdown is preferred for production** (4 FP files vs 15).

## Why the knockdown approach worked better
The gate is additive and can be compensated by the base logit — if the temporal CNN
learns a high logit for a single-ring event, the gate can only add a fixed negative
offset, not suppress it entirely. Knockdown negatives change *what the temporal CNN
learns*, which is a deeper fix.
