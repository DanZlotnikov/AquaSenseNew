# carp_presence_v4

**F/B envelope channels — explicit ring co-activation cue.**

## Motivation
Empirical analysis of ACQ GT events shows that 75% of real fish crossings produce strong
activation (|z| > 1.0) in BOTH the front ring and the back ring simultaneously, compared
to only 30% of random noise windows. The waveform-shape correlation between rings is not
present (Pearson r ≈ -0.03 for both events and random), so the discriminative signal is
purely amplitude-based co-activation.

## Architecture
Same single-branch temporal CNN as v1–v3, but input width is **14 channels** instead of 12.

Two engineered channels appended to every window after normalisation:
- **ch 13 — F_env(t)**: max |z-score| across the 6 front electrodes (F15 F37 F18 F32 F45 F67) at time t
- **ch 14 — B_env(t)**: max |z-score| across the 6 back electrodes (B15 B37 B18 B32 B45 B67) at time t

These give the model an explicit, compact representation of "how activated is each ring
right now", so it can learn to require both envelopes to be elevated at the same time
without encoding the rule in a hard threshold.

## Dataset
Same split and recordings as v3 (`data/carp_presence_v4/`).
Window shape: `(39, 14)` — 12 raw channels + 2 envelope channels.

## Performance vs v3 (ACQ_5_3_2026, threshold=0.5, merge_gap=79)
|                          | v3     | v4     |
|--------------------------|--------|--------|
| Total GT detections      | 427    | 427    |
| Total ML detections      | 736    | 809    |
| Over-count (ML − GT)     | +309   | +382   |
| Over-count %             | +72.4% | +89.5% |
| GT=0 files with FPs      | 15     | 24     |
| GT>0 files missed        | 0      | 0      |
| Best val_loss (ensemble) | 0.173  | 0.206  |

**v4 did not improve over v3.** The envelope channels carried information that was already
implicit in the raw electrode channels, and the 3-seed ensemble converged to a slightly
worse solution (best seed val_loss 0.206 vs 0.173 for v3). The FP rate increased
rather than decreased.

## Lessons
- The F/B envelope idea is sound but may require a stronger architectural inductive bias
  (e.g. a separate "co-activation" head, or a multiplicative interaction between F_env and
  B_env) rather than just feeding them as extra input channels.
- Alternatively, a custom loss term penalising detections where min(F_env_peak, B_env_peak)
  is below a threshold could enforce the co-activation requirement at training time.
- Production inference continues to use **carp_presence_v3**.
