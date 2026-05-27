# carp_presence_v2

**Multi-source dataset + sigma-floor normalisation.**

## Architecture
Identical to v1: single-branch 12-channel temporal CNN.
Input: `(batch, 39, 12)`.

## Key changes from v1
1. **All four data sources** included: `salted_water`, `salted_water_test`, `ACQ_5_3_2026`, `carp_old`.
2. **Sigma-floor normalisation** (`σ_eff = max(σ, 150)`): prevents flatline recordings from having their noise amplified to fish-like amplitude.
3. **GT=0 recordings as negatives** (50% of negatives per split): forces the model to learn that flat/quiet backgrounds are not fish events.
4. **Recording-level train/val/test split** with all three environments represented in every split.

## Split
Train ~544 pos (69%) | Val ~103 (13%) | Test ~139 (18%) | Neg ratio 4:1.

## Known issues fixed in v3
Guard zone around GT events used center ± WINDOW_SIZE//2, but GT events can be up to 146 samples wide.
Windows from the outer edges of long events were sampled as negatives despite overlapping fish signal,
causing multiple probability peaks per crossing and double-counting at inference.
