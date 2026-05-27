# carp_presence_v3

**Guard-zone fix — correct negative sampling around GT events.**

## Architecture
Identical to v1/v2: single-branch 12-channel temporal CNN.
Input: `(batch, 39, 12)`.

## Key change from v2
**Correct guard zone**: negative candidates are excluded from the full GT event span
`[event_start − GUARD, event_end + GUARD]` instead of just `[center − WINDOW_SIZE//2, center + WINDOW_SIZE//2]`.

This eliminates contradictory training examples: windows from the outer edge of a long event
were previously labelled negative despite containing fish signal, producing multiple probability
peaks per crossing and inflated false-positive counts at inference.

## Performance
3-seed ensemble. Best val_loss: seed1=0.173, seed2=0.220, seed3=0.227.

## Known limitations
Model sees only raw electrode amplitudes. It has no explicit representation of whether the
disturbance activates both the front (F) ring and back (B) ring simultaneously — the physically
expected signature of a real carp crossing the antenna aperture. This is addressed in v4.
