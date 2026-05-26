"""
Analyze GT event structure across all 4 carp sources.

For each source:
  - Event duration: end_index - start_index
  - Inter-event gap: start[i+1] - end[i]  (consecutive events in same file)

Reports percentile distributions and the bimodal split (if any) that
separates within-crossing dips from between-crossing gaps.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

BASE = _ROOT / "data" / "raw" / "carp"

SOURCES = {
    "salted_water":      BASE / "carp_salted/salted_water/output_report.csv",
    "salted_water_test": BASE / "carp_salted/salted_water/test/output_report.csv",
    "ACQ":               BASE / "ACQ_5_3_2026/output_report.csv",
    "carp_old":          BASE / "carp_old/output_report.csv",
}

all_durations = []
all_gaps      = []

for src, rep_path in SOURCES.items():
    rep   = pd.read_csv(rep_path)
    valid = rep[rep["is_valid"] == True] if "is_valid" in rep.columns else pd.DataFrame()
    if len(valid) == 0:
        print(f"[{src}]  no valid events")
        continue

    src_durations = []
    src_gaps      = []

    for fname, grp in valid.groupby("raw_data_file_name"):
        evs = grp.sort_values("start_index")
        starts = evs["start_index"].values
        ends   = evs["end_index"].values
        durs   = ends - starts
        src_durations.extend(durs.tolist())
        if len(evs) > 1:
            gaps = starts[1:] - ends[:-1]
            src_gaps.extend(gaps.tolist())

    all_durations.extend(src_durations)
    all_gaps.extend(src_gaps)

    def pct(arr, p): return int(np.percentile(arr, p)) if len(arr) else "-"

    print(f"\n[{src}]  events={len(src_durations)}  files-with-consecutive-events={len(src_gaps)>0}")
    print(f"  Event duration (samples @ 40Hz)")
    if src_durations:
        d = np.array(src_durations)
        print(f"    min={d.min():4d}  p10={pct(d,10):4d}  p25={pct(d,25):4d}  median={pct(d,50):4d}"
              f"  p75={pct(d,75):4d}  p90={pct(d,90):4d}  p99={pct(d,99):4d}  max={d.max():4d}")
        print(f"    seconds: min={d.min()/40:.2f}  median={np.median(d)/40:.2f}  max={d.max()/40:.2f}")
    if src_gaps:
        g = np.array(src_gaps)
        print(f"  Inter-event gap (samples @ 40Hz)  n={len(g)}")
        print(f"    min={g.min():4d}  p5={pct(g,5):4d}  p10={pct(g,10):4d}  p25={pct(g,25):4d}"
              f"  median={pct(g,50):4d}  p75={pct(g,75):4d}  p90={pct(g,90):4d}  max={g.max():4d}")
        print(f"    seconds: min={g.min()/40:.2f}  p10={np.percentile(g,10)/40:.2f}"
              f"  median={np.median(g)/40:.2f}  max={g.max()/40:.2f}")
        neg = (g < 0).sum()
        if neg:
            print(f"    WARNING: {neg} overlapping/negative gaps (events overlap in GT)")

print("\n" + "="*60)
print("COMBINED (all sources)")
if all_durations:
    d = np.array(all_durations)
    g = np.array(all_gaps)
    print(f"  Total events    : {len(d)}")
    print(f"  Total gaps      : {len(g)}")
    print(f"\n  Duration (samples):")
    print(f"    p5={int(np.percentile(d,5))}  p25={int(np.percentile(d,25))}"
          f"  median={int(np.median(d))}  p75={int(np.percentile(d,75))}"
          f"  p95={int(np.percentile(d,95))}  p99={int(np.percentile(d,99))}")
    print(f"    in seconds: median={np.median(d)/40:.2f}s  p95={np.percentile(d,95)/40:.2f}s")
    print(f"\n  Inter-event gap (samples):")
    print(f"    p1={int(np.percentile(g,1))}  p5={int(np.percentile(g,5))}"
          f"  p10={int(np.percentile(g,10))}  p25={int(np.percentile(g,25))}"
          f"  median={int(np.median(g))}  p75={int(np.percentile(g,75))}"
          f"  p95={int(np.percentile(g,95))}")
    print(f"    in seconds: p5={np.percentile(g,5)/40:.2f}s  median={np.median(g)/40:.2f}s"
          f"  p95={np.percentile(g,95)/40:.2f}s")
    neg = (g < 0).sum()
    print(f"    overlapping events (gap<0): {neg} ({100*neg/len(g):.1f}%)")

    print(f"\n  Gap histogram (samples):")
    bins = [0, 20, 40, 80, 120, 200, 400, 800, 2000, 99999]
    labels = ["0-20","20-40","40-80","80-120","120-200","200-400","400-800","800-2000","2000+"]
    counts, _ = np.histogram(g[g >= 0], bins=bins)
    for lbl, cnt in zip(labels, counts):
        bar = "#" * int(cnt / max(counts) * 40)
        print(f"    {lbl:>10s}: {cnt:4d}  {bar}")
