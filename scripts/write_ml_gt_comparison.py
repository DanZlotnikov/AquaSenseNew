import json
from pathlib import Path

with open("ml_vs_gt_rows.json") as f:
    data = json.load(f)

rows        = data["rows"]
total_gt    = data["total_gt"]
total_ml    = data["total_ml"]
total_diff  = data["total_diff"]
overall_pct = data["overall_pct"]

lines = []
lines.append("ML vs Ground Truth — ACQ_5_3_2026")
lines.append("Detection model: carp-only ensemble (20260517_*), threshold=0.50")
lines.append("GT source: output_report.csv, is_valid==True")
lines.append("=" * 72)
lines.append(f"  {'File':<22}  {'GT':>6}  {'ML':>6}  {'Diff':>7}  {'Diff %':>8}")
lines.append("-" * 72)

for fname, gt_n, ml_n, diff, pct in rows:
    if pct == float("inf"):
        pct_str = "   inf %"
    else:
        pct_str = f"{pct:+7.1f}%"
    lines.append(f"  {fname:<22}  {gt_n:>6}  {ml_n:>6}  {diff:>+7}  {pct_str}")

lines.append("-" * 72)
lines.append(f"  {'TOTAL':<22}  {total_gt:>6}  {total_ml:>6}  {total_diff:>+7}  {overall_pct:+.1f}%")
lines.append("=" * 72)
lines.append("")
lines.append(f"Files in dataset : 115")
lines.append(f"Files with GT>0  : {sum(1 for r in rows if r[1]>0)}")
lines.append(f"Files with ML>0  : {sum(1 for r in rows if r[2]>0)}")
lines.append(f"Files GT=0,ML=0  : {sum(1 for r in rows if r[1]==0 and r[2]==0)}")
lines.append(f"Files GT=0,ML>0  : {sum(1 for r in rows if r[1]==0 and r[2]>0)}  (pure false positives)")
lines.append(f"Files GT>0,ML=0  : {sum(1 for r in rows if r[1]>0 and r[2]==0)}  (missed detections)")

out = "\n".join(lines)
out_path = Path(r"C:\Users\Admin\repos\AquaSenseNew\data\raw\carp\ACQ_5_3_2026\ml_vs_gt_comparison.txt")
out_path.write_text(out, encoding="utf-8")
print(out)
print(f"\nSaved to {out_path}")
