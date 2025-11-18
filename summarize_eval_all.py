#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize EchoNet GP results into a single Markdown table.
- Robust to JSON schemas (string or float coverage keys like "0.90" or 0.9)
- Optional CLI to override default paths
- Writes Markdown table (summary_all.md) and a CSV (summary_all.csv)
- Prints the table to STDOUT as well

Usage (defaults assumed):
python summarize_eval_all.py \
  --base /mnt/4DHeartModel/experiments/EchoNet/outputs \
  --alphas 0.68 0.90 0.95

Or fully custom paths:
python summarize_eval_all.py \
  --uncal_eval   /path/eval_uncal/test/summary.json \
  --cal_eval     /path/eval_cal/test/summary.json \
  --uncal_temp   /path/temporal_uncal/test/summary_temporal.json \
  --cal_temp     /path/temporal_cal/test/summary_temporal.json \
  --out_md       /path/summary_all.md \
  --out_csv      /path/summary_all.csv
"""

from pathlib import Path
import argparse, json, csv
from typing import Dict, Any, List
import numpy as np

# ---------- helpers ----------

def load_json(p: Path) -> Dict[str, Any]:
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    with p.open('r') as f:
        return json.load(f)


def normalize_cov_keys(cov: Dict[Any, Any]) -> Dict[str, float]:
    """Return a dict with string keys normalized to 2-decimals when possible.
    Accepts keys like 0.9, '0.9', '0.90'.
    """
    out = {}
    for k, v in cov.items():
        try:
            # Try parse to float then back to str with up to 2 decimals
            kk = float(k)
            # keep 2 decimals (avoid scientific)
            key_str = f"{kk:.2f}".rstrip('0').rstrip('.') if kk % 1 != 0 else f"{int(kk)}"
        except Exception:
            key_str = str(k)
        out[key_str] = float(v)
    return out


def get_cov_value(cov_norm: Dict[str, float], alpha: float) -> float:
    """Fetch coverage value for a given alpha with multiple fallbacks."""
    # Try '0.68', '0.9', '0.90', '68%', etc.
    # Build candidate keys in priority order
    cands: List[str] = []
    # exact 2 decimals
    cands.append(f"{alpha:.2f}")
    # minimal decimal
    if alpha % 1 == 0:
        cands.append(f"{int(alpha)}")
    else:
        cands.append(str(alpha))              # e.g., '0.9'
        cands.append(f"{alpha:.1f}")         # e.g., '0.9'
    # percent style
    cands.append(f"{alpha*100:.0f}%")
    for k in cands:
        if k in cov_norm:
            return float(cov_norm[k])
    # final brute force: try float equality with small tol
    for k, v in cov_norm.items():
        try:
            if abs(float(k) - alpha) < 1e-9:
                return float(v)
        except Exception:
            continue
    return float('nan')


def extract_eval(summary: Dict[str, Any], alphas: List[float]) -> Dict[str, float]:
    cov_raw = summary.get('overall', {}).get('coverage_mean', {})
    cov = normalize_cov_keys(cov_raw)
    out = {
        'CRPS': float(summary.get('overall', {}).get('crps_mean', float('nan')))
    }
    for a in alphas:
        out[f"cov_{a}"] = get_cov_value(cov, a)
    return out


def extract_temporal(summary: Dict[str, Any]) -> Dict[str, float]:
    o = summary.get('overall_mean_drop_%', {})
    return {
        'TV1↓%': float(o.get('tv1', float('nan'))),
        'TV2↓%': float(o.get('tv2', float('nan'))),
        'HF↓%':  float(o.get('hf',  float('nan'))),
    }


def make_table(unc_eval: Dict[str, float], cal_eval: Dict[str, float],
               unc_temp: Dict[str, float], cal_temp: Dict[str, float],
               alphas: List[float]) -> List[List[str]]:
    rows = []
    header = ['Metric', 'Uncalibrated', 'Calibrated', 'Delta (Cal-Uncal)']
    rows.append(header)

    def fmt(x, prec=5):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return 'NA'
        return f"{x:.{prec}f}"

    # CRPS
    crps_u = unc_eval.get('CRPS', float('nan'))
    crps_c = cal_eval.get('CRPS', float('nan'))
    rows.append(['CRPS ↓', fmt(crps_u), fmt(crps_c), fmt(crps_c - crps_u)])

    # Coverage
    for a in alphas:
        key = f"cov_{a}"
        cu = unc_eval.get(key, float('nan'))
        cc = cal_eval.get(key, float('nan'))
        rows.append([f"Coverage@{a:.2f}", fmt(cu,3), fmt(cc,3), fmt(cc - cu,3)])

    # Temporal metrics
    for name, k in [('TV1↓%', 'TV1↓%'), ('TV2↓%', 'TV2↓%'), ('HF↓%', 'HF↓%')]:
        tu = unc_temp.get(k, float('nan'))
        tc = cal_temp.get(k, float('nan'))
        rows.append([name, fmt(tu,2), fmt(tc,2), fmt(tc - tu,2)])

    return rows


def write_markdown(rows: List[List[str]], out_md: Path):
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open('w') as f:
        # header
        f.write('| ' + ' | '.join(rows[0]) + ' |\n')
        f.write('| ' + ' | '.join(['---'] * len(rows[0])) + ' |\n')
        # body
        for r in rows[1:]:
            f.write('| ' + ' | '.join(r) + ' |\n')
    print(f"[SAVE] Markdown -> {out_md}")


def write_csv(rows: List[List[str]], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerows(rows)
    print(f"[SAVE] CSV -> {out_csv}")


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, default=Path('/mnt/4DHeartModel/experiments/EchoNet/outputs'))
    ap.add_argument('--uncal_eval', type=Path, default=None)
    ap.add_argument('--cal_eval',   type=Path, default=None)
    ap.add_argument('--uncal_temp', type=Path, default=None)
    ap.add_argument('--cal_temp',   type=Path, default=None)
    ap.add_argument('--alphas', type=float, nargs='*', default=[0.68, 0.90, 0.95])
    ap.add_argument('--out_md', type=Path, default=None)
    ap.add_argument('--out_csv', type=Path, default=None)
    args = ap.parse_args()

    base = args.base
    uncal_eval = args.uncal_eval or (base / 'eval_uncal/test/summary.json')
    cal_eval   = args.cal_eval   or (base / 'eval_cal/test/summary.json')
    uncal_temp = args.uncal_temp or (base / 'temporal_uncal/test/summary_temporal.json')
    cal_temp   = args.cal_temp   or (base / 'temporal_cal/test/summary_temporal.json')

    out_md = args.out_md or (base / 'summary_all.md')
    out_csv= args.out_csv or (base / 'summary_all.csv')

    eval_uncal = load_json(uncal_eval)
    eval_cal   = load_json(cal_eval)
    temp_uncal = load_json(uncal_temp)
    temp_cal   = load_json(cal_temp)

    unc = extract_eval(eval_uncal, args.alphas)
    cal = extract_eval(eval_cal,   args.alphas)
    t_unc = extract_temporal(temp_uncal)
    t_cal = extract_temporal(temp_cal)

    rows = make_table(unc, cal, t_unc, t_cal, args.alphas)

    # Print to stdout
    for r in rows:
        print('\t'.join(r))

    write_markdown(rows, out_md)
    write_csv(rows, out_csv)


if __name__ == '__main__':
    main()
