"""Score a G2VLM predictions JSON with the benchmark-faithful metrics.

  python score_results.py --predictions preds_vsi.json --benchmark vsi
"""
import argparse
import json

import metrics as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--benchmark", choices=["vsi", "vsti"], required=True)
    ap.add_argument("--out", default=None, help="optional JSON path for the score summary")
    args = ap.parse_args()

    payload = json.load(open(args.predictions))
    records = payload["results"] if isinstance(payload, dict) else payload

    scored = [M.process_results(dict(r), r["prediction"], args.benchmark) for r in records]
    overall, per_type = M.aggregate_results(scored, args.benchmark)

    print(f"\n=== {args.benchmark.upper()}-Bench | G2VLM | n={len(records)} ===")
    for k in sorted(per_type):
        print(f"  {k:<48} {per_type[k]*100:6.2f}")
    print(f"  {'OVERALL':<48} {overall:6.2f}")

    summary = {
        "benchmark": args.benchmark,
        "n": len(records),
        "overall": overall,
        "per_type": {k: per_type[k] * 100 for k in per_type},
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nwrote {args.out}")
    return summary


if __name__ == "__main__":
    main()
