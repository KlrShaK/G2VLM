"""Materialize the canonical random-200 subset of VSI-Bench / VSTI-Bench to JSON.

Run this in the *baseline* envs so the subset is identical to what SpatialStack /
VLM-3R evaluate on:
  - VSI : envs/spatialstack-qwen35  (datasets 3.6.0)
  - VSTI: envs/vlm3r_env            (datasets 2.16.1)

Both subsets are `dataset.shuffle(seed=42).select(range(200))`, which is exactly
SpatialStack's `process_docs_random200`. The output JSON is a plain list of docs
consumed by eval_g2vlm_bench.py (which runs in g2vlm_env, no `datasets` needed).
"""
import argparse
import ast
import json
import os

import datasets

VSI_VIDEO_ROOT = "/cluster/scratch/spanwar/datasets/eval_cache/hf-home/vsibench"
VSI_CACHE_DIR = "/cluster/scratch/spanwar/datasets/eval_cache/vsibench"
VSTI_VIDEO_ROOT = "/cluster/work/igp_psr/spanwar/datasets/vstibench"
VSTI_TEST_JSON = "/cluster/work/igp_psr/spanwar/datasets/vstibench/test.json"


def _norm_options(opts):
    """VSTI test.json stores options as a string repr of a list; HF as a real list."""
    if opts is None:
        return None
    if isinstance(opts, str):
        try:
            return list(ast.literal_eval(opts))
        except (ValueError, SyntaxError):
            return [opts]
    return list(opts)


def load_vsi():
    ds = datasets.load_dataset(
        "nyu-visionx/VSI-Bench", "full", cache_dir=VSI_CACHE_DIR, split="test"
    )
    ds = ds.shuffle(seed=42).select(range(min(200, len(ds))))
    out = []
    for d in ds:
        out.append({
            "id": d["id"],
            "benchmark": "vsi",
            "question": d["question"],
            "question_type": d["question_type"],
            "options": _norm_options(d["options"]),
            "ground_truth": d["ground_truth"],
            "mc_answer": None,
            "video_path": os.path.join(
                VSI_VIDEO_ROOT, d["dataset"], d["scene_name"] + ".mp4"
            ),
        })
    return out


def load_vsti():
    # Prefer the canonical HF dataset (matches VLM-3R's vstibench.yaml); fall back
    # to the local test.json if the HF cache is unavailable.
    try:
        ds = datasets.load_dataset("Journey9ni/vstibench", split="test")
    except Exception as e:  # noqa: BLE001
        print(f"[vsti] HF load failed ({e}); falling back to local test.json")
        ds = datasets.Dataset.from_list(json.load(open(VSTI_TEST_JSON)))
    ds = ds.shuffle(seed=42).select(range(min(200, len(ds))))
    out = []
    for d in ds:
        out.append({
            "id": d["id"],
            "benchmark": "vsti",
            "question": d["question"],
            "question_type": d["question_type"],
            "options": _norm_options(d.get("options")),
            "ground_truth": d["ground_truth"],
            "mc_answer": d.get("mc_answer"),
            "video_path": os.path.join(VSTI_VIDEO_ROOT, d["video_path"]),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["vsi", "vsti"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    docs = load_vsi() if args.benchmark == "vsi" else load_vsti()

    # Sanity checks
    from collections import Counter
    missing = [d["video_path"] for d in docs if not os.path.exists(d["video_path"])]
    types = Counter(d["question_type"] for d in docs)
    print(f"[{args.benchmark}] {len(docs)} docs")
    print(f"[{args.benchmark}] question_type distribution: {dict(types)}")
    if missing:
        print(f"[{args.benchmark}] WARNING: {len(missing)} videos missing, e.g. {missing[:3]}")
    else:
        print(f"[{args.benchmark}] all {len(docs)} videos present on disk")

    with open(args.out, "w") as f:
        json.dump(docs, f, indent=2)
    print(f"[{args.benchmark}] wrote {args.out}")


if __name__ == "__main__":
    main()
