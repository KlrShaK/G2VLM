# G2VLM on VSI-Bench & VSTI-Bench (random-200)

A self-contained harness to evaluate **G2VLM-2B-MoT** on the spatial-reasoning video-QA
benchmarks **VSI-Bench** (used by SpatialStack) and **VSTI-Bench** (introduced by VLM-3R),
on the canonical 200-sample random subsets, scored with each benchmark's *own* metric code.

> **TL;DR finding:** G2VLM is a few-view geometry model, not a video-QA model. It
> degenerates into repetitive gibberish beyond ~8 frames, and even when coherent it
> answers in depth-estimation prose instead of the requested option-letter / single-word
> format. **All four runs (VSI/VSTI × 8f/32f) score 0.00** vs SpatialStack's VSI
> random-200 baseline of 67.45. See [`RESULTS.md`](RESULTS.md) for full per-type tables.

---

## Why this is a faithful comparison

- **Subsets are the canonical ones.** Both are `dataset.shuffle(seed=42).select(200)` — the
  exact operation in SpatialStack's `process_docs_random200`. The VSI subset's sample ids
  were verified to match SpatialStack's `vsibench_random200` byte-for-byte.
- **Prompts are byte-identical** to each benchmark's `vsibench_doc_to_text`
  (pre-prompt + question + options + post-prompt). Verified: 0/400 mismatches.
- **Scoring is copied verbatim** from the benchmarks' `lmms_eval` task utils
  (`exact_match` for multiple-choice, Mean Relative Accuracy `MRA:.5:.95:.05` for numeric;
  VSI MCA vs `ground_truth`, VSTI MCA vs `mc_answer`). See `metrics.py`.
- **Frames** are sampled uniformly with decord (`np.linspace(0, N-1, k)`), identical to
  lmms-eval / `llava_onevision.load_video`. The benchmark standard is **32 frames**; VSTI
  questions explicitly reference "frame X of 32", so 32 is required for them to be meaningful.

## The key finding in detail

A frame-count sweep (1→32) and a resolution sweep (224/336/448/768 px) on sample
questions showed:

| frames | output |
|---|---|
| 1–4  | coherent (depth-style prose; occasionally a bare letter) |
| 8    | borderline coherent |
| 16   | degrades ("…the new observer's new observer…") |
| 32   | **collapses** ("…from the new new new new observer…") at *every* resolution |

So the failure tracks the **number of frames**, not the per-frame token budget — lowering
resolution does not help. Because of this we ran **both** a faithful 32-frame config
(documents the protocol-level result, ≈0 from degeneration) and a coherent 8-frame config
(shows the model's real behavior). Both score 0:

| Benchmark | @8f (coherent) | @32f (faithful) | Baseline (32f) |
|---|---:|---:|---:|
| VSI-Bench  | 0.00 | 0.00 | SpatialStack-Qwen3.5-4B: **67.45** |
| VSTI-Bench | 0.00 † | 0.00 | VLM-3R: no completed score in-repo |

† VSTI-8f is a coherence check only — its "frame X of 32" questions cannot be honored with
8 frames.

Even in the coherent 8-frame regime the 0.00 is genuine, not a parser artifact: 121/200 VSI
predictions begin with "The" (full sentences), only 4/200 are a clean option letter and
1/200 a bare number — and none were correct. G2VLM ignores the "answer with the letter /
single word" instruction and instead emits geometry prose such as
*"the printer is ~2.0 meters from the observer"*.

---

## Layout

```
eval_benchmarks/
├── dump_subset.py        # Stage 1: materialize the canonical random-200 subsets
├── eval_g2vlm_bench.py   # Stage 2: G2VLM inference harness (frame sampling + prompts)
├── metrics.py            # Stage 3: VSI/VSTI scoring, copied verbatim from the benchmarks
├── score_results.py      # Stage 3: score a predictions file -> per-type + overall
├── debug_frames.py       # diagnostic: frame-count sweep
├── debug_pixel.py        # diagnostic: resolution sweep at 32 frames
├── vsi_random200.json    # canonical VSI subset (200 docs)
├── vsti_random200.json   # canonical VSTI subset (200 docs)
├── preds_<bench>_<k>f_*.json   # raw predictions for the four runs
├── RESULTS.md            # full per-type result tables + analysis
└── README.md
../slurm_scripts/run_bench_eval.sh   # A100 SLURM wrapper (inference + auto-score)
```

## How to reproduce

**Stage 1 — subsets (run in the baseline envs so subsets match exactly).** `datasets` is
not in `g2vlm_env`; use the benchmark envs:

```bash
# VSI (envs/spatialstack-qwen35, datasets 3.6.0)
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  /cluster/work/igp_psr/spanwar/envs/spatialstack-qwen35/bin/python dump_subset.py \
  --benchmark vsi  --out vsi_random200.json

# VSTI (envs/vlm3r_env, datasets 2.16.1)
HF_HOME=/cluster/scratch/spanwar/datasets/eval_cache HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  /cluster/work/igp_psr/spanwar/envs/vlm3r_env/bin/python dump_subset.py \
  --benchmark vsti --out vsti_random200.json
```

**Stage 2+3 — inference + scoring on an A100** (runs in `g2vlm_env`; the SLURM script
auto-scores at the end):

```bash
# args: <vsi|vsti> [limit, 0=all] [num_frames, default 32]
sbatch slurm_scripts/run_bench_eval.sh vsi  0 32
sbatch slurm_scripts/run_bench_eval.sh vsi  0 8
sbatch slurm_scripts/run_bench_eval.sh vsti 0 32
sbatch slurm_scripts/run_bench_eval.sh vsti 0 8
```

Score an existing predictions file manually:

```bash
python score_results.py --predictions preds_vsi_8f_<ts>.json --benchmark vsi
```

## Operational notes

- The student QOS (`es_schin/BAUG-IGP-PRS-HPC-STUDENTS`) caps ~4 CPU / 24 GiB / 1 GPU per
  user, so the jobs run **one at a time** (use `--mem-per-cpu=6G`).
- The inference process occasionally **hangs on CUDA exit after writing all predictions**.
  Predictions are checkpointed after every sample, so just score the JSON with
  `score_results.py` and `scancel` the hung job.
