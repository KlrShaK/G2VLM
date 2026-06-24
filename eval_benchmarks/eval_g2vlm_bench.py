"""Run G2VLM on a materialized VSI/VSTI random-200 subset and save predictions.

GPU step. Reuses the G2VLM inference API exactly as inference_chat.py does, and
builds the prompt exactly as the benchmarks' `vsibench_doc_to_text` does, so the
only thing that changes vs the baselines is the model.

  python eval_g2vlm_bench.py --subset vsi_random200.json --benchmark vsi \
      --out preds_vsi.json --num-frames 32
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

# repo root (parent of eval_benchmarks/) for g2vlm_utils + modeling imports
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from g2vlm_utils import (  # noqa: E402
    build_transform,
    load_model_and_tokenizer,
    process_conversation,
)

MCA_TYPES = {
    "vsi": {
        "object_rel_direction_easy", "object_rel_direction_medium",
        "object_rel_direction_hard", "object_rel_distance", "route_planning",
        "obj_appearance_order",
    },
    "vsti": {
        "obj_obj_relative_pos_nf", "obj_obj_relative_pos_ud",
        "obj_obj_relative_pos_lr", "camera_obj_rel_dist_v1",
        "camera_obj_rel_dist_v2", "camera_obj_rel_dist_v3",
        "camera_movement_direction",
    },
}

PRE_PROMPT = "These are frames of a video."
MCA_POST = "Answer with the option's letter from the given choices directly."
NA_POST = "Please answer the question using a single word or phrase."


def build_prompt(doc, benchmark):
    """Verbatim replica of vsibench_doc_to_text (default lmms_eval_specific_kwargs)."""
    question = doc["question"]
    if doc["question_type"] in MCA_TYPES[benchmark]:
        options = "Options:\n" + "\n".join(doc["options"])
        return "\n".join([PRE_PROMPT, question, options, MCA_POST])
    return PRE_PROMPT + "\n" + question + "\n" + NA_POST


def load_frames(video_path, num_frames):
    """Uniform `num_frames` sampling via decord -- matches lmms-eval load_video."""
    import decord

    vr = decord.VideoReader(video_path, num_threads=1)
    total = len(vr)
    idx = np.linspace(0, total - 1, num_frames, dtype=int)
    batch = vr.get_batch(idx.tolist()).asnumpy()  # (T, H, W, 3)
    return [Image.fromarray(batch[i]).convert("RGB") for i in range(batch.shape[0])]


def save(out_path, payload):
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--benchmark", choices=["vsi", "vsti"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-frames", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=32)
    ap.add_argument("--pixel", type=int, default=768)
    ap.add_argument(
        "--model-path",
        default="/cluster/work/igp_psr/spanwar/G2VLM/models/G2VLM-2B-MoT",
    )
    ap.add_argument("--limit", type=int, default=0, help="0=all; else first N (smoke test)")
    args = ap.parse_args()

    docs = json.load(open(args.subset))
    if args.limit:
        docs = docs[: args.limit]

    # resume support: keep already-finished ids
    done = {}
    if os.path.exists(args.out):
        try:
            prev = json.load(open(args.out))
            for r in prev.get("results", []):
                done[str(r["id"])] = r
            print(f"Resuming: {len(done)} already done")
        except Exception:  # noqa: BLE001
            pass

    print("Loading model...")
    model, tokenizer, new_token_ids, _vit_t, dino_transform = load_model_and_tokenizer(args)
    image_transform = build_transform(pixel=args.pixel, model_path=args.model_path)
    print(f"Model loaded ({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)")

    metadata = {
        "model_path": args.model_path,
        "benchmark": args.benchmark,
        "subset": os.path.abspath(args.subset),
        "num_frames": args.num_frames,
        "pixel": args.pixel,
        "max_length": args.max_length,
    }
    results = list(done.values())
    frame_cache = {}

    for i, doc in enumerate(docs):
        did = str(doc["id"])
        if did in done:
            continue
        vp = doc["video_path"]
        try:
            if vp not in frame_cache:
                frame_cache[vp] = load_frames(vp, args.num_frames)
            frames = frame_cache[vp]
        except Exception as e:  # noqa: BLE001
            print(f"  [{i+1}/{len(docs)}] id={did} FRAME-FAIL {vp}: {e}")
            prediction = ""
        else:
            prompt = build_prompt(doc, args.benchmark)
            imgs, conversation = process_conversation(frames, prompt)
            t0 = time.time()
            prediction = model.chat_with_recon(
                tokenizer, new_token_ids, image_transform, dino_transform,
                images=imgs, prompt=conversation,
                max_length=args.max_length, do_sample=False,
            )
            print(f"  [{i+1}/{len(docs)}] id={did} {doc['question_type']} "
                  f"({time.time()-t0:.1f}s) -> {prediction!r}")

        results.append({
            "id": doc["id"],
            "benchmark": args.benchmark,
            "question_type": doc["question_type"],
            "question": doc["question"],
            "options": doc["options"],
            "ground_truth": doc["ground_truth"],
            "mc_answer": doc["mc_answer"],
            "prediction": prediction,
        })
        # free the cache once the last question for a video is consumed
        save(args.out, {"metadata": metadata, "results": results})

    print(f"Done: {len(results)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
